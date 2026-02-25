"""
GNN Stream cho DualMamba4Rec.

Pipeline:
    1. GraphBuilder: đọc toàn bộ training interactions,
       xây dựng Global Item Transition Graph (directed, weighted)
       
    2. LightGCNPropagation: với mỗi session trong batch,
       lấy subgraph liên quan và propagate features qua L hops
       
    3. GNNStream: wrapper kết hợp 2 thành phần trên

Về Global Graph:
    - Node: mỗi item ID là một node
    - Edge (u → v): u được click ngay trước v trong session nào đó
    - Weight w_{uv} = count(u→v) / sum_w count(u→w)  [normalized transition prob]
    
Về LightGCN trên session subgraph:
    - Chỉ xét các node trong session hiện tại + direct neighbors trong global graph
    - L=2 hops propagation
    - Final representation: mean pooling qua các layer
    - Không dùng self-loop (để phân biệt rõ ràng thông tin local vs propagated)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple


class GlobalGraphBuilder:
    """
    Xây dựng Global Item Transition Graph từ tập huấn luyện.
    
    Graph được lưu dưới dạng sparse tensors để hiệu quả,
    không phụ thuộc vào DGL cho phần storage.
    
    Attributes:
        n_items (int): Tổng số items trong vocabulary
        edge_index (Tensor): (2, E) — [src; dst] indices
        edge_weight (Tensor): (E,) — normalized transition probabilities
    """

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self._is_built = False

        # Adjacency list dạng dict để build nhanh
        # adj[u][v] = count(u → v)
        self._adj_count: dict = defaultdict(lambda: defaultdict(int))

    def build_from_interaction_matrix(
        self,
        interaction_matrix,
        uid_field: str,
        iid_field: str,
        seq_field: str,
        device: torch.device,
    ):
        """
        Xây dựng graph từ RecBole interaction matrix (training data).
        
        RecBole lưu session data theo dạng:
            - interaction[seq_field]: (N, max_len) — padded item sequences
            - Hoặc interaction[iid_field]: item IDs nếu dạng flat
        
        Args:
            interaction_matrix: RecBole dataset object
            device: target device
        """
        print("[GraphBuilder] Building global item transition graph...")

        # Lấy tất cả interaction sequences từ dataset
        if seq_field in interaction_matrix.field2type:
            # Session-based: mỗi sample là một sequence
            seqs = interaction_matrix[seq_field]  # (N, max_len) tensor
            if isinstance(seqs, torch.Tensor):
                self._build_from_sequences_tensor(seqs)
            else:
                self._build_from_sequences_list(seqs)
        else:
            print("[GraphBuilder] Warning: seq_field not found, using user-grouped interactions")
            self._build_from_user_interactions(interaction_matrix, uid_field, iid_field)

        self._finalize(device)

    def build_from_sequences(
        self,
        sequences,  # list of lists or 2D tensor
        device: torch.device,
    ):
        """
        Build trực tiếp từ list of sequences.
        Dùng khi cần control thủ công.
        
        Args:
            sequences: list of list of item_ids (int), hoặc 2D tensor (padded với 0)
            device: target device
        """
        print("[GraphBuilder] Building from provided sequences...")
        if isinstance(sequences, torch.Tensor):
            self._build_from_sequences_tensor(sequences)
        else:
            self._build_from_sequences_list(sequences)
        self._finalize(device)

    def _build_from_sequences_tensor(self, seqs: torch.Tensor):
        """seqs: (N, max_len), padding = 0"""
        seqs = seqs.cpu().numpy() if isinstance(seqs, torch.Tensor) else seqs
        for seq in seqs:
            # Filter padding (item id = 0 là padding trong RecBole)
            items = [int(x) for x in seq if x > 0]
            for i in range(len(items) - 1):
                u, v = items[i], items[i + 1]
                if u != v:  # tránh self-loop
                    self._adj_count[u][v] += 1

    def _build_from_sequences_list(self, seqs):
        """seqs: list of list"""
        for seq in seqs:
            items = [int(x) for x in seq if x > 0]
            for i in range(len(items) - 1):
                u, v = items[i], items[i + 1]
                if u != v:
                    self._adj_count[u][v] += 1

    def _build_from_user_interactions(self, dataset, uid_field, iid_field):
        """Fallback: group by user và tạo transition từ consecutive clicks"""
        from collections import defaultdict as dd
        user_items = dd(list)
        uids = dataset[uid_field].cpu().numpy()
        iids = dataset[iid_field].cpu().numpy()
        for uid, iid in zip(uids, iids):
            user_items[uid].append(int(iid))
        for uid, items in user_items.items():
            for i in range(len(items) - 1):
                u, v = items[i], items[i + 1]
                if u != v:
                    self._adj_count[u][v] += 1

    def _finalize(self, device: torch.device):
        """Chuyển adj_count thành edge tensors với normalized weights."""
        if not self._adj_count:
            print("[GraphBuilder] Warning: empty graph!")
            self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            self.edge_weight = torch.zeros(0, device=device)
            self._is_built = True
            return

        src_list, dst_list, weight_list = [], [], []

        for u, neighbors in self._adj_count.items():
            total = sum(neighbors.values())
            for v, count in neighbors.items():
                src_list.append(u)
                dst_list.append(v)
                weight_list.append(count / total)  # normalize

        self.edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long, device=device
        )  # (2, E)
        self.edge_weight = torch.tensor(weight_list, dtype=torch.float32, device=device)

        n_edges = len(src_list)
        n_nodes = len(self._adj_count)
        print(f"[GraphBuilder] Graph built: {n_nodes} source nodes, {n_edges} edges")

        # Giải phóng memory
        self._adj_count.clear()
        self._is_built = True

    def get_neighbor_mask(self, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lấy subgraph liên quan đến item_ids trong session.
        
        Args:
            item_ids: (n_items_in_session,) — unique item IDs
        Returns:
            sub_src: source nodes của subgraph edges
            sub_dst: destination nodes của subgraph edges  
            sub_weight: edge weights của subgraph
        """
        if not self._is_built or self.edge_index.shape[1] == 0:
            return None, None, None

        # Tạo mask: giữ edges có src HOẶC dst trong item_ids
        item_set = set(item_ids.cpu().tolist())
        src = self.edge_index[0]  # (E,)
        dst = self.edge_index[1]  # (E,)

        # Boolean mask
        src_mask = torch.isin(src, item_ids.to(src.device))
        dst_mask = torch.isin(dst, item_ids.to(dst.device))
        edge_mask = src_mask | dst_mask  # 1-hop neighborhood

        if edge_mask.sum() == 0:
            return None, None, None

        sub_src = src[edge_mask]
        sub_dst = dst[edge_mask]
        sub_weight = self.edge_weight[edge_mask]

        return sub_src, sub_dst, sub_weight


class LightGCNLayer(nn.Module):
    """
    Một layer LightGCN: h_v = sum_{u∈N(v)} w_{uv} * h_u
    
    LightGCN loại bỏ self-loop, feature transformation và non-linearity
    để tránh over-smoothing và giảm parameters.
    
    Propagation:
        H^{(l+1)} = Â · H^{(l)}
    
    Trong đó Â = D^{-1/2} A D^{-1/2} (symmetric normalized)
    hoặc Â = D^{-1} A (row-normalized, dùng cho directed graph)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,         # (N, d) — node features
        edge_index: torch.Tensor, # (2, E) — [src, dst]
        edge_weight: torch.Tensor, # (E,) — edge weights
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x: node feature matrix (N, d)
            edge_index: (2, E)
            edge_weight: (E,)
            n_nodes: total number of nodes N
        Returns:
            x_new: (N, d) — propagated features
        """
        if edge_index.shape[1] == 0:
            return x

        src, dst = edge_index[0], edge_index[1]  # each (E,)

        # Aggregation: h_dst += w * h_src
        # Dùng scatter_add để vectorize
        # x[src]: (E, d) — source features
        # weight: (E, 1) — broadcast
        msg = x[src] * edge_weight.unsqueeze(-1)  # (E, d)

        # Scatter to destination nodes
        out = torch.zeros_like(x)  # (N, d)
        dst_expanded = dst.unsqueeze(-1).expand(-1, x.shape[-1])  # (E, d)
        out.scatter_add_(0, dst_expanded, msg)

        return out


class GNNStream(nn.Module):
    """
    GNN Channel trong DualMamba4Rec.
    
    Với mỗi batch:
    1. Lấy item embeddings từ shared embedding table
    2. Propagate qua L layers LightGCN trên global graph
    3. Lấy representations của các items trong session
    
    Kết quả: H^B — enriched item representations với
    cross-session collaborative information.
    
    Args:
        n_items (int): Vocabulary size (số lượng items)
        d_model (int): Embedding dimension
        n_layers (int): Số LightGCN layers
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        n_items: int,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.n_layers = n_layers

        # LightGCN layers
        self.gnn_layers = nn.ModuleList([
            LightGCNLayer() for _ in range(n_layers)
        ])

        # Layer norm cho mỗi GCN output
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Projection để align với Mamba stream nếu cần
        # (thực ra không cần vì cùng d_model, nhưng để linh hoạt)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.out_proj.weight)  # init as identity

        self.dropout = nn.Dropout(dropout)

        # Global graph (được set từ bên ngoài sau khi build)
        self.graph: Optional[GlobalGraphBuilder] = None

    def set_graph(self, graph: GlobalGraphBuilder):
        """Inject global graph vào GNN stream."""
        self.graph = graph

    def forward(
        self,
        session_items: torch.Tensor,  # (B, L) — item IDs (padded với 0)
        item_embeddings: torch.Tensor,  # (n_items+1, d_model) — toàn bộ embedding table
    ) -> torch.Tensor:
        """
        Args:
            session_items: (B, L) — batch of sessions, item IDs
            item_embeddings: (n_items+1, d_model) — full embedding table
                             (index 0 = padding)
        Returns:
            H_B: (B, L, d_model) — GNN-enriched item representations
        """
        B, L = session_items.shape
        device = session_items.device

        # ── Nếu không có graph, fallback về raw embeddings ──────────────────
        if self.graph is None or not self.graph._is_built:
            # Fallback: chỉ lookup embedding (tương đương GNN với 0 layer)
            out = item_embeddings[session_items]  # (B, L, d)
            return self.dropout(out)

        # ── Propagate trên global graph ──────────────────────────────────────
        # Global propagation: không giới hạn theo batch — dùng toàn bộ graph
        # Điều này tính ONCE per forward pass, sau đó lookup
        
        # NOTE: Để hiệu quả, ta propagate trên TOÀN BỘ item embedding table
        # (không chỉ items trong batch). Điều này cho phép vectorize toàn bộ.
        # Với n_items lớn (77K), đây là tradeoff memory vs speed.
        # Alternative: chỉ propagate subgraph của batch (xem _forward_subgraph)
        
        if self.n_items <= 50000:
            # Full graph propagation (với item vocab nhỏ/vừa)
            H = self._forward_full_graph(item_embeddings, device)
        else:
            # Subgraph propagation (với item vocab lớn như Tmall 77K)
            H = self._forward_subgraph_batched(
                session_items, item_embeddings, device
            )

        # ── Lookup embeddings cho items trong session ────────────────────────
        H_B = H[session_items]  # (B, L, d_model)
        H_B = self.dropout(H_B)

        return H_B

    def _forward_full_graph(
        self,
        item_embeddings: torch.Tensor,  # (n_items+1, d)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Propagate trên toàn bộ global graph.
        
        Returns:
            H_final: (n_items+1, d_model) — final item representations
        """
        edge_index = self.graph.edge_index  # (2, E)
        edge_weight = self.graph.edge_weight  # (E,)
        n_nodes = item_embeddings.shape[0]

        H = item_embeddings  # (n_items+1, d) — starting point
        layer_outputs = [H]  # để mean pooling sau

        for i, gnn_layer in enumerate(self.gnn_layers):
            H_new = gnn_layer(H, edge_index, edge_weight, n_nodes)
            H_new = self.layer_norms[i](H_new)
            H = H_new
            layer_outputs.append(H)

        # Mean pooling qua các layers (LightGCN style)
        H_final = torch.stack(layer_outputs, dim=0).mean(dim=0)  # (n_items+1, d)
        H_final = self.out_proj(H_final)

        return H_final

    def _forward_subgraph_batched(
        self,
        session_items: torch.Tensor,  # (B, L)
        item_embeddings: torch.Tensor,  # (n_items+1, d)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Subgraph propagation cho vocab lớn.
        Chỉ xây dựng subgraph từ unique items trong batch.
        
        Trade-off: Tiết kiệm memory nhưng không tận dụng đầy đủ 
        global collaborative signal như _forward_full_graph.
        
        Returns:
            H_out: (n_items+1, d_model) — partial update (chỉ batch items được cập nhật)
        """
        # Lấy unique items trong batch
        unique_items = session_items.unique()  # (M,)
        unique_items = unique_items[unique_items > 0]  # remove padding

        # Lấy subgraph edges liên quan đến batch items
        sub_src, sub_dst, sub_weight = self.graph.get_neighbor_mask(unique_items)

        # Bắt đầu từ full embedding table (sẽ update in-place)
        H = item_embeddings.clone()  # (n_items+1, d)
        layer_outputs = [H]

        if sub_src is not None:
            sub_edge_index = torch.stack([sub_src, sub_dst], dim=0)  # (2, E_sub)

            for i, gnn_layer in enumerate(self.gnn_layers):
                H_new = gnn_layer(H, sub_edge_index, sub_weight, H.shape[0])
                # Chỉ update các node liên quan
                involved = torch.cat([sub_src, sub_dst]).unique()
                H_update = H.clone()
                H_update[involved] = self.layer_norms[i](H_new[involved])
                H = H_update
                layer_outputs.append(H)

        H_final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        H_final = self.out_proj(H_final)

        return H_final
