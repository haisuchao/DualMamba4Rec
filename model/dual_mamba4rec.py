"""
DualMamba4Rec — Main Model

Ba chế độ chạy qua tham số `channel_mode` trong config:

    'mamba_only' : Chỉ Channel A (C-Mamba). Tương đương CMamba4Rec gốc.
                   Dùng để verify kết quả so với paper và baseline.

    'gnn_only'   : Chỉ Channel B (GNN Stream).
                   Dùng để đo đóng góp riêng của GNN.

    'dual'       : Cả hai channel, kết hợp qua AdaptiveGateFusion. (default)
                   Đây là mô hình đầy đủ DualMamba4Rec.

Lưu ý:
    - 'mamba_only' và 'gnn_only': không khởi tạo lớp fusion → ít params hơn.
    - 'dual': khởi tạo cả hai channel và AdaptiveGateFusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from collections import defaultdict

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from model.cmamba_block import CMambaEncoder
from model.gnn_stream import GNNStream, GlobalGraphBuilder
from model.cross_fusion import AdaptiveGateFusion, SessionAttentionAggregator

VALID_MODES = ('mamba_only', 'gnn_only', 'dual')


class DualMamba4Rec(SequentialRecommender):
    """
    DualMamba4Rec với 3 chế độ chạy.

    Config YAML:
        channel_mode: dual        # 'mamba_only' | 'gnn_only' | 'dual'

        # Chung
        embedding_size: 64
        max_seq_length: 50
        dropout_prob: 0.3
        loss_type: CE

        # Channel A — C-Mamba (dùng khi mode = 'mamba_only' hoặc 'dual')
        n_layers: 2
        d_state: 32
        d_conv_mamba: 4
        expand: 2
        conv_kernel: 3
        ffn_multiplier: 4

        # Channel B — GNN (dùng khi mode = 'gnn_only' hoặc 'dual')
        gnn_layers: 2
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ── Mode ─────────────────────────────────────────────────────────────
        self.channel_mode = config.get('channel_mode', 'dual')
        if self.channel_mode not in VALID_MODES:
            raise ValueError(
                f"channel_mode phải là một trong {VALID_MODES}, "
                f"nhận được: '{self.channel_mode}'"
            )

        # ── Hyperparameters ───────────────────────────────────────────────────
        self.d_model      = config['embedding_size']
        self.dropout_prob = config.get('dropout_prob', 0.3)
        self.loss_type    = config.get('loss_type', 'CE')

        # ── Shared Embedding ──────────────────────────────────────────────────
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length + 1, self.d_model
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        # ── Channel A: C-Mamba ────────────────────────────────────────────────
        if self.channel_mode in ('mamba_only', 'dual'):
            self.cmamba_encoder = CMambaEncoder(
                n_layers     = config.get('n_layers', 2),
                d_model      = self.d_model,
                d_state      = config.get('d_state', 32),
                d_conv_mamba = config.get('d_conv_mamba', 4),
                expand       = config.get('expand', 2),
                conv_kernel  = config.get('conv_kernel', 3),
                ffn_multiplier = config.get('ffn_multiplier', 4),
                dropout      = self.dropout_prob,
            )

        # ── Channel B: GNN ────────────────────────────────────────────────────
        if self.channel_mode in ('gnn_only', 'dual'):
            self.gnn_stream = GNNStream(
                n_items  = self.n_items,
                d_model  = self.d_model,
                n_layers = config.get('gnn_layers', 2),
                dropout  = self.dropout_prob,
            )
            self.graph_builder = GlobalGraphBuilder(n_items=self.n_items)
            self.gnn_stream.set_graph(self.graph_builder)

        # ── Fusion (chỉ khi dual) ─────────────────────────────────────────────
        if self.channel_mode == 'dual':
            self.fusion = AdaptiveGateFusion(
                d_model = self.d_model,
                dropout = self.dropout_prob,
            )

        # ── Session Aggregation & Output ──────────────────────────────────────
        self.session_aggregator = SessionAttentionAggregator(self.d_model)
        self.final_norm = nn.LayerNorm(self.d_model)

        # ── Loss ──────────────────────────────────────────────────────────────
        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError(f"loss_type '{self.loss_type}' không hỗ trợ")

        self.apply(self._init_weights)
        self._log_mode()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            constant_(module.bias, 0)

    def _log_mode(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_desc = {
            'mamba_only': 'Channel A only  (C-Mamba) — tương đương CMamba4Rec',
            'gnn_only'  : 'Channel B only  (GNN Stream)',
            'dual'      : 'Channel A + B   (C-Mamba + GNN + AdaptiveGateFusion)',
        }
        print(f"\n[DualMamba4Rec] Mode : {mode_desc[self.channel_mode]}")
        print(f"[DualMamba4Rec] Params: {n_params:,}\n")

    # ── Graph Pre-building ────────────────────────────────────────────────────

    def pre_build_graph(self, dataset):
        """
        Xây dựng Global Item Transition Graph từ training data.
        Chỉ cần gọi khi channel_mode = 'gnn_only' hoặc 'dual'.
        Nếu mode = 'mamba_only', hàm này không làm gì.
        """
        if self.channel_mode == 'mamba_only':
            print("[DualMamba4Rec] mode='mamba_only' — bỏ qua build graph.\n")
            return

        print("[DualMamba4Rec] Building global item transition graph...")
        device = next(self.parameters()).device

        try:
            inter_feat = dataset.inter_feat
            if self.ITEM_SEQ in inter_feat:
                self.graph_builder.build_from_sequences(
                    inter_feat[self.ITEM_SEQ], device=device
                )
            else:
                items = inter_feat[self.ITEM_ID]
                users = inter_feat.get(self.USER_ID)
                seqs = (
                    self._group_by_user(users, items)
                    if users is not None
                    else [items.tolist()]
                )
                self.graph_builder.build_from_sequences(seqs, device=device)
        except Exception as e:
            print(f"[DualMamba4Rec] Warning: Graph building failed ({e})")

        print("[DualMamba4Rec] Graph ready.\n")

    def _group_by_user(self, users, items):
        user_items = defaultdict(list)
        for u, i in zip(users.tolist(), items.tolist()):
            user_items[u].append(i)
        return list(user_items.values())

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq    : (B, L) — padded item ID sequences
            item_seq_len: (B,)   — actual lengths
        Returns:
            s: (B, d_model) — session representation
        """
        B, L = item_seq.shape
        padding_mask = (item_seq == 0)  # (B, L)

        # Embedding
        pos_ids = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        h0 = self.emb_dropout(
            self.item_embedding(item_seq) + self.position_embedding(pos_ids)
        )  # (B, L, d)

        # ── Chạy theo mode ────────────────────────────────────────────────────
        if self.channel_mode == 'mamba_only':
            h = self.cmamba_encoder(h0)                         # (B, L, d)

        elif self.channel_mode == 'gnn_only':
            h = self.gnn_stream(
                session_items    = item_seq,
                item_embeddings  = self.item_embedding.weight,
            )                                                    # (B, L, d)

        else:  # dual
            h_a = self.cmamba_encoder(h0)                       # (B, L, d)
            h_b = self.gnn_stream(
                session_items    = item_seq,
                item_embeddings  = self.item_embedding.weight,
            )                                                    # (B, L, d)
            h = self.fusion(h_a, h_b)                           # (B, L, d)

        # Session aggregation
        s = self.session_aggregator(h, item_seq_len, padding_mask)  # (B, d)
        return self.final_norm(s)

    # ── RecBole Interface ─────────────────────────────────────────────────────

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items    = interaction[self.POS_ITEM_ID]

        s = self.forward(item_seq, item_seq_len)

        if self.loss_type == 'CE':
            # weight[1:] bỏ padding index 0 → (n_items, d)
            # pos_items là 1-based → trừ 1 để align
            scores = torch.matmul(s, self.item_embedding.weight[1:].T)  # (B, n_items)
            return self.loss_fct(scores, pos_items - 1)
        else:  # BPR
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = (s * self.item_embedding(pos_items)).sum(-1)
            neg_score = (s * self.item_embedding(neg_items)).sum(-1)
            return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction) -> torch.Tensor:
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item    = interaction[self.ITEM_ID]

        s = self.forward(item_seq, item_seq_len)
        return (s * self.item_embedding(test_item)).sum(-1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        s = self.forward(item_seq, item_seq_len)                # (B, d)
        # weight[1:] → (n_items, d), bỏ padding index 0
        return torch.matmul(s, self.item_embedding.weight[1:].T)  # (B, n_items)