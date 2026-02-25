# recbole/model/sequential_recommender/conformer4rec.py

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeedForward


class ConformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, conv_kernel_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Multi-head Self-Attention (with causal mask)
        self.mha_norm = nn.LayerNorm(hidden_size)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_raises,
            dropout=dropout,
            batch_first=True
        )

        # 2. Convolution Module
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.conv_module = ConvolutionModule(hidden_size, conv_kernel_size, dropout)

        # 3. Feed-Forward (2 lần trong Conformer gốc, nhưng ta dùng 1 lần để gọn)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # MHSA
        residual = x
        x = self.mha_norm(x)
        # Apply causal mask
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        # Conv
        residual = x
        x = self.conv_norm(x)
        x = self.conv_module(x)
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


class ConvolutionModule(nn.Module):
    def __init__(self, hidden_size, kernel_size, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv1 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # keep length
            groups=hidden_size
        )
        self.glu = nn.GLU(dim=1)  # Gated Linear Unit
        self.pointwise_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x = self.layer.norm(x)
        x = x.transpose(1, 2)  # [B, D, L]

        x = self.pointwise_conv1(x)  # [B, 2D, L]
        x = self.glu(x)           # [B, D, L]

        x = self.depthwise_conv(x)  # [B, D, L]
        x = x.transpose(1, 2)       # [B, L, D]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(1, 2)       # [B, D, L]

        x = self.pointwise_conv2(x)  # [B, D, L]
        x = self.dropout(x)
        x = x.transpose(1, 2)        # [B, L, D]
        return x


class Conformer4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Load config
        self.hidden_size = config['hidden_size']
        self.num_layers = config['n_layers']
        self.num_heads = config['n_heads']
        self.dropout = config['dropout']
        self.kernel_size = config.get('conv_kernel_size', 31)  # default large kernel

        # Embeddings
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config['max_seq_length'], self.hidden_size)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(self.hidden_size, self.num_heads, self.kernel_size, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        batch_size, seq_len = item_seq.shape

        # Embeddings
        item_emb = self.item_embedding(item_seq)  # [B, L, D]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        pos_emb = self.position_embedding(pos_ids.unsqueeze(0))  # [1, L, D]
        x = item_emb + pos_emb
        x = self.dropout_layer(x)
        x = self.layer_norm(x)

        # Causal attention mask (lower triangular)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()  # [L, L]

        # Pass through Conformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Use last item's representation for prediction
        seq_output = x[:, -1, :]  # [B, D]
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores