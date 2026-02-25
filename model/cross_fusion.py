"""
Fusion & Aggregation modules cho DualMamba4Rec.

Gồm 2 class:
    AdaptiveGateFusion     : Kết hợp H^A (Mamba) và H^B (GNN) bằng learnable gate.
                             Không dùng cross-attention — đưa trực tiếp 2 output vào gate.
    SessionAttentionAggregator : Soft-attention aggregation thay cho last-item (CMamba4Rec).

Architecture của AdaptiveGateFusion:
    H^A, H^B
       │
       ├── cat([H^A, H^B]) → Linear(2d → d) → Sigmoid → gate
       │
       └── H^fused = gate ⊙ H^A + (1 - gate) ⊙ H^B
                   → LayerNorm
                   → FFN + residual + LayerNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGateFusion(nn.Module):
    """
    Kết hợp output của Mamba stream (H^A) và GNN stream (H^B)
    bằng learnable sigmoid gate — không qua cross-attention.

    Công thức:
        gate    = sigmoid(Linear(cat([H^A, H^B])))   # (B, L, d)
        H^fused = gate ⊙ H^A + (1 - gate) ⊙ H^B

    Gate phụ thuộc vào nội dung cả hai nguồn tại từng position,
    cho phép model tự học mức độ tin tưởng vào mỗi channel.

    Args:
        d_model (int)  : Feature dimension.
        dropout (float): Dropout sau fusion.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # Gate: nhìn vào cả hai nguồn để quyết định trọng số
        self.gate_linear = nn.Linear(2 * d_model, d_model, bias=True)
        # Init gần 0 → ban đầu gate ≈ 0.5 (balanced giữa hai nguồn)
        nn.init.normal_(self.gate_linear.weight, std=0.01)
        nn.init.constant_(self.gate_linear.bias, 0.0)

        # Post-fusion
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h_a: torch.Tensor,  # (B, L, d_model) — Mamba stream
        h_b: torch.Tensor,  # (B, L, d_model) — GNN stream
    ) -> torch.Tensor:
        """
        Returns:
            h_fused: (B, L, d_model)
        """
        # Gate dựa trên cả hai nguồn
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([h_a, h_b], dim=-1))
        )  # (B, L, d_model)

        # Gated interpolation
        h = gate * h_a + (1 - gate) * h_b  # (B, L, d_model)

        # Residual + Norm
        h = self.norm1(h + h_a)             # residual từ H^A (primary stream)

        # FFN + Residual + Norm
        h = self.norm2(h + self.ffn(h))

        return h


class SessionAttentionAggregator(nn.Module):
    """
    Attention-based session aggregation (từ CMamba4Rec paper).

    Thay vì chỉ lấy last-item, soft-attention weight tất cả items
    dựa trên độ liên quan với last item (current intent).

    Công thức:
        α_i = v^T · σ(W1·h_last + W2·h_i)
        s   = Σ_i α_i · h_i

    Args:
        d_model (int): Feature dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.v  = nn.Linear(d_model, 1,       bias=False)

    def forward(
        self,
        h: torch.Tensor,            # (B, L, d_model)
        seq_len: torch.Tensor,      # (B,) — actual lengths
        padding_mask: torch.Tensor = None,  # (B, L) — True = padding
    ) -> torch.Tensor:
        """
        Returns:
            s: (B, d_model) — session representation
        """
        B, L, d = h.shape

        # Lấy last non-padding item
        last_idx = (seq_len - 1).clamp(min=0)               # (B,)
        h_last = h.gather(
            dim=1,
            index=last_idx.view(B, 1, 1).expand(B, 1, d)
        ).squeeze(1)                                          # (B, d)

        # Attention score
        score = self.v(
            torch.sigmoid(self.W1(h_last).unsqueeze(1) + self.W2(h))
        ).squeeze(-1)                                         # (B, L)

        if padding_mask is not None:
            score = score.masked_fill(padding_mask, float('-inf'))

        alpha = torch.nan_to_num(F.softmax(score, dim=-1), nan=0.0)  # (B, L)

        return (alpha.unsqueeze(-1) * h).sum(dim=1)          # (B, d_model)