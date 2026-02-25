"""
C-Mamba Block — tái hiện từ CMamba4Rec paper.

Module này gồm hai phần:
    1. MambaBlock  : Wrapper của mamba_ssm.Mamba với pre-LayerNorm và residual.
    2. CMambaBlock : Conv1D → MambaBlock → DS Gate → FFN (building block chính).
    3. CMambaEncoder: Stack L × CMambaBlock.

Architecture trong mỗi C-Mamba Block:
    Input H^{l-1}
        │
        ├── Conv1D (local context extraction)  ← "C" trong C-Mamba
        │       │
        │       └──→ C^l  ──→  MambaBlock  ──→  M^l
        │                               │
        └── DS Gate (noise filtering)──→ Z^l = D ⊙ M + M
                                            │
                                          FFN + LayerNorm
                                            │
                                          H^l (output)

Về mamba_ssm.Mamba (thư viện chính thức, CUDA-optimized):
    Yêu cầu : CUDA >= 11.6, PyTorch >= 1.12, causal-conv1d >= 1.1.0
    Cài đặt : pip install causal-conv1d mamba-ssm
    Nguồn   : https://github.com/state-spaces/mamba

    mamba_ssm.Mamba là một COMPLETE BLOCK, bao gồm:
        - in_proj  : Linear(d_model → 2*d_inner)
        - conv1d   : depthwise Conv1d causal (kernel = d_conv)
        - x_proj   : Linear(d_inner → dt_rank + 2*d_state)
        - dt_proj  : Linear(dt_rank → d_inner)
        - out_proj : Linear(d_inner → d_model)
        - A_log, D : SSM learnable parameters
    Nó KHÔNG có LayerNorm và residual — MambaBlock thêm hai thứ này bên ngoài.

Phân biệt hai Conv1D trong CMambaBlock:
    conv_kernel  → Local Context Conv1D (bên NGOÀI, trước MambaBlock):
                    Mục đích: ngữ cảnh hóa item embeddings từ hàng xóm
                    Đây là đóng góp chính của CMamba4Rec paper
    d_conv_mamba → Mamba's internal depthwise conv1d (bên TRONG mamba_ssm.Mamba):
                    Mục đích: local mixing trong SSM pipeline
                    Mặc định = 4 theo paper Mamba gốc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


# ══════════════════════════════════════════════════════════════════════════════
# MambaBlock — Wrapper của mamba_ssm.Mamba
# ══════════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    Wrapper của mamba_ssm.Mamba với pre-LayerNorm, residual connection và Dropout.

    mamba_ssm.Mamba xử lý toàn bộ SSM logic (in_proj, causal conv1d,
    selective scan, SiLU gating, out_proj) bằng CUDA kernel tối ưu.
    Class này bổ sung thêm:
        - Pre-LayerNorm  : normalize input trước khi vào Mamba
        - Residual       : x + Mamba(norm(x))
        - Dropout        : regularization sau Mamba

    Architecture:
        x ──→ LayerNorm ──→ mamba_ssm.Mamba ──→ Dropout ──→ (+x) ──→ output
        │                                                      ↑
        └──────────────────────────────────────────────────────┘ (residual)

    Args:
        d_model      (int)  : Input/output dimension.
        d_state      (int)  : SSM state dimension N. CMamba4Rec dùng 32.
        d_conv       (int)  : Kernel size của depthwise conv1d bên trong Mamba. Thường = 4.
        expand       (int)  : Expansion factor. d_inner = expand * d_model. Thường = 2.
        dropout      (float): Dropout rate áp dụng sau Mamba, trước residual.
        bias         (bool) : Bias trong in_proj/out_proj của Mamba (mặc định False).
        use_fast_path (bool): Dùng CUDA fused kernel (True = nhanh hơn, yêu cầu float32).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=bias,
            use_fast_path=use_fast_path,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        return self.dropout(self.mamba(self.norm(x))) + x


# ══════════════════════════════════════════════════════════════════════════════
# DSGate, CMambaBlock, CMambaEncoder
# ══════════════════════════════════════════════════════════════════════════════

class DSGate(nn.Module):
    """
    Dense Selective Gate — lọc nhiễu và kết hợp thông tin adaptive.
    
    Từ paper CMamba4Rec:
        G = Conv1D(H^{l-1} · W0 + b0)
        δ(G) = G · W1 + b1
        D = SiLU(δ(G)) + σ(δ(G))
        Z = D ⊙ M + M                  ← residual gated fusion
    
    Args:
        d_model (int): Feature dimension
        conv_kernel (int): Kernel size for gate conv
    """

    def __init__(self, d_model: int, conv_kernel: int = 3):
        super().__init__()
        # W0, b0: linear projection trước conv
        self.linear0 = nn.Linear(d_model, d_model, bias=True)

        # Gate Conv1D
        self.gate_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,  # same padding
            groups=d_model,  # depthwise
            bias=True,
        )

        # W1, b1: final linear
        self.linear1 = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h_prev: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: (B, L, d_model) — input của block (H^{l-1})
            m:      (B, L, d_model) — output của Mamba (M^l)
        Returns:
            z: (B, L, d_model) — gated fused representation
        """
        # G = Conv1D(H^{l-1} · W0)
        g = self.linear0(h_prev)  # (B, L, d_model)
        g = self.gate_conv(g.transpose(1, 2)).transpose(1, 2)  # (B, L, d_model)

        # δ(G) = G · W1
        delta_g = self.linear1(g)  # (B, L, d_model)

        # D = SiLU(δ) + Sigmoid(δ)
        d = F.silu(delta_g) + torch.sigmoid(delta_g)  # (B, L, d_model)

        # Z = D ⊙ M + M  (gated residual)
        z = d * m + m
        return z


class CMambaBlock(nn.Module):
    """
    C-Mamba Block: Conv1D → Mamba → DS Gate → FFN

    Đây là building block chính của CMamba4Rec.
    Ta tái sử dụng trong DualMamba4Rec cho Channel A.

    Args:
        d_model      (int)  : Feature dimension.
        d_state      (int)  : SSM state dimension N (tham số của mamba_ssm.Mamba).
        d_conv_mamba (int)  : Kernel size của causal conv1d BÊN TRONG mamba_ssm.Mamba.
                              Khác với conv_kernel — đây là tham số nội tại của SSM block,
                              không phải Conv1D bên ngoài dùng để trích xuất local context.
        expand       (int)  : Expansion factor của mamba_ssm.Mamba. d_inner = expand * d_model.
        conv_kernel  (int)  : Kernel size của Conv1D BÊN NGOÀI (local context extraction).
                              Đây là "C" trong "C-Mamba", khác hoàn toàn với d_conv_mamba.
        ffn_multiplier (int): FFN hidden dim = d_model * ffn_multiplier.
        dropout      (float): Dropout rate.

    Phân biệt hai Conv1D:
        conv_kernel  → Local Context Conv1D (bên ngoài, trước Mamba):
                        Mục đích: ngữ cảnh hóa item embeddings từ hàng xóm
                        Đây là đóng góp chính của CMamba4Rec paper
        d_conv_mamba → Mamba's internal depthwise conv1d (bên trong mamba_ssm.Mamba):
                        Mục đích: local mixing trong SSM pipeline
                        Mặc định = 4 theo paper Mamba gốc
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv_mamba: int = 4,
        expand: int = 2,
        conv_kernel: int = 3,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ─── Layer Norm ──────────────────────────────────────────────────────
        # norm1: trước local conv (pre-norm cho conv)
        # norm2: trước FFN (pre-norm cho FFN)
        # Lưu ý: MambaBlock đã có norm riêng bên trong (pre-norm trước Mamba)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ─── Local Context Aggregation (Conv1D bên ngoài) ───────────────────
        # "same" padding: padding = kernel // 2 → output length = input length
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            bias=True,
        )
        self.local_norm = nn.LayerNorm(d_model)

        # ─── Global Dependency Modeling (mamba_ssm.Mamba qua MambaBlock) ────
        # MambaBlock = pre-LayerNorm + mamba_ssm.Mamba + Dropout + residual
        # mamba_ssm.Mamba nhận d_state, d_conv (=d_conv_mamba), expand
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv_mamba,   # kernel của conv1d BÊN TRONG Mamba
            expand=expand,
            dropout=dropout,
        )

        # ─── Dense Selective Gate ────────────────────────────────────────────
        self.ds_gate = DSGate(d_model=d_model, conv_kernel=conv_kernel)

        # ─── Feed-Forward Network ────────────────────────────────────────────
        d_ffn = d_model * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — input sequence
        Returns:
            h: (B, L, d_model) — output sequence

        Flow chi tiết:
            x
            │
            ├── [Step 1] pre-norm(x) → local_conv → local_norm → c  (local context)
            │   c = local_norm(Conv1D(norm1(x)) + x)
            │
            ├── [Step 2] c → MambaBlock → m
            │   MambaBlock bên trong đã làm: m = c + Mamba(norm(c))
            │   Nên m đã mang residual từ c
            │
            ├── [Step 3] DS Gate: gate dựa trên x (input gốc), áp lên m
            │   z = DSGate(h_prev=x, m=m)
            │
            └── [Step 4] pre-norm(z) → FFN → residual
                h = z + Dropout(FFN(norm2(z)))
        """
        # ── Step 1: Local Context Aggregation ───────────────────────────────
        # pre-norm → Conv1D → residual + norm
        # Conv1D nhận (B, d, L), nên cần transpose
        c = self.local_conv(
            self.norm1(x).transpose(1, 2)   # (B, d, L)
        ).transpose(1, 2)                    # (B, L, d)
        c = self.local_norm(c + x)           # residual + norm

        # ── Step 2: Mamba (global sequential modeling) ───────────────────────
        # MambaBlock: m = c + Mamba(pre_norm(c))  [residual bên trong]
        m = self.mamba(c)                    # (B, L, d_model)

        # ── Step 3: Dense Selective Gate ────────────────────────────────────
        # Gate dựa trên x gốc (không phải c hay m) — theo paper CMamba4Rec
        z = self.ds_gate(h_prev=x, m=m)     # (B, L, d_model)

        # ── Step 4: FFN + Residual ───────────────────────────────────────────
        h = z + self.dropout(self.ffn(self.norm2(z)))  # (B, L, d_model)

        return h


class CMambaEncoder(nn.Module):
    """
    Stack L × C-Mamba blocks.
    
    Args:
        n_layers (int): Số block
        d_model, d_state, ...: Truyền xuống CMambaBlock
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_state: int = 32,
        d_conv_mamba: int = 4,
        expand: int = 2,
        conv_kernel: int = 3,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv_mamba=d_conv_mamba,
                expand=expand,
                conv_kernel=conv_kernel,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            h: (B, L, d_model)
        """
        for block in self.blocks:
            x = block(x)
        return x
