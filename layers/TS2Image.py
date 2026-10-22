import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EndogenousTS2Image(nn.Module):
    """
    On-the-fly 1D-to-2D Multi-Channel Imaging module for Endogenous Time Series.
    Translates a 1D look-back window into a 3-channel 2D spatial representation
    compatible with DINOv3 Vision Foundation Model backbones.

    Channels:
      - Channel 1 (GASF): Gramian Angular Summation Field (Auto-correlation)
      - Channel 2 (GADF): Gramian Angular Difference Field (Directional phase transition, skew-symmetric)
      - Channel 3 (Periodic Folding): 2D inter-day x intra-day grid bilinearly interpolated to (L x L)

    All operations are strictly differentiable, executed on GPU tensors,
    and avoid arccos calls to guarantee numerical stability.
    """

    def __init__(
        self,
        seq_len: int = 168,
        img_size: int = 176,
        period_len: int = 24,
        eps_norm: float = 1e-5,
        eps_arc: float = 1e-6,
    ):
        super(EndogenousTS2Image, self).__init__()
        self.seq_len = seq_len
        self.period_len = period_len
        self.eps_norm = eps_norm
        self.eps_arc = eps_arc

        # If img_size is not explicitly given, compute nearest multiple of 16 >= seq_len
        if img_size is None or img_size <= 0:
            k = math.ceil(seq_len / 16)
            self.img_size = 16 * k
        else:
            self.img_size = img_size

        # Register ImageNet channel statistics as buffers (auto-move with device)
        self.register_buffer(
            "mean_rgb",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std_rgb",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x_en: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting 1D endogenous window to 3-channel 2D image.

        Args:
            x_en: Normalized endogenous series, shape (B, L, 1) or (B, L)

        Returns:
            I_en: Normalized 2D image tensor, shape (B, 3, img_size, img_size)
        """
        if x_en.dim() == 3:
            x_en = x_en.squeeze(-1)  # (B, L)
        elif x_en.dim() != 2:
            raise ValueError(f"Expected x_en to have 2 or 3 dims, got shape {x_en.shape}")

        b_sz, l_len = x_en.shape

        # --- Stage 2.1: Bounded Min-Max Angular Rescaling ---
        x_min = x_en.min(dim=-1, keepdim=True).values  # (B, 1)
        x_max = x_en.max(dim=-1, keepdim=True).values  # (B, 1)
        denom = (x_max - x_min) + self.eps_norm       # (B, 1)

        # Scale to [-1, 1] then clamp to [-1 + eps_arc, 1 - eps_arc]
        hat_x = 2.0 * ((x_en - x_min) / denom) - 1.0
        hat_x = torch.clamp(hat_x, -1.0 + self.eps_arc, 1.0 - self.eps_arc)

        # Auxiliary radial sine component: r_t = sqrt(1 - hat_x^2)
        # Clamped inside sqrt to guarantee strictly positive derivative
        hat_r = torch.sqrt(torch.clamp(1.0 - hat_x**2, min=self.eps_arc, max=1.0))

        # Vector preparations for batched outer products
        x_col = hat_x.unsqueeze(-1)  # (B, L, 1)
        x_row = hat_x.unsqueeze(1)   # (B, 1, L)
        r_col = hat_r.unsqueeze(-1)  # (B, L, 1)
        r_row = hat_r.unsqueeze(1)   # (B, 1, L)

        # --- Stage 2.2: Channel 1 (GASF - Gramian Angular Summation Field) ---
        # G_GASF = hat_x * hat_x^T - hat_r * hat_r^T (Symmetric)
        gasf = torch.bmm(x_col, x_row) - torch.bmm(r_col, r_row)  # (B, L, L)

        # --- Stage 2.3: Channel 2 (GADF - Gramian Angular Difference Field) ---
        # G_GADF = hat_r * hat_x^T - hat_x * hat_r^T (Skew-Symmetric)
        gadf = torch.bmm(r_col, x_row) - torch.bmm(x_col, r_row)  # (B, L, L)

        # --- Stage 2.4: Channel 3 (2D Periodic Folding Map) ---
        # Reshape sequence into (n_cycles, period_len) grid and bilinearly interpolate to (L, L)
        if l_len >= self.period_len:
            n_cycles = l_len // self.period_len
            trunc_len = n_cycles * self.period_len
            fold_seq = hat_x[:, -trunc_len:]  # Take most recent full cycles
            m_fold = fold_seq.view(b_sz, 1, n_cycles, self.period_len)
        else:
            m_fold = hat_x.view(b_sz, 1, 1, l_len)

        period_map = F.interpolate(
            m_fold, size=(l_len, l_len), mode="bilinear", align_corners=False
        ).squeeze(1)  # (B, L, L)

        # --- Stage 2.5: Stacking & Spatial Resizing to (H_img, W_img) ---
        stacked = torch.stack([gasf, gadf, period_map], dim=1)  # (B, 3, L, L)

        if l_len != self.img_size:
            img_raw = F.interpolate(
                stacked,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            img_raw = stacked

        # Shift from [-1, 1] to [0, 1]
        img_scaled = 0.5 * (img_raw + 1.0)

        # ImageNet channel normalization
        out = (img_scaled - self.mean_rgb) / self.std_rgb  # (B, 3, img_size, img_size)
        return out
