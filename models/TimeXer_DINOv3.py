import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from layers.TS2Image import EndogenousTS2Image
from layers.Embed import DataEmbedding_inverted


class DINOv3BackboneWrapper(nn.Module):
    """
    Wrapper around timm DINOv3 Vision Transformer backbone.
    Enforces the critical constraint:
    - Slices out and discards the 4 Register Tokens (indices 1..4).
    - Preserves [CLS] token (index 0) and spatial patch tokens (indices 5..).
    - Supports freezing regimes: 'all' (Regime 1), 'partial' (Regime 2), 'lora', 'none'.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
        img_size: int = 176,
        freeze_mode: str = "all",
    ):
        super(DINOv3BackboneWrapper, self).__init__()
        self.model_name = model_name
        self.freeze_mode = freeze_mode

        self.dino = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )
        self.d_dino = self.dino.embed_dim

        self._configure_freezing(freeze_mode)

    def _configure_freezing(self, freeze_mode: str):
        if freeze_mode == "all":
            # Regime 1: 100% Frozen Backbone
            for p in self.dino.parameters():
                p.requires_grad = False
        elif freeze_mode == "partial":
            # Regime 2: Unfreeze last 2 blocks + all LayerNorms
            for p in self.dino.parameters():
                p.requires_grad = False
            if hasattr(self.dino, "blocks"):
                for blk in self.dino.blocks[-2:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            for name, m in self.dino.named_modules():
                if isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                    for p in m.parameters():
                        p.requires_grad = True
        elif freeze_mode == "lora":
            # Regime 2 alternative: LoRA on QKV projections
            for p in self.dino.parameters():
                p.requires_grad = False
            try:
                from peft import LoraConfig, get_peft_model
                lora_cfg = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["qkv"],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.dino = get_peft_model(self.dino, lora_cfg)
            except Exception as e:
                print(f"[Warning] Failed to apply LoRA: {e}. Keeping backbone frozen.")
        elif freeze_mode == "none":
            for p in self.dino.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze_mode: '{freeze_mode}'. Choices: all, partial, lora, none.")

    def forward(self, img: torch.Tensor):
        """
        Args:
            img: 3-channel 2D image (B, 3, H, W)
        Returns:
            z_cls: (B, 1, D_dino)
            z_patch: (B, N_p, D_dino)
        """
        tokens = self.dino.forward_features(img)
        # CRITICAL CONSTRAINT:
        # Index 0: [CLS]
        # Indices 1..4: 4 Register Tokens (DISCARDED)
        # Indices 5..: N_p Spatial Patch Tokens
        z_cls = tokens[:, 0:1, :]
        z_patch = tokens[:, 5:, :]
        return z_cls, z_patch


class CrossAttentionFusion(nn.Module):
    """
    Step 5a: Exogenous-to-Global Cross-Attention.
    DINOv3 [CLS] token acts as Query to aggregate exogenous variate tokens.
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_cls: torch.Tensor, v_ex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_cls: (B, 1, D)
            v_ex:  (B, C_ex_all, D)
        Returns:
            v_fused: (B, 1, D)
        """
        attn_out, _ = self.cross_attn(query=v_cls, key=v_ex, value=v_ex)
        v_cls = self.norm1(v_cls + self.dropout(attn_out))
        ffn_out = self.ffn(v_cls)
        v_fused = self.norm2(v_cls + ffn_out)
        return v_fused


class FlattenHead(nn.Module):
    def __init__(self, in_features: int, target_window: int, head_dropout: float = 0.1):
        super(FlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(in_features, target_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return self.linear(self.dropout(self.flatten(x)))


class Model(nn.Module):
    """
    TimeXer-DINOv3: Hybrid Vision-Endogenous & Inverted-Exogenous Time-Series Forecasting.
    Replaces canonical TimeXer's 1D patch encoder with a pretrained DINOv3 Vision Transformer
    while preserving TimeXer's inverted variate embedding and adding a 2-step cross-modality fusion.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.patch_len = getattr(configs, "patch_len", 24)
        self.period_len = getattr(configs, "period_len", 24)
        self.dino_model_name = getattr(configs, "dino_model_name", "vit_small_patch16_dinov3")
        self.dino_img_size = getattr(configs, "dino_img_size", 176)
        self.dino_freeze = getattr(configs, "dino_freeze", "all")
        self.use_pooled_bottleneck = getattr(configs, "use_pooled_bottleneck", True)
        self.post_e_layers = getattr(configs, "post_e_layers", 1)

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        dropout = configs.dropout

        # --- Stage 2: 1D to 2D Multi-Channel Imaging ---
        self.ts2img = EndogenousTS2Image(
            seq_len=self.seq_len,
            img_size=self.dino_img_size,
            period_len=self.period_len,
        )

        # --- Stage 3: DINOv3 Vision Backbone Wrapper ---
        self.dino_backbone = DINOv3BackboneWrapper(
            model_name=self.dino_model_name,
            pretrained=True,
            img_size=self.dino_img_size,
            freeze_mode=self.dino_freeze,
        )
        d_dino = self.dino_backbone.d_dino

        # Dimension Adapters (Projection from D_dino -> d_model)
        self.proj_cls = nn.Sequential(
            nn.Linear(d_dino, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_patch = nn.Sequential(
            nn.Linear(d_dino, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Stage 4: Exogenous Inverted Embedding ---
        self.ex_embedding = DataEmbedding_inverted(
            c_in=self.seq_len,
            d_model=d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=dropout,
        )
        self.dummy_ex_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # --- Stage 5: 2-Step Cross-Modality Fusion ---
        # Step 5a: Exogenous-to-Global Cross-Attention
        self.cross_fusion = CrossAttentionFusion(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
        )

        # Step 5b: Global-to-Patch Post-Fusion Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.post_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.post_e_layers)

        # --- Stage 6: Projection Head ---
        if self.use_pooled_bottleneck:
            self.n_reduced = max(1, self.seq_len // self.patch_len)
            head_in_features = (self.n_reduced + 1) * d_model
        else:
            k = math.ceil(self.dino_img_size / 16)
            n_p = k * k
            head_in_features = (n_p + 1) * d_model

        self.head = FlattenHead(
            in_features=head_in_features,
            target_window=self.pred_len,
            head_dropout=dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        # --- Stage 1: Instance-Level Stationarization (RevIN) ---
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # Routing: Endogenous target is last column, Exogenous covariates are remaining columns
        x_en = x_enc[:, :, -1:]  # (B, L, 1)

        # --- Stage 2: 1D to 2D Multi-Channel Imaging ---
        img_en = self.ts2img(x_en)  # (B, 3, img_size, img_size)

        # --- Stage 3: DINOv3 Backbone & Token Decomposition (Register Tokens discarded) ---
        z_cls, z_patch = self.dino_backbone(img_en)  # z_cls: (B, 1, D_dino), z_patch: (B, N_p, D_dino)

        # Adapters projection
        v_en_cls = self.proj_cls(z_cls)      # (B, 1, d_model)
        v_en_patch = self.proj_patch(z_patch) # (B, N_p, d_model)

        # --- Stage 4: Exogenous Variate-Level Inverted Embedding ---
        if x_enc.shape[-1] > 1:
            x_ex = x_enc[:, :, :-1]  # (B, L, C_ex)
            v_ex = self.ex_embedding(x_ex, x_mark_enc)  # (B, C_ex + C_mark, d_model)
        else:
            v_ex = self.dummy_ex_token.expand(x_enc.shape[0], -1, -1)

        # --- Stage 5: Cross-Modality Fusion ---
        # Step 5a: Exogenous-to-Global Cross-Attention
        v_en_fused = self.cross_fusion(v_en_cls, v_ex)  # (B, 1, d_model)

        # Step 5b: Global-to-Patch Self-Attention
        z_in = torch.cat([v_en_patch, v_en_fused], dim=1)  # (B, N_p + 1, d_model)
        z_out = self.post_encoder(z_in)                    # (B, N_p + 1, d_model)

        # --- Stage 6: Token Bottleneck & FlattenHead ---
        if self.use_pooled_bottleneck:
            n_p = z_out.shape[1] - 1
            patches = z_out[:, :n_p, :]           # (B, N_p, d_model)
            fused_cls = z_out[:, -1:, :]          # (B, 1, d_model)

            # Adaptive average pooling across token dimension: (B, d_model, N_p) -> (B, d_model, N_reduced)
            pooled_patches = F.adaptive_avg_pool1d(
                patches.transpose(1, 2), self.n_reduced
            ).transpose(1, 2)
            z_final = torch.cat([pooled_patches, fused_cls], dim=1)  # (B, N_reduced + 1, d_model)
        else:
            z_final = z_out

        y_hat = self.head(z_final)  # (B, pred_len)
        y_hat = y_hat.unsqueeze(-1)  # (B, pred_len, 1)

        # RevIN De-Normalization
        if self.use_norm:
            target_mean = means[:, 0, -1:].unsqueeze(1)
            target_stdev = stdev[:, 0, -1:].unsqueeze(1)
            dec_out = y_hat * target_stdev + target_mean
        else:
            dec_out = y_hat

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} is not implemented for TimeXer_DINOv3.")
