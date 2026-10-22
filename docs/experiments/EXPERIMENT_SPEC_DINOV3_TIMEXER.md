# TimeXer-DINOv3 Technical Specification

**Technical Specification: Hybrid Vision-Endogenous & Inverted-Exogenous Time-Series Forecasting (TimeXer-DINOv3)**

| Field | Value |
|-------|-------|
| **Document Status** | Source of Truth (SoT) — Experimental Phase 1 |
| **Base Repository** | TimeXer (Commit `76011909357972bd55a27adba2e1be994d81b327`) |
| **Core Architectural Paradigm** | **Option A** — 1D Endogenous Time-Series → 2D Multi-Channel Imaging → Pretrained DINOv3 Backbone + Variate-Level Exogenous Cross-Attention Fusion |
| **Target Task** | Multivariate-to-Univariate Time-Series Forecasting with Exogenous Variables (`--features MS`) |

---

## 1. Executive Summary & Architectural Motivation

### 1.1. Problem Statement

In the canonical TimeXer architecture (`models/TimeXer.py`), the endogenous target series is partitioned into 1D temporal patches and processed alongside a learnable global token via a lightweight, randomly initialized 1D Transformer Encoder (typically 1 to 3 layers). At each encoder layer, the endogenous global token serves as a bridge to absorb information from exogenous covariates via Cross-Attention.

While effective, a shallow from-scratch 1D encoder has limited representational capacity for capturing complex non-linear phase transitions, multi-scale periodicities, and abrupt regime shifts (such as electricity price spikes).

### 1.2. Proposed Solution (Option A: Full Endogenous Backbone Replacement via 2D Imaging)

This specification defines the mathematical and engineering blueprint for replacing the entire endogenous 1D Transformer Encoder of TimeXer with **DINOv3** (specifically `dinov3_vits16` or `dinov3_vitb16`), a state-of-the-art self-supervised Vision Foundation Model trained with 2D Axial Rotary Position Embeddings (2D RoPE) and Register Tokens.

To resolve the modality mismatch between 1D time series and 2D vision backbones while strictly preserving DINOv3’s spatial inductive biases (Conv2d patchify filter bank and 2D RoPE), the normalized 1D endogenous look-back window is projected into a **3-channel 2D spatial representation** prior to entering DINOv3. Meanwhile, the exogenous covariates retain TimeXer’s highly efficient Variate-Level Inverted Embedding (`DataEmbedding_inverted`) and interact with DINOv3’s `[CLS]` global representation and spatial patch tokens via a dedicated **Post-Backbone Cross-Modality Fusion Module**.

---

## 2. Dataset Specification & Per-Dataset Adaptation Protocol

### 2.1. Primary Benchmark: Electricity Price Forecasting (EPF)

Located in `./dataset/EPF/`, the EPF benchmark comprises five distinct day-ahead electricity markets. Each dataset is sampled hourly ($1\text{h}$ resolution) over a 6-year period ($\approx 52{,}416$ timestamps) and split chronologically into Train, Validation, and Test sets at a **7 : 1 : 2** ratio.

| Dataset File | Market / Region       | Endogenous Target (1 Variate) | Exogenous Covariates ($C_{ex}=2$ Variates)              | Temporal Marks ($C_{mark}$)              | Look-back ($L$)     | Horizon ($H$)      |
|--------------|-----------------------|-------------------------------|---------------------------------------------------------|------------------------------------------|---------------------|--------------------|
| `NP.csv`     | Nord Pool (Nordic)    | OT (Zonal Price)              | Load Forecast, Hydro/Wind Generation Forecast           | 4 (Hour, Day, Weekday, Month)            | 168 (7 days)        | 24 (1 day)         |
| `PJM.csv`    | PJM (USA)             | OT (Zonal Price)              | System Load Forecast, Wind Generation Forecast          | 4 (Hour, Day, Weekday, Month)            | 168 (7 days)        | 24 (1 day)         |
| `BE.csv`     | Belgium               | OT (Zonal Price)              | Belgian Load Forecast, French Generation/Load           | 4 (Hour, Day, Weekday, Month)            | 168 (7 days)        | 24 (1 day)         |
| `FR.csv`     | France                | OT (Zonal Price)              | Load Forecast, Nuclear/Total Generation Forecast        | 4 (Hour, Day, Weekday, Month)            | 168 (7 days)        | 24 (1 day)         |
| `DE.csv`     | Germany               | OT (Zonal Price)              | Load Forecast, Renewable (Wind/Solar) Forecast          | 4 (Hour, Day, Weekday, Month)            | 168 (7 days)        | 24 (1 day)         |

### 2.2. Secondary Benchmark: Long-Term Forecasting with Exogenous Covariates (LTSF)

Configured in `./scripts/forecast_exogenous/`, these datasets evaluate long-horizon extrapolation:

- **Datasets**: ETTh1, ETTh2, ETTm1, ETTm2 ($C_{ex}=6$), Weather ($C_{ex}=20$), ECL ($C_{ex}=320$), Traffic ($C_{ex}=861$).
- **Windowing Protocol**: Look-back length $L=96$; Prediction horizons $H \in \{96, 192, 336, 720\}$.

### 2.3. Per-Dataset Adaptation Protocol

Each dataset $d \in \mathcal{D}$ **MUST** be trained and adapted as an independent model instance:

1. **Shared Universal Prior**: The pretrained DINOv3 backbone weights $\Theta_{\text{DINOv3}}$ serve as a universal structural feature extractor across all datasets.
2. **Dataset-Specific Adaptation Parameters ($\Phi_d$)**: Because each dataset exhibits distinct exogenous dimensionality $C_{ex}^{(d)}$, amplitude distributions, and cross-variate causal dynamics, the exogenous embedding layer, cross-attention fusion blocks, dimension adapters, and prediction head are initialized and optimized specifically for dataset $d$ and horizon $H$.

---

## 3. Rigorous Mathematical Formulation & Constraints

Let $B$ be the batch size, $L$ be the endogenous look-back window length (`seq_len`), $H$ be the prediction horizon (`pred_len`), and $C_{ex}$ be the number of exogenous series.

At each training step, the data loader (`data_provider/data_loader.py`) yields:

- Full input tensor: $\mathbf{X} \in \mathbb{R}^{B \times L \times (C_{ex} + 1)}$
- Temporal timestamp features: $\mathbf{X}_{mark} \in \mathbb{R}^{B \times L \times C_{mark}}$

In the `--features MS` setting, we slice $\mathbf{X}$ along the variate dimension into:

- **Endogenous Target Series**: $\mathbf{x}_{en} = \mathbf{X}_{:, :, -1:} \in \mathbb{R}^{B \times L \times 1}$
- **Exogenous Covariate Matrix**: $\mathbf{X}_{ex} = \mathbf{X}_{:, :, :-1} \in \mathbb{R}^{B \times L \times C_{ex}}$

### 3.1. Stage 1: Instance-Level Stationarization (RevIN)

To prevent non-stationary drift and amplitude saturation during 2D image mapping, Reversible Instance Normalization (`layers/StandardNorm.py`) is applied independently to each window instance $b \in \{1, \dots, B\}$:

$$
\mu_{en}^{(b)} = \frac{1}{L} \sum_{t=1}^{L} x_{en, t}^{(b)}, \quad
\left(\sigma_{en}^{(b)}\right)^2 = \frac{1}{L} \sum_{t=1}^{L} \left(x_{en, t}^{(b)} - \mu_{en}^{(b)}\right)^2
$$

$$
\tilde{\mathbf{x}}_{en}^{(b)} = \frac{\mathbf{x}_{en}^{(b)} - \mu_{en}^{(b)}}{\sqrt{\left(\sigma_{en}^{(b)}\right)^2 + \epsilon_{norm}}} \in \mathbb{R}^{L \times 1}
$$

where $\epsilon_{norm} = 10^{-5}$. The statistics $\bigl(\mu_{en}^{(b)}, \sigma_{en}^{(b)}\bigr)$ are cached to de-normalize the final prediction in Stage 6.

### 3.2. Stage 2: On-the-Fly 1D-to-2D Multi-Channel Imaging ($\mathcal{T}_{2D}$)

To leverage DINOv3’s 3-channel RGB Conv2d patch embedding and 2D spatial self-attention without information loss, the normalized 1D vector $\tilde{\mathbf{x}}_{en} \in \mathbb{R}^{B \times L}$ (squeezed along the last dimension) is transformed into a 3-channel 2D tensor $\mathbf{I}_{en} \in \mathbb{R}^{B \times 3 \times H_{img} \times W_{img}}$.

#### Constraint 1: Bounded Rescaling for Angular Projection

For each sample $b$, $\tilde{\mathbf{x}}_{en}^{(b)}$ is min-max scaled into $[-1 + \epsilon_{arc}, 1 - \epsilon_{arc}]$ to guarantee finite gradients. We use exact algebraic trigonometric identities to bypass $\arccos$ entirely and ensure 100% numerical stability:

$$
\hat{x}_t^{(b)} = \operatorname{clamp}\Biggl(
  2 \cdot \frac{\tilde{x}_{en, t}^{(b)} - \min_k \tilde{x}_{en, k}^{(b)}}{\max_k \tilde{x}_{en, k}^{(b)} - \min_k \tilde{x}_{en, k}^{(b)} + \epsilon_{norm}} - 1,
  \; -1 + \epsilon_{arc}, \; 1 - \epsilon_{arc}
\Biggr)
$$

where $\epsilon_{arc} = 10^{-6}$. Define the auxiliary sine radial component:

$$
\hat{r}_t^{(b)} = \sqrt{1 - \bigl(\hat{x}_t^{(b)}\bigr)^2} \in (0, 1]
$$

#### Channel 1: Gramian Angular Summation Field (GASF — Temporal Auto-Correlation)

GASF encodes static magnitude along the main diagonal and pairwise temporal correlation across $(i,j) \in \{1,\dots,L\}^2$:

$$
\mathbf{G}_{GASF}^{(b)} = \hat{\mathbf{x}}^{(b)}\bigl(\hat{\mathbf{x}}^{(b)}\bigr)^\top - \hat{\mathbf{r}}^{(b)}\bigl(\hat{\mathbf{r}}^{(b)}\bigr)^\top \in \mathbb{R}^{L \times L}
$$

*(Mathematical identity: $\cos(\phi_i + \phi_j) = \cos\phi_i\cos\phi_j - \sin\phi_i\sin\phi_j$ where $\phi_t = \arccos(\hat{x}_t)$.)*

#### Channel 2: Gramian Angular Difference Field (GADF — Directional Phase Transition)

Because $\cos(\phi_i + \phi_j)$ is symmetric ($\mathbf{G}_{GASF,i,j} = \mathbf{G}_{GASF,j,i}$), Channel 1 alone loses causal temporal directionality. GADF is strictly skew-symmetric and preserves the sign of temporal transitions $(t_i \to t_j)$:

$$
\mathbf{G}_{GADF}^{(b)} = \hat{\mathbf{r}}^{(b)}\bigl(\hat{\mathbf{x}}^{(b)}\bigr)^\top - \hat{\mathbf{x}}^{(b)}\bigl(\hat{\mathbf{r}}^{(b)}\bigr)^\top \in \mathbb{R}^{L \times L}
$$

*(Mathematical identity: $\sin(\phi_i - \phi_j) = \sin\phi_i\cos\phi_j - \cos\phi_i\sin\phi_j$.)*

#### Channel 3: 2D Periodic Folding Map (Multi-Cycle Structural Grid)

Given a fundamental period $P_{fold}$ (e.g., $P_{fold}=24$ hours for daily cycles in EPF, or $P_{fold}=16$ for $L=96$), the 1D sequence $\hat{\mathbf{x}}^{(b)} \in \mathbb{R}^L$ is reshaped into a 2D inter-day $\times$ intra-day grid:

$$
\mathbf{M}_{fold}^{(b)} = \operatorname{Reshape}\Bigl(\hat{\mathbf{x}}^{(b)}, \;\bigl(\lfloor L / P_{fold} \rfloor, P_{fold}\bigr)\Bigr)
$$

and upsampled via 2D bilinear interpolation to match the $(L \times L)$ spatial resolution:

$$
\mathbf{G}_{PERIOD}^{(b)} = \operatorname{Interpolate}_{\text{bilinear}}\bigl(\mathbf{M}_{fold}^{(b)}, \;\text{size}=(L,L)\bigr) \in \mathbb{R}^{L \times L}
$$

#### Constraint 2: Spatial Divisibility by DINOv3 Patch Size ($P_{dino}=16$)

DINOv3 ViT models require spatial dimensions $(H_{img}, W_{img})$ to be exact multiples of $P_{dino}=16$:

$$
H_{img} = W_{img} = 16 \times K, \quad K = \Bigl\lceil \frac{L}{16} \Bigr\rceil
\quad \text{(or a fixed target grid } K\text{)}
$$

- **For EPF ($L=168$)**: $K = \lceil 168/16 \rceil = 11 \implies H_{img}=W_{img}=176$. Number of spatial patches $N_p = 11 \times 11 = 121$.
- **For LTSF ($L=96$)**: $K = 96/16 = 6 \implies H_{img}=W_{img}=96$. Number of spatial patches $N_p = 6 \times 6 = 36$.

The three $(L \times L)$ matrices are stacked along the channel dimension, resized/padded to $(H_{img}, W_{img})$, and normalized using ImageNet channel statistics $\mu_{RGB}=[0.485, 0.456, 0.406]^\top$ and $\sigma_{RGB}=[0.229, 0.224, 0.225]^\top$ after shifting from $[-1,1]$ to $[0,1]$:

$$
\tilde{\mathbf{I}}_{en} = \frac{1}{2}\Biggl(
  \operatorname{Interpolate}_{\text{bilinear}}\Bigl(
    \operatorname{Stack}\bigl([\mathbf{G}_{GASF}, \mathbf{G}_{GADF}, \mathbf{G}_{PERIOD}], \text{dim}=1\bigr),
    (H_{img}, W_{img})
  \Bigr) + 1
\Biggr)
$$

$$
\mathbf{I}_{en} = \frac{\tilde{\mathbf{I}}_{en} - \mu_{RGB}}{\sigma_{RGB}} \in \mathbb{R}^{B \times 3 \times H_{img} \times W_{img}}
$$

### 3.3. Stage 3: DINOv3 Vision Backbone & Token Decomposition

The image tensor $\mathbf{I}_{en}$ is passed into the DINOv3 Vision Transformer backbone $f_{\text{DINOv3}}$ with hidden dimension $D_{dino}$ (384 for ViT-S/16, 768 for ViT-B/16) and $N_{reg}=4$ register tokens.

#### 1. Patchification & Token Prepending

$$
\mathbf{U}^{(0)} = \bigl[
  \mathbf{u}_{cls} \;\Vert\; \mathbf{U}_{reg} \;\Vert\;
  \operatorname{Flatten}_{2D}\bigl(\operatorname{Conv2d}_{16\times16}(\mathbf{I}_{en})\bigr)
\bigr] \in \mathbb{R}^{B \times (1 + 4 + N_p) \times D_{dino}}
$$

#### 2. 2D Axial RoPE Self-Attention across $M_{dino}$ Blocks

Inside each DINOv3 block, Queries and Keys of the $N_p = K \times K$ spatial tokens are rotated by 2D frequency matrices $\mathbf{R}_{\Theta}(h_i, w_i)$ where $(h_i, w_i) \in \{0,\dots,K-1\}^2$, while $\mathbf{u}_{cls}$ and $\mathbf{U}_{reg}$ bypass spatial rotation.

#### 3. Output Slicing & Register Token Removal (**CRITICAL CONSTRAINT**)

Let $\mathbf{U}^{(M_{dino})} \in \mathbb{R}^{B \times (5 + N_p) \times D_{dino}}$ be the final normalized hidden state of DINOv3. We **strictly discard** the 4 register tokens (indices 1 to 4), as they store internal attention artifacts rather than semantic content:

$$
\mathbf{z}_{cls} = \mathbf{U}_{:, 0:1, :}^{(M_{dino})} \in \mathbb{R}^{B \times 1 \times D_{dino}}
$$

$$
\mathbf{Z}_{patch} = \mathbf{U}_{:, 5:, :}^{(M_{dino})} \in \mathbb{R}^{B \times N_p \times D_{dino}}
$$

#### 4. Dimension Alignment (Adapter Projection to $d_{model}=D$)

To match TimeXer’s latent dimension $D$, we apply learnable linear projections followed by Layer Normalization:

$$
\hat{\mathbf{v}}_{en} = \operatorname{LN}\bigl(\mathbf{z}_{cls} \mathbf{W}_{proj\_cls} + \mathbf{b}_{proj\_cls}\bigr) \in \mathbb{R}^{B \times 1 \times D}
$$

$$
\hat{\mathbf{P}}_{en} = \operatorname{LN}\bigl(\mathbf{Z}_{patch} \mathbf{W}_{proj\_patch} + \mathbf{b}_{proj\_patch}\bigr) \in \mathbb{R}^{B \times N_p \times D}
$$

where $\mathbf{W}_{proj\_cls}, \mathbf{W}_{proj\_patch} \in \mathbb{R}^{D_{dino} \times D}$.

### 3.4. Stage 4: Exogenous Variate-Level Inverted Embedding

Following TimeXer (`layers/Embed.py`), the exogenous series $\mathbf{X}_{ex} \in \mathbb{R}^{B \times L \times C_{ex}}$ (normalized via RevIN) and temporal marks $\mathbf{X}_{mark} \in \mathbb{R}^{B \times L \times C_{mark}}$ are concatenated along the variate axis and transposed so that each variate’s entire look-back trajectory of length $L$ is embedded as a single token:

$$
\mathbf{X}_{ex\_all} = \bigl[\tilde{\mathbf{X}}_{ex} \;\Vert\; \mathbf{X}_{mark}\bigr]^\top \in \mathbb{R}^{B \times (C_{ex} + C_{mark}) \times L}
$$

$$
\mathbf{V}_{ex} = \operatorname{Dropout}\bigl(\mathbf{X}_{ex\_all} \mathbf{W}_{ex} + \mathbf{b}_{ex}\bigr) \in \mathbb{R}^{B \times (C_{ex} + C_{mark}) \times D}
$$

where $\mathbf{W}_{ex} \in \mathbb{R}^{L \times D}$.

### 3.5. Stage 5: Cross-Modality Fusion & Post-Fusion Propagation

In canonical TimeXer, Cross-Attention and Self-Attention alternate across layers, allowing exogenous information absorbed by the global token to propagate back into the endogenous patches. Because DINOv3 processes the endogenous branch monolithically before meeting $\mathbf{V}_{ex}$, a **2-Step Fusion-Propagation Block** is mathematically required after DINOv3:

#### Step 5a — Exogenous-to-Global Cross-Attention

The projected DINOv3 `[CLS]` token $\hat{\mathbf{v}}_{en} \in \mathbb{R}^{B \times 1 \times D}$ acts as the Query to aggregate relevant exogenous variate tokens $\mathbf{V}_{ex}$:

$$
\tilde{\mathbf{v}}_{en} = \operatorname{LN}\Bigl(
  \hat{\mathbf{v}}_{en} + \operatorname{MultiHeadAttn}\bigl(
    Q=\hat{\mathbf{v}}_{en},\; K=\mathbf{V}_{ex},\; V=\mathbf{V}_{ex}
  \bigr)
\Bigr) \in \mathbb{R}^{B \times 1 \times D}
$$

$$
\mathbf{v}_{en}^{fused} = \operatorname{LN}\bigl(
  \tilde{\mathbf{v}}_{en} + \operatorname{FFN}(\tilde{\mathbf{v}}_{en})
\bigr) \in \mathbb{R}^{B \times 1 \times D}
$$

#### Step 5b — Global-to-Patch Post-Fusion Self-Attention

We concatenate the spatial patch tokens $\hat{\mathbf{P}}_{en}$ with the exogenous-enriched global token $\mathbf{v}_{en}^{fused}$ and pass them through $M_{post}$ (default $M_{post}=1$) standard Transformer Encoder layers so that every local temporal patch is conditioned on the exogenous context:

$$
\mathbf{Z}_{en}^{in} = \bigl[ \hat{\mathbf{P}}_{en} \;\Vert\; \mathbf{v}_{en}^{fused} \bigr] \in \mathbb{R}^{B \times (N_p + 1) \times D}
$$

$$
\mathbf{Z}_{en}^{out} = \operatorname{PostEncoder}\bigl(\mathbf{Z}_{en}^{in}\bigr) \in \mathbb{R}^{B \times (N_p + 1) \times D}
$$

### 3.6. Stage 6: Projection Head (FlattenHead) & De-Normalization

#### 1. Flatten & Linear Projection

The fused representation $\mathbf{Z}_{en}^{out}$ is flattened across the token and channel dimensions into $\mathbb{R}^{B \times ((N_p + 1) \cdot D)}$ and mapped to the prediction horizon $H$:

$$
\tilde{\mathbf{y}} = \operatorname{Dropout}\bigl(\operatorname{Flatten}(\mathbf{Z}_{en}^{out})\bigr) \mathbf{W}_{head} + \mathbf{b}_{head} \in \mathbb{R}^{B \times H \times 1}
$$

where $\mathbf{W}_{head} \in \mathbb{R}^{((N_p + 1) \cdot D) \times H}$.

#### 2. Token Bottleneck Option (For High $N_p$)

When $N_p=121$ ($L=168$) and $D=512$, $(N_p+1)\cdot D = 62{,}464$, which can over-parameterize $\mathbf{W}_{head}$. To prevent overfitting, an optional 1D Adaptive Average Pooling or linear bottleneck reduces the $N_p$ patch tokens to $N_{reduced} = \lfloor L / P \rfloor$ tokens prior to flattening:

$$
\mathbf{Z}_{en}^{pooled} = \Biggl[
  \operatorname{AdaptiveAvgPool1d}_{N_{reduced}}\bigl(\mathbf{Z}_{en, :, :N_p, :}^{out}\bigr)
  \;\Vert\;
  \mathbf{Z}_{en, :, -1:, :}^{out}
\Biggr] \in \mathbb{R}^{B \times (N_{reduced} + 1) \times D}
$$

#### 3. RevIN De-Normalization

$$
\hat{\mathbf{y}}^{(b)} = \tilde{\mathbf{y}}^{(b)} \cdot \sqrt{\bigl(\sigma_{en}^{(b)}\bigr)^2 + \epsilon_{norm}} + \mu_{en}^{(b)} \in \mathbb{R}^{H \times 1}
$$

---

## 4. Codebase Integration & File-Level Modification Blueprint

To maintain backward compatibility with the existing TimeXer framework, modifications are modularized across the following files:

| File Path | Action | Detailed Engineering Responsibility |
|-----------|--------|-------------------------------------|
| `requirements.txt` | Modify | Add `timm>=1.0.9`, `transformers>=4.45.0`, `peft>=0.12.0` (optional for LoRA). |
| `layers/TS2Image.py` | **Create** | Implement `EndogenousTS2Image(nn.Module)` performing vector-form GASF, GADF, 2D Periodic Folding, bilinear interpolation to $(16K \times 16K)$, and ImageNet normalization on GPU tensors without CPU-GPU transfers. |
| `models/TimeXer_DINOv3.py` | **Create** | Implement `Model(nn.Module)` integrating: Normalize (RevIN) → EndogenousTS2Image → DINOv3BackboneWrapper (slicing out register tokens) → AdapterProjection → DataEmbedding_inverted → CrossAttentionFusion → PostEncoder → FlattenHead. |
| `exp/exp_basic.py` | Modify | Import `TimeXer_DINOv3` and register `'TimeXer_DINOv3': TimeXer_DINOv3` in `self.model_dict`. |
| `exp/exp_long_term_forecasting.py` | Modify | Update `_select_optimizer()` to support Differential Learning Rates (e.g., $\eta_{backbone}$ for unfrozen DINOv3 parameters vs. $\eta_{head}$ for TimeXer adapter/fusion/head parameters) and filter `requires_grad=True` parameters. |
| `run.py` | Modify | Add CLI arguments: `--dino_model_name` (default `'vit_small_patch16_dinov3'`), `--dino_img_size` (default `176`), `--dino_freeze` (choices: `'all'`, `'partial'`, `'lora'`, `'none'`), `--dino_lr` (default `1e-5`), `--period_len` (default `24`). |
| `scripts/forecast_exogenous/EPF/TimeXer_DINOv3.sh` | **Create** | Execution script for the 5 EPF datasets (NP, PJM, BE, FR, DE) under the 3 experimental regimes. |

---

## 5. Experimental Regimes & Hyperparameter Configurations

To rigorously evaluate whether DINOv3 improves representation quality over the vanilla 1D encoder, experiments **MUST** be executed across three controlled regimes.

### 5.1. The Three Experimental Regimes

| Regime ID | Configuration Name | DINOv3 Backbone State | Trainable Modules | Optimizer & Learning Rate |
|-----------|--------------------|-----------------------|-------------------|---------------------------|
| **Regime 0** | `Baseline_TimeXer` | Not used (Vanilla 1D Patch Encoder) | All TimeXer parameters ($\approx 0.5\text{M}$ – $2\text{M}$) | AdamW, $\eta = 10^{-4}$ |
| **Regime 1** | `DINOv3_Frozen_Probe` | 100% Frozen ($\nabla_{\Theta_{\text{DINOv3}}} \mathcal{L} = 0$) | AdapterProjection, DataEmbedding_inverted, CrossAttn, PostEncoder, FlattenHead | AdamW, $\eta_{head}=10^{-4}$, Weight Decay $=10^{-2}$ |
| **Regime 2** | `DINOv3_Partial_FT` | Unfreeze last 2 ViT blocks + all LayerNorm layers (or LoRA $r=8$, $\alpha=16$ on $Q,V$) | All Regime 1 modules + Top-2 DINOv3 blocks | Differential AdamW: $\eta_{head}=10^{-4}$, $\eta_{backbone}=10^{-5}$, Cosine Annealing |

### 5.2. Standard Hyperparameter Table (EPF Benchmark)

```yaml
# Core Task & Windowing
task_name: long_term_forecast
features: MS
seq_len: 168
label_len: 48
pred_len: 24
patch_len: 24          # Used for pooled token bottleneck N_reduced = 168 / 24 = 7

# DINOv3 & 2D Imaging Parameters
dino_model_name: vit_small_patch16_dinov3  # D_dino = 384, 12 layers, 4 register tokens
dino_img_size: 176                         # 11 × 16 = 176 → N_p = 121 spatial tokens
period_len: 24                             # 24-hour daily cycle for Channel 3 folding
use_pooled_bottleneck: True                # Pools 121 spatial tokens → 7 temporal tokens before FlattenHead

# TimeXer Fusion & Head Parameters
d_model: 256                               # Projected dimension D
n_heads: 8
d_ff: 512
post_e_layers: 1                           # Number of Post-Fusion Self-Attention layers
dropout: 0.1

# Optimization & Hardware Constraints
batch_size: 32                             # Reduce to 16 if GPU VRAM < 8GB
train_epochs: 10
patience: 3                                # Early stopping patience
learning_rate: 0.0001                      # Head & Fusion learning rate
dino_lr: 0.00001                           # Backbone fine-tuning learning rate (Regime 2)
loss: MSE
use_amp: True                              # Mixed precision FP16/BF16 mandatory for ViT speed
```

## 6. Evaluation Protocol, Sanity Checks & Acceptance Criteria

### 6.1. Quantitative Evaluation Metrics

Evaluated on the de-normalized predictions $\hat{\mathbf{Y}} \in \mathbb{R}^{N_{test} \times H}$ against ground truth $\mathbf{Y} \in \mathbb{R}^{N_{test} \times H}$ (`utils/metrics.py`):

$$
\text{MSE} = \frac{1}{N_{test} H} \sum_{i=1}^{N_{test}} \sum_{t=1}^{H} \bigl(Y_{i,t} - \hat{Y}_{i,t}\bigr)^2,
\quad
\text{MAE} = \frac{1}{N_{test} H} \sum_{i=1}^{N_{test}} \sum_{t=1}^{H} \bigl|Y_{i,t} - \hat{Y}_{i,t}\bigr|
$$

### 6.2. Mandatory Pre-Flight Sanity Checks

Before running full 10-epoch benchmarks, the implementation **MUST** pass these four unit assertions:

1. **Zero-NaN Imaging Check**  
   Verify that constant windows (where $\max_t \tilde{x}_t = \min_t \tilde{x}_t$, e.g., zero variance) do not produce `NaN` or `Inf` in $\mathbf{G}_{GASF}$ and $\mathbf{G}_{GADF}$ due to division by zero.

2. **Token Shape & Register Slicing Assertion**  
   Assert that for input $(B, 3, 176, 176)$, the raw DINOv3 output has shape $(B, 126, 384)$, and after slicing out indices $1..4$, $\mathbf{z}_{cls}$ has shape $(B, 1, 384)$ and $\mathbf{Z}_{patch}$ has shape $(B, 121, 384)$.

3. **Gradient Isolation Check (Regime 1)**  
   Verify that `sum(p.grad is not None for p in model.dino_backbone.parameters()) == 0` during `loss.backward()` in Regime 1.

4. **Endogenous-Exogenous Alignment Check**  
   Verify that `x_enc[:, :, -1:]` is routed to the 2D Imaging + DINOv3 branch and `x_enc[:, :, :-1]` is routed to `DataEmbedding_inverted`.

---

**Document Status**: This specification (`EXPERIMENT_SPEC_DINOV3_TIMEXER.md`) is established as the **Source of Truth (SoT)** with complete mathematical constraints (algebraic formulas for GASF/GADF avoiding $\arccos$ gradient issues, mandatory discarding of the 4 DINOv3 Register Tokens, Post-Fusion Propagation mechanism, and the 3-regime fine-tuning strategy for each dataset).
```
