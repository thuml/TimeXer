# Kết quả nhánh TimeXer-DINOv3: không cải thiện so với TimeXer

*Ghi ngày 22/10/2026. Nhánh này nằm ngoài roadmap PERIN (`docs/plan/proof-roadmap.md`); spec ở `docs/experiments/EXPERIMENT_SPEC_DINOV3_TIMEXER.md`. Thuật ngữ theo `CONTEXT.md`.*

**Kết luận:** TimeXer-DINOv3 thua TimeXer trên cả 5 thị trường EPF ở cả hai regime. Probe tuyến tính cho thấy weight DINOv3 pretrained giữ lại *ít* thông tin về chuỗi hơn cả weight khởi tạo ngẫu nhiên. Đề xuất: dừng nhánh, quay về G2/G3.

---

## 1. Kết quả benchmark (EPF, `seq_len 168`, `pred_len 24`, MSE / MAE)

Nguồn: `result_long_term_forecast.txt`, `docs/results/raw/results.csv`. DINOv3 chỉ chạy 1 seed (2021); so cùng seed với TimeXer.

| Thị trường | TimeXer (seed 2021) | Regime 1: frozen | Δ MSE | Regime 2: partial FT | Δ MSE |
|---|---|---|---|---|---|
| NP | 0.2365 / 0.2673 | 0.2710 / 0.2964 | +14.6% | 0.2520 / 0.2879 | +6.6% |
| PJM | 0.0969 / 0.1947 | 0.1319 / 0.2230 | +36.1% | 0.1190 / 0.2132 | +22.8% |
| BE | 0.3794 / 0.2423 | 0.4153 / 0.2645 | +9.5% | 0.4125 / 0.2595 | +8.7% |
| FR | 0.3908 / 0.2113 | 0.4573 / 0.2333 | +17.0% | 0.4480 / 0.2253 | +14.6% |
| DE | 0.4664 / 0.4213 | 0.4920 / 0.4482 | +5.5% | 0.4854 / 0.4409 | +4.1% |
| **AVG** | **0.3140 / 0.2674** | **0.3535 / 0.2931** | **+12.6%** | **0.3434 / 0.2854** | **+9.4%** |

- So với mean 3 seed của G1 (AVG 0.3119), Regime 2 tệ hơn 10.1%. Std giữa các seed của TimeXer (≤ 0.017) nhỏ hơn nhiều so với độ chênh, nên dù mới 1 seed thì kết luận vẫn vững.
- Cả hai regime đều thua iTransformer (0.335) và PatchTST (0.330) theo số paper báo.
- Chi phí: backbone 21.6M tham số; Regime 2 mất khoảng 2.5 giờ cho 5 thị trường trên RTX 2050.

## 2. Weight: không tải nhầm

timm 1.0.30, repo `timm/vit_small_patch16_dinov3.lvd1689m`: ViT-S/16, 21.6M tham số, `embed_dim=384`, `num_prefix_tokens=5` (CLS + 4 register), output `(B, 126, 384)` với ảnh 176×176, mean/std ImageNet. Repo timm không bị gated (tải được mà không cần HF token); repo `facebook/dinov3-*` của Meta thì bị gated. License: `dinov3-license`, không phải MIT/Apache.

## 3. Probe đặc trưng (`docs/brainstorm/probes/dinov3_feature_probe.py`)

Chỉ dùng target (OT) của NP, tính trong không gian RevIN của từng window. Ridge fit trên 4563 window train (stride 8), đánh giá trên 436 window test. Đo hai việc: dự báo 24h kế tiếp, và tái tạo 24h cuối của input. Chạy khoảng 3 phút, không train model.

| Đầu vào cho ridge | Forecast MSE ↓ | Recon R² ↑ |
|---|---|---|
| Seasonal naive (lặp lại 24h trước) | 1.190 | — |
| Chuỗi gốc (168 giá trị) | **1.141** | 1.000 |
| DINOv3 pretrained, pool 7 + CLS (đúng như model) | 1.559 | 0.555 |
| DINOv3 pretrained, đủ 121 patch + CLS | 1.406 | 0.771 |
| DINOv3 random-init, pool 7 + CLS | 1.319 | 0.840 |
| DINOv3 random-init, đủ 121 patch + CLS | 1.141 | 0.919 |

MSE ở đây chỉ dùng để so tương đối, không so được với số benchmark. Giới hạn: probe tuyến tính, không có covariate, chỉ NP, một lần chia dữ liệu; phản ánh sát Regime 1 nhưng chưa chứng minh trực tiếp cho Regime 2.

## 4. Nguyên nhân (xếp theo mức bằng chứng)

1. **Mục tiêu tiền huấn luyện ngược với nhu cầu dự báo** (có bằng chứng từ probe). DINO học đặc trưng bất biến với crop, flip, color jitter. Ảnh GASF/GADF lại mã hoá giá trị chuỗi bằng cường độ pixel, và lật ảnh tương đương đảo chiều thời gian. Pretrained kém hơn random-init ở cả dự báo lẫn tái tạo.
2. **Pool 121 → 7 theo thứ tự raster** (có bằng chứng từ probe; `models/TimeXer_DINOv3.py`, Stage 6). `adaptive_avg_pool1d` gộp các token theo hàng của lưới 11×11, mỗi nhóm cắt ngang khoảng 1.5 hàng và không ứng với khoảng thời gian nào. R² tái tạo giảm từ 0.77 xuống 0.56.
3. **Ảnh không thêm thông tin.** Trường hợp tốt nhất (random-init, đủ patch) chỉ ngang ridge trên chuỗi gốc. 168 giá trị bị nội suy thành 3×176×176 pixel; mỗi patch phủ khoảng 15×15 giờ; channel periodic folding lặp mỗi hàng khoảng 24 lần.
4. **Target không có đường đi trực tiếp tới head.** TimeXer gốc có patch embedding tuyến tính từ chuỗi gốc; ở bản này, khi backbone làm mất thông tin thì không có đường bù lại.
5. **Covariate được fuse quá muộn và quá nông.** Fuse một lần sau backbone, cộng 1 layer self-attention; TimeXer cho global token cross-attend với covariate ở mỗi layer.
6. **Protocol chưa cân** (không giải thích được mức chênh, nhưng cần ghi lại): `d_model` 256 so với 512; 1 seed so với 3; cả hai đều bỏ `x_dec`, trong khi reference baseline theo ADR-0002 là TimeXer-F.

## 5. Nếu mở lại nhánh này

Chỉ mở lại khi đổi cả hai thứ: backbone học bằng tái tạo pixel (MAE, như VisionTS), và một nhánh residual giữ encoder 1D của TimeXer. Với backbone đóng băng, tính sẵn feature một lần thay vì chạy forward ở mọi epoch.
