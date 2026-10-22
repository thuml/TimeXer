# Báo cáo ngân sách GPU và Phân tích Seed (G0.4)

*Thời gian đo: 28/09/2026. Môi trường: NVIDIA GeForce RTX 2050 (4096 MiB VRAM), PyTorch CUDA trên Windows.*

## 1. Kết quả đo thực nghiệm trên EPF-NP (1 epoch)

- **Cấu hình chạy:**
  - Mô hình: `TimeXer` (`e_layers=3`, `d_model=512`, `d_ff=512`, `patch_len=24`)
  - Dataset: `EPF/NP.csv` (`seq_len=168`, `pred_len=24`, `features=MS`)
  - Batch size: `4` (9.125 bước huấn luyện / epoch)
- **Thời gian đo được:**
  - **1 epoch:** `326.13 giây` (~**5.4 phút**)
  - Tốc độ huấn luyện trung bình: ~`0.028 – 0.033 giây / iter`
  - Đánh giá tập test (10.460 mẫu): ~`15 – 20 giây`
- **Tài nguyên VRAM chiếm dụng:**
  - Peak VRAM: ~`1.5 – 2.0 GB` (hoàn toàn an toàn trên card 4GB, không bị OOM).

## 2. Ước lượng thời gian huấn luyện đầy đủ

Thông thường trên bộ dữ liệu EPF, mô hình hội tụ và dừng sớm (Early Stopping với `patience=3`) sau khoảng **5 – 8 epochs**.

| Kịch bản | Thời gian 1 lần chạy (1 seed, ~6 epochs) | 3 Seeds / Dataset | 5 Seeds / Dataset | Toàn bộ 5 Dataset EPF (3 seeds) | Toàn bộ 5 Dataset EPF (5 seeds) |
|---|---|---|---|---|---|
| **Batch size 4** (như paper gốc NP, DE) | ~30 – 35 phút | **~1.5 – 1.8 giờ** | **~2.5 – 3.0 giờ** | **~7.5 – 9.0 giờ** | **~13 – 15 giờ** |

*Ghi chú:* Đối với PJM, BE, FR, script gốc `scripts/forecast_exogenous/EPF/TimeXer.sh` dùng `batch_size 16` (thay vì 4), nên thời gian mỗi epoch của các dataset này sẽ giảm khoảng 3 lần (~1.8 phút/epoch).

## 3. Quyết định về số lượng Seed (3 seeds vs 5 seeds)

1. **Khẳng định tính khả thi:**
   - Nếu chạy **3 seeds**: Tổng thời gian toàn bộ EPF rơi vào khoảng **~5 – 6 giờ** (có thể chạy qua đêm hoặc chia theo từng đợt).
   - Nếu chạy **5 seeds**: Tổng thời gian sẽ lên tới **~10 – 12 giờ**.
2. **Khuyến nghị kết luận:**
   - **Giai đoạn G0 & G1 thăm dò / tái lập ban đầu:** Sử dụng **3 seeds** (`2021, 2022, 2023`) là tối ưu nhất về mặt chi phí thời gian phần cứng local (RTX 2050), giảm được **40% thời gian chờ đợi** mà vẫn tính được `mean ± std`.
   - **Giai đoạn chốt Paper / Main track (G2, G6):** Sau khi xác nhận method PERIN có hiệu quả ở G3, sẽ cân nhắc bổ sung 2 seed còn lại nếu cần nộp main track hội nghị top-tier.
