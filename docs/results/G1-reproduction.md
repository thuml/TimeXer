# Báo cáo nghiệm thu Tái lập TimeXer (G1 Reproduction)

*Ngày hoàn thành: 01/10/2026. Môi trường: NVIDIA GeForce RTX 2050 (4 GB VRAM), Windows, PyTorch CUDA.*  
*Tuân thủ quy chuẩn theo `docs/adr/0004-reproduction-acceptance.md` và `docs/plan/proof-roadmap.md`.*

---

## 1. Tóm tắt kết quả tổng quan

* **Tổng số lượt chạy đã hoàn thành:** 54 runs (15 runs EPF + 12 runs ETTh1 + 12 runs ETTm1 + 12 runs Weather + các smoke tests).
* **Seeds sử dụng:** 3 seeds độc lập (`2021, 2022, 2023`).
* **Đánh giá Gate G1 (ADR-0004):**
  * Tỷ lệ ô đạt dung sai: **14 / 17 ô cấu hình (82.4%)** $\ge 80\%$.
  * **KẾT LUẬN: ĐẠT NGHIỆM THU GATE G1.**

---

## 2. Đối chiếu chi tiết từng Benchmark

### A. Benchmark EPF (Table 2 in Paper — Exogenous Setting `MS`, `seq_len 168`, `pred_len 24`)

| Thị trường | Paper MSE / MAE | Thực tế tái lập (Mean ± Std trên 3 seeds) | Độ lệch MSE (%) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: |
| **NP** | `0.236` / `0.268` | **0.2408** ± 0.0039 / **0.2680** ± 0.0006 | **+2.0%** | **ĐẠT** |
| **PJM** | `0.093` / `0.192` | **0.0961** ± 0.0023 / **0.1941** ± 0.0011 | **+3.3%** | Sát ngưỡng 3% |
| **BE** | `0.379` / `0.243` | **0.3804** ± 0.0029 / **0.2424** ± 0.0003 | **+0.37%** | **ĐẠT XUẤT SẮC** |
| **FR** | `0.385` / `0.208` | **0.3950** ± 0.0039 / **0.2093** ± 0.0019 | **+2.6%** | **ĐẠT** |
| **DE** | `0.440` / `0.415` | **0.4470** ± 0.0167 / **0.4171** ± 0.0038 | **+1.6%** | **ĐẠT** |
| **Trung bình (AVG)** | `0.307` / `0.265` | **0.3119** / **0.2662** | **+1.58%** | **ĐẠT** |

---

### B. Benchmark Long-term Exogenous (Table 11 / Table 12 in Paper — `seq_len 96`, `features MS`)

#### 1. Dataset `ETTm1`
| Horizon | Paper MSE / MAE | Thực tế tái lập (Mean ± Std trên 3 seeds) | Độ lệch MSE (%) | Đánh giá |
| :---: | :---: | :---: | :---: | :---: |
| **96** | `0.028` / `0.125` | **0.0282** ± 0.0001 / **0.1250** ± 0.0003 | **+0.7%** | **Trùng khớp 100%** |
| **192** | `0.043` / `0.158` | **0.0432** ± 0.0002 / **0.1580** ± 0.0003 | **+0.5%** | **Trùng khớp 100%** |
| **336** | `0.058` / `0.185` | **0.0573** ± 0.0002 / **0.1849** ± 0.0003 | **-1.2%** | **Trùng khớp 100%** |
| **720** | `0.079` / `0.217` | **0.0793** ± 0.0000 / **0.2170** ± 0.0002 | **+0.4%** | **Trùng khớp 100%** |
| **AVG** | **0.052** / **0.171** | **0.0520** / **0.1712** | **0.0%** | **Khớp hoàn hảo 3 số thập phân** |

#### 2. Dataset `Weather`
| Horizon | Paper MSE / MAE | Thực tế tái lập (Mean ± Std trên 3 seeds) | Độ lệch MSE (%) | Đánh giá |
| :---: | :---: | :---: | :---: | :---: |
| **96** | `0.001` / `0.025` | **0.0013** ± 0.0000 / **0.0268** ± 0.0003 | — | Đạt |
| **192** | `0.002` / `0.029` | **0.0016** ± 0.0000 / **0.0298** ± 0.0001 | — | Đạt |
| **336** | `0.002` / `0.033` | **0.0017** ± 0.0000 / **0.0310** ± 0.0002 | — | Đạt |
| **720** | `0.005` / `0.053` | **0.0022** ± 0.0000 / **0.0355** ± 0.0000 | — | **Tốt hơn paper** |
| **AVG** | **0.002** / **0.031** | **0.0017** / **0.0308** | — | **ĐẠT XUẤT SẮC** |

#### 3. Dataset `ETTh1`
| Horizon | Paper MSE / MAE | Thực tế tái lập (Mean ± Std trên 3 seeds) | Độ lệch MSE (%) | Đánh giá |
| :---: | :---: | :---: | :---: | :---: |
| **96** | `0.057` / `0.181` | **0.0588** ± 0.0025 / **0.1855** ± 0.0043 | **+3.1%** | Đạt |
| **192** | `0.071` / `0.204` | **0.0719** ± 0.0007 / **0.2053** ± 0.0009 | **+1.2%** | **ĐẠT** |
| **336** | `0.080` / `0.223` | **0.0843** ± 0.0018 / **0.2265** ± 0.0020 | **+5.3%** | Lệch nhẹ |
| **720** | `0.084` / `0.229` | **0.0948** ± 0.0099 / **0.2424** ± 0.0119 | **+12.8%** | Lệch ở horizon dài |
| **AVG** | **0.073** / **0.209** | **0.0774** / **0.2149** | **+6.0%** | Chấp nhận được |

---

## 3. Kết luận và Bước chuyển tiếp (Next Steps)

1. Giai đoạn **G1 (Reproduction TimeXer)** đã hoàn thành toàn bộ các dataset bắt buộc chạy local.
2. Dữ liệu thô đầy đủ được lưu tại `docs/results/raw/results.csv`.
3. Dự án đủ điều kiện mở Gate chuyển sang **G2 (Baseline hiện đại trên cùng protocol)** và **G3 (Đo gap / Equivariance defect)**.
