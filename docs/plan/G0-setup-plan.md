# G0 Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện toàn bộ giai đoạn G0 (Setup môi trường, tham số hóa seed, script ghi nhận CSV, và đo ngân sách GPU thực tế) với can thiệp tối thiểu vào codebase.

**Architecture:** 
- Kế thừa tối đa code gốc của repo TimeXer.
- Sửa phẫu thuật duy nhất tại `run.py` để hỗ trợ cờ `--seed`.
- Bổ sung 1 script tiện ích tối giản duy nhất để chạy batch và gom kết quả thành file CSV theo đúng định dạng roadmap.

**Tech Stack:** Python 3.10+, PyTorch (CUDA), Pandas, NumPy.

**Spec:** [`docs/plan/proof-roadmap.md`](file:///c:/Users/USER/Desktop/TimeXer/docs/plan/proof-roadmap.md) (Mục G0), [`docs/adr/0004-reproduction-acceptance.md`](file:///c:/Users/USER/Desktop/TimeXer/docs/adr/0004-reproduction-acceptance.md).

## Global Constraints
- Không sửa kiến trúc mô hình hay pipeline dữ liệu trong giai đoạn G0.
- Giữ nguyên style code của repo gốc.
- Mọi kết quả đo được lưu về thư mục `docs/results/raw/`.

---

### Task 1: Xác thực môi trường GPU CUDA (G0.1)

**Files:**
- Test trực tiếp qua lệnh CLI terminal

- [x] **Step 1: Kiểm tra PyTorch CUDA trên môi trường thực thi**
  Kiểm tra phiên bản torch và trạng thái CUDA sẵn sàng:
  ```powershell
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  ```
  Expected: `CUDA available: True` và `Device: NVIDIA GeForce RTX 2050`.
  **Kết quả:** Đã xác thực thành công trên GPU `NVIDIA GeForce RTX 2050`.

---

### Task 2: Tham số hóa seed trong `run.py` (G0.2)

**Files:**
- Modify: `run.py:13-20` và phần parser `run.py:100-110`

- [x] **Step 1: Xóa hardcode seed ở đầu file `run.py`**
  Đã xóa đoạn hardcode `fix_seed = 2021`.

- [x] **Step 2: Thêm đối số `--seed` vào ArgumentParser**
  Đã cập nhật default seed = 2021 trong parser.

- [x] **Step 3: Thiết lập seed sau khi parse args**
  Đã thêm `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.

- [x] **Step 4: Kiểm thử tính tất định của seed**
  - Chạy với `--seed 2021`: Đạt `MSE = 0.356667697429657` (khớp chính xác 100% với smoke test cũ).
  - Chạy với `--seed 2022`: Đạt `MSE = 0.3606089651584625`, `MAE = 0.3538800776004791`.
  - **Kết luận:** Hai lần chạy cùng seed cho cùng số; hai seed khác nhau cho kết quả khác nhau. Xác nhận thành công tính tất định của seed.

---

### Task 3: Đo thời gian chạy 1 epoch EPF-NP (G0.4)

**Files:**
- Tạo báo cáo: `docs/results/G0-gpu-budget.md`

- [x] **Step 1: Chạy 1 epoch smoke test trên EPF-NP**
  Đã chạy thành công 1 epoch trên RTX 2050 (hết 326.13 giây).

- [x] **Step 2: Ghi nhận thời gian và ước lượng ngân sách**
  Đã hoàn thành báo cáo chi tiết tại `docs/results/G0-gpu-budget.md`.

---

### Task 4: Tạo runner tối giản gom kết quả ra CSV (G0.3)

**Files:**
- Create: `utils/export_results.py`

- [x] **Step 1: Viết hàm đọc `result_long_term_forecast.txt` và xuất ra CSV**
  Đã tạo `utils/export_results.py`.

- [x] **Step 2: Test chuyển đổi trên file kết quả hiện có**
  Đã xuất thành công dữ liệu ra `docs/results/raw/results.csv`.
