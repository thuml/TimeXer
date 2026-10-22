# Roadmap chứng minh: từ tái lập TimeXer đến PERIN

*Tạo ngày 25/09/2026. Quy tắc lấy từ ADR-0001: không viết code PERIN khi giai đoạn 1–4 chưa qua gate, và không sửa ngưỡng của gate sau khi đã thấy kết quả. Thuật ngữ theo `CONTEXT.md`.*

```
G0 Setup ─► G1 Reproduction ─► G2 Baseline hiện đại ─► G3 Đo gap ─┐
                                                                  ├─► G5 Triển khai PERIN ─► G6 Thí nghiệm đầy đủ và paper
G4 Kiểm tra toán (chạy song song từ bây giờ, không tốn GPU) ──────┘
```

Kết quả của mỗi giai đoạn ghi vào `docs/results/G<n>-<tên>.md`, kèm CSV thô trong `docs/results/raw/`.

---

## G0: Setup (khoảng 0.5 ngày)

| Việc | Chi tiết | Xong khi |
|---|---|---|
| G0.1 Cài môi trường GPU | Anaconda hiện có `torch 2.9.1+cpu`, chưa có CUDA. Cài bản torch CUDA phù hợp với RTX 2050 vào môi trường riêng (`conda create -n timexer`). Cài `requirements.txt`, nhưng **không** hạ torch xuống 2.0.0 nếu không cần | `torch.cuda.is_available()` trả về True |
| G0.2 Sửa seed | `run.py:14` hard-code `fix_seed = 2021`, còn `--seed` không được dùng. Đổi thành seed lấy từ tham số (ADR-0004) | Hai lần chạy cùng seed cho cùng số; hai seed khác nhau cho số khác nhau |
| G0.3 Ghi kết quả | Script chạy nhiều seed rồi gom `result_long_term_forecast.txt` thành CSV: dataset, horizon, model, seed, MSE, MAE, thời gian | Có `docs/results/raw/*.csv` |
| G0.4 Đo thời gian | Chạy 1 epoch EPF-NP để ước lượng ngân sách GPU của G1–G2 | Có bảng thời gian ước tính |

---

## G1: Reproduction TimeXer

**Mục tiêu:** chứng minh code, dữ liệu và protocol của ta khớp với paper, để mọi mức cải thiện về sau có nghĩa.

**Số liệu mục tiêu** (paper, arXiv 2402.19072, Table 2 EPF, MSE / MAE):

| | NP | PJM | BE | FR | DE | AVG |
|---|---|---|---|---|---|---|
| TimeXer | 0.236 / 0.268 | 0.093 / 0.192 | 0.379 / 0.243 | 0.385 / 0.208 | 0.440 / 0.415 | 0.307 / 0.265 |

**Long-term exogenous** (Table 11, trung bình 4 horizon, MSE / MAE): ETTh1 0.073 / 0.209 · ETTm1 0.052 / 0.171 · Weather 0.002 / 0.031. Còn ECL 0.327 / 0.408 và Traffic 0.498 / 0.350 thì chạy trên GPU thuê hoặc bỏ qua (ADR-0004).

**Kiểm tra thêm, lấy cùng lần chạy:** iTransformer và PatchTST trên EPF. Paper báo AVG 0.335 và 0.330. Nếu baseline tái lập được mà TimeXer thì không, lỗi nằm ở TimeXer chứ không nằm ở pipeline.

**Gate G1:**
- ≥ 80% số ô bắt buộc nằm trong dung sai $\max(2\,\text{std},\,3\%)$.
- Thứ hạng TimeXer > iTransformer > RLinear trên EPF AVG được giữ nguyên.

**Nếu trượt:** so các bước tiền xử lý (split 70/10/20 trong `Dataset_Custom`, `scale=True`, `inverse=False`), rồi `patch_len`, rồi `batch_size`. Chỉ sang G2 khi đã hiểu nguyên nhân chênh lệch.

---

## G2: Baseline hiện đại trên cùng protocol

**Mục tiêu:** biết hiện ai mạnh nhất. Chọn reference baseline cho từng dataset theo ADR-0002.

### Dataset

| Nhóm | Dataset | Vai trò |
|---|---|---|
| Chính (có future-known covariate thật) | EPF: NP, PJM, BE, FR, DE (có sẵn) | Bằng chứng chính. Covariate là *dự báo*, nên không bị rò rỉ oracle |
| Chính (level-dependent theo vật lý) | Tải điện theo nhiệt độ: GEFCom2014-L, hoặc tải ENTSO-E kèm nhiệt độ; **cần xác minh giấy phép và cách tải** | Kiểm tra phía covariate của symmetry mismatch |
| Phụ | DAG TSF-X: 7 bộ có future exogenous (repo `decisionintelligence/DAG`); **cần xác minh nội dung từng bộ** | Tính tổng quát |
| Kiểm tra không gây hại | ETTh1, ETTm1, Weather (setting MS, past-only) | PERIN không được làm tệ đi |

### Mô hình

| Nhóm | Mô hình | Nguồn |
|---|---|---|
| Có sẵn trong repo | TimeXer, iTransformer, PatchTST, TiDE, TemporalFusionTransformer, TimeMixer, TSMixer, DLinear, Nonstationary_Transformer | `models/` |
| Hạ tầng cho reference baseline | **TimeXer-F**: TimeXer + future-known covariate. Đây là bước chuẩn bị baseline, không phải PERIN | Sửa nhỏ `models/TimeXer.py` sau cờ `--use_future_exo` |
| Code ngoài, chuyên cho covariate | DAG, CrossLinear (KDD'25), CITRAS (cần kiểm tra có code không) | GitHub của tác giả |
| Thống kê và chuyên EPF | LEAR (epftoolbox), Ridge có covariate tương lai | epftoolbox |
| Foundation model (tham chiếu, ADR-0003) | Chronos-2 (zero-shot, có covariate), TiRex-2, TabPFN-TS | HF / pip |

**Protocol:** cùng split, cùng `seq_len` (168 cho EPF), 5 seed. Metric là MSE/MAE trên thang chuẩn hoá toàn cục (như paper), kèm MAE ở thang gốc cho EPF để so với LEAR.

**Sản phẩm:** `docs/results/G2-leaderboard.md`, mỗi dataset một bảng mean ± std, kèm kiểm định Diebold–Mariano theo cặp so với reference baseline.

**Gate G2:**
- Mỗi dataset chính có một reference baseline được chọn rõ ràng.
- Ghi lại TimeXer-F tốt hơn TimeXer bao nhiêu (probe dự đoán 5–30%).

Nếu một mô hình ngoài (ví dụ LEAR hay TabPFN-TS) vượt xa mọi deep model, phải ghi lại ngay, vì điều đó đổi reference baseline.

---

## G3: Đo gap trước khi cải thiện

**Mục tiêu:** trả lời bằng số hai câu hỏi. Symmetry mismatch có lớn đủ trên dữ liệu thật không? Và nó nằm ở phía target hay phía covariate?

| Việc | Cách làm |
|---|---|
| G3.1 Equivariance defect $\hat D_c$ | Mở rộng `docs/brainstorm/probes/epf_probe.py`. Hồi quy target đã chuẩn hoá theo normalized window, lần có và lần không có window statistics của kênh $c$ (GBM và MLP). Dùng sample-split cộng cross-fitting theo khối thời gian (VIM kiểu Williamson et al.), báo cáo *cỡ hiệu ứng* kèm CI phụ thuộc giả định hồi quy. **Không** gọi là kiểm định $D=0$, vì Shah–Peters chứng minh không tồn tại kiểm định hợp lệ đều (ADR-0006) |
| G3.2 Statistics extrapolation risk | Với mỗi kênh, tính tỷ lệ window ở tập test có mean hoặc std nằm ngoài khoảng [1%, 99%] của tập train |
| G3.3 Headroom trên backbone thật | Chạy chẩn đoán: reference baseline + window statistics thô đưa vào global token. **Không tune**, chỉ để đo trần cải thiện. Đây *không* phải PERIN, vì không có đặc trưng bị chặn, không chọn kênh, không có nhánh thang gốc |
| G3.4 Phân loại dataset | Vẽ biểu đồ ($\hat D_c$, extrapolation risk) cho từng kênh. Định lý 3 dự đoán PERIN chỉ có lợi ở góc "defect cao, rủi ro thấp" |

**Gate G3** (ngưỡng viết trước, ADR-0001):

| Kết quả | Quyết định |
|---|---|
| ≥ 2 dataset chính có $\hat D$ với CI không chứa 0 **và** headroom G3.3 ≥ 5% MSE so với reference baseline | **GO** sang G5 |
| Như trên, nhưng chỉ phía target có ý nghĩa, phía covariate thì không | GO, nhưng thu hẹp câu chuyện. Viết ADR-0006 điều chỉnh phạm vi |
| Headroom < 5% ở mọi dataset chính | **NO-GO** với paper method. Chuyển sang paper phân tích hoặc workshop (TSALM) với định lý và chẩn đoán |

---

## G4: Kiểm tra toán (đang chạy song song)

| Việc | Sản phẩm |
|---|---|
| G4.1 Nghiên cứu tài liệu gốc: lý thuyết đẳng biến, maximal invariant, ngoại suy, kiểm định invariance, prior art về chuẩn hoá | `docs/research/normalization-symmetry-math.md`. **Xong 25/09** |
| G4.2 Phân loại từng định lý: *đã biết* / *hệ quả của X* / *mới* / *sai* | Bảng verdict trong file trên. **Xong 25/09**: mọi định lý đều "đã biết" hoặc "hệ quả"; hai khẳng định bị đánh sai (NST; "kiểm định trước huấn luyện"). Khung đóng góp chốt ở ADR-0006 |
| G4.3 Viết lại phát biểu chính xác (giả thiết, nhóm không compact, moment) và chứng minh đầy đủ | `docs/theory/theorems.md` |
| G4.4 Rà prior art lần cuối: α-RevIN, Berthelier 2026, NoRIN, APT, Non-stationary Transformer | Bảng khác biệt trong `docs/theory/theorems.md` |

**Gate G4:**
- Không còn định lý nào bị đánh "sai".
- Có ít nhất một phát biểu được đánh "mới", hoặc có một hệ quả chưa ai nêu cho covariate.

Nếu mọi thứ đều "đã biết", đóng góp chỉ còn là thực nghiệm và chẩn đoán. Khi đó phải ghi ADR và hạ kỳ vọng về nơi nộp.

---

## G5: Triển khai PERIN (chỉ khi G1–G4 đều qua)

**Mỗi thành phần là một cờ riêng để làm ablation:**
- `--use_future_exo`: đã có từ G2.
- `--exo_norm {revin, global, asinh}`.
- `--stat_features {none, target, exo, all}` và `--stat_encoding {raw, rank_clip}`.
- `--channel_select {all, defect}`: chọn kênh bằng $\hat D_c$ từ G3.
- `--raw_branch`: nhánh ở thang gốc.
- `--var_id`: identity embedding cho token ngoại sinh.

**Kiểm thử bắt buộc** (viết test trước khi viết code):
- **Property test đẳng biến.** Với mọi phép affine $g$ tác động lên target, output phải biến đổi đúng theo $g$. Với các kênh covariate *không* được chọn, output phải bất biến.
- **Property test khi tắt mọi cờ.** Output phải trùng TimeXer-F từng bit (cùng seed).

---

## G6: Thí nghiệm đầy đủ và quyết định paper

- **Ma trận thí nghiệm:** PERIN cắm vào mọi backbone của G2 × mọi dataset × 4 horizon × 5 seed.
- **Ablation:** từng cờ của G5. So với Non-stationary Transformer (baseline gần nhất, vì nó đưa lại thống kê mọi kênh), α-RevIN, thống kê target làm feature kiểu Lag-Llama, nhánh song song kiểu RPP, SAN, FAN, NoRIN, APT, và global scaling (ADR-0006).
- **Kiểm chứng lý thuyết:**
  1. Tương quan giữa $\hat D_c$ (G3) và mức cải thiện thực tế khi giữ window statistics của kênh $c$.
  2. Đường cong đánh đổi khi cố ý dịch thống kê giữa train và test.
- **Tiêu chí main track:** PERIN thắng reference baseline có ý nghĩa thống kê (5 seed, Diebold–Mariano) trên ≥ 60% cặp dataset × horizon chính, và $\hat D_c$ dự đoán đúng dấu cải thiện. Nếu không đạt thì nộp workshop.

---

## Ngân sách ước tính (sẽ cập nhật sau G0.4)

| Giai đoạn | Số lần chạy | GPU |
|---|---|---|
| G1 | EPF 5 × 5 seed; ETTh1, ETTm1, Weather 3 × 4 horizon × 5 seed; iTransformer và PatchTST EPF 2 × 5 × 5 | Local RTX 2050 |
| G2 | Khoảng 12 mô hình × 5–10 dataset × 5 seed | Local, cộng GPU thuê cho foundation model và ECL/Traffic |
| G3 | Probe chạy CPU; headroom khoảng 2 dataset × 5 seed | Local |
