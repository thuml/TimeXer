# TimeXer research

Dự án nghiên cứu nâng cấp TimeXer (dự báo chuỗi thời gian có biến ngoại sinh). Mục tiêu là tìm và chứng minh một khoảng trống đủ cho một paper. File này là bảng thuật ngữ dùng chung trong docs, ADR và code.

## Language

### Dữ liệu và bài toán

**Target**:
Chuỗi cần dự báo; trong repo là cột cuối (`OT`) của setting `MS`.
_Avoid_: endogenous (chỉ dùng khi trích paper gốc), output

**Covariate**:
Chuỗi phụ trợ dùng để dự báo target nhưng bản thân không phải đối tượng dự báo.
_Avoid_: exogenous variable (trừ khi trích TimeXer), feature, auxiliary series

**Past-only covariate**:
Covariate chỉ biết đến thời điểm dự báo (ví dụ các cảm biến khác trong Traffic).

**Future-known covariate**:
Covariate có giá trị cho cả horizon tại thời điểm dự báo: lịch, sự kiện đã lên lịch, hoặc *dự báo* của một biến khác (dự báo phụ tải và gió trong EPF).
_Avoid_: future exogenous, known input

**Window**:
Một mẫu gồm lookback của target, covariate, và phần horizon của các future-known covariate.

**Horizon**:
Các bước tương lai cần dự báo (`pred_len`).

### Chuẩn hoá và đối xứng

**Window statistics**:
Mean và std của từng kênh, tính trên phần lookback của một window.
_Avoid_: instance stats, RevIN stats

**Normalized window**:
Window sau khi mỗi kênh trừ mean và chia std của chính nó. Đây là thứ duy nhất một mô hình họ RevIN nhìn thấy.

**Symmetry choice**:
Tập phép biến đổi affine theo từng kênh mà dự báo của mô hình phải *đẳng biến* (target) hoặc *bất biến* (covariate) với chúng. RevIN là một symmetry choice cụ thể, không phải một bước tiền xử lý trung lập.
_Avoid_: normalization strategy (khi đang nói về tính chất toán học)

**Symmetry mismatch**:
Tình trạng khi symmetry choice của mô hình không khớp với tính chất của kỳ vọng có điều kiện thật. Hệ quả là một sai số Bayes không khử được.

**Level-dependent effect**:
Hiệu ứng của covariate lên target phụ thuộc vào mức tuyệt đối của covariate (ngưỡng nhiệt độ, đường cong merit-order), không chỉ vào hình dạng.

**Absolute-unit effect**:
Hiệu ứng của covariate lên target tính theo đơn vị gốc của target, không tỷ lệ với độ biến động của target trong lookback.

**Equivariance defect**:
Phần sai số giảm được khi mô hình được biết window statistics của một tập kênh. Đây là thước đo độ lớn của symmetry mismatch cho từng kênh.
_Avoid_: gap (quá mơ hồ), importance

**Statistics extrapolation risk**:
Rủi ro khi window statistics ở tập test nằm ngoài vùng đã gặp lúc huấn luyện. Rủi ro này chỉ phát sinh khi mô hình được biết window statistics.

### Quy trình nghiên cứu

**Reproduction**:
Chạy lại code gốc và đạt số liệu của paper TimeXer trong dung sai đã định (ADR-0004).

**Reference baseline**:
Mô hình mạnh nhất *không* chứa đóng góp của ta trên cùng protocol. Method mới phải thắng mô hình này, không phải TimeXer gốc (ADR-0002).
_Avoid_: vanilla baseline

**Evidence gate**:
Điều kiện go/no-go ở cuối mỗi giai đoạn của roadmap (ADR-0001).

**PERIN**:
Tên method đề xuất: *Partially Equivariant Reversible Instance Normalization* (ADR-0005).
_Avoid_: PEN, ExoShift (tên làm việc cũ)
