---
status: proposed
---

# Tên method: PERIN (Partially Equivariant Reversible Instance Normalization)

Đóng góp là thay symmetry choice cố định của RevIN (đẳng biến theo target, bất biến theo mọi covariate) bằng một symmetry choice *theo từng kênh*, được chọn dựa trên equivariance defect. Tên cần nói được hai điều: đây là họ hàng của RevIN (để reviewer định vị ngay), và điểm khác nằm ở "partial equivariance". Chọn **PERIN**.

Tìm kiếm ngày 25/09/2026 không thấy "PERIN" hay "SyRIN" trong tài liệu dự báo chuỗi thời gian. Tên phải được kiểm tra lại ngay trước khi nộp paper.

## Considered Options

- **ExoShift** (tên ở v0.4): gắn với hướng dịch phân phối do sự kiện, mà hướng đó đã bị thay. Bị bác.
- **PEN** (tên ở v0.5): quá chung chung, dễ trùng.
- **SyRIN** (Symmetry-aware RevIN): nghe dễ lẫn. Giữ làm dự phòng.
- **NoRIN, α-RevIN, IN-Flow**: đã có người dùng.

## Consequences

Nếu giai đoạn 3 cho thấy chỉ có phía target đóng góp (covariate-side defect ≈ 0), tên vẫn đúng: partial equivariance áp dụng cho target. Nhưng câu chuyện "covariate" phải thu hẹp lại, xem roadmap.
