---
status: accepted
---

# Không triển khai method trước khi qua các evidence gate

Brainstorm v0.3 → v0.5 đã ba lần đổi hướng: H1 và H2 hoá ra đã có người làm, còn H3 chỉ đúng một phần. Nguyên nhân chung là ta đưa ra giả thuyết trước khi đo. Vì vậy, từ giờ công việc đi theo roadmap cố định trong `docs/plan/proof-roadmap.md`. Mỗi giai đoạn kết thúc bằng một evidence gate có ngưỡng định lượng viết ra *trước* khi chạy:

1. **Reproduction** TimeXer.
2. **Baseline hiện đại** trên cùng protocol.
3. **Đo gap:** headroom và equivariance defect.
4. **Kiểm tra toán** với nguồn chính thống.
5. **Triển khai** PERIN.
6. **Thí nghiệm đầy đủ.**

Không viết code method (giai đoạn 5) khi giai đoạn 1–4 chưa qua gate.

## Considered Options

- **Triển khai PERIN ngay rồi đo sau.** Nhanh hơn khoảng 2 tuần. Bị bác vì probe EPF cho thấy thành phần phía covariate chỉ nằm trong khoảng nhiễu. Nếu gap không có thật, 2 tuần đó thành công cốc.
- **Chạy giai đoạn 2–4 song song.** Chấp nhận một phần: giai đoạn 4 (đọc tài liệu) chạy song song với 1–3 vì không tốn GPU.

## Consequences

- Ngưỡng của gate không được sửa sau khi đã thấy kết quả. Nếu muốn đổi, phải viết ADR mới giải thích lý do.
- Kết quả âm tính vẫn phải ghi lại trong `docs/results/`, vì nó quyết định có rút về paper phân tích hay không.
