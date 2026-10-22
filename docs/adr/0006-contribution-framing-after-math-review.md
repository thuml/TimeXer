---
status: accepted
---

# Đóng góp của PERIN là symmetry choice hỗn hợp theo kênh, không phải "định lý mới" hay "đưa thống kê trở lại"

Kết quả đối chiếu nguồn gốc (`docs/research/normalization-symmetry-math.md`) cho thấy:
- đặc trưng hoá họ RevIN là kết quả đã biết (Lehmann–Casella 1998; Bloem-Reddy & Teh 2020; Kaba et al. 2023; Saied & Fleuret 2026);
- công thức gap là MRE location–scale cộng phép chiếu $L^2$;
- $D_S$ là LOCO/VIM, và không có kiểm định $D_S=0$ nào hợp lệ đều;
- nửa "không định danh ngoài support" đã có (Christiansen et al. 2022; Dong & Ma 2023);
- mỗi thành phần của phương pháp đều có tiền lệ: Non-stationary Transformer (đưa lại thống kê *mọi* kênh), α-RevIN, Lag-Llama/DeepAR, Residual Pathway Priors.

Vì vậy paper trình bày các kết quả toán là **Mệnh đề** có trích nguồn, và đặt đóng góp ở bốn điểm chưa ai làm:
1. **Symmetry choice hỗn hợp theo kênh** khi có covariate. Đây đúng là trường hợp Saied–Fleuret và Berthelier loại trừ tường minh.
2. **Hệ quả A:** hiệu ứng phụ thuộc mức của covariate, có cận dưới định lượng, kiểm trên dataset có vật lý đảm bảo.
3. **Mệnh đề 3(c):** đặc trưng bị chặn dẫn tới sai số ngoại suy $\le B^2$, nằm trong hệ số 4 của cận minimax.
4. **Đánh giá có kiểm soát** trên dataset có covariate thật.

## Consequences

- **Baseline bắt buộc thêm cho G2/G6:** NST (de-stationary attention), α-RevIN, "thống kê target làm feature" kiểu Lag-Llama, và nhánh song song kiểu RPP. Thắng RevIN thôi là không đủ.
- **Đổi cách nói về $D_c$:** là *ước lượng cỡ hiệu ứng* (sample-split cộng cross-fit), không phải "kiểm định trước khi huấn luyện".
- **Rủi ro lớn nhất:** nếu G3 cho thấy defect phía covariate ≈ 0 trên cả dữ liệu tải–nhiệt độ, thì điểm 1 và 2 yếu đi. Khi đó paper chỉ còn là α-RevIN có lý thuyết, không đủ cho main track.
