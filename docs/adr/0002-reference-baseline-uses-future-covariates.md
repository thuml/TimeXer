---
status: accepted
---

# Reference baseline là TimeXer có future-known covariate cộng baseline mạnh nhất, không phải TimeXer gốc

TimeXer gốc bỏ qua `x_dec`/`x_mark_dec` (`models/TimeXer.py:157-170`). Trên EPF, nó không dùng dự báo phụ tải và gió cho ngày tới, dù thông tin này có sẵn trong CSV. Probe ở v0.5 cho thấy chỉ riêng việc thêm các covariate này đã giảm MSE 5–30%.

Nếu so method mới với TimeXer gốc, ta sẽ nhận nhầm công của covariate tương lai (vốn đã có ở DAG, CITRAS, TiDE, Chronos-2) thành công của PERIN. Vì vậy, reference baseline trên mỗi dataset là mô hình tốt nhất trong các lựa chọn sau:
- TimeXer + future-known covariate (gọi là **TimeXer-F**);
- TimeXer-F + Non-stationary de-stationary factors;
- các baseline hiện đại ở giai đoạn 2 chạy trên cùng protocol.

## Consequences

- TimeXer gốc chỉ xuất hiện ở hàng "reproduction", không bao giờ là mốc để tính % cải thiện.
- Dataset không có future-known covariate (ECL, Weather, ETT, Traffic ở setting `MS`) chỉ dùng để kiểm tra phía past-only và để chứng minh method không làm hại, không dùng làm bằng chứng chính.
