# Nâng cấp TimeXer v0.4: khoảng trống toán học cho paper cải tiến mô hình

*24/09/2026. Bản này chuyển hướng từ paper "chứng minh" (v0.3) sang paper **cải tiến mô hình**. Phần phân tích ở v0.3 giờ chỉ làm động lực (motivation), không còn là đóng góp chính.*

---

## 0. Trả lời ngắn

**Có khoảng trống để cải tiến**, và nó trùng với trực giác thứ 2 và thứ 3 của bạn. Tuy vậy, cần phát biểu lại cho đúng về toán:

> TimeXer **không có cơ chế giải mã horizon** (horizon decoder). Biến ngoại sinh chỉ đi vào dự báo qua một vector toàn cục $g \in \mathbb R^d$ và một head tuyến tính phẳng (flatten), còn việc khử chuẩn hoá luôn dùng thống kê **quá khứ**. Hệ quả là mô hình không biểu diễn hiệu quả được *phản ứng theo thời gian* của target trước một cú sốc ngoại sinh đã biết xảy ra trong horizon. Cú sốc có thể là tạm thời (spike) hoặc kéo dài (dịch mức, tức là phân phối của target đổi ngay trong cửa sổ dự báo).

Có ba lỗ hổng toán học cụ thể, kèm cách sửa và bằng chứng sơ bộ (Mục 2):

| # | Lỗ hổng | Cách sửa | Sơ bộ (tổng hợp, 3 seed) |
|---|---|---|---|
| H1 | Không có **tính đẳng biến theo thời gian** giữa biến ngoại sinh tương lai và đầu ra | Head cục bộ dùng chung trọng số theo từng bước $h$ | −8 đến −10% (spike, spike hiếm) |
| H2 | Không có **trạng thái tích luỹ** cú sốc trong horizon, nên không biểu diễn được dịch mức kéo dài | SSM tuyến tính trên horizon, được kích bởi biến ngoại sinh | **−23%** (dịch mức) |
| H3 | **Khử chuẩn hoá bằng thống kê quá khứ** nên không dịch được phân phối theo cú sốc đã biết | Khử chuẩn hoá có điều kiện theo biến ngoại sinh: $\hat\mu_h, \hat\sigma_h$ | Cộng dịch chuyển ở thang gốc: −17% so với thang chuẩn hoá (E2) |

**Về khối transformer cũ:** thay khối encoder *không phải* là khoảng trống. Bằng chứng: iTransformer dùng đúng khối Post-LN cổ điển mà vẫn thắng TimeXer trên Traffic, và các thí nghiệm v0.2 cho thấy Pre-LN/QK-norm/gate không giúp gì. Kiến trúc hiện đại **chỉ có lý do toán học ở một chỗ**: bộ giải mã horizon cần *theo dõi trạng thái* (H2). Đây đúng là lập luận của TiRex (xLSTM) và là lý do TiRex-2 dùng xLSTM cho covariate tương lai. Như vậy "nâng cấp khối" nên vào **decoder**, không vào encoder.

**Về Traffic:** bằng chứng ở Traffic thật ra ủng hộ một lỗ hổng *khác*, xem Mục 1.3.

---

## 1. Đánh giá ba trực giác của bạn

### 1.1 "Khối transformer còn cũ": không phải nút thắt

- **Cùng một khối mà kết quả ngược nhau:** iTransformer dùng *cùng* encoder Post-LN và softmax cổ điển, nhưng thắng TimeXer ở Traffic đa biến (0.428 so với 0.466, Bảng 3 của paper). Khác biệt vì thế nằm ở **cách tổ chức token**, không nằm ở khối.
- **Thí nghiệm v0.2:** MOD (Pre-LN, QK-norm) và ABS (null KV, gate) không cải thiện ở 2 và 6 lớp.
- **Theo toán:** lợi ích của Pre-LN nằm ở gradient khi mạng sâu (Xiong et al. 2020). TimeXer chỉ có 1–3 lớp.

→ Nên dùng khối hiện đại làm **mặc định** (Pre-LN/RMSNorm, SwiGLU, RoPE) nhưng **không nhận đó là đóng góp**.

### 1.2 "Biến ngoại sinh bị nén thành một token nên mất khả năng suy diễn tương lai": đúng, nhưng lý do toán học khác với trực giác

Token $e_c = W x_c$ vẫn *chứa* thông tin (Mệnh đề 1 ở v0.2), kể cả khi mở rộng sang tương lai: $W \in \mathbb R^{d \times (L+H)}$. Vấn đề nằm ở **đường đi từ token đến đầu ra**:

$$\hat y_{t+h} = \big[W_{head}\,\mathrm{vec}(P, g)\big]_h,\qquad g = g_0 + \textstyle\sum_c \alpha_c\,W_oW_v e_c .$$

Giả sử một sự kiện ở vị trí tương lai $\tau$ tạo phản ứng $A\,\kappa(h - \tau)$. Đáp án đúng là một **toán tử Toeplitz**: cùng một kernel $\kappa$ trượt theo $\tau$. Kiến trúc hiện tại phải học riêng từng cặp $(\tau, h)$, tức là **$O(H^2)$ tham số hiệu dụng không chia sẻ**, thay vì $O(|\kappa|)$.

Với sự kiện hiếm (xác suất $p$ mỗi bước), mỗi cặp $(\tau, h)$ chỉ xuất hiện khoảng $Np$ lần. Sai số ước lượng vì thế tăng theo $H/|\kappa|$ lần. Đây chính là "mất khả năng suy diễn tương lai" phát biểu dưới dạng **thiếu tính đẳng biến tịnh tiến** (translation equivariance) và **độ phức tạp mẫu** (sample complexity).

### 1.3 "Sự kiện bất ngờ làm lệch phân phối cả nội sinh lẫn ngoại sinh, bằng chứng ở Traffic"

Có hai trường hợp khác nhau về toán, cần tách ra:

**(a) Sự kiện đã biết trước** (lịch nghỉ, dự báo thời tiết, dự báo phụ tải, thông báo thay đổi giá). Thông tin nằm trong $x_{t+1:t+H}$, nên kiến trúc có thể dự báo được sự kiện. Đây là chỗ H1–H3 phát huy tác dụng. Kèm theo là vấn đề khử chuẩn hoá:

$$\hat y_{t+h} = \sigma_{past}\, f_h + \mu_{past}.$$

Khi horizon có dịch mức $\Delta$ tại $\tau$, mạng phải xuất ra $\Delta/\sigma_{past}$ ở thang chuẩn hoá, một đại lượng phụ thuộc cửa sổ (Mệnh đề 3 của v0.2). Các phương pháp chuẩn hoá thích ứng như SAN, Dish-TS, FAN, CDFM có dự báo $\mu_{fut}$, nhưng **chỉ từ lịch sử target**. Chúng không thể lường trước một cú sốc chỉ được báo trong biến ngoại sinh. Sai số không thể khử của chúng là $\mathbb E\,\mathrm{Var}(\mu_{fut} \mid \mathcal F^{past}_y) > 0$.

Với biến ngoại sinh **thưa** (cờ sự kiện), RevIN còn gặp một lỗi số học. Nếu cửa sổ quá khứ không có sự kiện nào, $\sigma = \sqrt{\varepsilon} \approx 0.003$, nên giá trị tương lai bị khuếch đại khoảng 300 lần. Trong thí nghiệm, 15% cửa sổ của E2 và khoảng 60% cửa sổ của E1r rơi vào trường hợp này.

**(b) Sự kiện không biết trước** (tai nạn giao thông ở Traffic). Khi đó $\mathbb E[y_{t+h} \mid \mathcal F_t]$ là phản ứng sự kiện *tích chập với phân phối thời điểm xảy ra*. Dự báo tối ưu theo MSE **buộc phải làm mượt spike** (vấn đề "double penalty"). Không kiến trúc điểm nào tránh được điều này. Đó đúng là hiện tượng Phụ lục G của paper mô tả: bắt được xu hướng nhưng trượt spike.

Ở Traffic, biến ngoại sinh là 861 cảm biến **chỉ có quá khứ**, nên thông tin hữu ích chỉ là **lan truyền không gian có độ trễ** (tắc nghẽn phía trên lan xuống dưới). Bằng chứng trong paper khớp với điều này:
- Biến thể "Concatenate" (cho các token ngoại sinh tương tác với nhau) thắng TimeXer trên Traffic: 0.149 so với 0.156 (Bảng 8).
- iTransformer thắng ở chế độ đa biến.

→ **Traffic là bằng chứng cho tương tác giữa các biến ngoại sinh và lead-lag cục bộ theo thời gian, không phải cho sự kiện tương lai.** Muốn chứng minh H1–H3 thì cần dataset có covariate tương lai đã biết (Mục 4).

---

## 2. Thí nghiệm sự kiện (tổng hợp, dựa trên code gốc TimeXer, 3 seed)

**Các bài toán:**
- **E1:** cờ sự kiện thưa ($p = 0.02$), đã biết cả trong tương lai, gây spike $3\,[1, 0.7, 0.4]$.
- **E1r:** như E1 nhưng hiếm hơn ($p = 0.005$).
- **E2:** thông báo trước một bước nhảy mức ±2 có dấu, **mức kéo dài** trong horizon (phân phối target đổi ngay trong cửa sổ dự báo).

Mọi bài toán đều thêm một covariate liên tục. $L = 96$, $H = 24$.

MSE trên tập test đã chuẩn hoá; phần trăm là so với FUT:

| Biến thể | E1 (spike) | E1r (spike hiếm) | E2 (dịch mức kéo dài) |
|---|---|---|---|
| Ridge có biến ngoại sinh tương lai (tham chiếu tuyến tính) | 0.225 | 0.287 | 0.019 |
| V0: TimeXer gốc, chỉ quá khứ | 0.691 | 0.601 | 0.099 |
| **FUT: một token, trải qua quá khứ + tương lai** | 0.310 | 0.390 | 0.040 |
| FUT-raw: như FUT nhưng biến ngoại sinh không qua RevIN | 0.307 (−1%) | 0.384 (−1%) | 0.040 (+1%) |
| ALN-norm: head cục bộ đẳng biến, cộng trước khi khử chuẩn hoá | 0.286 (−8%) | 0.355 (−9%) | 0.040 (+1%) |
| **ALN-raw:** head cục bộ đẳng biến, cộng **sau** khi khử chuẩn hoá (thang gốc) | **0.278 (−10%)** | **0.351 (−10%)** | 0.033 (−16%) |
| **ALN-ssm:** ALN-raw + SSM đường chéo trên horizon | 0.280 (−10%) | 0.359 (−8%) | **0.031 (−23%)** |

**Đọc kết quả:**
1. **H1 được xác nhận:** head đẳng biến giảm 8–10% so với token đơn, kể cả khi token đơn *đã thấy* tương lai.
2. **H2 được xác nhận:** chỉ SSM (có trạng thái tích luỹ) cải thiện mạnh ở dịch mức kéo dài (−23%). Head cục bộ bán kính $r = 3$ không biểu diễn được phản ứng dạng bậc thang.
3. **H3 được xác nhận một phần:** ở E2, cộng dịch chuyển ở thang gốc (ALN-raw) tốt hơn cộng ở thang chuẩn hoá (ALN-norm): 0.033 so với 0.040. Lỗi $\sqrt\varepsilon$ của RevIN chỉ gây hại khoảng 1%, vì mạng học cách chịu được.
4. **Ridge vẫn tốt nhất**, vì các bài toán này **tuyến tính theo cấu trúc**. Paper thật cần thêm phản ứng *phi tuyến* với sự kiện, ví dụ biên độ spike phụ thuộc giờ trong ngày hoặc mức hiện tại, và dữ liệu thật. Mọi con số ở đây là chỉ báo sơ bộ về cơ chế, không phải kết quả để đăng.

---

## 3. Mô hình đề xuất: TimeXer + bộ giải mã horizon theo cú sốc ngoại sinh

*Tên làm việc: **ExoShift**, cần kiểm tra trùng tên.*

### 3.1 Phát biểu

Encoder giữ nguyên ý tưởng của TimeXer: patch self-attention, global token, cross-attention theo biến. Mình chỉ thêm một **bộ giải mã horizon** $\mathcal D$ và thay RevIN bằng **khử chuẩn hoá có điều kiện**:

$$\hat y_{t+h} \;=\; \underbrace{\mu_{past} + a_h}_{\hat\mu_h} \;+\; \underbrace{\sigma_{past}\, e^{b_h}}_{\hat\sigma_h}\; f_h(\text{encoder}),$$

$$u_h = \phi\big(\tilde x_{t+h-r\,:\,t+h+r},\; g\big),\qquad s_h = \Lambda s_{h-1} + B u_h,\qquad [a_h, b_h] = C s_h + D u_h,$$

Trong đó:
- $\phi$ dùng **chung cho mọi $h$** (đẳng biến theo thời gian, sửa H1);
- $\Lambda = \mathrm{diag}(\lambda_i)$ với $\lambda_i \in (0, 1]$ là SSM tuyến tính trên horizon (sửa H2);
- $(a_h, b_h)$ dịch và co giãn phân phối theo từng bước, có điều kiện theo biến ngoại sinh (sửa H3);
- $\tilde x$ là biến ngoại sinh ở **thang robust (asinh) kèm mask**, không dùng RevIN theo cửa sổ, để tránh lỗi $\sqrt\varepsilon$.

### 3.2 Tính chất toán học (có thể viết thành mệnh đề trong paper)

- **(T1) Tổng quát hoá RevIN và SAN/Dish-TS.**
  - Khi $a = b = 0$, mô hình quy về RevIN.
  - Khi $(a, b)$ chỉ phụ thuộc $y_{past}$, mô hình tương đương họ SAN/Dish-TS.
  - Khi $(a, b)$ phụ thuộc $x_{fut}$: nếu $\mu_{fut}$ đo được theo $\mathcal F^{fut}$, phần sai số không khử được $\mathbb E\,\mathrm{Var}(\mu_{fut} \mid \mathcal F^{past}_y)$ của họ SAN bị loại bỏ.
- **(T2) Đẳng biến và độ phức tạp mẫu.** Vì $\phi$ dùng chung theo $h$, ánh xạ từ biến ngoại sinh tương lai sang $a_h$ là một toán tử (phi tuyến) bất biến tịnh tiến. Số tham số phụ thuộc bán kính $r$, không phụ thuộc $H$. So với head phẳng, cận sai số ước lượng giảm khoảng $H/(2r+1)$ lần khi sự kiện hiếm.
- **(T3) Độ biểu diễn của phản ứng xung.** Phản ứng tuyến tính của $a_h$ với một xung tại $\tau$ có dạng $\sum_i c_i \lambda_i^{h-\tau}$ cộng thêm phần có giá hữu hạn từ $\phi$.
  - $\lambda = 1$ cho phản ứng bậc thang (dịch mức kéo dài);
  - $\lambda$ nhỏ cho phản ứng thoáng qua;
  - tổng các hàm mũ xấp xỉ được mọi kernel liên tục trên horizon hữu hạn.

  Head cục bộ thuần tuý chỉ biểu diễn được kernel có giá $\le r$. Đây là lý do toán học cho một **khối hiện đại (SSM/linear recurrence) đặt đúng chỗ**.
- **(T4) Ổn định số.** Với covariate thưa, RevIN theo cửa sổ cho $|\tilde x| \sim |x|/\sqrt\varepsilon$. Dùng asinh hoặc robust scaling toàn cục thì bị chặn.

### 3.3 Vì sao không trùng với công trình đã có (theo tìm kiếm của mình, cần rà thêm)

| Thành phần | Đã có | Khác biệt của ExoShift |
|---|---|---|
| Căn chỉnh covariate tương lai theo thời gian | TFT (decoder), TiDE (chiếu theo từng bước), CITRAS (KV shift), Chronos-2 (group attention cùng patch), TiRex-2 (xLSTM hai chiều) | **Không mới khi đứng riêng.** Chỉ là nền tảng bắt buộc |
| Dự báo thống kê tương lai để khử chuẩn hoá | SAN (NeurIPS'23), Dish-TS, FAN (NeurIPS'24), CDFM | Các phương pháp này **chỉ dùng lịch sử target**. ExoShift điều kiện theo **biến ngoại sinh tương lai** (T1). Chưa tìm thấy ai làm |
| SSM hoặc recurrence trên horizon, kích bởi covariate | TiRex-2 dùng xLSTM cho covariate tương lai *trong encoder* | ExoShift dùng recurrence để **sinh dịch chuyển phân phối** $(a_h, b_h)$ và phân tách phản ứng thoáng qua và kéo dài (T3) |
| Sự kiện ngoại sinh | EventTSF (IJCAI'26): sự kiện dạng văn bản, diffusion | ExoShift dùng covariate số và mô hình điểm hoặc quantile, gọn nhẹ và có cấu trúc |

→ **Đóng góp lõi** là T1 cộng T3: *khử chuẩn hoá có điều kiện theo biến ngoại sinh, sinh bởi một SSM trên horizon*. H1 là phần hạ tầng cần có.

---

## 4. Kế hoạch thí nghiệm

**Dữ liệu** (công khai, không cần thu thập):
- **Có covariate tương lai:** EPF (5 thị trường, dự báo ngày tới), EDF, Bike Sharing (thời tiết, ngày lễ), các task có covariate động đã biết trong fev-bench (30 task).
- **Đặc trưng lịch làm sự kiện:** ngày lễ, cuối tuần trên các dataset sẵn có.
- **Bộ tổng hợp:** E1/E1r/E2 cộng các biến thể phi tuyến; quét tần suất sự kiện $p \in \{0.001, \dots, 0.05\}$ để vẽ đường "sai số theo độ hiếm" (kiểm chứng T2).
- **Traffic:** chỉ dùng cho câu hỏi lead-lag, ưu tiên thấp.

**Baseline:**
- TimeXer, **TimeXer + tương lai** (mốc tối thiểu), CITRAS, DAG, TiDE, TFT;
- SAN, Dish-TS, FAN cắm vào TimeXer + tương lai (để cô lập T1);
- Chronos-2, TiRex-2 nếu chạy được;
- ridge/LEAR.

**Ablation:**
1. head phẳng / head cục bộ đẳng biến / cục bộ + SSM / chỉ cumsum;
2. RevIN / SAN / khử chuẩn hoá có điều kiện theo biến ngoại sinh;
3. RevIN / asinh cho covariate thưa;
4. phổ $\lambda$ học được, để diễn giải phản ứng thoáng qua và kéo dài;
5. encoder Post-LN / Pre-LN (để cho thấy phần block không phải nguồn cải thiện).

**Điều kiện tiếp tục** (1–2 tuần GPU): trên EPF và Bike Sharing, ExoShift phải thắng **TimeXer + tương lai** và **TimeXer + tương lai + SAN** một cách có ý nghĩa thống kê (≥5 seed) trên ≥60% cặp dataset × horizon. Nếu chỉ thắng trên dữ liệu tổng hợp, chuyển thành paper phân tích hoặc workshop.

**Nơi nộp:** ICML 2027 / NeurIPS 2027. Mệnh đề T1–T4 kèm các đường cong độ hiếm và phổ $\lambda$ đủ làm câu chuyện cho main track, *với điều kiện* dữ liệu thật xác nhận.

---

## 5. Rủi ro

1. **Hiệu ứng trên dữ liệu thật có thể nhỏ:** EPF có ít sự kiện rời rạc. Nên ưu tiên dataset có ngày lễ hoặc khuyến mãi trong fev-bench.
2. **Reviewer có thể nói đây là TiDE + SAN ghép lại:** cần T1 và T3 rõ ràng, cùng ablation cho thấy điều kiện theo biến ngoại sinh là thứ tạo khác biệt.
3. **Mô hình tuyến tính mạnh trên quan hệ tuyến tính:** luôn báo cáo ridge/LEAR và thiết kế thêm tác động phi tuyến.
4. **Toàn bộ số liệu ở đây là CPU, $d = 64$, 3 seed, dữ liệu tổng hợp.** Chưa chạy ExoShift đầy đủ (có cả $b_h$) trên dữ liệu thật.

## Nguồn

- TimeXer (paper bạn gửi; Bảng 3, 8 và Phụ lục G) · https://github.com/thuml/TimeXer
- SAN: https://openreview.net/pdf?id=5BqDSw8r5j · FAN: https://arxiv.org/html/2409.20371v1 · CDFM: https://arxiv.org/html/2503.02609v1
- EventTSF (IJCAI 2026): https://arxiv.org/abs/2508.13434
- TiRex: https://arxiv.org/abs/2505.23719 · TiRex-2: https://arxiv.org/html/2607.01204v1 · Chronos-2: https://arxiv.org/html/2510.15821v1
- CITRAS: https://arxiv.org/html/2503.24007v3 · TFT: https://arxiv.org/abs/1912.09363
- fev-bench: https://arxiv.org/pdf/2509.26468
- Code và kết quả: `timexer_diagnostics.zip` (thêm `events.py`, `res_events*.json`)
