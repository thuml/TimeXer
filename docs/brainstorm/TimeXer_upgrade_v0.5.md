# Nâng cấp TimeXer v0.5: điểm yếu, bản đồ mô hình hiện đại, và khoảng trống toán học còn mở

*25/09/2026. Bản này rà lại v0.4 sau khi kiểm tra tài liệu: H1 đã có người làm, H2/T3 là mô hình cổ điển, còn H3 hẹp nhưng vẫn mở. Tài liệu này tìm hướng đứng vững hơn cho paper và cho việc nâng cấp repo.*

> **Errata (25/09, sau G4):** các phát biểu toán ở Mục 4 đã được đối chiếu với nguồn gốc. Bản sửa chính xác nằm ở `docs/research/normalization-symmetry-math.md` Mục B, gồm Mệnh đề 0, Định lý 1, Định lý 2 và Mệnh đề 3. Khẳng định rằng NST chỉ đưa lại thống kê target là **sai**. "Đặc trưng bị chặn" cần Mệnh đề 3(c) để biện minh, không suy ra được từ Định lý 3.
>
> **Quy trình thực hiện:** xem `docs/plan/proof-roadmap.md` và các ADR trong `docs/adr/`. Thuật ngữ xem `CONTEXT.md`. Mục 6 bên dưới được thay bằng roadmap đó.

---

## 0. Trả lời ngắn

**Hướng đề xuất:** *Sự lệch đối xứng (symmetry mismatch) của chuẩn hoá trong dự báo có biến ngoại sinh.*

RevIN (`use_norm`) trong TimeXer, và trong gần như mọi mô hình hiện đại kể cả Chronos-2, biến bộ dự báo thành một hàm có hai tính chất:
- **đẳng biến affine** (equivariant) theo target;
- **bất biến affine** (invariant) theo *từng* biến ngoại sinh.

Đây là một lựa chọn *đối xứng* ngầm định. Nó sai khi:
- hiệu ứng của covariate phụ thuộc **mức tuyệt đối** (ngưỡng nhiệt độ, đường cong merit-order của giá điện, đường cong công suất gió);
- covariate tác động lên target theo **đơn vị tuyệt đối** thay vì tỷ lệ với $\sigma_{past}$ (dịch mức do sự kiện: chính là H3 của v0.4).

Cả hai là **cùng một định lý**, xem Mục 4:

$$\text{Gap}_{RevIN} = \mathbb E\big[\sigma_y^2\,\mathrm{Var}_w(r \mid u)\big],\qquad r = \frac{\mathbb E[y_{fut}\mid z] - \mu_y}{\sigma_y},$$

trong đó $u$ là cửa sổ đã chuẩn hoá. Đại lượng này đo được từ dữ liệu, nên có thể **kiểm định** kênh nào cần giữ mức và kênh nào nên chuẩn hoá.

**Trạng thái khoảng trống** (chi tiết ở Mục 3):
- Berthelier et al. (3/2026) phân tích RevIN nhưng **không có định lý nào** và **nói rõ bỏ qua covariate**.
- Chronos-2 chuẩn hoá mỗi covariate theo thống kê ngữ cảnh.
- TimeXer tắt luôn cơ chế đưa lại thống kê của Non-stationary Transformer: encoder được gọi không kèm `tau`, `delta` ở `models/TimeXer.py:170`.
- Chưa thấy ai làm lý thuyết bất biến hay đẳng biến cho phía covariate.

**Bằng chứng sơ bộ trên EPF** (Mục 5, probe chứ chưa phải TimeXer thật):
- Covariate tương lai giảm MSE 5–30%.
- Giữ đối xứng một phần kèm thống kê tốt hơn cả RevIN thuần lẫn bỏ hẳn chuẩn hoá.
- Lợi ích đến chủ yếu từ **thống kê target** và **tương tác target × covariate**. Thống kê covariate riêng lẻ trên EPF chỉ nằm trong khoảng nhiễu.

Nghĩa là Hệ quả A **chưa được xác nhận** và cần dữ liệu tải–nhiệt độ.

**Mức rủi ro:** trung bình. Toán dựa trên lý thuyết đẳng biến đã biết (Elesedy & Zaidi 2021; Bloem-Reddy & Teh 2020), nên đóng góp nằm ở *việc nhận diện*, phép chẩn đoán đo được, và phương pháp. Mảng normalization rất đông (RevIN, SAN, FAN, Dish-TS, IN-Flow, NoRIN, APT, TimeAPN...). Muốn tách khỏi họ này thì bắt buộc câu chuyện phải xoay quanh *biến ngoại sinh*, và phải có lợi ích rõ trên dữ liệu thật (EPF, tải điện theo nhiệt độ).

---

## 1. Điểm yếu của TimeXer (đã đối chiếu code)

| # | Điểm yếu | Vị trí trong code | Hệ quả toán học | Đã có người sửa? |
|---|---|---|---|---|
| W1 | **Không dùng covariate tương lai.** `x_dec` và `x_mark_dec` bị bỏ qua. Ở EPF, cột `Grid load forecast` và `Wind power forecast` là *dự báo cho ngày tới* nhưng mô hình chỉ thấy 168h quá khứ | `models/TimeXer.py:157-170` | Bỏ phí thông tin quan trọng nhất của bài toán EPF (LEAR dùng nó) | **Có:** DAG, CITRAS, TFT, TiDE, Chronos-2, TiRex-2. Chỉ là hạ tầng |
| W2 | **RevIN chuẩn hoá cả covariate theo cửa sổ.** Mức tuyệt đối của tải, gió, nhiệt độ bị xoá | `models/TimeXer.py:159-163` | Mô hình bất biến affine theo từng covariate, nên hiệu ứng phụ thuộc mức (ngưỡng, merit-order) không biểu diễn được (Định lý 1) | **Chưa:** xem Mục 3 |
| W3 | **Có code Non-stationary Transformer nhưng không đưa lại thống kê** (encoder gọi không kèm `tau`, `delta`) | `models/TimeXer.py:170` | Over-stationarization toàn phần, cho cả target lẫn covariate | **Đã có một phần:** Non-stationary Transformer đưa lại `x_raw` cùng mean/std của *mọi* kênh vào attention (`models/Nonstationary_Transformer.py:114-124`), nhưng không có khung đối xứng và không chọn kênh |
| W4 | **Token ngoại sinh không có định danh biến.** `Linear(seq_len→d)` dùng chung cho mọi biến và mọi đặc trưng thời gian | `layers/Embed.py:129-143` | Nhánh ngoại sinh là hàm tập hợp đối xứng: $g=\sum_c \mathrm{softmax}_c(q^\top k_c)\,v_c$. Muốn phân biệt vai trò các biến có phân phối giống nhau thì phải suy luận in-context | STID (2022) cho thấy identity embedding quan trọng. Là cải tiến kỹ thuật, không phải paper |
| W5 | **Gộp covariate bằng softmax** (tổ hợp lồi, trọng số ≥ 0, tổng bằng 1) | `EncoderLayer` cross-attn | Hiệu ứng *cộng* của nhiều covariate bị chia tỷ lệ theo số biến đang hoạt động; muốn tổng hợp K biến cùng lúc phải dùng khoảng K head | Họ in-context regression (TimesFM XReg, TabPFN-TS, TFMAdapter) giải quyết theo cách khác |
| W6 | **Lỗi $\sqrt\varepsilon$ với covariate thưa.** `1e-5` cho $\sigma\approx0.0032$, giá trị bị khuếch đại khoảng 300 lần | `models/TimeXer.py:162` | Mất ổn định số | Chronos-2 dùng asinh. Việc kỹ thuật |
| W7 | **Head phẳng qua một global token** (xem v0.4) | `FlattenHead` | Với `e_layers=1` và $d<H$, phần đóng góp của covariate nằm trong không gian con $\le d$ chiều | TiDE/TFT dùng decoder theo từng bước |
| W8 | **Dự báo điểm với MSE** | `exp/` | Double penalty với sự kiện có thời điểm bất định | Chronos-2 và TiRex-2 dự báo quantile |

**Kết luận:** W1, W4, W6, W7, W8 là nâng cấp repo (nên làm, nhưng không phải đóng góp). **W2 và W3 là chỗ có toán mở.** W5 thuộc mảng in-context regression, đang rất đông.

---

## 2. Bản đồ mô hình hiện đại (2024–2026)

**Mô hình huấn luyện theo từng dataset, có biến ngoại sinh:**
- TimeXer (NeurIPS'24): chỉ dùng quá khứ.
- TFT, TiDE: decoder theo bước với covariate tương lai.
- CITRAS (2025): KV shift, dùng covariate tương lai.
- DAG (2025, 12 bộ TSF-X, có mã nguồn): tương quan theo thời gian và theo kênh, dùng covariate tương lai.
- CrossLinear (KDD'25): embedding tương quan chéo, plug-and-play.
- ExoTST, XLinear (2026), FTimeXer (2026).
- EventTSF (IJCAI'26): sự kiện dạng văn bản.

**Foundation model có covariate:**
- Chronos-2 (10/2025): group attention, chuẩn hoá $\sinh^{-1}((v-\mu)/\sigma)$ **theo từng chiều, dùng thống kê ngữ cảnh**, không có định danh biến.
- TiRex-2 (7/2026): xLSTM, time mixer hai chiều, grouped attention bất đối xứng.
- TimesFM-2.5 + XReg: ridge in-context trên covariate, không end-to-end.
- TabPFN-TS: hồi quy dạng bảng in-context; **tốt nhất trên nhóm task có covariate của fev-bench**.
- ChronosX, UniCA, TFMAdapter (CIKM'25: Bayesian ridge cộng GP theo từng instance), LightGTS-Cov, CITRAS-FM, TS-ICL.

**Bằng chứng thực nghiệm đáng chú ý:**
- Berthelier, Baranova et al. (TSALM@ICLR'26, EDF): với target $\sqrt{\text{Traffic}}+\text{Elec}-\text{Solar}^2$, **Chronos-2 thêm covariate không cải thiện gì** (0.186 lên 0.200 ở 168-24), trong khi TabPFN-TS giảm xuống 0.001. Ngay cả khi *chính target* được đưa vào làm covariate, Chronos-2 vẫn còn nMSE 0.048 ở 168-24.
- Lipiecki & Weron (9/2026): TSFM **chưa thay thế được** mô hình EPF chuyên biệt như LEAR hay DNN có covariate.
- Cheong et al. (2/2026): hiệu quả covariate của TSFM trên dự báo tải điện rất thất thường.

**Họ normalization:** RevIN, Non-stationary Transformer, Dish-TS, SAN, FAN, IN-Flow (KDD'25), GAS-Norm, WDAN, TimeAPN, NoRIN (5/2026: phi tuyến Johnson S_U), APT (11/2025: tham số affine điều kiện theo timestamp), cmIN (Berthelier 3/2026).
- **Tất cả** chỉ xét phía target, hoặc coi mọi kênh như nhau.
- APT gần nhất (điều kiện theo timestamp), nhưng không xét covariate tổng quát và không có lý thuyết.

---

## 3. Kiểm tra khoảng trống

| Khoảng trống | Ai đã hoặc đang làm | Trạng thái |
|---|---|---|
| Dùng covariate tương lai, căn chỉnh theo bước (v0.4 H1) | TiDE, TFT, CITRAS, DAG, TAT, Chronos-2, TiRex-2, LightGTS-Cov | **Đã xong.** Chỉ là hạ tầng |
| Phản ứng xung dạng tổng hàm mũ, SSM trên horizon (v0.4 H2/T3) | Box–Tiao (1975, intervention/transfer function), DeepState (NeurIPS'18) | **Cổ điển** |
| Thống kê horizon dự báo từ lịch sử target | SAN, Dish-TS, FAN, GAS-Norm, TimeAPN | **Đông** |
| In-context regression trên covariate | TimesFM XReg, TabPFN-TS, TFMAdapter, TS-ICL, lý thuyết ICL (2502.03383) | **Đang nóng.** Nhóm EDF gợi ý đúng hướng này |
| Covariate đầu vào là *dự báo* có sai số tăng theo h (errors-in-variables) | Thực hành trong ngành năng lượng (huấn luyện trên hindcast); 2501.06180 dùng dự báo phụ tải xác suất cho EPF | Mở một phần. Khó benchmark vì cần kho dự báo lịch sử |
| **Lý thuyết đối xứng của chuẩn hoá khi có covariate:** bất biến theo covariate, đẳng biến theo target, và hiệu ứng theo đơn vị tuyệt đối | Berthelier 3/2026 tự nhận bỏ qua covariate và không có định lý; Non-stationary Transformer đưa lại thống kê mọi kênh nhưng không có lý thuyết; α-RevIN (2605.13678) bỏ bớt thống kê một phần, mọi kênh như nhau; NoRIN, APT không xét covariate | **Mở.** Đề xuất ở Mục 4 |

---

## 4. Hướng đề xuất: lệch đối xứng của chuẩn hoá

### 4.1 Thiết lập

**Cửa sổ và thống kê:**
- Cửa sổ $z = (y_{1:L},\,x_{1:L+H})$ với $x\in\mathbb R^{(L+H)\times C}$.
- Thống kê: $s=(\mu_y,\sigma_y,\mu_1,\sigma_1,\dots,\mu_C,\sigma_C)$, tính trên phần quá khứ.
- Cửa sổ đã chuẩn hoá: $u = \big(\tfrac{y-\mu_y}{\sigma_y}, \tfrac{x_c-\mu_c}{\sigma_c}\big)$.

**Nhóm tác động:** $G=\{(a_0,b_0,\dots,a_C,b_C): a_i>0\}$, với $y\mapsto a_0y+b_0$ và $x_c\mapsto a_cx_c+b_c$.

Ánh xạ $z\mapsto(s,u)$ là song ánh (khi $\sigma>0$), và $u$ là **bất biến cực đại** (maximal invariant) của $G$.

**Họ RevIN:**
$$\mathcal F_{RevIN} = \{\,f(z)=\mu_y+\sigma_y\,\varphi(u)\,\}$$
Đây đúng là tập hàm $f(g\cdot z)=a_0f(z)+b_0$ với mọi $g\in G$. TimeXer, PatchTST, iTransformer, CITRAS và DAG (nếu dùng RevIN) đều thuộc họ này. Chronos-2 thuộc họ này cộng thêm asinh.

**Mục tiêu chuẩn hoá theo điều kiện:** gọi $m(z)=\mathbb E[y_{fut}\mid z]$ và $r(z)=(m(z)-\mu_y)/\sigma_y$.

### 4.2 Định lý 1 (khoảng cách Bayes do lệch đối xứng)

Với loss MSE:

$$\min_{f\in\mathcal F_{RevIN}}\mathbb E\|y_{fut}-f(z)\|^2 \;-\; \mathbb E\|y_{fut}-m(z)\|^2 \;=\; \mathbb E\big[\sigma_y^2\,\|r(z)-\varphi^*(u)\|^2\big],$$

trong đó
$$\varphi^*(u)=\frac{\mathbb E[\sigma_y^2 r\mid u]}{\mathbb E[\sigma_y^2\mid u]}.$$

Khoảng cách bằng 0 khi và chỉ khi $r(z)$ là hàm đo được theo $u$, tức $m$ là $G$-đẳng biến.

*Chứng minh (phác thảo):* Pythagoras cho kỳ vọng có điều kiện. Viết $f=\mu_y+\sigma_y\varphi(u)$. Cực tiểu hoá $\mathbb E[\sigma_y^2\|r-\varphi(u)\|^2]$ theo từng giá trị $u$ là một bài bình phương tối thiểu có trọng số. □

~~Định lý này là một trường hợp của Elesedy & Zaidi.~~ **Đã sửa:** Elesedy & Zaidi cần nhóm compact, nên không áp dụng được. Định lý 1 thực chất là công thức MRE location–scale (Lehmann & Casella 1998, eq. 3.44) cộng phép chiếu $L^2$, nên phải trình bày như *Mệnh đề*. Cái mới là **cách phân rã theo từng kênh** và hai hệ quả dưới đây.

### 4.3 Hai hệ quả, hai thất bại thực tế

**Hệ quả A (phía covariate: hiệu ứng phụ thuộc mức).**

Giả sử $y_{t+h}=\psi(x_{t+h})+\text{(phần đẳng biến)}$ với $\psi$ không affine, ví dụ:
- đường cong chữ U của tải theo nhiệt độ quanh $\theta\approx18^\circ C$;
- đường cong merit-order lồi của giá theo phụ tải dư (residual load).

Khi đó $r$ phụ thuộc $\mu_c,\sigma_c$ khi đã biết $u$. Hai cửa sổ có *cùng hình dạng* nhiệt độ, một ở 5–10°C và một ở 25–30°C, cho phản ứng tải **ngược dấu**. Mô hình RevIN buộc phải dự báo trung bình của hai trường hợp.

Cận dưới: $\text{Gap}\ge \mathbb E[\sigma_y^2]\cdot p_{lo}\,p_{hi}\,\|\Delta r\|^2$ khi $\sigma_y$ độc lập.

**Hệ quả B (phía target: hiệu ứng theo đơn vị tuyệt đối).**

Nếu một covariate gây dịch mức $\Delta$ *cố định theo đơn vị gốc*, thì $r\ni\Delta/\sigma_y$ phụ thuộc $\sigma_y$ mà $u$ không biết. Đây chính là **H3 của v0.4**: ALN-raw thắng ALN-norm (0.033 so với 0.040) vì cộng ở thang gốc phá vỡ đẳng biến theo $a_0$.

→ v0.4 H3 và W2 là **hai mặt của cùng một định lý**.

### 4.4 Định lý 2 (phân rã theo kênh và phép chẩn đoán đo được)

Gọi $\mathcal F_S=\{\mu_y+\sigma_y\varphi(u,s_S)\}$ là họ được thấy thống kê của tập kênh $S$. Khi đó:

$$\text{Gap}(\varnothing)-\text{Gap}(S)=\mathbb E\big[\sigma_y^2\,\|\mathbb E_w[r\mid u,s_S]-\mathbb E_w[r\mid u]\|^2\big]=:D_S\;\ge 0 .$$

**Độ lệch đẳng biến $D_c$ của từng covariate** ước lượng được bằng hai bộ hồi quy, một có và một không có $s_c$, trên target đã chuẩn hoá. ~~Như vậy có một phép kiểm định trước khi huấn luyện.~~ **Đã sửa:** $D_c$ là LOCO/VIM, chỉ *ước lượng cỡ hiệu ứng* được (sample-split cộng cross-fit). Không tồn tại kiểm định $D_c=0$ hợp lệ đều (Shah & Peters 2020). $D_c$ cũng không cộng tính giữa các kênh.

### 4.5 Định lý 3 (đánh đổi khi thống kê bị dịch chuyển)

Phá vỡ bất biến không miễn phí:
- Nếu thống kê ở tập test $s_c^{test}$ nằm ngoài support của tập train, thì mọi bộ học dùng $s_c$ đều không định danh được $m$ ở vùng đó. Có thể dựng hai phân phối trùng nhau trên support train nhưng khác nhau ngoài support, nên cận minimax $\ge c>0$.
- Ngược lại, mô hình bất biến ngoại suy *đúng* nếu $m$ thật sự đẳng biến.

Vì vậy lựa chọn tối ưu cho từng kênh cân bằng **$D_c$ (thiên lệch do bất biến)** với **độ dịch chuyển thống kê train→test của kênh c** (rủi ro ngoại suy).

**Dự đoán ban đầu cho EPF:** mức giá trôi theo giá nhiên liệu nên giữ đẳng biến theo target; mức tải và gió ổn định theo mùa nên phá bất biến theo covariate là an toàn.

**Probe ở Mục 5 chỉ ủng hộ một phần:**
- Thống kê của target giúp nhiều hơn thống kê của covariate.
- FR cho thấy đúng rủi ro ngoại suy mà Định lý 3 cảnh báo.

Dự đoán này cần sửa lại sau khi chạy với mô hình thật.

### 4.6 Ghi chú cho foundation model (in-context)

TSFM không biết đơn vị, nên *buộc phải* bất biến theo covariate. Khi đó Hệ quả A chỉ khắc phục được bằng cách **ước lượng in-context**. Điều này chỉ khả thi khi ngữ cảnh bao phủ vùng ngưỡng (một điều kiện định danh).

Dữ liệu EDF cho thấy Chronos-2 thất bại cả ở trường hợp dễ, nghĩa là vấn đề của TSFM chủ yếu là *năng lực hồi quy in-context* chứ không riêng đối xứng. **Paper này nên nhắm mô hình có giám sát và fine-tuning, không nhắm zero-shot.**

### 4.7 Phương pháp: chuẩn hoá đẳng biến theo từng phần (tên: **PERIN**, Partially Equivariant Reversible Instance Normalization; xem ADR-0005)

$$\hat y = \mu_y+\sigma_y\,\varphi\big(u,\ \rho(s_S)\big)\;+\;\underbrace{\psi\big(x^{fut}_{raw}\big)}_{\text{nhánh thang gốc, tuỳ chọn}}$$

- **$u$:** giữ RevIN để tối ưu dễ.
- **$\rho(s_S)$:** thống kê của các kênh được chọn, đưa vào qua **đặc trưng bị chặn** (rank hoặc quantile theo phân phối train, clip về $[0,1]$). Nhờ vậy ngoài support thì mô hình ngoại suy hằng số, đúng theo tinh thần Định lý 3. Đưa vào như token thống kê, hoặc như FiLM cho global token.
- **$S$:** chọn bằng phép chẩn đoán $D_c$ (Định lý 2), hoặc học bằng gate thưa.
- **$\psi$:** nhánh thang gốc cho hiệu ứng theo đơn vị tuyệt đối (Hệ quả B, kế thừa ALN-raw của v0.4).
- **Kỹ thuật:** asinh cho covariate thưa (W6), identity embedding cho token ngoại sinh (W4), và dùng covariate tương lai (W1).

---

## 5. Bằng chứng sơ bộ trên EPF (dữ liệu có sẵn trong repo)

**Script:** `docs/brainstorm/probes/epf_probe.py`.

**Thiết lập:**
- Mốc dự báo mỗi ngày: dùng 168h quá khứ để dự báo 24h tới.
- Chia 70/10/20 theo thời gian như `Dataset_Custom`; tập train gộp cả phần val. Khoảng 1.500 mẫu train mỗi thị trường.
- Scaling toàn cục như repo.
- Mô hình: Ridge (CV theo alpha) và MLP 256×256 (3 seed).
- MSE tính trên giá đã chuẩn hoá toàn cục.

Đây là **probe thông tin** (mỗi biến thể cho mô hình thấy gì), không phải TimeXer thật.

**Các biến thể:**
- **V1:** RevIN mọi kênh, chỉ dùng quá khứ. Tương đương TimeXer gốc.
- **V2:** V1 + dự báo tải và gió cho 24h tới, chuẩn hoá bằng thống kê quá khứ. Đây là "FUT".
- **V3a:** V2 + $(\mu,\log\sigma)$ của covariate. Phá bất biến covariate.
- **V3b:** V2 + $(\mu,\log\sigma)$ của target.
- **V3:** V2 + cả hai loại thống kê.
- **V4:** không RevIN, chỉ scaling toàn cục (bỏ hoàn toàn đối xứng).

**MLP MSE (mean ± std, 3 seed):**

| | NP | PJM | BE | FR | DE |
|---|---|---|---|---|---|
| V1 past-only | 0.279±.002 | 0.107±.002 | 0.457±.013 | 0.556±.031 | 0.491±.002 |
| V2 +future exo | 0.246±.003 | 0.102±.005 | 0.401±.003 | 0.508±.031 | 0.342±.011 |
| V3a +exo stats | 0.237±.014 | 0.109±.011 | 0.398±.006 | 0.509±.037 | 0.337±.006 |
| V3b +target stats | 0.221±.005 | 0.097±.007 | 0.388±.010 | 0.732±.133 | 0.335±.010 |
| V3 +cả hai | **0.207±.001** | **0.097±.004** | 0.395±.003 | **0.506±.024** | **0.326±.019** |
| V4 global, không RevIN | 0.261±.007 | 0.129±.010 | 0.479±.009 | 0.777±.377 | 0.370±.013 |

**Ridge MSE:**

| | NP | PJM | BE | FR | DE |
|---|---|---|---|---|---|
| V1 | 0.245 | 0.090 | 0.454 | 0.491 | 0.442 |
| V2 | 0.214 | 0.079 | 0.387 | 0.443 | 0.307 |
| V3a | 0.216 | 0.080 | 0.388 | 0.435 | 0.307 |
| V3b | 0.205 | 0.078 | 0.374 | 0.451 | 0.302 |
| V3 | 0.204 | 0.079 | 0.375 | 0.443 | 0.301 |
| V4 | 0.223 | 0.111 | 0.385 | 0.449 | 0.334 |

**Đọc kết quả (thẳng thắn):**

1. **W1 được xác nhận mạnh.** Dùng dự báo tải và gió cho ngày tới giảm MSE 5–30% ở cả 5 thị trường (DE: 0.491 → 0.342). Riêng việc này đã là nâng cấp repo đáng làm, dù không mới.
2. **Bỏ hẳn đối xứng (V4) luôn tệ hơn giữ đối xứng một phần (V3).** FR không ổn định (±0.377). Phù hợp với Định lý 3: phá bất biến toàn bộ thì trả giá bằng rủi ro ngoại suy.
3. **Phía covariate (Hệ quả A) chỉ được ủng hộ yếu trên EPF.** V3a so với V2 chỉ khác trong khoảng nhiễu, trừ có thể NP (−4% ± 6%). Phía target (V3b) rõ hơn: NP −10%, PJM −5%, BE −3%.
4. **Có tương tác.** Ở NP, cả hai loại thống kê cùng lúc (0.207) tốt hơn từng loại riêng (0.237; 0.221). Điều này khớp với merit-order: giá phụ thuộc đồng thời mức nhiên liệu (thể hiện qua mức giá) và mức phụ tải dư.
5. **FR minh hoạ rủi ro ngoại suy.** Thống kê target làm FR tệ đi rất nhiều (0.732 ± 0.133). Đây đúng là tình huống Định lý 3 cảnh báo, và là lý do đề xuất đặc trưng bị chặn $\rho(s)$ thay cho $s$ thô.

**Giới hạn:**
- ~1.500 mẫu và MLP nhỏ, chưa phải TimeXer.
- Chưa có dataset nhiệt độ–tải, nơi vật lý dự đoán Hệ quả A mạnh nhất.
- Chỉ 3 seed.

**Hệ quả cho kế hoạch:**
- Câu chuyện paper nên đặt trọng tâm vào *đối xứng theo từng kênh, và đánh đổi giữa bias với ngoại suy*, không chỉ "giữ mức covariate".
- **Việc tiếp theo bắt buộc:** chạy probe này trên một bộ tải–nhiệt độ trước khi đầu tư thêm.

---

## 6. Kế hoạch

### 6.1 Nâng cấp repo (theo thứ tự, mỗi bước là một flag để ablation)

1. `--use_future_exo`: ghép $x_{t+1:t+H}$ vào token ngoại sinh. Với EPF, cột dự báo đã có sẵn trong CSV (W1).
2. `--exo_norm {revin, global, asinh}`: tách chuẩn hoá covariate khỏi chuẩn hoá target (W2, W6).
3. `--stat_tokens {none, target, exo, all}`: đưa $\rho(s)$ vào dưới dạng token (W3, Định lý 2).
4. `--var_id`: identity embedding cho token ngoại sinh (W4).
5. `--raw_branch`: nhánh $\psi$ ở thang gốc (Hệ quả B).
6. Script chẩn đoán `tools/equivariance_defect.py`: ước lượng $D_c$ cho mỗi dataset.

### 6.2 Dữ liệu

- **EPF** (5 thị trường, có sẵn, covariate là dự báo thật nên không bị rò rỉ oracle).
- **Tải điện theo nhiệt độ:** GEFCom2014-L; ENTSO-E kèm nhiệt độ; các task có covariate nhiệt độ trong fev-bench.
- **DAG TSF-X** (7 bộ).
- **Bộ tổng hợp:** chữ U, ngưỡng, merit-order; quét độ lệch thống kê train/test để vẽ đường cong đánh đổi của Định lý 3.

### 6.3 Baseline và ablation

- **Backbone:** TimeXer, iTransformer, PatchTST+exo, CITRAS, DAG, TiDE, CrossLinear. Cắm PERIN vào *mọi* backbone để chứng minh tính plug-in.
- **Chuẩn hoá:** RevIN, Non-stationary Transformer (de-stationary attention), SAN, FAN, NoRIN, APT, "không chuẩn hoá" + global scaling.
- **Tham chiếu mạnh:** LEAR cho EPF, TabPFN-TS, Chronos-2 fine-tune.
- **Kiểm chứng lý thuyết:**
  1. Tương quan giữa $\hat D_c$ và mức cải thiện thực tế của việc giữ mức kênh c.
  2. Đường cong đánh đổi theo độ dịch chuyển thống kê.

### 6.4 Điều kiện tiếp tục

Trên EPF và ít nhất một bộ tải–nhiệt độ, PERIN phải thắng **TimeXer + covariate tương lai + RevIN** và **+ Non-stationary Transformer** (≥5 seed, có ý nghĩa thống kê) trên ≥60% cặp dataset × horizon. $\hat D_c$ phải dự đoán được dấu của mức cải thiện. Nếu không đạt, rút về workshop (TSALM) dưới dạng paper phân tích.

---

## 7. Reviewer sẽ công kích gì

1. **"Định lý 1 là lý thuyết đẳng biến sẵn có."** Đúng. Cần nói thẳng điều đó, trích Elesedy & Zaidi và Bloem-Reddy & Teh, và đặt đóng góp ở: (i) nhận diện RevIN là một lựa chọn đối xứng theo từng kênh; (ii) phép chẩn đoán $D_c$; (iii) đánh đổi của Định lý 3; (iv) thống nhất với v0.4 H3.
2. **"Chỉ là Non-stationary Transformer cho covariate."** Khác ở chỗ có lý thuyết chọn kênh, có đặc trưng bị chặn để chống ngoại suy, và có nhánh thang gốc. Cần ablation cho thấy de-stationary attention không đủ.
3. **"Chỉ là không chuẩn hoá covariate thôi."** Phải có baseline global scaling. Định lý 3 dự đoán global scaling thua khi thống kê covariate bị dịch chuyển, nên cần dataset có dịch chuyển để chứng minh.
4. **"Lợi ích nhỏ trên dữ liệu thật."** Chọn dataset mà vật lý đảm bảo hiệu ứng mức (merit-order, nhiệt độ).

---

## Nguồn

- TimeXer: https://github.com/thuml/TimeXer · Non-stationary Transformer: https://arxiv.org/abs/2205.14415 · RevIN: https://openreview.net/forum?id=cGDAkQo1C0p
- Berthelier et al., On the Role of RevIN (3/2026): https://arxiv.org/html/2603.11869
- Berthelier, Baranova et al., Chronos-2 vs TabPFN-TS covariate (TSALM@ICLR'26): https://arxiv.org/abs/2605.12200
- Chronos-2: https://arxiv.org/abs/2510.15821 · TiRex-2: https://arxiv.org/abs/2607.01204 · fev-bench: https://arxiv.org/abs/2509.26468
- TFMAdapter (CIKM'25): https://arxiv.org/abs/2509.13906 · UniCA: https://arxiv.org/abs/2506.22039 · ChronosX: https://arxiv.org/abs/2503.12107 · LightGTS-Cov: https://arxiv.org/abs/2602.10412
- DAG: https://arxiv.org/abs/2509.14933 · CITRAS: https://arxiv.org/abs/2503.24007 · CrossLinear (KDD'25): https://arxiv.org/abs/2505.23116 · ExoTST: https://arxiv.org/abs/2410.12184 · TAT: https://arxiv.org/abs/2507.10349
- SAN: https://openreview.net/forum?id=5BqDSw8r5j · FAN: https://arxiv.org/abs/2409.20371 · Dish-TS: https://arxiv.org/abs/2302.14829 · NoRIN: https://arxiv.org/abs/2605.10823 · APT: https://arxiv.org/abs/2511.12945 · RevIN contradictions: https://arxiv.org/abs/2510.04667
- Lipiecki & Weron, TSFM cho EPF (9/2026): https://arxiv.org/abs/2609.00089 · Cheong et al., TSFM cho tải điện (2/2026): https://arxiv.org/abs/2602.05390
- Transformers làm least squares in-context cho chuỗi thời gian: https://arxiv.org/abs/2502.03383 · Event Signature Transfer: https://arxiv.org/abs/2609.23074
- DeepState (NeurIPS'18): https://papers.nips.cc/paper/8004 · STID: https://arxiv.org/abs/2208.05233
- Lý thuyết đẳng biến: Elesedy & Zaidi, "Provably Strict Generalisation Benefit for Equivariant Models" (ICML 2021); Bloem-Reddy & Teh, "Probabilistic Symmetries and Invariant Neural Networks" (JMLR 2020)
