# Đối chiếu nguồn gốc: toán của "lệch đối xứng do chuẩn hoá (RevIN) khi có biến ngoại sinh"

Ngày kiểm tra: 2026-09-25. Đối tượng kiểm tra: Mục 4 và 5 của `docs/brainstorm/TimeXer_upgrade_v0.5.md`, gồm (a) đặc trưng hoá họ RevIN, (b) Định lý 1, (c) Định lý 2, (d) Định lý 3 và (e) phương pháp PEN.

Nguyên tắc: mỗi khẳng định đi kèm nguồn gốc (paper hoặc code chính thức). Trích dẫn nguyên văn để trong ngoặc kép, bằng tiếng Anh. Chỗ nào chỉ đọc được abstract thì ghi rõ.

---

## Tóm tắt kết luận

1. **Định lý 1 không phải hệ quả của Elesedy & Zaidi.** Kết quả của họ cần nhóm compact, độ đo Haar chuẩn hoá và phân phối đầu vào bất biến theo G. Nhóm affine $(a>0,b\in\mathbb R)$ không compact, và dữ liệu chuỗi thời gian không có phân phối bất biến. Định lý 1 thực chất là **phép tính MRE (minimum risk equivariant) cổ điển cho họ location–scale** (Lehmann & Casella 1998, Thm 3.17 và công thức (3.44)), áp dụng cho bài toán dự báo, cộng với phép chiếu $L^2$ có trọng số lên các hàm đo được theo $u$. Phần đặc trưng hoá "RevIN = mọi hàm đẳng biến" là Lemma 8 của Bloem-Reddy & Teh (không cần compact), là dạng canonicalization của Kaba et al. (2023), và với trường hợp một kênh thì là Characterization 1 của Saied & Fleuret (2026).
2. **Định lý 2 là Pythagoras (chain rule) cho kỳ vọng có điều kiện.** $D_S$ chính là variable importance kiểu Williamson et al., hay LOCO tổng thể của Lei et al. với MSE ở thang gốc. Phép kiểm "$D_S=0$" là kiểm định conditional mean independence, và **không có kiểm định nào hợp lệ đều trên mọi phân phối** (Shah & Peters 2020; Lundborg et al. 2024). Phiên bản theo phân phối của phép chẩn đoán này đã có: Chiu, Sharp & Bloem-Reddy kiểm định conditional equivariance cho nhóm locally compact.
3. **Nửa "không định danh" của Định lý 3 đã có sẵn** (Christiansen et al. 2022, Prop. 4.7; Dong & Ma 2023, Prop. 4.1). Nửa "mô hình bất biến ngoại suy đúng" cần thêm một giả định không kiểm được: $m$ phải đẳng biến cả *ngoài* support. Câu "đặc trưng bị chặn ⇒ ngoại suy hằng số, đúng tinh thần Định lý 3" là **nói quá**: Định lý 3 không ủng hộ quy tắc ngoại suy nào. Cần một mệnh đề riêng, xem Mục B.
4. **Phương pháp PEN có tiền lệ gần cho từng thành phần:**
   - Đưa thống kê của target làm feature: Lag-Llama, DeepAR bản GluonTS, pytorch-forecasting.
   - Đưa lại thống kê của *mọi* kênh, kể cả covariate: Non-stationary Transformer, theo code chính thức. **Khẳng định trong v0.5 rằng NST "chỉ đưa lại thống kê target" là sai.**
   - Bỏ bớt thống kê một phần: α-RevIN.
   - Nhánh không đẳng biến song song với nhánh đẳng biến: Residual Pathway Priors.
   - Chọn nhóm con theo lattice: partial canonicalization.

   Không tìm thấy công trình nào trình bày RevIN như một lựa chọn đối xứng *theo từng kênh* (target đẳng biến, covariate bất biến) kèm lý thuyết và phép chẩn đoán. Đây là khoảng trống còn lại, nhưng hẹp hơn v0.5 tưởng.

---

## 1. Lý thuyết bất biến/đẳng biến và khoảng cách Bayes của mô hình bất biến

### 1.1 Elesedy & Zaidi (ICML 2021): họ chứng minh gì, và giả định là gì

**Nguồn:** B. Elesedy, S. Zaidi, "Provably Strict Generalisation Benefit for Equivariant Models", ICML 2021, PMLR 139. <http://proceedings.mlr.press/v139/elesedy21a/elesedy21a.pdf> · arXiv:2102.10333

**Giả định (Mục 4.1, nguyên văn):**
- "Let G be a compact group. The Haar measure is the unique invariant measure on G … We assume normalisation such that λ(G) = 1, which is always possible when G is compact."
- Toán tử lấy trung bình trên orbit là $(\mathcal Q f)(x)=\int_G \psi(g^{-1})f(gx)\,d\lambda(g)$.
- Không gian hàm dùng tích trong $\langle f,h\rangle_\mu$ với "μ be a G-invariant measure on X".

**Lemma 1** (phân rã trực giao): "Let U be any subspace of V that is closed under Q … Then U admits an orthogonal decomposition into symmetric and anti-symmetric parts U = S ⊕ A."

Bài báo nói rõ tính chất then chốt là "(D) Q is self-adjoint with respect to ⟨·,·⟩_μ. The last of these is critical and depends on the G-invariance of μ."

**Lemma 6:** "Let X ∼ μ where μ is a G-invariant distribution on X. Let Y = f*(X)+ξ … f* is G-equivariant. Then, for any f ∈ V, the generalisation gap satisfies Δ(f, Qf) = … = ‖f_A‖²_μ."

Như vậy hướng của họ là: **target đã đẳng biến**, và họ đo lợi ích khi chiếu một predictor bất kỳ lên phần đẳng biến. Theorem 7 và 13 là trường hợp hồi quy tuyến tính với $X\sim N(0,\sigma^2 I)$ và nhóm compact tác động qua biểu diễn trực giao.

**Có áp dụng được cho nhóm affine $\{(a,b):a>0\}$ không? Không.**
- Nhóm này không compact nên không có Haar *xác suất*, và $\mathcal Q$ không định nghĩa được.
- Không có độ đo xác suất nào trên $\mathbb R^n$ bất biến dưới phép tịnh tiến và co giãn. Chiu & Bloem-Reddy viết thẳng: "If G is non-compact, then P_X cannot be G-invariant, but P_{Y|X} may be" (arXiv:2307.15834v2, Mục 1.1.1).
- Kết quả của E&Z đo *lợi ích* khi target đối xứng. Định lý 1 của ta đo *chi phí* khi target **không** đối xứng. Hai hướng này ngược nhau.

**Kết luận:** câu trong v0.5 "Định lý này là một trường hợp của … Elesedy & Zaidi" phải sửa. E&Z chỉ nên được trích làm *bối cảnh* (phân rã symmetric và anti-symmetric), không phải làm nguồn cho Định lý 1.

### 1.2 Công cụ đúng cho nhóm không compact: maximal invariant và representative equivariant

**Nguồn:** B. Bloem-Reddy, Y. W. Teh, "Probabilistic Symmetries and Invariant Neural Networks", JMLR 21(90):1–61, 2020. <https://www.jmlr.org/papers/volume21/19-322/19-322.pdf>

**Định nghĩa maximal invariant (Mục 4.1):** "A maximal invariant statistic … is an invariant statistic M : X → S such that M(x₁) = M(x₂) implies x₂ = g·x₁ for some g ∈ G."

**Lemma 8** không có giả định compact. Nguyên văn: "For a group G acting measurably on Borel spaces X and Y, a representative equivariant τ : X → G, as defined in (15) [τ(g·x) = g·τ(x)], has the following properties: (i) The function M : X → X defined by M(x) = τ_x⁻¹·x is a maximal invariant. (ii) For any mapping b : X → Y, the function f(x) = τ_x · b(τ_x⁻¹·x) … is G-equivariant."

Áp vào bài toán của ta:
- Đặt $\tau(z)=s=(\mu_y,\sigma_y,\mu_1,\sigma_1,\dots)$, xem như một phần tử của $G$.
- Khi đó $u=\tau(z)^{-1}\cdot z$ là cửa sổ đã chuẩn hoá, và $f(z)=\mu_y+\sigma_y\varphi(u)$ chính là dạng (ii).
- Chiều ngược lại, mọi hàm đẳng biến đều có dạng này, vì $f(z)=\tau(z)\cdot f(\tau(z)^{-1}z)=\tau(z)\cdot f(u)$. Đây là một dòng đại số.

**Giới hạn của B-R&T:** Theorem 7, 9, 10 (biểu diễn noise-outsourcing cho phân phối có điều kiện) đều giả định "G a compact group … Assume that P_X is G-invariant". Họ ghi: "Versions of Theorem 7 may hold for non-compact groups … but require considerably more technical details" (Mục 4.1) và "Theorem 9 may hold for non-compact groups, but our proof for compact groups makes use of the normalized Haar measure" (Mục 4.2).

Như vậy **chỉ Lemma 8 là dùng trực tiếp được**. Các định lý noise-outsourcing thì không.

**Cùng cấu trúc ở dạng canonicalization:**
- **Kaba et al.** "Equivariance with Learned Canonicalization Functions", ICML 2023, PMLR 202. <https://arxiv.org/abs/2211.06489>. Phương trình (2): "φ(x) = ρ'(h(x)) f(ρ(h(x))⁻¹ x) … Equivariance in Equation (2) is obtained for any prediction function if the canonicalization function is itself G-equivariant." RevIN là một canonicalization *cố định* (không học) cho nhóm affine theo từng kênh.
- Mục 3.3 của Kaba et al. ("Partial Canonicalization and Lattice of Subgroups") khớp với ý chọn nhóm con theo kênh (Định lý 2): "set inclusion in the subgroup lattice is equivalent to increased expressivity for the corresponding models."
- **Puny et al.** "Frame Averaging for Invariant and Equivariant Network Design", ICLR 2022. <https://arxiv.org/abs/2110.03336>. Họ xử lý tường minh nhóm không compact E(d), dùng tâm (centroid) làm frame cho phép tịnh tiến: "This theorem can be used to prove universality results if the backbone model is universal, even for non-compact groups (e.g., the Euclidean motion group)."

### 1.3 Nguồn đúng cho *khoảng cách Bayes*: lý thuyết MRE cho họ location–scale

**Nguồn:** E. L. Lehmann, G. Casella, *Theory of Point Estimation*, 2nd ed., Springer, 1998, Ch. 3, Mục 3.3 "Location-Scale Families", tr. 173–175. Bản text tham khảo: <https://archive.org/stream/cshourave_gmail_TPE/>

**Theorem 3.17 (nguyên văn, OCR đã sửa):** "Let δ₀ be any estimator of ξ satisfying (3.35) [δ(a+bx)=a+bδ(x), b>0] and δ₁ any estimator of τ taking on positive values only and satisfying δ₁(a+bx)=bδ₁(x) … Then, δ satisfies (3.35) if and only if it is of the form δ(x) = δ₀(x) − w(z)δ₁(x), where z is given by (3.30) [maximal invariant]."

**MRE dưới loss bình phương, công thức (3.44):** $w^*(z)=E_{0,1}[\delta_0(X)\delta_1(X)\mid z]\,/\,E_{0,1}[\delta_1^2(X)\mid z]$.

**Đối chiếu với ta:** chọn $\delta_0=\mu_y$ và $\delta_1=\sigma_y$. Theorem 3.17 chính là đặc trưng hoá họ RevIN, với $u$ là $z$ của Lehmann–Casella. Công thức (3.44) có đúng dạng
$$\varphi^*(u)=\frac{E[\sigma_y(m-\mu_y)\mid u]}{E[\sigma_y^2\mid u]}=\frac{E[\sigma_y^2 r\mid u]}{E[\sigma_y^2\mid u]}.$$

**Khác biệt duy nhất:** Lehmann–Casella tính kỳ vọng tại tham số chuẩn $(\xi,\tau)=(0,1)$, vì rủi ro của estimator đẳng biến là hằng số khi nhóm tác động bắc cầu lên không gian tham số. Ta không có họ tham số. Ta lấy kỳ vọng dưới phân phối dữ liệu thật $P$, vốn *không* bất biến. Chứng minh vẫn y hệt: cực tiểu hoá có điều kiện theo maximal invariant.

**Tài liệu cùng dòng về *dự báo* đẳng biến** (predictor chứ không phải estimator): Zhou & Nayak, "Pitman closest equivariant estimators and predictors under location–scale models", J. Statist. Plann. Inference 142 (2012) 1367–1377. <https://www.sciencedirect.com/science/article/abs/pii/S0378375811004599>. Chỉ đọc được metadata.

**Kết luận câu hỏi 1:** Định lý 1 = phép tính MRE location–scale (Lehmann–Casella Thm 3.17 + (3.44)), chuyển sang dự báo dưới phân phối không bất biến, và cho nhóm tích theo kênh trong đó covariate chỉ bất biến. Nên trích Lehmann–Casella làm nguồn chính, Bloem-Reddy & Teh Lemma 8 cho phần biểu diễn, và Kaba et al. cho góc nhìn canonicalization. Không nên gọi nó là "định lý mới". Nên gọi là "proposition (standard)".

### 1.4 Lyle et al., Augerino, bất biến mềm và bất biến một phần

- **Lyle, van der Wilk, Kwiatkowska, Gal, Bloem-Reddy**, "On the Benefits of Invariance in Neural Networks", arXiv:2005.00178 (2020). <https://arxiv.org/abs/2005.00178>. So sánh data augmentation với feature averaging. Chỉ đọc abstract: "feature averaging reduces generalization error when used with convex losses", và cảnh báo "the 'learned invariance' fails outside of the training distribution". Feature averaging theo nghĩa của họ cần nhóm có độ đo xác suất, nên cũng thuộc khung compact.
- **Benton, Finzi, Izmailov, Wilson**, "Learning Invariances in Neural Networks" (Augerino), NeurIPS 2020. <https://arxiv.org/abs/2010.11882>. Học một phân phối tham số hoá trên các phép augmentation (độ rộng). Đây là cách học *mức độ* bất biến. Có thể dùng làm baseline "học độ bất biến theo kênh", ví dụ augmentation affine trên từng covariate.
- **Finzi, Benton, Wilson**, "Residual Pathway Priors for Soft Equivariance Constraints", NeurIPS 2021. <https://arxiv.org/abs/2112.01388>. Mô hình = nhánh đẳng biến + nhánh không đẳng biến, mỗi nhánh có prior phương sai khác nhau. Họ báo cáo RPP "resilient to approximate or misspecified symmetries". **Đây là tiền lệ trực tiếp cho nhánh thang gốc $\psi(x_{raw})$ của PEN.**
- **Petrache & Trivedi**, "Approximation-Generalization Trade-offs under (Approximate) Group Equivariance", NeurIPS 2023. <https://arxiv.org/abs/2305.17592>. Họ xét "model mis-specification i.e. when the model symmetries don't align with the data symmetries". Giả định nhóm hữu hạn hoặc compact: "we first restrict to finite G. However, this setting can be extended to compact groups which are doubling …" (Mục 2).
- **Huang, Levie, Villar**, "Approximately Equivariant Graph Networks", NeurIPS 2023. <https://arxiv.org/abs/2308.10436>. Có "bias-variance formula" cho việc chọn nhóm con, nhưng cũng dùng "a compact group G with Haar measure" và "μ is a S_N-invariant distribution" (Lemma 1).
- **Chen, Dobriban, Lee**, "A Group-Theoretic Framework for Data Augmentation", JMLR 21 (2020). <https://www.jmlr.org/papers/volume21/20-163/20-163.pdf>. Mục 6 có đánh đổi bias–variance dưới "approximate invariance". Bias đo bằng khoảng cách Wasserstein giữa $X$ và $gX$. Khung này là augmentation (lấy trung bình trên orbit), không phải canonicalization.

**Nhận xét:** toàn bộ văn liệu "đánh đổi khi đối xứng bị đặt sai" trong ML mà tôi tìm được đều giả định nhóm compact hoặc hữu hạn, và thường giả định phân phối đầu vào bất biến. Trường hợp nhóm affine không compact, phân phối không bất biến và đối xứng hỗn hợp theo kênh (target đẳng biến, covariate bất biến) **không được các bài này bao phủ**. Tuy vậy, về toán thì trường hợp của ta *dễ hơn*: hành động là free, nên có cross-section toàn cục, và công thức gap là phép chiếu $L^2$ sơ cấp. Không nên bán nó như một khó khăn kỹ thuật.

---

## 2. Maximal invariant cho nhóm affine theo kênh và normalization equivariance

### 2.1 Kiểm tra khẳng định (a) của v0.5

**Khẳng định (a) đúng**, với điều kiện hạn chế vào $Z_+=\{z:\sigma_y>0,\ \sigma_c>0\ \forall c\}$:
- $G=(\mathbb R\times\mathbb R_{>0})^{1+C}$ tác động *free* lên $Z_+$.
- $\tau(z)=s$ thoả $\tau(g\cdot z)=g\circ\tau(z)$.
- $z\mapsto(s,u)$ là song ánh $Z_+\to G\times U$, với $U$ là tập cửa sổ có phần quá khứ của mỗi kênh có mean 0 và std 1.

Theo Lemma 8 (i) của B-R&T, $u$ là maximal invariant. Đặc trưng hoá $\mathcal F_{RevIN}=\{f: f(g z)=a_0f(z)+b_0\}$ đúng như đã nói ở 1.2.

**Bốn điểm cần ghi rõ trong paper:**

1. **Epsilon.** Code TimeXer dùng `stdev = torch.sqrt(torch.var(x_enc, ...) + 1e-5)` (`models/TimeXer.py:162`). Khi đó đẳng biến theo scale chỉ gần đúng, và sai số lớn khi cửa sổ gần hằng. Saied & Fleuret chứng minh rằng ánh xạ đẳng biến *chính xác* bắt buộc có "for std(y)=0 one necessarily has f(y)=μ(y)𝟙". Với covariate thưa (nhiều cửa sổ toàn 0), $u_c$ bằng 0 và điều này vô hại. Với target, cần nói rằng định lý áp dụng trên $Z_+$.
2. **Đặc trưng thời gian.** `x_mark` không bị chuẩn hoá. Về hình thức, $G$ tác động tầm thường lên các toạ độ này, và chúng thuộc $u$.
3. **Các biến thể RevIN khác vẫn nằm trong họ.** Affine học được $(\gamma,\beta)$ của RevIN, asinh của Chronos-2, và phép Johnson-SU của NoRIN (áp lên giá trị đã chuẩn hoá bằng median/MAD, tham số dùng chung giữa các instance) đều chỉ là một phần của $\varphi$. Họ hàm không đổi.
4. **Trọng số.** Loss tính trên thang gốc (TimeXer de-normalize trước khi tính MSE) thì tương ứng với trọng số $\sigma_y^2$ trong Định lý 1. Loss tính trên thang chuẩn hoá (Berthelier et al. khuyến nghị) thì cực tiểu hoá $E\|r_Y-\varphi(u)\|^2$ *không trọng số*, nên nghiệm là $E_P[r\mid u]\neq\varphi^*$, trừ khi $\sigma_y$ độc lập có điều kiện với $r$ khi biết $u$. Đây là một hệ quả nhỏ nhưng đo được. Nên đưa vào paper để giải thích phát hiện của Berthelier.

### 2.2 Herbreteau et al. (NeurIPS 2023)

**Nguồn:** S. Herbreteau, E. Moebel, C. Kervrann, "Normalization-Equivariant Neural Networks with Application to Image Denoising", NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html> · arXiv:2306.05037

- Định nghĩa NE: $f(ax+b\mathbf 1)=af(x)+b\mathbf 1$ với $a>0$ (scale + shift).
- Cách làm: thay kiến trúc bằng "affine-constrained convolutions and channel-wise sort pooling layers". Họ **không** dùng normalize–process–denormalize.
- **Không nhắc tới time series hay RevIN**, theo abstract và tìm kiếm trong trang arXiv.

### 2.3 Saied & Fleuret, arXiv:2605.08193 (5/2026)

**Nguồn:** Y. Saied, F. Fleuret, "Normalization Equivariance for Arbitrary Backbones, with Application to Image Denoising", arXiv:2605.08193v3. <https://arxiv.org/html/2605.08193>

**Characterization 1 (nguyên văn):** "A function f:ℝ^d→ℝ^d is normalization-equivariant if and only if there exists a function g:ℳ→ℝ^d such that, for all y with std(y)>0, f(y)=std(y)g(T_NE(y))+μ(y)𝟙. Moreover, for std(y)=0 one necessarily has f(y)=μ(y)𝟙."

**Về RevIN, có nhắc, nhưng chỉ một câu:** "Reversible Instance Normalization (RevIN) uses a related normalize-process-denormalize template, motivated empirically by distribution shift in time-series forecasting." Họ cũng viết: "We show that the same template, with global μ and std statistics, is exactly the form taken by every normalization-equivariant map."

**Về đa kênh, họ loại trừ tường minh:** "Pooling statistics per channel would instead correspond to enforcing a larger, channel-wise affine equivariance, which we do not assume." Họ không bàn chi phí xấp xỉ khi target không đẳng biến.

**Hệ quả cho novelty:** đây là bài gần nhất, và đã nói ra ý "RevIN = khuôn mẫu đặc trưng hoá NE". Paper của ta **phải trích Saied & Fleuret** cho phần đặc trưng hoá. Phần còn lại cho ta:
- nhóm theo kênh, trộn đẳng biến (target) và bất biến (covariate), đúng cái họ loại trừ;
- chi phí Bayes khi đặt sai đối xứng;
- việc chọn kênh.

### 2.4 Berthelier et al. (3/2026)

**Nguồn:** G. Berthelier, T. Nabil, E. Le Naour, R. Niamke, S. Perlaza, G. Neglia, "On the Role of Reversible Instance Normalization", arXiv:2603.11869. <https://arxiv.org/html/2603.11869>

- Không có định lý. Không nói tới equivariance.
- Loại trừ covariate: "A more general setting would include multivariate inputs and outputs, exogenous covariates, and probabilistic predictions. Although we anticipate similar conclusions, we do not consider it here."
- **Câu quan trọng (Mục 4.2, Conditional shift):** "Overall, robustness to conditional changes likely requires retaining dependence on the original statistics, although doing so in an architecture-agnostic manner is not straightforward and ongoing research." Đây là *động cơ* của ta, được nói ra nhưng chưa được hình thức hoá. Nên trích như vậy.
- Kết luận khác: affine $(\alpha,\beta)$ thừa; huấn luyện trong không gian đã chuẩn hoá tổng quát hoá tốt hơn (xem điểm 4 ở Mục 2.1).

---

## 3. Ngoại suy và định danh khi thống kê test nằm ngoài support train

### 3.1 Christiansen, Pfister, Jakobsen, Gnecco, Peters (IEEE TPAMI 2022)

**Nguồn:** "A Causal Framework for Distribution Generalization", IEEE TPAMI 2022, DOI 10.1109/TPAMI.2021.3094760. arXiv:2006.07433. <https://arxiv.org/abs/2006.07433>

- **Assumption 2 (Extrapolation of F):** "For all f̃, f̄ ∈ F with f̃(x) = f̄(x) for all x ∈ supp(X), it holds that f̃ ≡ f̄."
- **Prop. 4.4:** dưới Assumption 1 và 2, mô hình tổng quát hoá được tới các can thiệp mở rộng support.
- **Prop. 4.7 (Impossibility of extrapolation):** "Assume that F = {f : ℝ^d → ℝ | f is continuous}. Let I be a well-behaved set of support-extending interventions on X, such that supp_I(X) \ supp(X) has non-empty interior. Then, (P_M, M) does not admit generalization to I, even if Assumption 1 is satisfied. In particular, for any function f̄ ∈ F and any c > 0, there exists a model M̃ …" (worst-case risk vượt quá cực tiểu ít nhất c).
- Phần giới thiệu: "strong assumptions on the functional relationship between X and Y are needed whenever the interventions extend the training support of X."

**Ủng hộ Định lý 3 (a).** Nửa âm của Định lý 3 là **hệ quả trực tiếp** của Prop. 4.7, hoặc của lập luận hai điểm cơ bản hơn. Không mới.

**Góc nhìn sắc hơn cho nửa dương:** họ $\mathcal F_{RevIN}$ thoả Assumption 2 trên *bão hoà orbit* $G\cdot\mathrm{supp}(Z)$. Hai hàm trong họ trùng nhau trên $\mathrm{supp}(Z)$ thì trùng nhau trên $G\cdot\mathrm{supp}(Z)$, vì chúng cùng một $\varphi$ trên $\mathrm{supp}(u)$.

Vậy **đẳng biến là một quy tắc ngoại suy**, cùng loại với "ngoại suy tuyến tính" trong NILE của họ. Nó *đúng* khi và chỉ khi $m$ thật sự đẳng biến trên vùng test. Điều này **không kiểm được từ dữ liệu train** khi $s$ ở test ra khỏi support.

Định lý 3 trong v0.5 nên được viết lại thành **so sánh hai giả định cấu trúc**, không phải "bất biến an toàn, phá bất biến rủi ro".

### 3.2 Dong & Ma (ICLR 2023)

**Nguồn:** K. Dong, T. Ma, "First Steps Toward Understanding the Extrapolation of Nonlinear Models to Unseen Domains", ICLR 2023 (spotlight). arXiv:2211.11719. <https://arxiv.org/abs/2211.11719>

- **Prop. 4.1 (lower bound):** với lớp mạng ReLU hai tầng, nếu $Q$ có khối lượng dương trên các điểm cách xa $\mathrm{supp}(P)$, thì có hai mạng "that agree on P perfectly but behave very differently on Q". Nguyên văn phần giới thiệu: "even two-layer neural networks cannot be guaranteed to extrapolate outside the support of the training distribution without further assumptions on the domain shift."
- **Kết quả dương:** "the family of nonlinear models of the form f(x)=∑ f_i(x_i) … can extrapolate to unseen distributions, if the covariance of the features is well-conditioned". Điều kiện: phân phối biên của từng nhóm feature không dịch chuyển, chỉ phân phối đồng thời dịch chuyển.

**Liên hệ trực tiếp với ta:** chia feature thành $x_1=u$ và $x_2=\rho(s_S)$. Nếu $\varphi(u,\rho)=\varphi_1(u)+\varphi_2(\rho)$, và biên của $u$ lẫn biên của $\rho(s)$ ở test đều được train bao phủ, thì mô hình ngoại suy được tới các *tổ hợp mới* (hình dạng quen, mức quen, nhưng cặp đôi chưa thấy).

Việc clip $\rho$ vào $[0,1]$ đảm bảo *biên* của $\rho$ luôn nằm trong đoạn train, về mặt giá trị. Nhưng nó **không** đảm bảo phân phối của $\rho$ nằm trong support: mọi giá trị ngoài vùng dồn về điểm biên. Đây là gợi ý thiết kế có cơ sở: đưa thống kê vào theo cấu trúc *cộng tính* hoặc tương tác bậc thấp, ví dụ FiLM một tầng, thay vì trộn tự do.

### 3.3 Xu et al. (ICLR 2021)

**Nguồn:** K. Xu, M. Zhang, J. Li, S. S. Du, K. Kawarabayashi, S. Jegelka, "How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks", ICLR 2021. arXiv:2009.11848. <https://arxiv.org/abs/2009.11848>

- **Theorem 1 (Linear extrapolation):** "Suppose we train a two-layer ReLU MLP f : ℝ^d → ℝ with squared loss in the NTK regime. For any direction v ∈ ℝ^d, let x₀ = tv. As t → ∞, f(x₀ + hv) − f(x₀) → β_v · h for any h > 0 …". Hệ quả họ nêu: MLP chỉ khớp ngoài vùng train những "functions that are almost-linear along the directions away from the origin."
- **Theorem 2:** với target tuyến tính, cần support train "covers all directions".

**Liên hệ:** đưa $(\mu,\log\sigma)$ *thô* vào MLP làm đầu ra ngoại suy xấp xỉ tuyến tính theo $s$ khi $s$ ra khỏi vùng train. Điều này phù hợp định tính với sự bùng nổ ở FR (V3b: 0.732 ± 0.133).

Đặc trưng bị chặn (rank/quantile rồi clip) biến phần phụ thuộc vào $s$ thành hằng số ngoài vùng. Như vậy **mô tả "bounded features ⇒ constant extrapolation" là đúng về hành vi mô hình**, trong phạm vi phần phụ thuộc $s$ chỉ đi qua $\rho(s)$.

Nhưng đó là *một* quy tắc ngoại suy. Nó chỉ tối ưu dưới giả định "hiệu ứng của thống kê bị chặn", xem Mệnh đề 3(c) ở Mục B. Nó **không** được Định lý 3 ngụ ý.

### 3.4 Kết luận câu hỏi 3

| Khẳng định trong v0.5 | Đánh giá |
|---|---|
| "Ngoài support train thì không định danh, cận minimax ≥ c > 0" | Đúng. Đã biết (Christiansen Prop. 4.7; Dong–Ma Prop. 4.1). Cần nêu rõ lớp hàm, vì $c$ phụ thuộc lớp (Mục B, Mệnh đề 3a). |
| "Mô hình bất biến ngoại suy *đúng* nếu $m$ thật sự đẳng biến" | Đúng nhưng gần như lặp lại giả thiết. Cần thêm: đẳng biến phải đúng *ngoài* support, và $\mathrm{supp}_Q(u)\subseteq\mathrm{supp}_P(u)$. Không kiểm được. |
| "Đặc trưng bị chặn ⇒ ngoại suy hằng số, *đúng tinh thần* Định lý 3" | Mô tả hành vi thì đúng. Biện minh thì sai. Cần mệnh đề riêng dưới giả định hiệu ứng bị chặn, và so với Xu et al. |

---

## 4. Kiểm định bất biến từ dữ liệu và ước lượng $D_S$

### 4.1 $D_S$ là gì theo ngôn ngữ thống kê

Gọi $r_Y=(y_{fut}-\mu_y)/\sigma_y$. Hai đồng nhất thức sau cho ta đọc $D_S$ theo ngôn ngữ thống kê:
- $\sigma_y^2\|r_Y-\varphi\|^2=\|y_{fut}-\mu_y-\sigma_y\varphi\|^2$, tức MSE ở thang gốc.
- $E_Q[r_Y\mid z]=r$, với $dQ\propto\sigma_y^2dP$.

Từ đó:
$$D_S=\min_{\mathcal F_\varnothing}E\|y-f\|^2-\min_{\mathcal F_S}E\|y-f\|^2 .$$

Đây chính là:
- **LOCO ở mức tổng thể** với loss bình phương, trên thang gốc. Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, "Distribution-Free Predictive Inference for Regression", JASA 113(523), 2018, Mục 6 "Model-Free Variable Importance: LOCO". <https://arxiv.org/abs/1604.04173>. Bài gốc dùng sai số tuyệt đối; bản bình phương là biến thể.
- **Variable importance "oracle predictiveness" của Williamson et al.**: "a population-level contrast between the oracle predictiveness of all available features versus all features except those under consideration". B. D. Williamson, P. B. Gilbert, N. R. Simon, M. Carone, "A General Framework for Inference on Algorithm-Agnostic Variable Importance", JASA 118(543):1645–1658, 2023. <https://arxiv.org/abs/2004.03683>

Với $\mathcal F_S$, "feature bị loại" là $s_S$, còn $u$ luôn có mặt.

### 4.2 Suy luận hợp lệ: làm được gì, không làm được gì

**Ước lượng và khoảng tin cậy khi $D_S>0$:** dùng ước lượng plug-in cross-fitted của Williamson et al. Họ tiệm cận hiệu quả, dùng được với learner ML.

**Tại null $D_S=0$ có suy biến.** Williamson et al. Mục 3.4: "When ψ₀,s = 0 … the influence function of ψ_{n,s} is identically zero … standard Wald-type confidence intervals and tests … will typically have incorrect coverage or type I error". Cách xử lý: **sample-splitting**, ước lượng hai predictiveness trên hai nửa dữ liệu khác nhau. Họ cũng đề xuất kiểm định $\delta$-null $H_0:\psi\in[0,\delta]$, hợp với mục đích "kênh nào đáng giữ mức".

**Tính khó (hardness):**
- **Shah & Peters**, "The Hardness of Conditional Independence Testing and the Generalised Covariance Measure", Annals of Statistics 48(3):1514–1538, 2020. <https://arxiv.org/abs/1804.07203>. **Theorem 2 (No-free-lunch):** "Given any n ∈ ℕ, α ∈ (0,1), M ∈ (0,∞], and any potentially randomised test ψ_n that has valid level α for the null hypothesis P_{0,M}, we have that P_Q(ψ_n = 1) ≤ α for all Q ∈ Q_{0,M}. Thus ψ_n cannot have power against any alternative." Ở đây null là CI trên các phân phối liên tục tuyệt đối.
- **Lundborg, Kim, Shah, Samworth**, "The Projected Covariance Measure for Assumption-Lean Variable Significance Testing", Annals of Statistics 52(6):2851–2878, 2024. <https://arxiv.org/abs/2211.02039>. Họ áp tính khó này cho đúng null của ta, tức conditional mean independence: "this property is not testable without further restrictions on the null hypothesis: if (X, Y, Z) have a density … the power of any test at any alternative is at most its size. This comes as a direct consequence of the untestability of the smaller conditional independence null (Shah and Peters, 2020)." PCM là kiểm định đúng công cụ cho $H_0: E[r\mid u,s_S]=E[r\mid u]$. Nó hợp lệ khi các hồi quy phụ hội tụ đủ nhanh.

**Hệ quả cho "phép chẩn đoán trước khi huấn luyện":**
1. **Không** có kiểm định "kênh $c$ có cần giữ mức không" mà hợp lệ đều trên mọi phân phối. Mọi bảo đảm phụ thuộc vào chất lượng hồi quy $E[r\mid u]$, trong khi $u$ có số chiều rất cao (cả cửa sổ).
2. $\hat D_S$ nên được báo cáo như một *ước lượng có khoảng tin cậy phụ thuộc giả định* (cross-fitting + sample-splitting), không phải một "test".
3. $\hat D_S$ có thể âm trên mẫu hữu hạn. Hai mô hình phải cùng học tốt $u$, nếu không thì $\hat D_S$ đo khác biệt năng lực học chứ không đo đối xứng.
4. $\hat D_S$ tính trên $P$ (train) **không** dự báo được lợi ích trên test khi $Q(s\mid u)$ khác $P(s\mid u)$ (Mục 3).

### 4.3 Kiểm định đối xứng theo phân phối: phiên bản mạnh của $D=0$

**Nguồn:**
- K. Chiu, A. Sharp, B. Bloem-Reddy, "Randomization Tests for Conditional Group Symmetry", Electronic Journal of Statistics. arXiv:2412.14391. <https://arxiv.org/abs/2412.14391>
- Bản trước: K. Chiu, B. Bloem-Reddy, "Non-parametric Hypothesis Tests for Distributional Group Symmetry", arXiv:2307.15834. <https://arxiv.org/html/2307.15834v2>

**Nội dung:**
- Kiểm định "symmetry (invariance or equivariance) of a conditional distribution under the action of a specified locally compact group", với "finite-sample Type I error control".
- Đặc trưng hoá (Mục 1.1.1 / Theorem 3 của v2): "P_{Y|X} is equivariant if and only if (τ̃,X) ⊥⊥ τ̃⁻¹Y | γ(X), with τ̃|X ∼ ζ(X,·)".
- Khi hành động free: "the inversion kernel is δ_{τ(x)}. In this case, τ(gx)=gτ(x)" (Prop. 1).
- Không cần $P_X$ bất biến.

**Áp vào ta:** $\tau(z)=s$, $\gamma(z)=u$, $\tau^{-1}Y=r_Y$. Điều kiện trở thành
$$\text{"}P(y_{fut}\mid z)\text{ đẳng biến"}\iff s\perp\!\!\!\perp r_Y\mid u .$$

**Định lý 2 của ta là phiên bản trung bình (mean) của đúng phép thử này.** $D_{\text{all}}=0$ khi và chỉ khi $E_Q[r\mid u,s]=E_Q[r\mid u]$, và điều này là hệ quả của CI nói trên.

Đây là prior art trực tiếp cho "phép chẩn đoán đo được". Phải trích, và phải định vị $D_S$ là:
- một **độ đo cỡ hiệu ứng** trên thang MSE, theo kênh, tách target và covariate;
- chứ không phải một kiểm định mới.

Chiu–Sharp–Bloem-Reddy không nói tới time series, chuẩn hoá hay nhóm location–scale. Đây chỉ là nhận định theo phần đọc được.

**Công cụ nên dùng:** Williamson et al. (sample-split, cross-fitted) cho cỡ hiệu ứng $D_S$ theo từng kênh hoặc nhóm kênh. PCM (Lundborg et al.) nếu cần p-value cho conditional mean independence. Chiu–Sharp–Bloem-Reddy nếu muốn kiểm định đẳng biến của cả phân phối có điều kiện. Cả ba đều phải kèm cảnh báo về tính khó.

---

## 5. Prior art: đưa thống kê chuẩn hoá trở lại, và chuẩn hoá covariate

### 5.1 Non-stationary Transformer (Liu et al., NeurIPS 2022): v0.5 mô tả sai

**Nguồn:** Y. Liu, H. Wu, J. Wang, M. Long, "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting", NeurIPS 2022. arXiv:2205.14415. <https://arxiv.org/abs/2205.14415> · Code: <https://github.com/thuml/Nonstationary_Transformers>

**Over-stationarization:** "the stationarized series deprived of inherent non-stationarity can be less instructive for real-world bursty events forecasting. This problem, termed over-stationarization in this paper, leads Transformers to generate indistinguishable temporal attentions for different series".

**De-stationary Attention, Eq. (6):**
$$\log\tau=\mathrm{MLP}(\sigma_x,x),\quad \Delta=\mathrm{MLP}(\mu_x,x),\quad \mathrm{Attn}=\mathrm{Softmax}\Big(\tfrac{\tau\,Q'K'^\top+\mathbf 1\Delta^\top}{\sqrt{d_k}}\Big)V' .$$
- $\tau\in\mathbb R_+$ là vô hướng, xấp xỉ $\sigma_x^2$.
- $\Delta\in\mathbb R^{S}$ xấp xỉ $K\mu_Q$.
- Nguyên văn: "we apply a multilayer perceptron as the projector to learn de-stationary factors τ, Δ from the statistics μ_x, σ_x of unstationarized x individually".

**Code chính thức** (`ns_models/ns_Transformer.py`):
- `tau = self.tau_learner(x_raw, std_enc).exp()` và `delta = self.delta_learner(x_raw, mean_enc)`, với `x_raw: B x S x E`, `stats: B x 1 x E`.
- `Projector` có tầng đầu `nn.Linear(2 * enc_in, ...)`.

Như vậy projector nhận **chuỗi thô và thống kê của cả E kênh**. Ở chế độ MS, E kênh gồm cả covariate.

**Hệ quả:**
- NST phá cả đẳng biến target lẫn bất biến covariate, bằng thống kê *thô, không chặn*, qua 1 + S con số điều biến attention.
- Câu trong v0.5 ("Non-stationary Transformer chỉ đưa lại thống kê của target", bảng Mục 3) **sai và phải sửa**.
- Điểm khác thật sự của PEN so với NST:
  - chọn kênh;
  - đặc trưng bị chặn;
  - đường đưa thống kê vào (token/FiLM so với attention);
  - lý thuyết.
- Ablation "NST có đủ không" (v0.5 §7.2) trở nên bắt buộc.

### 5.2 RevIN (Kim et al., ICLR 2022)

**Nguồn:** T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo, "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022. <https://openreview.net/forum?id=cGDAkQo1C0p> · Code: <https://github.com/ts-kim/RevIN>

- Chuẩn hoá theo instance và theo kênh, affine học được $(\gamma,\beta)$, rồi đảo ngược ở đầu ra.
- Động cơ thuần thực nghiệm (distribution shift). Không có khung đối xứng.
- OpenReview chặn tải PDF tự động. Mô tả này dựa trên trang dự án, README và các bài trích lại.

### 5.3 DeepAR (Salinas et al., IJF 2020) và code GluonTS

**Nguồn:** D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski, "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", Int. J. Forecasting 36(3), 2020. arXiv:1704.04110. <https://arxiv.org/abs/1704.04110>

- **Mục 3.3 "Scale handling":** "dividing the autoregressive inputs z_{i,t} … by an item-dependent scale factor ν_i, and conversely multiplying the scale-dependent likelihood parameters by the same factor … ν_i = 1 + (1/t₀)∑ z_{i,t}". Đây là đẳng biến *theo scale* (không theo shift), và chỉ gần đúng vì có "+1".
- **Covariate:** "We standardize all covariates to have zero mean and unit variance". Chuẩn hoá *toàn cục*, không theo cửa sổ. **DeepAR giữ nguyên mức tuyệt đối của covariate**, tức không bất biến theo covariate.
- **Code GluonTS** (`gluonts/torch/model/deepar/module.py`): `static_feat = torch.cat((embedded_cat, feat_static_real, scale.log()), dim=1)`. Log của scale target được đưa vào làm feature. Đây là **tiền lệ của V3b** (phá đẳng biến target bằng thống kê target), dùng log thô, không chặn.

### 5.4 Lag-Llama (Rasul et al., 2023/2024)

**Nguồn:** K. Rasul et al., "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting", arXiv:2310.08278v3. <https://arxiv.org/abs/2310.08278> · Code: <https://github.com/time-series-foundation-models/lag-llama>

- **Mục 4.4 "Value Scaling" (nguyên văn):** "We can then replace the time series x^i_{1:C} in the window by {(x^i_t − μ^i)/σ^i}. We also incorporate μ^i and σ^i as time independent real-valued covariates for each token, to give the model information of the statistics of the inputs, which we call summary statistics." Trên thực tế họ dùng robust scaling (median, IQR).
- **Code** (`lag_llama/model/module.py`): `static_feat = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)`. Như vậy feature thực tế là $\log(1+|\text{loc}|)$ (mất dấu) và $\log(\text{scale})$, không chặn.
- Lag-Llama là mô hình đơn biến, nên không có covariate để bàn.

### 5.5 Chronos-2 (arXiv:2510.15821)

**Nguồn:** A. F. Ansari et al., "Chronos-2: From Univariate to Universal Forecasting" (Technical Report), arXiv:2510.15821. <https://arxiv.org/abs/2510.15821>

- **Mục 3.1 "Robust Scaling":** "We adopt standardization … and introduce an additional step: applying the sinh⁻¹ transformation to the standardized values." Công thức $\tilde v_{t,d}=\sinh^{-1}\big((v_{t,d}-\mu_d)/\sigma_d\big)$, "where μ_d and σ_d are the mean and standard deviation of the historical values". Áp cho **mọi chiều $d$**, cả target lẫn covariate (Hình 1: "Input time series (targets and covariates) are first normalized").
- **Mục 3.4:** $\hat y=\mu_d+\sigma_d\sinh(\hat z)$.
- **Kết luận:** Chronos-2 nằm **chính xác** trong $\mathcal F_{RevIN}$, vì asinh là một phần của $\varphi$. Covariate bất biến hoàn toàn. Khẳng định trong v0.5 đúng.

### 5.6 Moirai (Woo et al., ICML 2024)

**Nguồn:** G. Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", ICML 2024. arXiv:2402.02592. <https://arxiv.org/abs/2402.02592>

- **Mục 3.1:** "(non-learnable) instance normalization (Kim et al., 2022) is applied to inputs/outputs".
- Code `uni2ts/module/packed_scaler.py`: `PackedStdScaler` tính loc/scale theo nhóm `(sample_id, variate_id)`, tức **theo từng biến, kể cả covariate động**. Moirai cũng thuộc $\mathcal F_{RevIN}$ về phía covariate.

### 5.7 TimesFM, phần covariate (XReg)

**Nguồn:** code chính thức <https://github.com/google-research/timesfm>, các file `src/timesfm/utils/xreg_lib.py` và `timesfm_2p5_base.py`. Mô hình gốc: A. Das, W. Kong, R. Sen, Y. Zhou, "A decoder-only foundation model for time-series forecasting", ICML 2024.

- Covariate đi qua một **hồi quy tuyến tính (ridge) in-context**, theo hai chế độ: "xreg + timesfm" (fit XReg lên target rồi TimesFM dự báo phần dư) hoặc "timesfm + xreg".
- Cột covariate số được chuẩn hoá bằng mean/std **gộp trên toàn bộ hàng train của batch** (`x_mean = np.mean(x_train, axis=0)`), không theo từng cửa sổ.
- Target được chuẩn hoá theo input (`normalize_xreg_target_per_input=True`).

**Kết luận:** TimesFM giữ mức covariate, nhưng chỉ qua một mô hình *tuyến tính*. Lớp hàm affine đóng dưới phép affine của covariate, nên không biểu diễn được hiệu ứng mức phi tuyến (chữ U). Câu v0.5 §4.6 "TSFM … buộc phải bất biến theo covariate" đúng với Chronos-2 và Moirai, nhưng **không đúng với TimesFM-XReg**.

### 5.8 TFT (Lim et al., IJF 2021) và pytorch-forecasting

- **TFT:** B. Lim, S. Ö. Arık, N. Loeff, T. Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", IJF 37(4), 2021. Code chính thức `google-research/tft/data_formatters/electricity.py`: `StandardScaler` fit **theo từng entity** trên dữ liệu train, cho cả target và real inputs. Đây là chuẩn hoá theo chuỗi, cố định, không theo cửa sổ. Mức thay đổi giữa các cửa sổ của cùng một chuỗi vẫn được giữ.
- **pytorch-forecasting:** `EncoderNormalizer` ("fitted on each encoder sequence separately") cộng `add_target_scales` ("add the center and scale of the unnormalized timeseries as features"). Đây là **RevIN + thống kê target làm feature**, đã có sẵn như một tuỳ chọn phần mềm.

### 5.9 Các bài điều phối viên yêu cầu kiểm tra

| Bài | Nội dung liên quan (nguồn gốc) | Covariate tách khỏi target? | Có lý thuyết? |
|---|---|---|---|
| **α-RevIN**, Yu, Jiang, Yang, "Three-Stage Learning Unlocks Strong Performance in Simple Models for LTSF" (STAIR), arXiv:2605.13678 (5/2026). <https://arxiv.org/abs/2605.13678> | Mục "α-RevIN: Partial Reversible Normalization": "standard RevIN implies a strong assumption: local mean and variance changes are mainly non-stationary factors that should be removed. In some datasets, mean or scale changes may themselves contain useful information for future prediction." Và: "When α = 1, α-RevIN reduces to standard RevIN; when α = 0, it is equivalent to no RevIN; and when 0 < α < 1, the model partially removes local statistics". α cố định, dùng chung. Có biến thể chỉ-mean và chỉ-std. Công thức đọc từ PDF bị mất ký hiệu; dạng hợp lý là $(x_c-\alpha\mu_c)/\sigma_c^{\alpha}$, cần kiểm lại trên bản PDF. | **Không.** Benchmark LTSF chuẩn, mọi kênh như nhau. Không thấy từ "exogenous" hay "covariate". | **Không.** |
| **RS-GLinear**, arXiv:2510.03788 | Chỉ dùng RevIN chuẩn. Không có α-RevIN. | Không | Không |
| Vilhes, Gasso, Alaya, "Does Normalization Choice Matter for Causal Large Time-Series Models?", arXiv:2606.09954 (TSALM@ICLR'26). <https://arxiv.org/abs/2606.09954> | Rò rỉ thông tin tương lai do chuẩn hoá trong mô hình causal (decoder). | Không (không thấy covariate trong toàn văn) | Không (thực nghiệm) |
| Hu, Gauthier, Naima, "Reversible Residual Normalization Alleviates Spatio-Temporal Distribution Shift", arXiv:2604.15838. <https://arxiv.org/abs/2604.15838> | Chuẩn hoá khả nghịch có GNN cho dữ liệu không gian–thời gian. Có câu "the latent conditional distribution … becomes robust and approximately invariant". | Không | Không có định lý về bất biến |
| Berthelier et al., arXiv:2603.11869 | Câu "robustness to conditional changes likely requires retaining dependence on the original statistics …" (Mục 4.2). Loại trừ covariate tường minh. | Không | Không |
| Saied & Fleuret, arXiv:2605.08193 | Đặc trưng hoá NE ⇔ normalize–process–denormalize. Nhắc tới RevIN. | Loại trừ per-channel tường minh | **Có** (đặc trưng hoá), không có gap |

**Khác, ít liên quan:**
- NoRIN (arXiv:2605.10823): Johnson-SU sau median/MAD theo kênh, vẫn thuộc $\mathcal F_{RevIN}$.
- "Noise or Signal?" (arXiv:2510.04667): robust statistics, không có covariate.
- "Temporal Window Smoothing of Exogenous Variables" (arXiv:2507.05284): whitening covariate bằng thống kê *toàn cục*. Có áp instance norm lên cả endogenous lẫn exogenous. Không có lý thuyết.

### 5.10 Có ai đã nói "RevIN là một lựa chọn đối xứng" hoặc "bất biến theo mức covariate" chưa?

**Tìm được gần nhất (quan trọng nhất cho novelty):**
1. Saied & Fleuret (2026): RevIN "uses a related normalize-process-denormalize template" và template này là dạng của mọi ánh xạ NE. **Đã có ý "RevIN ≡ đẳng biến"**, nhưng chỉ cho một nhóm toàn cục, và loại trừ per-channel tường minh.
2. Berthelier et al. (2026): nói trực giác "cần giữ phụ thuộc vào thống kê gốc", không hình thức hoá, không có covariate.
3. α-RevIN (2026): "mean or scale changes may themselves contain useful information", giải pháp thực nghiệm, không theo kênh, không có covariate.
4. NST (2022): đưa lại thống kê mọi kênh, gọi hiện tượng là "over-stationarization", không có khung đối xứng.

**Không tìm thấy** công trình nào phát biểu:
- (i) RevIN khi có covariate là đẳng biến theo target *và* bất biến theo từng covariate;
- (ii) chi phí Bayes của lựa chọn đó;
- (iii) phép chọn kênh dựa trên $D_S$.

Tìm kiếm đã thử nhiều tổ hợp từ khoá (RevIN, equivariance, canonicalization, maximal invariant, exogenous, covariate normalization). Nhưng "không tìm thấy" không bằng "không tồn tại". Cần tìm lại trên OpenReview của ICLR/NeurIPS 2026 trước khi nộp.

---

## 6. Những điểm reviewer sẽ nói "đã biết"

1. **Đặc trưng hoá họ RevIN** đã có ở: Lehmann–Casella Thm 3.17 (1998), B-R&T Lemma 8 (2020), Kaba et al. eq. (2) (2023), Saied–Fleuret Characterization 1 (2026, có nhắc RevIN). → Trình bày như "standard", trích đủ bốn nguồn.
2. **Công thức gap và $\varphi^*$** là MRE location–scale (Lehmann–Casella (3.44)) cộng phép chiếu $L^2$. → "Proposition", không phải "Theorem". Đóng góp nằm ở việc áp cho nhóm tích theo kênh (target và covariate) và ở hệ quả A/B.
3. **$D_S$** là LOCO / VIM (Lei et al. 2018; Williamson et al. 2023). **Phép thử đẳng biến có điều kiện** đã có (Chiu–Sharp–Bloem-Reddy). **Tính không kiểm được** đã biết (Shah–Peters; Lundborg et al.).
4. **Không định danh ngoài support** đã biết (Christiansen et al.; Dong–Ma). **MLP ngoại suy tuyến tính** đã biết (Xu et al.).
5. **Đưa thống kê target làm feature** đã có: Lag-Llama, GluonTS-DeepAR, pytorch-forecasting. **Đưa lại thống kê mọi kênh** đã có: NST. **Bỏ bớt thống kê một phần** đã có: α-RevIN. **Nhánh không đẳng biến song song** đã có: RPP (Finzi et al. 2021). **Không chuẩn hoá theo cửa sổ cho covariate** là mặc định cũ của DeepAR (covariate chuẩn hoá toàn cục) và TFT (theo entity).
6. **Bias–variance khi chọn nhóm con** đã có: Huang–Levie–Villar 2023; Petrache–Trivedi 2023; Kaba et al. §3.3 (lattice). Các bài này dùng nhóm compact, nhưng reviewer vẫn sẽ xem ý tưởng là quen thuộc.
7. **Hệ quả B** (hiệu ứng theo đơn vị tuyệt đối phá đẳng biến $a_0$) là một ví dụ của "m không đẳng biến". Nó đúng, nhưng tầm thường khi đã có Định lý 1. Giá trị của nó là *thực nghiệm* (v0.4 H3), không phải lý thuyết.

**Phần có thể bảo vệ như đóng góp:**
- (i) Đối xứng *theo kênh, hỗn hợp* khi có covariate. Đây đúng là trường hợp Saied–Fleuret và Berthelier loại trừ.
- (ii) Hệ quả A: hiệu ứng phụ thuộc mức của covariate (chữ U nhiệt độ–tải, merit-order), có cận dưới định lượng và dataset có vật lý đảm bảo.
- (iii) Phiên bản chính xác của đánh đổi (Mục B, Mệnh đề 3), dẫn tới thiết kế cộng tính + chặn, và kiểm chứng bằng đường cong theo độ dịch chuyển.
- (iv) Đánh giá có kiểm soát trên dataset có covariate thật, có NST, α-RevIN, kiểu Lag-Llama và RPP làm baseline.

---

## A. Bảng phán quyết

| Mục | Phán quyết | Lý do / nguồn |
|---|---|---|
| (a) $\mathcal F_{RevIN}$ = tập hàm đẳng biến theo target, bất biến theo covariate; $u$ là maximal invariant | **Đã biết.** Đúng khi hạn chế vào $Z_+$ ($\sigma>0$). | Lehmann–Casella 1998 Thm 3.17; B-R&T 2020 Lemma 8; Kaba et al. 2023 eq. (2); Saied–Fleuret 2026 Char. 1 (một kênh). Cần ghi chú epsilon và $Z_+$. |
| Định lý 1 (gap Bayes, $\varphi^*$ có trọng số) | **Hệ quả của Lehmann–Casella (3.44) + phép chiếu $L^2$ có trọng số.** **Không** phải hệ quả của Elesedy–Zaidi (họ cần nhóm compact, Haar, $\mu$ bất biến, và đo hướng ngược lại). | Sửa hai chỗ: "gap = 0 ⇔ $m$ trùng P-hầu khắp nơi với một hàm đẳng biến" (chỉ trên support); gap có thể bằng 0 dù cơ chế không đẳng biến, nếu trên support $s$ là hàm của $u$. Thêm nhận xét trọng số (loss thang gốc ⇔ $\varphi^*$; loss thang chuẩn hoá ⇔ $E_P[r\mid u]$). |
| Hệ quả A (cận dưới $p_{lo}p_{hi}\|\Delta r\|^2$) | **Đúng, cần sửa cách viết**: $\text{Gap}\ge E[\sigma_y^2]\,E_u[p_{lo}(u)p_{hi}(u)\|\Delta r(u)\|^2]$ khi $\sigma_y\perp(u,r)$ | Phương sai của hỗn hợp hai điểm. Là ví dụ, không phải định lý mới. |
| Hệ quả B | **Đúng, tầm thường** khi đã có Định lý 1 | Giá trị nằm ở thực nghiệm (v0.4 H3). |
| Định lý 2 (phân rã, $D_S\ge0$) | **Hệ quả (Pythagoras / chain rule).** $D_S$ = LOCO/VIM tổng thể với MSE thang gốc. | Lei et al. 2018; Williamson et al. 2023. Sửa: $E_w$ phải là kỳ vọng dưới $Q\propto\sigma_y^2P$; $D_S$ đơn điệu nhưng **không cộng tính** giữa các kênh (chain rule phụ thuộc thứ tự). |
| "Phép kiểm định trước khi huấn luyện" dựa trên $D_c$ | **Sai như đang viết.** Không có kiểm định hợp lệ đều (Shah–Peters Thm 2; Lundborg et al. §1). Phiên bản phân phối đã có (Chiu–Sharp–Bloem-Reddy). | Viết lại thành "ước lượng cỡ hiệu ứng, sample-split + cross-fit, CI phụ thuộc giả định hồi quy; $\delta$-null test". Không dự báo lợi ích khi $P(s\mid u)$ dịch chuyển. |
| Định lý 3, nửa âm (không định danh ngoài support) | **Đã biết** | Christiansen et al. 2022 Prop. 4.7; Dong–Ma 2023 Prop. 4.1. Cần nêu lớp hàm; cận là $(B^2/4)E_Q[\sigma_y^2\mathbf 1_A]$ (Mục B). |
| Định lý 3, nửa dương ("bất biến ngoại suy đúng") | **Cần sửa.** Đúng chỉ khi đẳng biến đúng *ngoài* support và $\mathrm{supp}_Q(u)\subseteq\mathrm{supp}_P(u)$. Đó là giả định cấu trúc không kiểm được (Christiansen Assumption 2 trên bão hoà orbit). | Đánh đổi thật là giữa *hai giả định cấu trúc*, không phải giữa "bias" và "rủi ro". |
| "Đặc trưng bị chặn ⇒ ngoại suy hằng số, đúng tinh thần Định lý 3" | **Mô tả đúng, biện minh sai.** | Cần Mệnh đề 3(c) (hiệu ứng bị chặn ⇒ sai số ≤ $B^2$, trong khoảng hệ số 4 của cận minimax), so với Xu et al. Thm 1. Clip biên không bảo đảm support đồng thời → cấu trúc cộng tính (Dong–Ma). |
| Phương pháp PEN | **Từng thành phần đều có tiền lệ; tổ hợp theo kênh cho covariate + chặn + chọn bằng $D_S$ thì chưa thấy.** | Thống kê target làm feature: Lag-Llama, GluonTS-DeepAR, pytorch-forecasting. Thống kê mọi kênh: NST. Bỏ một phần: α-RevIN. Nhánh thang gốc ≈ RPP (Finzi et al.). Chọn nhóm con ≈ Kaba §3.3, Huang et al. |
| Khẳng định v0.5 "NST chỉ đưa lại thống kê target" | **Sai** | Code chính thức: projector nhận `x_raw` và `std_enc` / `mean_enc` của cả E kênh. |
| Khẳng định v0.5 "Chronos-2 thuộc họ này + asinh" | **Đúng** | Chronos-2 §3.1 và §3.4 |
| Khẳng định v0.5 "TSFM buộc phải bất biến theo covariate" | **Đúng một phần.** Chronos-2 và Moirai đúng; TimesFM-XReg thì không (giữ mức, nhưng tuyến tính). | Code TimesFM `xreg_lib.py` |

---

## B. Các phát biểu đề xuất (đã sửa)

**Thiết lập chung:**
- Cửa sổ $z=(y_{1:L},x_{1:L+H})$. Thống kê chỉ tính trên phần quá khứ.
- $Z_+=\{z:\sigma_y>0,\sigma_c>0\ \forall c\}$, giả định $P(Z_+)=1$ (bỏ epsilon).
- $G=(\mathbb R\times\mathbb R_{>0})^{1+C}$ tác động theo kênh, $(b,a)\cdot v=av+b$.
- $\tau(z)=s\in G$ và $u=\tau(z)^{-1}\cdot z$.
- $m(z)=E[y_{fut}\mid z]$, $r=(m-\mu_y)/\sigma_y$, $r_Y=(y_{fut}-\mu_y)/\sigma_y$.
- Giả định $E\|y_{fut}\|^2<\infty$ và $E\sigma_y^2<\infty$.
- Định nghĩa $dQ=\frac{\sigma_y^2}{E\sigma_y^2}dP$.

**Mệnh đề 0 (đặc trưng hoá, standard).** Hàm đo được $f:Z_+\to\mathbb R^H$ thoả $f(g\cdot z)=a_0f(z)+b_0\mathbf 1$ với mọi $g\in G$ khi và chỉ khi $f(z)=\mu_y+\sigma_y\varphi(u)$, với $\varphi$ đo được. $u$ là maximal invariant, và $z\mapsto(s,u)$ là song ánh $Z_+\to G\times U$.
*Nguồn:* Lehmann–Casella Thm 3.17; B-R&T Lemma 8; Kaba et al. (2); Saied–Fleuret Char. 1.

**Định lý 1 (chi phí Bayes của đối xứng RevIN).**
$$\inf_{f\in\mathcal F_{RevIN}}E\|y_{fut}-f\|^2-E\|y_{fut}-m\|^2=E\big[\sigma_y^2\|r-\varphi^*(u)\|^2\big]=E[\sigma_y^2]\;E_Q\big\|r-E_Q[r\mid u]\big\|^2,$$
với
$$\varphi^*(u)=E_Q[r\mid u]=\frac{E[\sigma_y(m-\mu_y)\mid u]}{E[\sigma_y^2\mid u]},$$
và infimum đạt tại $\varphi^*$.
- Gap = 0 khi và chỉ khi $m=\mu_y+\sigma_y\varphi^*(u)$ P-hầu khắp nơi, tức $m$ trùng P-h.k.n. với một hàm $G$-đẳng biến. Đây là tính chất *trên support của P*.
- **Nhận xét 1:** $\varphi^*$ là dạng dự báo của MRE location–scale, Lehmann–Casella (3.44).
- **Nhận xét 2:** huấn luyện với MSE không trọng số trên thang chuẩn hoá cho $\varphi_0=E_P[r\mid u]$, có gap thang gốc lớn hơn đúng một lượng $E[\sigma_y^2\|\varphi_0-\varphi^*\|^2]$.

**Định lý 2 (chuỗi lồng nhau và cỡ hiệu ứng theo kênh).** Với $S\subseteq\{0,1,\dots,C\}$, đặt $\mathcal F_S=\{\mu_y+\sigma_y\varphi(u,s_S)\}$. Khi đó:
- $\text{Gap}(S)=E[\sigma_y^2]\,E_Q\|r-E_Q[r\mid u,s_S]\|^2$.
- Với $S\subseteq T$: $\text{Gap}(S)-\text{Gap}(T)=E\big[\sigma_y^2\|E_Q[r\mid u,s_T]-E_Q[r\mid u,s_S]\|^2\big]\ge0$.
- $D_S:=\text{Gap}(\varnothing)-\text{Gap}(S)$ bằng hiệu MSE Bayes thang gốc giữa $\mathcal F_\varnothing$ và $\mathcal F_S$ (LOCO/VIM tổng thể).
- $D_S=0$ khi và chỉ khi $E_Q[r\mid u,s_S]=E_Q[r\mid u]$ hầu khắp nơi.
- Nếu $P(y_{fut}\mid z)$ đẳng biến, tức $s\perp\!\!\!\perp r_Y\mid u$ (Chiu–Sharp–Bloem-Reddy), thì $D_S=0$ với mọi $S$.
- $D$ **không cộng tính**: $D_{S\cup T}=D_S+[\text{Gap}(S)-\text{Gap}(S\cup T)]$.
- **Suy luận:** ước lượng bằng sample-split + cross-fit (Williamson et al. §3.4). Không tồn tại kiểm định $H_0:D_S=0$ hợp lệ đều trên mọi phân phối có mật độ (Lundborg et al., hệ quả của Shah–Peters Thm 2).

**Mệnh đề 3 (ngoại suy: hai giả định cấu trúc).**

*Thiết lập:* $Q$ là phân phối test, khác $P$ chỉ ở biên của $z$ ($P(y\mid z)$ giữ nguyên). $A=\{z:(u,s_S)\notin\mathrm{supp}_P(u,s_S)\}$. Mô hình đúng: $m=\mu_y+\sigma_y h(u,s_S)$, với lớp
$$\mathcal H_B=\{h:\ \|h(u,s)-h(u,s')\|_\infty\le B\ \ \forall u,s,s'\}.$$

- **(a) Cận dưới, đã biết.** Với mọi bộ học $\hat f$ dùng $n$ mẫu i.i.d. từ $P$:
$$\sup_{h\in\mathcal H_B}E_Q\|m-\hat f\|^2\ \ge\ \tfrac{B^2}{4}\,E_Q[\sigma_y^2\mathbf 1_A] .$$
*Chứng minh:* hai điểm $h_0(u)$ và $h_0(u)+B\mathbf 1_A$ cho cùng phân phối dữ liệu dưới $P$. Dùng $(a-v)^2+(b-v)^2\ge(a-b)^2/2$. Đây là dạng định lượng của Christiansen Prop. 4.7 / Dong–Ma Prop. 4.1.
- **(b) Giả định đẳng biến.** Nếu $h$ không phụ thuộc $s_S$ *trên toàn miền* (không chỉ trên $\mathrm{supp}\,P$), và $\mathrm{supp}_Q(u,s_{S^c})\subseteq\mathrm{supp}_P(u,s_{S^c})$, thì predictor Bayes của $\mathcal F_{S^c}$ học trên $P$ cũng là Bayes dưới $Q$. Đây là Christiansen Prop. 4.3/4.4, với $\mathcal F_{RevIN}$ thoả Assumption 2 trên bão hoà orbit. Nếu giả định sai, sai số test là $E_Q[\sigma_y^2\|h-\varphi^*_P\|^2]$. Đại lượng này **không** bị $D_S$ (tính dưới $P$) chặn khi $Q(s\mid u)\ne P(s\mid u)$.
- **(c) Giả định hiệu ứng bị chặn, cùng đặc trưng clip.** Gọi $\Pi$ là phép chiếu theo toạ độ của $s_S$ lên hộp train; tương đương với rank/quantile theo train rồi clip. Nếu $h\in\mathcal H_B$ và $(u,\Pi s_S)\in\mathrm{supp}_P(u,s_S)$ Q-h.k.n., thì predictor tổng thể $\mu_y+\sigma_y h(u,\Pi s_S)$ có sai số test trên $A$ không vượt quá $B^2E_Q[\sigma_y^2\mathbf 1_A]$. Mức này nằm trong hệ số 4 so với (a).
  - So sánh: MLP ReLU nhận $s$ thô ngoại suy tuyến tính theo tia (Xu et al. Thm 1), nên không có cận đều theo $B$.
  - Với giả định Lipschitz theo $s$, thay $B$ bằng $L\cdot\mathrm{dist}(s_S,\text{hộp})$.
  - Điều kiện support *đồng thời* không tự có khi clip theo biên. Dạng cộng tính $\varphi_1(u)+\varphi_2(\Pi s_S)$ chỉ cần phủ biên (Dong–Ma).

*Cách đọc:* chọn $S$ là chọn giữa (b), vốn không có bias trên support khi $D_S=0$, và (c), vốn không có bias trên support và có sai số ngoại suy $\le B^2$. Không lựa chọn nào an toàn vô điều kiện. Thực nghiệm cần quét độ dịch chuyển của $s_S$ để vẽ đường đánh đổi.

---

## C. Danh mục bắt buộc trích dẫn

**Lý thuyết đối xứng và equivariant estimation**
1. E. L. Lehmann, G. Casella. *Theory of Point Estimation*, 2nd ed. Springer, 1998. Ch. 3, §3.3, Thm 3.17, eq. (3.44).
2. B. Bloem-Reddy, Y. W. Teh. Probabilistic Symmetries and Invariant Neural Networks. *JMLR* 21(90):1–61, 2020. <https://www.jmlr.org/papers/v21/19-322.html>
3. S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio, S. Ravanbakhsh. Equivariance with Learned Canonicalization Functions. *ICML 2023*, PMLR 202. <https://arxiv.org/abs/2211.06489>
4. O. Puny, M. Atzmon, H. Ben-Hamu, I. Misra, A. Grover, E. J. Smith, Y. Lipman. Frame Averaging for Invariant and Equivariant Network Design. *ICLR 2022*. <https://arxiv.org/abs/2110.03336>
5. B. Elesedy, S. Zaidi. Provably Strict Generalisation Benefit for Equivariant Models. *ICML 2021*, PMLR 139. <http://proceedings.mlr.press/v139/elesedy21a.html> (chỉ làm bối cảnh)
6. C. Lyle, M. van der Wilk, M. Kwiatkowska, Y. Gal, B. Bloem-Reddy. On the Benefits of Invariance in Neural Networks. arXiv:2005.00178, 2020.
7. G. Benton, M. Finzi, P. Izmailov, A. G. Wilson. Learning Invariances in Neural Networks. *NeurIPS 2020*. <https://arxiv.org/abs/2010.11882>
8. M. Finzi, G. Benton, A. G. Wilson. Residual Pathway Priors for Soft Equivariance Constraints. *NeurIPS 2021*. <https://arxiv.org/abs/2112.01388>
9. M. Petrache, S. Trivedi. Approximation-Generalization Trade-offs under (Approximate) Group Equivariance. *NeurIPS 2023*. <https://arxiv.org/abs/2305.17592>
10. N. Huang, R. Levie, S. Villar. Approximately Equivariant Graph Networks. *NeurIPS 2023*. <https://arxiv.org/abs/2308.10436>
11. S. Chen, E. Dobriban, J. H. Lee. A Group-Theoretic Framework for Data Augmentation. *JMLR* 21, 2020. <https://www.jmlr.org/papers/v21/20-163.html>

**Normalization equivariance**
12. S. Herbreteau, E. Moebel, C. Kervrann. Normalization-Equivariant Neural Networks with Application to Image Denoising. *NeurIPS 2023*. <https://arxiv.org/abs/2306.05037>
13. Y. Saied, F. Fleuret. Normalization Equivariance for Arbitrary Backbones, with Application to Image Denoising. arXiv:2605.08193, 2026. <https://arxiv.org/abs/2605.08193>

**Ngoại suy và định danh**
14. R. Christiansen, N. Pfister, M. E. Jakobsen, N. Gnecco, J. Peters. A Causal Framework for Distribution Generalization. *IEEE TPAMI*, 2022. DOI 10.1109/TPAMI.2021.3094760. <https://arxiv.org/abs/2006.07433>
15. K. Dong, T. Ma. First Steps Toward Understanding the Extrapolation of Nonlinear Models to Unseen Domains. *ICLR 2023*. <https://arxiv.org/abs/2211.11719>
16. K. Xu, M. Zhang, J. Li, S. S. Du, K. Kawarabayashi, S. Jegelka. How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks. *ICLR 2021*. <https://arxiv.org/abs/2009.11848>

**Kiểm định và variable importance**
17. R. D. Shah, J. Peters. The Hardness of Conditional Independence Testing and the Generalised Covariance Measure. *Ann. Statist.* 48(3):1514–1538, 2020. <https://arxiv.org/abs/1804.07203>
18. A. R. Lundborg, I. Kim, R. D. Shah, R. J. Samworth. The Projected Covariance Measure for Assumption-Lean Variable Significance Testing. *Ann. Statist.* 52(6):2851–2878, 2024. DOI 10.1214/24-AOS2447. <https://arxiv.org/abs/2211.02039>
19. B. D. Williamson, P. B. Gilbert, N. R. Simon, M. Carone. A General Framework for Inference on Algorithm-Agnostic Variable Importance. *JASA* 118(543):1645–1658, 2023. <https://arxiv.org/abs/2004.03683>
20. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. Distribution-Free Predictive Inference for Regression. *JASA* 113(523), 2018. <https://arxiv.org/abs/1604.04173>
21. K. Chiu, A. Sharp, B. Bloem-Reddy. Randomization Tests for Conditional Group Symmetry. *Electronic Journal of Statistics* (arXiv:2412.14391, supersedes arXiv:2307.15834). <https://arxiv.org/abs/2412.14391>

**Forecasting: chuẩn hoá và thống kê**
22. T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. *ICLR 2022*. <https://openreview.net/forum?id=cGDAkQo1C0p>
23. Y. Liu, H. Wu, J. Wang, M. Long. Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting. *NeurIPS 2022*. <https://arxiv.org/abs/2205.14415>. Code: <https://github.com/thuml/Nonstationary_Transformers>
24. G. Berthelier, T. Nabil, E. Le Naour, R. Niamke, S. Perlaza, G. Neglia. On the Role of Reversible Instance Normalization. arXiv:2603.11869, 2026.
25. Z. Yu, G. Jiang, J. Yang. Three-Stage Learning Unlocks Strong Performance in Simple Models for Long-Term Time Series Forecasting (α-RevIN). arXiv:2605.13678, 2026.
26. D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. *Int. J. Forecasting* 36(3), 2020. <https://arxiv.org/abs/1704.04110>. Code GluonTS: `gluonts/torch/model/deepar/module.py`.
27. K. Rasul et al. Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. arXiv:2310.08278, 2023/2024.
28. A. F. Ansari et al. Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821, 2025.
29. G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, D. Sahoo. Unified Training of Universal Time Series Forecasting Transformers (Moirai). *ICML 2024*. <https://arxiv.org/abs/2402.02592>
30. A. Das, W. Kong, R. Sen, Y. Zhou. A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM). *ICML 2024*. Code XReg: <https://github.com/google-research/timesfm>
31. B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecasting* 37(4), 2021.

**Nên trích thêm (đối chiếu baseline):** Vilhes et al. arXiv:2606.09954; Hu et al. arXiv:2604.15838; NoRIN arXiv:2605.10823; pytorch-forecasting (`EncoderNormalizer`, `add_target_scales`); Zhou & Nayak, *JSPI* 142 (2012) 1367–1377.

---

### Giới hạn của lần kiểm tra này

- RevIN (OpenReview) chặn tải PDF tự động. Mô tả RevIN dựa trên trang dự án và các bài trích lại.
- Với Lyle et al., Augerino, RPP, Vilhes et al. và Hu et al., tôi chỉ đọc abstract hoặc grep toàn văn, chưa đọc kỹ từng định lý.
- Công thức α-RevIN trong PDF bị mất ký hiệu khi trích text. Dạng $(x-\alpha\mu)/\sigma^\alpha$ là suy đoán hợp lý, cần kiểm lại.
- Trang/số hiệu định lý của Lehmann–Casella lấy từ bản OCR của lần in thứ 2 (1998).
- "Không tìm thấy prior art trực tiếp" chỉ là kết quả tìm kiếm tới 2026-09-25, không phải bằng chứng là không có.
