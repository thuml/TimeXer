---
status: accepted
---

# Phạm vi: mô hình huấn luyện theo dataset và fine-tuning, không nhắm zero-shot foundation model

Foundation model (Chronos-2, TiRex-2, TimesFM) không biết đơn vị của covariate. Vì vậy chúng *buộc phải* bất biến theo covariate, và symmetry mismatch chỉ khắc phục được bằng hồi quy in-context. Mảng này đang rất đông (TabPFN-TS, TFMAdapter, TimesFM XReg), và bằng chứng của nhóm EDF (arXiv 2605.12200) cho thấy nút thắt của Chronos-2 là năng lực hồi quy in-context chứ không riêng đối xứng. Do đó PERIN nhắm các mô hình huấn luyện theo dataset (TimeXer, iTransformer, PatchTST, DAG, CITRAS, TiDE) và bước fine-tuning foundation model. Zero-shot foundation model chỉ đóng vai trò baseline tham chiếu.

## Consequences

- Mọi định lý đều phát biểu cho họ mô hình huấn luyện có giám sát trên một phân phối.
- Phần in-context chỉ là một ghi chú (remark), không phải đóng góp.
