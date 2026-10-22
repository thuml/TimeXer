---
status: accepted
---

# Tiêu chí reproduction và giới hạn phần cứng

Máy local chỉ có RTX 2050 (4 GB VRAM), trong khi paper chạy trên RTX 4090 24 GB. Thêm nữa, `run.py:14` hard-code `fix_seed = 2021` và bỏ qua `--seed`, nên không lặp lại được "5 lần chạy" như paper (Table 9).

Quyết định:

1. **Sửa seed trước tiên.** Seed phải lấy từ `--seed`. Mọi con số báo cáo là mean ± std trên 5 seed.
2. **Dung sai.** Một ô được coi là tái lập khi $|\text{MSE}_{ours} - \text{MSE}_{paper}| \le \max(2\,\text{std}_{paper},\ 3\%\cdot\text{MSE}_{paper})$. Nếu paper không báo std cho ô đó thì dùng ngưỡng 3%.
3. **Phạm vi bắt buộc chạy local:** EPF (5 thị trường, Table 2), cùng các bộ long-term exogenous ETTh1, ETTm1 và Weather (Table 11).
4. **Chạy trên GPU thuê hoặc bỏ qua:** ECL và Traffic (321 và 862 kênh, `d_model` 512, có thể không vừa 4 GB). Không đổi hyperparameter để "vừa GPU" mà vẫn gọi là reproduction. Nếu phải đổi, ghi là *adapted*.
5. **Gate:** ≥ 80% số ô bắt buộc nằm trong dung sai. Nếu không đạt, dừng lại và tìm nguyên nhân trước khi so baseline.

## Considered Options

- **Tin số liệu của paper mà không chạy lại.** Bị bác: mọi mức cải thiện sau này phải đo trên cùng code, cùng seed và cùng phần cứng.
