import torch
import torch.nn.functional as F
from layers.TS2Image import EndogenousTS2Image


def test_zero_nan_check():
    """Assertion 1: Constant windows must not produce NaN or Inf."""
    print("Running Test 1: Zero-NaN Assertion on Constant / Zero Variance Windows...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = EndogenousTS2Image(seq_len=168, img_size=176, period_len=24).to(device)

    # 1. All zeros
    x_zero = torch.zeros(4, 168, 1, device=device, requires_grad=True)
    out_zero = module(x_zero)
    assert not torch.isnan(out_zero).any(), "NaN found in all-zeros output!"
    assert not torch.isinf(out_zero).any(), "Inf found in all-zeros output!"
    loss_zero = out_zero.sum()
    loss_zero.backward()
    assert x_zero.grad is not None, "Gradient is None for all-zeros!"
    assert not torch.isnan(x_zero.grad).any(), "NaN found in all-zeros gradient!"
    assert not torch.isinf(x_zero.grad).any(), "Inf found in all-zeros gradient!"

    # 2. Constant non-zero
    x_const = (torch.ones(4, 168, 1, device=device) * 42.0).requires_grad_(True)
    out_const = module(x_const)
    assert not torch.isnan(out_const).any(), "NaN found in constant output!"
    assert not torch.isinf(out_const).any(), "Inf found in constant output!"
    loss_const = out_const.sum()
    loss_const.backward()
    assert x_const.grad is not None, "Gradient is None for constant input!"
    assert not torch.isnan(x_const.grad).any(), "NaN found in constant gradient!"
    assert not torch.isinf(x_const.grad).any(), "Inf found in constant gradient!"

    print(" -> PASSED: Zero-NaN assertion passed for constant and zero-variance inputs.")


def test_shapes_and_divisibility():
    """Assertion 2: Shapes and patch size divisibility (16 * K)."""
    print("Running Test 2: Output Shapes and Divisibility Check...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EPF: L=168, img_size=176 (16 * 11)
    epf_module = EndogenousTS2Image(seq_len=168, img_size=176, period_len=24).to(device)
    x_epf = torch.randn(8, 168, 1, device=device)
    out_epf = epf_module(x_epf)
    assert out_epf.shape == (8, 3, 176, 176), f"Expected (8, 3, 176, 176), got {out_epf.shape}"
    assert out_epf.shape[-1] % 16 == 0 and out_epf.shape[-2] % 16 == 0

    # LTSF: L=96, img_size=96 (16 * 6)
    ltsf_module = EndogenousTS2Image(seq_len=96, img_size=96, period_len=24).to(device)
    x_ltsf = torch.randn(8, 96, 1, device=device)
    out_ltsf = ltsf_module(x_ltsf)
    assert out_ltsf.shape == (8, 3, 96, 96), f"Expected (8, 3, 96, 96), got {out_ltsf.shape}"
    assert out_ltsf.shape[-1] % 16 == 0 and out_ltsf.shape[-2] % 16 == 0

    print(" -> PASSED: Shapes and 16-divisibility verified for both EPF and LTSF configurations.")


def test_mathematical_symmetries():
    """Assertion 3: Verify GASF symmetry and GADF skew-symmetry directly."""
    print("Running Test 3: Mathematical Symmetries (GASF & GADF)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set img_size == seq_len so no interpolation alters the raw GASF/GADF matrices
    module = EndogenousTS2Image(seq_len=168, img_size=168, period_len=24).to(device)
    x = torch.randn(2, 168, device=device)

    # Compute raw hat_x and hat_r inside
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    hat_x = torch.clamp(2.0 * ((x - x_min) / ((x_max - x_min) + 1e-5)) - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)
    hat_r = torch.sqrt(torch.clamp(1.0 - hat_x**2, min=1e-6, max=1.0))

    x_col, x_row = hat_x.unsqueeze(-1), hat_x.unsqueeze(1)
    r_col, r_row = hat_r.unsqueeze(-1), hat_r.unsqueeze(1)

    gasf = torch.bmm(x_col, x_row) - torch.bmm(r_col, r_row)
    gadf = torch.bmm(r_col, x_row) - torch.bmm(x_col, r_row)

    # GASF symmetry: G == G^T
    gasf_diff = torch.abs(gasf - gasf.transpose(-1, -2)).max().item()
    assert gasf_diff < 1e-5, f"GASF symmetry violated! Max diff: {gasf_diff}"

    # GADF skew-symmetry: G == -G^T
    gadf_diff = torch.abs(gadf + gadf.transpose(-1, -2)).max().item()
    assert gadf_diff < 1e-5, f"GADF skew-symmetry violated! Max diff: {gadf_diff}"

    # GADF main diagonal must be exactly 0
    diag_max = torch.abs(torch.diagonal(gadf, dim1=-2, dim2=-1)).max().item()
    assert diag_max < 1e-5, f"GADF diagonal is non-zero! Max diag: {diag_max}"

    print(" -> PASSED: GASF is symmetric and GADF is skew-symmetric with zero diagonal.")


def test_backward_gradient_flow():
    """Assertion 4: End-to-end gradient flow without NaN/Inf."""
    print("Running Test 4: Autograd and Backpropagation Flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = EndogenousTS2Image(seq_len=168, img_size=176, period_len=24).to(device)

    x = torch.randn(4, 168, 1, device=device, requires_grad=True)
    out = module(x)
    loss = (out ** 2).mean()
    loss.backward()

    assert x.grad is not None, "Gradient is None!"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN!"
    assert not torch.isinf(x.grad).any(), "Gradient contains Inf!"
    assert (x.grad.abs().sum() > 0).item(), "Gradient is entirely zero!"

    print(" -> PASSED: Autograd flows smoothly with non-zero, finite gradients.")


if __name__ == "__main__":
    test_zero_nan_check()
    test_shapes_and_divisibility()
    test_mathematical_symmetries()
    test_backward_gradient_flow()
    print("\nALL SANITY CHECKS FOR TS2Image PASSED SUCCESSFULLY!")
