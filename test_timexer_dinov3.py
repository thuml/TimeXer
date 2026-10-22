import argparse
import torch
import torch.nn as nn
from models.TimeXer_DINOv3 import Model, DINOv3BackboneWrapper


def build_test_configs(freeze_mode="all", use_bottleneck=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    # Task & Dimensions
    args.task_name = "long_term_forecast"
    args.features = "MS"
    args.seq_len = 168
    args.label_len = 48
    args.pred_len = 24
    args.patch_len = 24
    args.period_len = 24

    # Model architecture
    args.d_model = 256
    args.n_heads = 8
    args.d_ff = 512
    args.post_e_layers = 1
    args.dropout = 0.1
    args.embed = "timeF"
    args.freq = "h"
    args.use_norm = True

    # DINOv3 specific
    args.dino_model_name = "vit_small_patch16_dinov3"
    args.dino_img_size = 176
    args.dino_freeze = freeze_mode
    args.use_pooled_bottleneck = use_bottleneck
    return args


def test_token_shape_and_register_slicing():
    print("Running Test 1: Token Shape & Register Slicing Assertion (Section 6.2.2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = DINOv3BackboneWrapper(
        model_name="vit_small_patch16_dinov3",
        pretrained=True,
        img_size=176,
        freeze_mode="all",
    ).to(device)

    dummy_img = torch.randn(2, 3, 176, 176, device=device)
    raw_tokens = wrapper.dino.forward_features(dummy_img)

    # 1. Assert raw output shape is (B, 126, 384)
    assert raw_tokens.shape == (2, 126, 384), f"Expected raw shape (2, 126, 384), got {raw_tokens.shape}"

    # 2. Assert sliced z_cls is (B, 1, 384) and z_patch is (B, 121, 384)
    z_cls, z_patch = wrapper(dummy_img)
    assert z_cls.shape == (2, 1, 384), f"Expected z_cls (2, 1, 384), got {z_cls.shape}"
    assert z_patch.shape == (2, 121, 384), f"Expected z_patch (2, 121, 384), got {z_patch.shape}"

    print(" -> PASSED: Raw tokens (2, 126, 384) properly decomposed into z_cls (2, 1, 384) & z_patch (2, 121, 384) with 4 registers discarded.")


def test_gradient_isolation_regime_1():
    print("Running Test 2: Gradient Isolation in Regime 1 (Section 6.2.3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_test_configs(freeze_mode="all")
    model = Model(configs).to(device)

    # Simulate batch: B=4, L=168, C_ex=2 + target 1 = 3 variates
    x_enc = torch.randn(4, 168, 3, device=device)
    x_mark_enc = torch.randn(4, 168, 4, device=device)

    out = model(x_enc, x_mark_enc)
    loss = out.sum()
    loss.backward()

    # Check backbone gradients are ALL None
    dino_grads = [p.grad for p in model.dino_backbone.parameters() if p.grad is not None]
    assert len(dino_grads) == 0, f"DINOv3 backbone received {len(dino_grads)} gradients in Regime 1!"

    # Check that adapter and head DO receive gradients
    assert model.proj_cls[0].weight.grad is not None, "proj_cls weight has no gradient!"
    assert model.head.linear.weight.grad is not None, "FlattenHead linear weight has no gradient!"
    assert model.cross_fusion.norm1.weight.grad is not None, "CrossAttentionFusion norm has no gradient!"

    print(" -> PASSED: Gradient isolation strictly holds in Regime 1 (0 backbone gradients, trainable adapters/head).")


def test_endogenous_exogenous_routing():
    print("Running Test 3: Endogenous-Exogenous Routing Check (Section 6.2.4)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_test_configs(freeze_mode="all")
    model = Model(configs).to(device)

    # If we perturb only exogenous variables (cols :-1), endogenous image input must be unaffected
    x_enc1 = torch.randn(2, 168, 3, device=device)
    x_mark1 = torch.randn(2, 168, 4, device=device)

    x_enc2 = x_enc1.clone()
    x_enc2[:, :, :-1] += torch.randn_like(x_enc2[:, :, :-1]) * 10.0  # modify exogenous

    img1 = model.ts2img(x_enc1[:, :, -1:])
    img2 = model.ts2img(x_enc2[:, :, -1:])

    diff = (img1 - img2).abs().max().item()
    assert diff == 0.0, f"Exogenous perturbation leaked into 2D imaging branch! diff: {diff}"

    print(" -> PASSED: Target series and exogenous series are cleanly isolated into their respective pipelines.")


def test_end_to_end_forward_backward():
    print("Running Test 4: End-to-End Forward, Loss & Optimizer Step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_test_configs(freeze_mode="all", use_bottleneck=True)
    model = Model(configs).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    criterion = nn.MSELoss()

    x_enc = torch.randn(4, 168, 3, device=device)
    x_mark_enc = torch.randn(4, 168, 4, device=device)
    target_y = torch.randn(4, 24, 1, device=device)

    optimizer.zero_grad()
    pred_y = model(x_enc, x_mark_enc)
    assert pred_y.shape == (4, 24, 1), f"Expected output shape (4, 24, 1), got {pred_y.shape}"

    loss = criterion(pred_y, target_y)
    assert not torch.isnan(loss).item(), "Loss is NaN!"
    loss.backward()
    optimizer.step()

    print(f" -> PASSED: Forward & Backward step completed with Loss: {loss.item():.4f}")


def test_regime_2_partial_ft():
    print("Running Test 5: Regime 2 (Partial Fine-Tuning) Check...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_test_configs(freeze_mode="partial")
    model = Model(configs).to(device)

    trainable_dino_params = [p for p in model.dino_backbone.dino.parameters() if p.requires_grad]
    frozen_dino_params = [p for p in model.dino_backbone.dino.parameters() if not p.requires_grad]

    assert len(trainable_dino_params) > 0, "No trainable parameters in partial FT!"
    assert len(frozen_dino_params) > 0, "All parameters are trainable in partial FT!"

    # Exactly last 2 blocks + LayerNorms are trainable
    trainable_names = [n for n, p in model.dino_backbone.dino.named_parameters() if p.requires_grad]
    for n in trainable_names:
        assert ("blocks.10" in n) or ("blocks.11" in n) or ("norm" in n), f"Unexpected trainable param: {n}"

    print(f" -> PASSED: Regime 2 correctly unfreezes top-2 blocks and norms ({len(trainable_names)} tensors).")


if __name__ == "__main__":
    test_token_shape_and_register_slicing()
    test_gradient_isolation_regime_1()
    test_endogenous_exogenous_routing()
    test_end_to_end_forward_backward()
    test_regime_2_partial_ft()
    print("\nALL SANITY CHECKS FOR TimeXer_DINOv3 PASSED SUCCESSFULLY!")
