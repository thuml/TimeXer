"""Ridge probe: how much forecasting info survives TS2Image -> DINOv3 features?
Endogenous-only (OT) on EPF NP. Compares ridge on raw lookback vs ridge on
frozen DINOv3 features (pretrained / random init). Usage: python docs/brainstorm/probes/dinov3_feature_probe.py
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from sklearn.linear_model import RidgeCV

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.insert(0, root)
from layers.TS2Image import EndogenousTS2Image

L, H = 168, 24
y = pd.read_csv(f"{root}/dataset/EPF/NP.csv")["OT"].values.astype(np.float32)
n = len(y); n_tr = int(n * 0.7); n_te = int(n * 0.2)
y = (y - y[:n_tr].mean()) / y[:n_tr].std()

def windows(lo, hi, stride):
    idx = np.arange(lo, hi - L - H + 1, stride)
    X = np.stack([y[i:i + L] for i in idx]); Y = np.stack([y[i + L:i + L + H] for i in idx])
    mu = X.mean(1, keepdims=True); sd = np.sqrt(X.var(1, keepdims=True) + 1e-5)
    return (X - mu) / sd, (Y - mu) / sd   # RevIN space, same as the model

Xtr, Ytr = windows(0, n_tr, 8)
Xte, Yte = windows(n - n_te - L, n, 24)
print(f"train windows {len(Xtr)}, test windows {len(Xte)}")

dev = "cuda"
ts2img = EndogenousTS2Image(seq_len=L, img_size=176, period_len=24).to(dev)

@torch.no_grad()
def feats(model, X, pooled):
    out = []
    for i in range(0, len(X), 256):
        x = torch.from_numpy(X[i:i + 256]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            t = model.forward_features(ts2img(x)).float()
        cls, patch = t[:, 0], t[:, 5:]
        if pooled:   # exactly what the model does: raster-order 1D pool 121 -> 7
            p = F.adaptive_avg_pool1d(patch.transpose(1, 2), L // 24).transpose(1, 2)
        else:        # keep the 11x11 grid
            p = patch
        out.append(torch.cat([cls[:, None], p], 1).flatten(1).cpu().numpy())
    return np.concatenate(out)

def ridge(name, Ftr, Fte):
    t = time.time()
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ftr, Ytr)
    mse = ((m.predict(Fte) - Yte) ** 2).mean()
    # can the features even reconstruct the last 24 input steps?
    r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ftr, Xtr[:, -24:])
    rec = 1 - ((r.predict(Fte) - Xte[:, -24:]) ** 2).mean() / Xte[:, -24:].var()
    print(f"{name:42s} dim={Ftr.shape[1]:6d}  forecast MSE={mse:.4f}  recon R2(last24)={rec:.3f}  ({time.time()-t:.0f}s)")

print("RevIN-space MSE (lower is better). Baseline 'repeat last day' and ridge on raw window:")
print(f"{'seasonal naive (repeat last 24h)':42s} forecast MSE={((Xte[:, -24:] - Yte) ** 2).mean():.4f}")
ridge("ridge on raw lookback (168)", Xtr, Xte)

for pre in (True, False):
    torch.manual_seed(0)
    m = timm.create_model("vit_small_patch16_dinov3", pretrained=pre, img_size=176, num_classes=0).to(dev).eval()
    tag = "pretrained" if pre else "random-init"
    for pooled in (True, False):
        ridge(f"DINOv3 {tag}, {'pooled 7+CLS (as model)' if pooled else 'full 121+CLS'}",
              feats(m, Xtr, pooled), feats(m, Xte, pooled))
    del m; torch.cuda.empty_cache()
