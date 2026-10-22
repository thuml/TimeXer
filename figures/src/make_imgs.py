"""Render the real TS2Image channels of one EPF-NP window, plus series glyphs, as PNGs for the architecture slide.
Usage (from repo root): python figures/src/make_imgs.py <out_dir>"""
import os, sys
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, root)
from layers.TS2Image import EndogenousTS2Image

df = pd.read_csv(f"{root}/dataset/EPF/NP.csv")
i0 = 24 * 400
win = df.iloc[i0:i0 + 168]
y = win["OT"].values.astype(np.float32)
x = torch.from_numpy((y - y.mean()) / y.std())[None]
img = EndogenousTS2Image(seq_len=168, img_size=176, period_len=24)(x)[0]
# undo ImageNet normalisation to get the [0, 1] image the backbone actually sees
img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
img = img.clamp(0, 1).numpy()
for k, name in enumerate(["gasf", "gadf", "fold"]):
    plt.imsave(f"{out}/{name}.png", img[k], cmap="viridis", dpi=200)

# RGB input with the 11 x 11 ViT patch grid
fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
ax.imshow(img.transpose(1, 2, 0)); ax.set_xticks([]); ax.set_yticks([])
for t in range(0, 177, 16):
    ax.axhline(t - .5, color="white", lw=0.8); ax.axvline(t - .5, color="white", lw=0.8)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.savefig(f"{out}/patches.png", bbox_inches="tight", pad_inches=0, transparent=True); plt.close(fig)


def glyph(series, color, name, w=3.0, h=0.9):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.plot(series, color=color, lw=2.2); ax.axis("off")
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight", pad_inches=0.02, transparent=True); plt.close(fig)


glyph(y, "#5170ff", "endo")
glyph(win.iloc[:, 1].values, "#5ce1e6", "exo1")
glyph(win.iloc[:, 2].values, "#69f2c4", "exo2")
