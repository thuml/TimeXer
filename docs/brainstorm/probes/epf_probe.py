"""Probe: how much information do (a) future exogenous and (b) absolute covariate level carry on EPF?
Daily-origin samples (predict next 24h from past 168h), chronological 70/10/20 split like Dataset_Custom.
Variants differ only in what the model sees; target is always predicted in RevIN space and de-normalized,
except V4 (global scaling). Metric: MSE/MAE on globally standardized price (TimeXer convention)."""
import sys, numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
L, H, EPS = 168, 24, 1e-5
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../dataset/EPF/")

def build(market):
    df = pd.read_csv(root + market + ".csv")
    X = df.iloc[:, 1:].values.astype(float)            # load_fc, wind_fc, price(OT)
    n = len(X); ntr = int(n * 0.7); nte = int(n * 0.2); nva = n - ntr - nte
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0)
    Z = (X - mu) / sd                                   # global StandardScaler (as in repo)
    starts = np.arange(L, n - H + 1, 24)
    rows = []
    for s in starts:
        past, fut = Z[s - L:s], Z[s:s + H]
        m, v = past.mean(0), np.sqrt(past.var(0) + EPS)
        rows.append((s, past, fut, m, v))
    return rows, ntr + nva

def feats(rows, variant):
    F, T, M, S, G = [], [], [], [], []
    for s, past, fut, m, v in rows:
        pn = (past - m) / v                             # instance norm (RevIN / use_norm)
        fn = (fut[:, :2] - m[:2]) / v[:2]               # future exo normalized with past stats
        if variant == "V1_past_only":
            f = pn.ravel()
        elif variant == "V2_fut_revin":
            f = np.r_[pn.ravel(), fn.ravel()]
        elif variant == "V3_fut_revin+stats":
            f = np.r_[pn.ravel(), fn.ravel(), m, np.log(v)]
        elif variant == "V3a_exo_stats":
            f = np.r_[pn.ravel(), fn.ravel(), m[:2], np.log(v[:2])]
        elif variant == "V3b_tgt_stats":
            f = np.r_[pn.ravel(), fn.ravel(), m[2:], np.log(v[2:])]
        elif variant == "V4_global":
            f = np.r_[past.ravel(), fut[:, :2].ravel()]
        F.append(f); G.append(fut[:, 2]); M.append(m[2]); S.append(v[2])
        T.append(fut[:, 2] if variant == "V4_global" else (fut[:, 2] - m[2]) / v[2])
    return np.array(F), np.array(T), np.array(M), np.array(S), np.array(G)

VARS = ["V1_past_only", "V2_fut_revin", "V3_fut_revin+stats", "V4_global"]
def run(market, seeds=(0, 1, 2)):
    rows, cut = build(market)
    tr = [r for r in rows if r[0] + H <= cut]; te = [r for r in rows if r[0] >= cut]
    out = {}
    for var in VARS:
        Ftr, Ttr, _, _, _ = feats(tr, var); Fte, _, Mte, Ste, Gte = feats(te, var)
        fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-8
        Ftr, Fte = (Ftr - fm) / fs, (Fte - fm) / fs
        def denorm(P): return P if var == "V4_global" else P * Ste[:, None] + Mte[:, None]
        r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ftr, Ttr)
        pr = denorm(r.predict(Fte))
        mlp = []
        for sd in seeds:
            nn = MLPRegressor((256, 256), alpha=1e-3, early_stopping=True, max_iter=400,
                              random_state=sd, learning_rate_init=1e-3).fit(Ftr, Ttr)
            pm = denorm(nn.predict(Fte)); mlp.append(((pm - Gte) ** 2).mean())
        out[var] = (((pr - Gte) ** 2).mean(), np.abs(pr - Gte).mean(), np.mean(mlp), np.std(mlp))
    return out

if __name__ == "__main__":
    import os
    if os.environ.get("VARS"): VARS[:] = os.environ["VARS"].split(",")
    for mk in sys.argv[1:] or ["NP", "PJM", "BE", "FR", "DE"]:
        res = run(mk)
        for k, (rm, ra, mm, ms) in res.items():
            print(f"{mk:4s} {k:22s} ridge MSE {rm:.3f} MAE {ra:.3f} | MLP MSE {mm:.3f}±{ms:.3f}", flush=True)
