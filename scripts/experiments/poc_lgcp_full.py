"""B-full PoC — 격자 베이지안 포아송 + I·S·W 공변량 + ICAR 공간항 (라플라스).
  log λ_g = β0 + β_I·logI_g + β_S·logS_g + β_W·logW_g + φ_g,  φ ~ ICAR(τ)
  y_g ~ Poisson(E_g·λ_g).  MAP[β,φ] + Hessian → 사후(점추정+분포 일관).
물리식 R=I·S·W가 맞으면 β_I≈β_S≈β_W≈1 로 나와야 함(베이지안 검증).
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from pfire import config, regimes, io, experts, weather

# ---- 1. I·S·W per pole (물리 MoE) ----
master = io.load_master(); gate, order = regimes.compute_gate(master)
feats = experts.build_ignition_features(master)
S = np.clip(master["S_p"].to_numpy().astype(float), 1e-3, 1)
W = np.clip(weather.season_weather(master, source="features"), 1e-3, 1)
per = []
for r in order:
    w = config.EXPERT_WEIGHTS[r]; ws = sum(w.values()); sc = np.zeros(len(S))
    for k, wt in w.items(): sc += wt * feats[k]
    per.append(sc / ws)
I = np.clip(np.sum(gate * np.stack(per, 1), 1), 1e-3, 1)
lon = master["lon"].to_numpy(); lat = master["lat"].to_numpy()
pole = pd.DataFrame({"lon": lon, "lat": lat, "logI": np.log(I), "logS": np.log(S), "logW": np.log(W)})

# ---- 2. 격자 비닝 + 발화 ----
fire = pd.read_csv("outputs/fire_cause_labeled.csv")[["lon", "lat"]].dropna()
cx = np.floor(pole.lon / 0.1).astype(int); cy = np.floor(pole.lat / 0.1).astype(int)
pole["cx"], pole["cy"] = cx, cy
fcx = np.floor(fire.lon / 0.1).astype(int); fcy = np.floor(fire.lat / 0.1).astype(int)
fcnt = pd.Series(list(zip(fcx, fcy))).value_counts().to_dict()
g = pole.groupby(["cx", "cy"]).agg(E=("lon", "size"), logI=("logI", "mean"),
                                   logS=("logS", "mean"), logW=("logW", "mean")).reset_index()
g["y"] = [fcnt.get((a, b), 0) for a, b in zip(g.cx, g.cy)]
g = g[g.E >= 20].reset_index(drop=True)
ng = len(g); print(f"격자 {ng} | 총발화 {int(g.y.sum())} | 발화>0 {int((g.y>0).sum())}")

# ---- 3. ICAR 인접 (격자 rook 인접) ----
idx = {(a, b): i for i, (a, b) in enumerate(zip(g.cx, g.cy))}
Wadj = np.zeros((ng, ng))
for (a, b), i in idx.items():
    for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        j = idx.get((a + da, b + db))
        if j is not None: Wadj[i, j] = 1
D = np.diag(Wadj.sum(1)); Q = D - Wadj            # ICAR 정밀도 구조
TAU = 1.0; EPS = 1e-3                              # 공간정밀도(고정) + ridge(proper화)

# ---- 4. 설계행렬 + 라플라스 ----
X = np.c_[np.ones(ng), g.logI, g.logS, g.logW]    # 절편 + logI/S/W
y = g.y.values.astype(float); E = g.E.values.astype(float)
p = X.shape[1]; SB2 = 100.0                        # β 약정보 사전

def unpack(t): return t[:p], t[p:]                 # β, φ
def nlp(t):
    b, phi = unpack(t); eta = X @ b + phi; lam = E * np.exp(eta)
    ll = lam.sum() - (y * eta).sum()
    pr = (b @ b) / (2 * SB2) + 0.5 * TAU * (phi @ Q @ phi) + 0.5 * EPS * (phi @ phi)
    return float(ll + pr)
def grad(t):
    b, phi = unpack(t); eta = X @ b + phi; lam = E * np.exp(eta)
    gb = X.T @ (lam - y) + b / SB2
    gp = (lam - y) + TAU * (Q @ phi) + EPS * phi
    return np.concatenate([gb, gp])
t0 = np.zeros(p + ng)
res = minimize(nlp, t0, jac=grad, method="L-BFGS-B", options={"maxiter": 500})
b, phi = unpack(res.x)
eta = X @ b + phi; lam = E * np.exp(eta); lamv = E * np.exp(eta)  # = λ·E weight

# ---- 5. Hessian → 사후공분산 (점추정+분포) ----
Hbb = X.T @ (X * lamv[:, None]) + np.eye(p) / SB2
Hbp = X.T * lamv                                   # (p,ng)
Hpp = np.diag(lamv) + TAU * Q + EPS * np.eye(ng)
H = np.block([[Hbb, Hbp], [Hbp.T, Hpp]])
cov = np.linalg.inv(H)
# 격자 η 사후 분산: J=[X, I] → var(η_g)=J cov Jᵀ
J = np.c_[X, np.eye(ng)]
eta_var = np.einsum("gi,ij,gj->g", J, cov, J)
eta_sd = np.sqrt(np.clip(eta_var, 0, None))
g["lam"] = np.exp(eta); g["lo"] = np.exp(eta - 1.645 * eta_sd); g["hi"] = np.exp(eta + 1.645 * eta_sd); g["sd"] = eta_sd

print("\n=== 회귀계수 (logλ) — 물리식이면 β_I,β_S,β_W ≈ 1 ===")
for nm, bi, si in zip(["절편", "β_logI", "β_logS", "β_logW"], b, np.sqrt(np.diag(cov)[:p])):
    print(f"  {nm:8s} = {bi:+.3f} ± {si:.3f}")
print(f"\nSpearman ρ(λ_점추정, 실제발화) = {spearmanr(g.lam, g.y).correlation:.3f}")
print(f"불확실성: 발화>0 sd={g[g.y>0].sd.mean():.3f} vs 발화0 sd={g[g.y==0].sd.mean():.3f}")
print("\n상위 위험격자 — 점추정 λ + 90%CI (한 모델):")
for _, r in g.sort_values("lam", ascending=False).head(5).iterrows():
    print(f"  ({int(r.cx)},{int(r.cy)}) y={int(r.y):2d}  λ={r.lam:.4f} [{r.lo:.4f},{r.hi:.4f}] sd={r.sd:.3f}")
print("\n→ I·S·W 공변량 + ICAR 공간항. 점추정·분포 일관 산출 ✅")
