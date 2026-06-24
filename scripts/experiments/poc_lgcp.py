"""LGCP-lite PoC — 격자 베이지안 포아송 GLM (라플라스 근사).
한 모델에서 점추정(사후평균 강도) + 불확실성(사후 신뢰구간)이 동시에 나오는지 시연.
  log λ_g = x_g·β,  y_g ~ Poisson(E_g·λ_g),  β ~ N(0, τ²I)
  MAP β + Hessian → 사후공분산 → 격자별 λ 평균·90% CI.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

# ---- 1. 데이터: 전주(공변량·노출) + 발화(카운트) ----
pf = pd.read_parquet("used_dataset/poles/pole_features.parquet")          # lon,lat,elevation,slope,yanggan_days...
ov = pd.read_parquet("used_dataset/poles/pole_static_overlay.parquet")[["pole_id", "mu_flammability", "mu_dist_to_forest"]]
fw = pd.read_parquet("used_dataset/weather/pole_fwi_obs.parquet")[["pole_id", "fwi_q90_obs"]]
pole = pf.merge(ov, on="pole_id").merge(fw, on="pole_id")
fire = pd.read_csv("outputs/fire_cause_labeled.csv")[["lon", "lat"]].dropna()

# ---- 2. 0.1° 격자 비닝 ----
def cell(lon, lat): return (np.floor(lon / 0.1).astype(int), np.floor(lat / 0.1).astype(int))
pole["cx"], pole["cy"] = cell(pole["lon"].values, pole["lat"].values)
fcx, fcy = cell(fire["lon"].values, fire["lat"].values)
fire_cnt = pd.Series(list(zip(fcx, fcy))).value_counts().to_dict()

COV = ["elevation", "slope", "yanggan_days", "fwi_q90_obs", "mu_flammability", "mu_dist_to_forest"]
g = pole.groupby(["cx", "cy"]).agg(E=("pole_id", "size"), **{c: (c, "mean") for c in COV}).reset_index()
g["y"] = [fire_cnt.get((cx, cy), 0) for cx, cy in zip(g["cx"], g["cy"])]
g = g[g["E"] >= 20].reset_index(drop=True)   # 표본 적은 격자 제외
print(f"격자 {len(g)} | 총발화 {int(g['y'].sum())} | 발화>0 격자 {int((g['y']>0).sum())}")

# ---- 3. 설계행렬 (표준화 + 절편) ----
X = g[COV].values.astype(float)
X = (X - X.mean(0)) / X.std(0)
X = np.c_[np.ones(len(g)), X]                # 절편
y = g["y"].values.astype(float)
E = g["E"].values.astype(float)
tau2 = 4.0                                    # β 사전분산(약정보)

# ---- 4. MAP + 라플라스 (사후 = N(β_map, H⁻¹)) ----
def nlp(b):
    eta = X @ b; lam = E * np.exp(eta)
    return float(lam.sum() - (y * (np.log(E) + eta)).sum() + (b @ b) / (2 * tau2))
def grad(b):
    eta = X @ b; lam = E * np.exp(eta)
    return X.T @ (lam - y) + b / tau2
res = minimize(nlp, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
b = res.x
W = E * np.exp(X @ b)
H = (X * W[:, None]).T @ X + np.eye(X.shape[1]) / tau2   # 사후정밀도
cov = np.linalg.inv(H)                                    # 사후공분산

# ---- 5. 격자별 사후 강도: 점추정 + 분포 (한 모델!) ----
eta = X @ b
eta_sd = np.sqrt(np.einsum("gi,ij,gj->g", X, cov, X))     # var(η_g)=x H⁻¹ x
lam_mean = np.exp(eta)                                     # 점추정(사후평균 강도)
lam_lo = np.exp(eta - 1.645 * eta_sd)                      # 90% credible
lam_hi = np.exp(eta + 1.645 * eta_sd)

print("\n=== 회귀계수 (log-강도) — β와 사후 sd ===")
names = ["절편"] + COV
for nm, bi, si in zip(names, b, np.sqrt(np.diag(cov))):
    sig = "***" if abs(bi) > 2 * si else ("*" if abs(bi) > si else "")
    print(f"  {nm:18s} β={bi:+.3f} ± {si:.3f} {sig}")

print("\n=== 한 모델에서 점추정 + 불확실성 (상위 위험격자 5개) ===")
g["lam_mean"], g["lam_lo"], g["lam_hi"], g["lam_sd"] = lam_mean, lam_lo, lam_hi, eta_sd
top = g.sort_values("lam_mean", ascending=False).head(5)
print(f"{'격자(cx,cy)':14s} {'발화y':>5s} {'λ평균':>9s} {'90%CI':>20s} {'불확실(sd)':>9s}")
for _, r in top.iterrows():
    print(f"({int(r.cx)},{int(r.cy)})".ljust(14) +
          f" {int(r.y):5d} {r.lam_mean:9.4f}  [{r.lam_lo:.4f}, {r.lam_hi:.4f}]   {r.lam_sd:7.3f}")

# 점추정 랭킹이 실제 발화와 맞나(검증)
from scipy.stats import spearmanr
rho = spearmanr(g["lam_mean"], g["y"]).correlation
print(f"\n점추정 λ vs 실제 발화수 Spearman ρ = {rho:.3f} (랭킹 타당성)")
print(f"불확실성: 발화>0 격자 평균 sd={g[g.y>0]['lam_sd'].mean():.3f} vs 발화0 격자 sd={g[g.y==0]['lam_sd'].mean():.3f}")
print("→ 한 베이지안 식에서 점추정(λ평균)과 분포(90%CI)가 동시 산출됨 ✅")
