# 보고서 그림 10 (fig_submission_uncertainty.png). 리포지토리 루트에서 실행: python scripts/figures/fig_submission_uncertainty.py
import sys
sys.path.insert(0, "/home/dlwhdtmd/OR-project/OR")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.make_figures import setup_korean_font
setup_korean_font()

d = pd.read_csv("outputs/submissions/submission_p0.02.csv")  # 4-regime + risk_lo/hi
assert "corridor" in set(d["regime"].unique()), "submission_p0.02.csv가 아직 3레짐(stale)"
d["unc"] = (d["risk_hi"] - d["risk_lo"]).clip(lower=0)
pos = d[d["decision"] == 1].copy()
neg = d[d["decision"] == 0]
umed = pos["unc"].median()
print(f"고위험 {len(pos):,} | credible폭 mean={pos['unc'].mean():.4f} median={umed:.4f}")
print("레짐별 평균 불확실성:", pos.groupby("regime")["unc"].mean().round(4).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

# (left) high-risk colored by posterior uncertainty (credible width)
ax = axes[0]
ax.scatter(neg["lon"][::40], neg["lat"][::40], s=0.8, c="#eeeeee", alpha=.4)
sc = ax.scatter(pos["lon"], pos["lat"], s=4, c=pos["unc"], cmap="plasma",
                vmin=pos["unc"].quantile(.05), vmax=pos["unc"].quantile(.95), alpha=.8)
ax.set_title("고위험 전주 — 사후 신뢰구간 폭", fontsize=11)
ax.set_xlabel("경도"); ax.set_ylabel("위도")
cb = fig.colorbar(sc, ax=ax, shrink=.8); cb.set_label("신뢰구간 폭 (risk_hi - risk_lo)")

# (right) decision x uncertainty quadrant: confident vs uncertain high-risk
ax = axes[1]
ax.scatter(neg["lon"][::40], neg["lat"][::40], s=0.8, c="#eeeeee", alpha=.4)
conf = pos[pos["unc"] < umed]; unce = pos[pos["unc"] >= umed]
ax.scatter(conf["lon"], conf["lat"], s=4, c="#1a9850", alpha=.75,
           label=f"확신 고위험 ({len(conf):,}) → 즉시조치")
ax.scatter(unce["lon"], unce["lat"], s=4, c="#d73027", alpha=.75,
           label=f"불확실 고위험 ({len(unce):,}) → 현장확인")
ax.set_title("운영 분류: 확신 vs 불확실 고위험", fontsize=11)
ax.set_xlabel("경도"); ax.legend(loc="upper right", markerscale=3, fontsize=8.5)

fig.suptitle("제출 위험 + 사후 불확실성",
             fontsize=12.5, y=.98)
out = "outputs/figures/fig_submission_uncertainty.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
