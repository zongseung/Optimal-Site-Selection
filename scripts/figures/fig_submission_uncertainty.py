# 보고서 그림 10 (fig_submission_uncertainty.png). 리포지토리 루트에서 실행: python scripts/figures/fig_submission_uncertainty.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ax.set_title("High-risk poles colored by posterior uncertainty\n(credible-interval width)", fontsize=11)
ax.set_xlabel("lon"); ax.set_ylabel("lat")
cb = fig.colorbar(sc, ax=ax, shrink=.8); cb.set_label("credible width (risk_hi - risk_lo)")

# (right) decision x uncertainty quadrant: confident vs uncertain high-risk
ax = axes[1]
ax.scatter(neg["lon"][::40], neg["lat"][::40], s=0.8, c="#eeeeee", alpha=.4)
conf = pos[pos["unc"] < umed]; unce = pos[pos["unc"] >= umed]
ax.scatter(conf["lon"], conf["lat"], s=4, c="#1a9850", alpha=.75,
           label=f"confident high-risk ({len(conf):,}) -> act now")
ax.scatter(unce["lon"], unce["lat"], s=4, c="#d73027", alpha=.75,
           label=f"uncertain high-risk ({len(unce):,}) -> field-verify")
ax.set_title("Operational triage: confident vs uncertain high-risk", fontsize=11)
ax.set_xlabel("lon"); ax.legend(loc="upper right", markerscale=3, fontsize=8.5)

fig.suptitle("Final submission with uncertainty - high-risk poles + Bayesian posterior credible band",
             fontsize=12.5, y=.98)
out = "outputs/figures/fig_submission_uncertainty.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
