# 제출 위험분류 지도(fig_submission_map.png; 보고서 미참조, 참고용). 루트에서 실행: python scripts/figures/fig_submission_map.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("outputs/submissions/test_hanjeon.csv")          # pole_id, lon, lat, decision
sub = pd.read_csv("outputs/submissions/submission.csv")[["pole_id", "regime"]]
t = t.merge(sub, on="pole_id", how="left")
pos = t[t["decision"] == 1]
neg = t[t["decision"] == 0]
print(f"전체 {len(t)} | 고위험(1) {len(pos)} ({len(pos)/len(t)*100:.2f}%)")
print("레짐별 고위험:", pos["regime"].value_counts().to_dict())

REG_COL = {"yeongdong": "#1f6fb4", "corridor": "#d62728",
           "yeongseo": "#2ca02c", "mountain": "#9467bd"}
REG_EN = {"yeongdong": "Yeongdong", "corridor": "Corridor (foehn)",
          "yeongseo": "Yeongseo", "mountain": "Mountain"}

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

# (left) all poles: safe=light grey, high-risk=red
ax = axes[0]
ax.scatter(neg["lon"][::20], neg["lat"][::20], s=1.2, c="#dddddd", alpha=.5, label="safe (0)")
ax.scatter(pos["lon"], pos["lat"], s=3, c="#d62728", alpha=.7, label="high-risk (1)")
ax.set_title(f"Submitted high-risk poles ({len(pos):,}, 2%)", fontsize=12)
ax.legend(loc="upper right", markerscale=4, fontsize=9); ax.set_xlabel("lon"); ax.set_ylabel("lat")

# (right) high-risk colored by regime
ax = axes[1]
ax.scatter(neg["lon"][::40], neg["lat"][::40], s=0.8, c="#eeeeee", alpha=.4)
for r, c in REG_COL.items():
    m = pos["regime"] == r
    ax.scatter(pos["lon"][m], pos["lat"][m], s=3.5, c=c, alpha=.75,
               label=f"{REG_EN[r]} ({int(m.sum()):,})")
ax.set_title("High-risk poles by regime (regime-anchor allocation)", fontsize=12)
ax.legend(loc="upper right", markerscale=3, fontsize=8.5); ax.set_xlabel("lon")

fig.suptitle("test_hanjeon.csv - final risk classification map (Gangwon, 1,387,831 poles)",
             fontsize=13, y=.98)
out = "outputs/figures/fig_submission_map.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
