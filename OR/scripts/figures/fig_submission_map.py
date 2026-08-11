# 제출 위험분류 지도(fig_submission_map.png; 보고서 미참조, 참고용). 루트에서 실행: python scripts/figures/fig_submission_map.py
import sys
sys.path.insert(0, "/home/dlwhdtmd/OR-project/OR")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.make_figures import setup_korean_font
setup_korean_font()

t = pd.read_csv("outputs/submissions/submission.csv")[
    ["pole_id", "lon", "lat", "decision", "regime"]]   # 최신 제출본(regime 포함)
pos = t[t["decision"] == 1]
neg = t[t["decision"] == 0]
print(f"전체 {len(t)} | 고위험(1) {len(pos)} ({len(pos)/len(t)*100:.2f}%)")
print("레짐별 고위험:", pos["regime"].value_counts().to_dict())

REG_COL = {"yeongdong": "#1f6fb4", "corridor": "#d62728",
           "yeongseo": "#2ca02c", "mountain": "#9467bd"}
REG_EN = {"yeongdong": "영동", "corridor": "양간지풍 회랑",
          "yeongseo": "영서", "mountain": "산간"}

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

# (left) all poles: safe=light grey, high-risk=red
ax = axes[0]
ax.scatter(neg["lon"][::20], neg["lat"][::20], s=1.2, c="#dddddd", alpha=.5, label="안전 (0)")
ax.scatter(pos["lon"], pos["lat"], s=3, c="#d62728", alpha=.7, label="고위험 (1)")
ax.set_title(f"제출 고위험 전주 ({len(pos):,}, 2%)", fontsize=12)
ax.legend(loc="upper right", markerscale=4, fontsize=9); ax.set_xlabel("경도"); ax.set_ylabel("위도")

# (right) high-risk colored by regime
ax = axes[1]
ax.scatter(neg["lon"][::40], neg["lat"][::40], s=0.8, c="#eeeeee", alpha=.4)
for r, c in REG_COL.items():
    m = pos["regime"] == r
    ax.scatter(pos["lon"][m], pos["lat"][m], s=3.5, c=c, alpha=.75,
               label=f"{REG_EN[r]} ({int(m.sum()):,})")
ax.set_title("체제별 고위험 전주", fontsize=12)
ax.legend(loc="upper right", markerscale=3, fontsize=8.5); ax.set_xlabel("경도")

fig.suptitle("최종 제출 위험분류 지도 (강원 전주 1,387,831본)",
             fontsize=13, y=.98)
out = "outputs/figures/fig_submission_map.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
