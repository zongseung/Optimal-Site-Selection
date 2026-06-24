# 보고서 그림 5 (fig_goseong_corridor.png). 리포지토리 루트에서 실행: python scripts/figures/fig_goseong_corridor.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CX, CY = 128.50, 38.21  # GOSEONG_2019_LONLAT

new = pd.read_csv("outputs/submissions/submission.csv")  # pole_id, lon, lat, risk_score, regime
new["pn"] = new["risk_score"].rank(method="average") / len(new) * 100  # global pctile 0-100
old = pd.read_parquet("outputs/pole_risk_pctile.parquet")[["pole_id", "risk_pctile"]]
df = new.merge(old, on="pole_id", how="left")  # risk_pctile = old 3-regime (0-100)

lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
dist_km = np.sqrt(((lat - CY) * 111.0) ** 2 + ((lon - CX) * 87.3) ** 2)
n5 = dist_km <= 5.0
mp_old = df["risk_pctile"].to_numpy()[n5].mean() / 100
mp_new = df["pn"].to_numpy()[n5].mean() / 100
print(f"고성 5km 전주={int(n5.sum())} | 3레짐={mp_old:.3f} | 4레짐 within={mp_new:.3f}")

box = (np.abs(lon - CX) < 0.13) & (np.abs(lat - CY) < 0.14)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
for ax, (col, tt, mp) in zip(axes, [("risk_pctile", "Before: 3-regime", mp_old),
                                    ("pn", "After: 4-regime within (corridor)", mp_new)]):
    pc = df[col].to_numpy()
    sc = ax.scatter(lon[box], lat[box], c=pc[box], cmap="inferno", s=11, vmin=0, vmax=100, alpha=.85)
    ax.add_patch(Circle((CX, CY), 5 / 87.3, fill=False, ec="cyan", lw=1.8, ls="--"))
    ax.scatter([CX], [CY], marker="*", s=560, c="cyan", ec="black", lw=1.4, zorder=5,
               label="2019 Goseong fire (powerline)")
    ax.set_title(f"{tt}\n5km mean pctile = {mp:.3f}", fontsize=11)
    ax.set_xlabel("lon"); ax.legend(loc="lower left", fontsize=8)
axes[0].set_ylabel("lat")
cb = fig.colorbar(sc, ax=axes, shrink=.82, pad=.02); cb.set_label("global risk percentile (0-100)")
fig.suptitle("Goseong 2019 powerline wildfire - risk before vs after corridor regime (5km sanity)",
             fontsize=12.5, y=.99)
out = "outputs/figures/fig_goseong_corridor.png"
fig.savefig(out, dpi=135, bbox_inches="tight")
print("saved", out)
