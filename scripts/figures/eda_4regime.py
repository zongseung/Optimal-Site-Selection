"""EDA 04·05 그림을 4레짐(영동·회랑·영서·산악)으로 재생성. 보고서 그림 1·2.
리포지토리 루트에서 실행: python scripts/figures/eda_4regime.py"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/dlwhdtmd/OR-project")
from scripts.make_figures import setup_korean_font
from pfire import config, regimes, io, experts, weather
setup_korean_font()

master = io.load_master()
gate, order = regimes.compute_gate(master)
feats = experts.build_ignition_features(master)
S = np.clip(master["S_p"].to_numpy().astype(float), 1e-3, 1)
W = np.clip(weather.season_weather(master, source="features"), 1e-3, 1)
per = []
for r in order:
    w = config.EXPERT_WEIGHTS[r]; ws = sum(w.values()); sc = np.zeros(len(S))
    for k, wt in w.items(): sc += wt * feats[k]
    per.append(sc / ws)
I = np.clip(np.sum(gate * np.stack(per, 1), 1), 1e-3, 1)
R = I * S * W
reg = np.array([order[i] for i in gate.argmax(1)])

KO = {"yeongdong": "영동", "corridor": "회랑", "yeongseo": "영서", "mountain": "산악"}
ORD = ["yeongdong", "corridor", "yeongseo", "mountain"]
labs = [KO[r] for r in ORD]
m = master.to_pandas() if hasattr(master, "to_pandas") else master
get = lambda c: m[c].to_numpy().astype(float)

# ---- 04. 체제별 핵심 파생변수 박스플롯 ----
VARS = [("yanggan_days", "양간일수"), ("fwi_q90", "FWI_q90"), ("mu_flammability", "연료가연성"),
        ("mu_dist_to_forest", "산림거리"), (None, "발화성향I"), (None, "위험R")]
data = {"발화성향I": I, "위험R": R}
for col, nm in VARS:
    if col and col in m.columns: data[nm] = get(col)
    elif col and col not in m.columns: data[nm] = None
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle("체제별 핵심 파생변수 분포 (4레짐; MoE 근거; outlier 숨김)", fontsize=13, weight="bold")
for ax, (col, nm) in zip(axes.ravel(), VARS):
    v = data.get(nm)
    if v is None:
        ax.set_visible(False); continue
    ax.boxplot([v[reg == r] for r in ORD], tick_labels=labs, showfliers=False)
    ax.set_title(nm, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("outputs/figures/eda/04_regime_boxplots.png", dpi=130, bbox_inches="tight")
print("saved 04_regime_boxplots (4레짐)")

# ---- 05. R=I·S·W 분산기여 (전역 + 체제별) ----
lI, lS, lW, lR = np.log(I), np.log(S), np.log(W), np.log(R)
def contrib(idx):
    lr = lR[idx]; vr = lr.var()
    return [np.cov(x[idx], lr)[0, 1] / vr for x in (lI, lS, lW)]
glob = contrib(np.ones(len(R), bool))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].bar(["I", "S", "W"], glob, color=["#4c72b0", "#55a868", "#c44e52"])
for i, v in enumerate(glob): ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
ax[0].set_title("R 을 지배하는 항 (전역)"); ax[0].set_ylabel("log R 분산기여 (합=1)")
x = np.arange(len(ORD)); wd = 0.25
for k, (nm, c) in enumerate(zip(["I", "S", "W"], ["#4c72b0", "#55a868", "#c44e52"])):
    vals = [contrib(reg == r)[k] for r in ORD]
    ax[1].bar(x + (k - 1) * wd, vals, wd, label=nm, color=c)
ax[1].set_xticks(x); ax[1].set_xticklabels(labs); ax[1].legend()
ax[1].set_title("체제별 I·S·W 기여 (4레짐)"); ax[1].set_ylabel("log R 분산기여")
fig.tight_layout()
fig.savefig("outputs/figures/eda/05_isw_contribution.png", dpi=130, bbox_inches="tight")
print("saved 05_isw_contribution (4레짐)")
print("전역 기여 I/S/W:", [round(v, 3) for v in glob])
for r in ORD: print(f"  {KO[r]}:", [round(v, 3) for v in contrib(reg == r)])
