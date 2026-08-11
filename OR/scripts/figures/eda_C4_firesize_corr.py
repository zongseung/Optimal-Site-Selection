"""그림 4c — '확산축' 지표: 화재 규모 vs 당일 기상의 Spearman 상관.

FFMC·ISI만 채택하는 근거 중 '확산' 부분을 실측으로 보인다.
화재 규모(safemap_fire_weather.ar)와 당일 관측 기상의 Spearman ρ:
풍속·ISI는 유의(+), BUI·종합 FWI·FFMC는 규모와 무의미.
(FFMC는 규모가 아닌 '발화' 신호 — 별도 recall 근거. DMC·DC는 매칭 데이터에 없어 분석 제외.)

used_dataset/ READ-ONLY. 실행: .venv/bin/python scripts/figures/eda_C4_firesize_corr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pfire import config  # noqa: E402

FIG_DIR = ROOT / "outputs" / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
if Path(FONT_PATH).exists():
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# 표시 변수(모두 관측 기반, 동일 n) — 라벨, 컬럼
VARS = [
    ("풍속", "ws"),
    ("ISI", "isi_obs"),
    ("종합 FWI", "fwi_obs"),
    ("FFMC", "ffmc_obs"),
    ("BUI", "bui_obs"),
]
F_FIRE_WEATHER = config.FIRE / "safemap_fire_weather.parquet"


def main() -> None:
    df = pl.read_parquet(F_FIRE_WEATHER)
    rows = []
    for label, col in VARS:
        sub = df.select(["ar", col]).drop_nulls()
        x, y = sub["ar"].to_numpy(), sub[col].to_numpy()
        rho, p = stats.spearmanr(x, y)
        rows.append((label, float(rho), float(p), int(len(x))))
    rows.sort(key=lambda r: r[1])  # ρ 오름차순(아래→위 증가)
    n_common = rows[0][3]

    labels = [r[0] for r in rows]
    rhos = [r[1] for r in rows]
    ps = [r[2] for r in rows]
    sig = [pv < 0.05 for pv in ps]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ypos = np.arange(len(labels))
    colors = ["#d1495b" if s else "#b8bcc2" for s in sig]
    ax.barh(ypos, rhos, color=colors, edgecolor="#555", linewidth=0.6)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)

    for i, (rho, pv, s) in enumerate(zip(rhos, ps, sig)):
        star = " ★" if s else ""
        off = 0.006 if rho >= 0 else -0.006
        ha = "left" if rho >= 0 else "right"
        ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"
        ax.text(rho + off, i, f"ρ={rho:+.2f} ({ptxt}){star}",
                va="center", ha=ha, fontsize=9,
                color="#7a1020" if s else "#555")

    ax.set_xlabel("Spearman ρ  (화재 규모 ar 와의 상관)")
    ax.set_xlim(-0.30, 0.52)
    ax.set_title(f"화재 규모 vs 당일 기상 — 확산축 상관 (n={n_common})", fontsize=12)
    # 범례 대용 주석
    ax.text(0.98, 0.05,
            "★ p<0.05 유의(빨강) · 회색=무의미\n"
            "DMC·DC: 발화점–기상 매칭 데이터에 없어 분석 제외\n"
            "FFMC는 규모가 아닌 '발화' 신호(별도 recall 근거)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9))
    ax.grid(axis="x", ls=":", alpha=0.4)

    out = FIG_DIR / "eda_C4_firesize_corr.png"
    fig.savefig(out)
    print(f"saved: {out}")
    for label, rho, p, n in rows:
        print(f"  {label:<8} rho={rho:+.3f} p={p:.3f} n={n}")


if __name__ == "__main__":
    main()
