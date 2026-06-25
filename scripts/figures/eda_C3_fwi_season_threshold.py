"""그림 4의 짝 — 기상 '시간 집계' 근거.

산불조심기간(2~5월) 일별 ISI/FFMC 분포에 평균·q90·고위험 임계선을 얹어,
(1) 왜 2~5월로 시즌 집계하는지(비시즌 대비 분포가 우측 이동),
(2) 왜 평균·90분위수·고위험일수 비율을 통계량으로 쓰는지,
(3) 임계 ISI 10.0·FFMC 90.0이 관측소 일별 q90에 정합함을 한 장으로 보인다.

used_dataset/·pfire/ READ-ONLY(import만). 실행: .venv/bin/python scripts/figures/eda_C3_fwi_season_threshold.py
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

SEASON = set(config.SEASON_MONTHS)  # (2,3,4,5)
PANELS = [
    # col, label, threshold, color, x표시상한, 텍스트박스 위치
    ("isi", "ISI (초기확산)", config.DAILY_ISI_HIGH_THRESHOLD, "#d1495b", 30.0, "right"),
    ("ffmc", "FFMC (미세연료 발화준비)", config.DAILY_FFMC_HIGH_THRESHOLD, "#e07a1f", None, "left"),
]


def main() -> None:
    df = pl.read_parquet(config.F_FWI_STATION_DAILY)
    season = df.filter(pl.col("month").is_in(list(SEASON)))
    off = df.filter(~pl.col("month").is_in(list(SEASON)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    for ax, (col, label, thr, color, xcap, boxloc) in zip(axes, PANELS):
        s = season[col].drop_nulls().to_numpy()
        o = off[col].drop_nulls().to_numpy()
        # 통계량은 원자료(클립 전)에서 산출 — 그림 표시만 클립
        mean_v = float(np.mean(s))
        q90_v = float(np.quantile(s, 0.90))
        hi_frac = float(np.mean(s >= thr)) * 100.0

        lo = float(min(s.min(), o.min()))
        hi = float(max(s.max(), o.max()))
        xhi = hi if xcap is None else xcap
        bins = np.linspace(lo, xhi, 55)
        sc = np.clip(s, lo, xhi)  # 꼬리 outlier는 마지막 bin에 모음
        oc = np.clip(o, lo, xhi)

        ax.hist(oc, bins=bins, density=True, histtype="step", color="#9aa0a6",
                linewidth=1.3, label="비시즌(6~1월)")
        ax.hist(sc, bins=bins, density=True, color=color, alpha=0.55,
                label="산불조심기간(2~5월)")
        ax.axvspan(thr, xhi, color=color, alpha=0.10)  # 고위험 꼬리

        ax.axvline(mean_v, color="#1f6feb", ls="--", lw=1.4)
        ax.axvline(q90_v, color="black", ls=":", lw=1.6)
        ax.axvline(thr, color=color, ls="-", lw=1.8)
        ax.set_xlim(lo, xhi)

        # 통계 요약 텍스트박스를 빈 코너에 배치(주석 겹침 방지)
        txt = (f"평균 {mean_v:.1f}\n"
               f"q90 {q90_v:.1f}\n"
               f"고위험 임계 {thr:.0f} (≈q90)\n"
               f"→ 고위험일 비율 {hi_frac:.1f}%")
        bx = 0.97 if boxloc == "right" else 0.03
        ha = "right" if boxloc == "right" else "left"
        ax.text(bx, 0.70, txt, transform=ax.transAxes, ha=ha, va="top",
                fontsize=9.0, color="#222",
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9))

        ax.set_title(label, fontsize=11)
        ax.set_xlabel(col.upper() + (f"  ({xcap:.0f} 초과는 우측 끝에 합산)" if xcap else ""))
        ax.set_ylabel("밀도")
        ax.legend(loc="upper right" if boxloc == "left" else "upper center",
                  fontsize=8.5)

    fig.suptitle(
        "산불조심기간(2~5월) 일별 ISI·FFMC 분포",
        fontsize=12.5, y=1.02,
    )
    out = FIG_DIR / "eda_C3_fwi_season_threshold.png"
    fig.savefig(out)
    print(f"saved: {out}")
    # 보고서 캡션용 수치 재출력
    for col, _label, thr, *_ in PANELS:
        s = season[col].drop_nulls().to_numpy()
        print(f"  {col}: mean={np.mean(s):.2f} q90={np.quantile(s,0.9):.2f} "
              f"hi_frac(>= {thr})={np.mean(s>=thr)*100:.1f}%")


if __name__ == "__main__":
    main()
