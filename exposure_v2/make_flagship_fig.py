"""exposure_v2 flagship 그림 — 옛 포화 OR vs v2.0 변별 dose (+ 2019 고성).

저장된 산출물만 사용(재계산 없음):
  outputs/submissions/submission.csv  → p_exposure(옛 OR), lon/lat/regime
  outputs/pole_dose_v2.parquet        → dose01(v2.0)

실행:  .venv/bin/python exposure_v2/make_flagship_fig.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pfire import config  # noqa: E402
from sweep_threshold_figs import setup_korean_font, load_admin_polygons, draw_admin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("flagship")


def main() -> int:
    setup_korean_font()
    admin = load_admin_polygons()

    sub = pl.read_csv(config.SUBMISSIONS / "submission.csv").sort("pole_id")
    dose = pl.read_parquet(config.OUT / "pole_dose_v2.parquet").sort("pole_id")
    df = sub.join(dose.select(["pole_id", "dose01"]), on="pole_id", how="left")

    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    pe = df["p_exposure"].to_numpy()
    d01 = df["dose01"].to_numpy()
    regime = df["regime"].to_numpy()
    yd = regime == "yeongdong"

    gx, gy = config.GOSEONG_2019_LONLAT

    def iqr(x):
        return float(np.subtract(*np.quantile(x, [0.75, 0.25])))

    fig, ax = plt.subplots(1, 2, figsize=(15, 8.4))
    panels = [
        (ax[0], pe, f"(a) 옛 노출 OR — 포화\n영동 IQR={iqr(pe[yd]):.3f} (변별 거의 없음)", "P(노출) OR"),
        (ax[1], d01, f"(b) v2.0 발화가중 dose — 변별\n영동 IQR={iqr(d01[yd]):.3f} (회랑이 보임)", "dose 백분위"),
    ]
    for a, val, ttl, lab in panels:
        sc = a.scatter(lon, lat, c=val, s=0.35, cmap="inferno", vmin=0, vmax=1,
                       rasterized=True, linewidths=0)
        draw_admin(a, admin)
        a.scatter([gx], [gy], marker="*", s=240, c="#39ff14", edgecolors="black",
                  linewidths=0.8, zorder=5, label="2019 고성 발화점")
        a.annotate("", xy=(gx + 0.22, gy), xytext=(gx, gy),
                   arrowprops=dict(arrowstyle="-|>", color="#39ff14", lw=2), zorder=5)
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
        cb.set_label(lab, fontsize=10)
        a.set_title(ttl, fontsize=12, fontweight="bold")
        a.set_xlabel("경도"); a.set_ylabel("위도")
        a.set_aspect("equal", adjustable="datalim")
        a.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("풍하 노출 재설계: 포화 OR → 변별 가능한 발화가중 dose (v2.0)\n"
                 "★ 2019 고성(KEPCO 특고압선 아크) · 화살표=양간지풍 풍하(동향)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_flagship_exposure.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("저장: %s", out)
    logger.info("영동 IQR: 옛 OR=%.4f → v2.0 dose=%.4f (%.0f배)",
                iqr(pe[yd]), iqr(d01[yd]), iqr(d01[yd]) / max(iqr(pe[yd]), 1e-9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
