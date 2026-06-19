"""위험 백분위 지수(0~100) — 표시용 재척도.

근거: R=I·S·W 는 1보다 작은 세 성분의 곱이라 절대값이 작다(mean≈0.06, max≈0.33).
0/1 decision 은 순위로만 결정되므로 영향이 없지만, 제출/그림에 raw 값(예: 0.011)을
그대로 쓰면 "위험=0.011"로 보여 정성평가에서 약해 보인다. 순위 백분위(0=최저,
100=최고)로 재표시하면 의미 보존 + 가독성↑. decision·순위 불변.

산출:
  - outputs/pole_risk_pctile.parquet  (pole_id, risk, risk_pctile)  ← 어떤 제출에도 조인
  - outputs/submissions/submission_pctile.csv  (기존 decision + risk_pctile)
  - outputs/figures_pctile/risk_raw_vs_pctile_map.png  (원시 R vs 백분위 비교 지도)

실행:
    .venv/bin/python scripts/risk_percentile.py
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

from pfire import config  # noqa: E402
from pfire.risk_index import risk_percentile, risk_percentile_by_group  # noqa: E402
from sweep_threshold_figs import (  # noqa: E402
    compute_production_risk, pick_production_R,
    setup_korean_font, load_admin_polygons, draw_admin,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pctile")

FIGDIR = config.OUT / "figures_pctile"


def fig_raw_vs_pctile(pole_xy, R, pct, fire_xy, admin):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2))
    for ax, vals, ttl, cb_lab, vmax in (
        (axes[0], R, "(a) 원시 위험 R = I·S·W×배율\n(mean=%.3f, max=%.3f → 색이 눌림)"
         % (R.mean(), R.max()), "위험 R (raw)", float(R.max())),
        (axes[1], pct, "(b) 위험 백분위 지수 0~100\n(순위 동일·decision 불변, 색 full-range)",
         "위험 백분위", 100.0),
    ):
        sc = ax.scatter(pole_xy[:, 0], pole_xy[:, 1], c=vals, s=0.4,
                        cmap="inferno", vmin=0, vmax=vmax, rasterized=True,
                        linewidths=0)
        draw_admin(ax, admin)
        if fire_xy is not None and len(fire_xy):
            ax.scatter(fire_xy[:, 0], fire_xy[:, 1], marker="*", s=14,
                       c="#39ff14", edgecolors="black", linewidths=0.2, zorder=4)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cb_lab, fontsize=10)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xlabel("경도"); ax.set_ylabel("위도")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("위험 표시: 원시 R vs 백분위 — 같은 순위, 다른 가독성",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = FIGDIR / "risk_raw_vs_pctile_map.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    logger.info("지도 저장: %s", p)


def main() -> int:
    setup_korean_font()
    admin = load_admin_polygons()

    master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy = \
        compute_production_risk()
    R, rname, agree = pick_production_R(R_phys, R_mult)
    logger.info("채택 R = %s (재현 일치 %.4f)", rname, agree)

    pct = risk_percentile(R)                                  # 글로벌 백분위
    pct_reg = risk_percentile_by_group(R, regime_lbl)         # 체제 내 백분위
    pole_id = master["pole_id"].to_numpy()
    logger.info("글로벌 백분위: min=%.2f p50=%.2f max=%.2f (R 고유값=%d/%d)",
                pct.min(), np.median(pct), pct.max(),
                int(np.unique(R).size), len(R))

    # 1) 조인용 parquet (글로벌 + 체제내 백분위)
    out_pq = config.OUT / "pole_risk_pctile.parquet"
    pl.DataFrame({"pole_id": pole_id, "risk": R, "regime": regime_lbl,
                  "risk_pctile": np.round(pct, 2),
                  "risk_pctile_regime": np.round(pct_reg, 2)}) \
        .write_parquet(out_pq)
    logger.info("parquet 저장: %s", out_pq)

    # 2) 기존 제출에 백분위 컬럼 추가 (decision 그대로)
    ref = config.SUBMISSIONS / "submission.csv"
    if ref.exists():
        sub = pl.read_csv(ref)
        pdf = pl.DataFrame({"pole_id": pole_id,
                            "risk_pctile": np.round(pct, 2),
                            "risk_pctile_regime": np.round(pct_reg, 2)})
        sub2 = sub.join(pdf, on="pole_id", how="left")
        out_csv = config.SUBMISSIONS / "submission_pctile.csv"
        sub2.write_csv(out_csv)
        logger.info("제출(백분위 컬럼 추가) 저장: %s | decision=%d (%.2f%%)",
                    out_csv, int(sub2["decision"].sum()),
                    100.0 * sub2["decision"].mean())
        # 정렬 진단: 위험(1) 전주의 글로벌 vs 체제내 백분위 분포.
        d1 = np.asarray(sub2["decision"].to_numpy()) == 1
        gp = sub2["risk_pctile"].to_numpy()[d1]
        rp = sub2["risk_pctile_regime"].to_numpy()[d1]
        logger.info("위험(1) 백분위 — 글로벌: min=%.1f p50=%.1f | 체제내: min=%.1f p50=%.1f",
                    float(np.min(gp)), float(np.median(gp)),
                    float(np.min(rp)), float(np.median(rp)))
        logger.info("  → 체제내 min 이 글로벌 min 보다 높을수록 배분 decision 과 정렬됨")
    else:
        logger.warning("기준 submission.csv 없음 → 제출 CSV 증강 생략")

    # 3) 비교 지도 (강원 박스 발화점만)
    m = ((fire_xy[:, 1] >= 36.9) & (fire_xy[:, 1] <= 38.7) &
         (fire_xy[:, 0] >= 126.9) & (fire_xy[:, 0] <= 129.6))
    fig_raw_vs_pctile(pole_xy, R, pct, fire_xy[m], admin)

    # 예시 해석 출력: 상위 컷이 백분위로 어디인가
    for q in (0.02, 0.05, 0.10, 0.15):
        thr_pct = 100.0 * (1.0 - q)
        logger.info("  상위 %.0f%% 위험 = 백분위 ≥ %.1f", q * 100, thr_pct)
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
