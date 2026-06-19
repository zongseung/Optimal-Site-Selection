"""exposure_v2 v2.0 프로토타입 — OR 확률 vs 발화가중 기대도즈 (포화 해제 확인).

같은 입력(영동 부분집합·동일 발화원·바람)으로 두 결합을 *나란히* 계산:
  OR_P(p)  = 1 − ∏_g (1 − rbar_gp)            # 현 방식(기대 OR) → 포화 예상
  dose(p)  = Σ_g  I_g · rbar_gp               # v2.0 (발화가중 누적) → 변별 기대
  rbar_gp  = (1/S)Σ_s exp(−d/L_s)·fuel_p·(1+β·south_p),  L_s=L0(1+α·max(0,align_s)·ws_s/5)

핵심 질문: **영동 *내* 에서 dose 가 OR_P보다 변별(std/IQR)이 큰가?** (포화 풀리나)

실행:  .venv/bin/python exposure_v2/proto_v0_dose.py
산출:  exposure_v2/fig_v0_or_vs_dose.png  + 콘솔 통계
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import config, experts, exposure_engine, geo, io, regimes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v0dose")

N_TARGET = 25000     # 영동 타깃 전주 부분집합
N_SOURCE = 600       # 발화원(영동 I 상위 부분집합)
N_SIMS = 48          # 바람 표집
MAXD = config.SPREAD_MAX_DIST_KM
L0 = config.SPREAD_LENGTH_SCALE_KM
ALPHA = config.SPREAD_WIND_ANISO
BETA = config.SPREAD_SOUTHNESS_BETA


def main() -> int:
    rng = np.random.default_rng(config.SEED)
    logger.info("로드 + I 계산 …")
    master = io.load_master()
    gate, order = regimes.compute_gate(master)
    regime_lbl = np.array(order)[gate.argmax(1)]
    I, _ = experts.ignition_propensity(master, gate, order)

    yd = regime_lbl == "yeongdong"
    idx_yd = np.nonzero(yd)[0]
    logger.info("영동 전주=%d", idx_yd.size)

    # 타깃 전주: 영동에서 무작위 부분집합(타깃 표집은 각 타깃의 발화원 밀도에 영향 없음)
    tgt = rng.choice(idx_yd, size=min(N_TARGET, idx_yd.size), replace=False)
    tgt.sort()
    # 발화원: **실제 운영 발화후보(전역 I 상위, ~8000)** — 포화 regime 재현 핵심.
    #   (600개로 줄이면 밀도가 낮아 포화가 안 생겨 테스트 무효였음.)
    src_pool = np.asarray(exposure_engine.ignition_candidates(I, gate, order), dtype=np.int64)
    Ig = np.clip(I[src_pool], 1e-6, None)
    logger.info("발화원(실제 후보)=%d", src_pool.size)

    lonlat = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    km = geo.lonlat_to_km(lonlat)
    tx, ty = km[tgt, 0], km[tgt, 1]
    sx, sy = km[src_pool, 0], km[src_pool, 1]
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)[tgt]
    south = (master["mu_southness"].to_numpy().astype(np.float64) * 2 - 1)[tgt]
    south_fac = 1.0 + BETA * south

    # 영동 양간 서풍 prior
    wd = (rng.normal(270.0, 20.0, N_SIMS)) % 360.0
    ws = rng.uniform(4.0, 14.0, N_SIMS)
    theta = np.deg2rad(wd + 180.0)            # 풍하 방향(rad)

    n = tgt.size
    prod_term = np.ones(n)     # ∏(1−rbar) → OR_P=1−prod
    dose = np.zeros(n)         # Σ I_g·rbar
    logger.info("커널 누적: 발화원 %d × 타깃 %d × 바람 %d …", src_pool.size, n, N_SIMS)
    for gi in range(src_pool.size):
        dx = tx - sx[gi]
        dy = ty - sy[gi]
        d = np.sqrt(dx * dx + dy * dy)
        near = d <= MAXD
        if not near.any():
            continue
        dn = d[near]
        phi = np.arctan2(dy[near], dx[near])
        # 바람 표집 평균 reach (Bernoulli 없이 기대값)
        align = np.cos(phi[:, None] - theta[None, :])         # (K, S)
        L = L0 * (1.0 + ALPHA * np.maximum(0.0, align) * (ws[None, :] / 5.0))
        reach_s = np.exp(-dn[:, None] / np.maximum(L, 1e-6))  # (K, S)
        rbar = reach_s.mean(axis=1) * fuel[near] * south_fac[near]
        rbar = np.clip(rbar, 0.0, 1.0)
        prod_term[near] *= (1.0 - rbar)
        dose[near] += Ig[gi] * rbar

    OR_P = 1.0 - prod_term

    def stats(name, x):
        q = np.quantile(x, [0.5, 0.9, 0.99])
        logger.info("  %-10s mean=%.4f std=%.4f IQR=%.4f p50=%.3f p90=%.3f p99=%.3f frac>0.95=%.3f",
                    name, x.mean(), x.std(),
                    np.quantile(x, .75) - np.quantile(x, .25),
                    q[0], q[1], q[2], float(np.mean(x > 0.95)))

    logger.info("=== 영동 내 분포 (포화 해제 확인) ===")
    stats("OR_P", OR_P)
    dose01 = (dose - dose.min()) / (dose.max() - dose.min() + 1e-12)
    stats("dose(norm)", dose01)
    # 변별력 지표: 고유값 비율, 정규화 std 대비
    logger.info("  변별: OR_P std=%.4f vs dose std=%.4f → dose/OR std비=%.1f배",
                OR_P.std(), dose01.std(), dose01.std() / max(OR_P.std(), 1e-9))

    # 그림: 분포 비교
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(OR_P, bins=60, color="#c1121f", alpha=0.85)
    ax[0].set_title(f"OR_P (current) — saturated?\nstd={OR_P.std():.3f}, frac>0.95={np.mean(OR_P>0.95):.2f}",
                    fontsize=11)
    ax[0].set_xlabel("OR P(exposure)"); ax[0].set_ylabel("# poles (Yeongdong)")
    ax[1].hist(dose01, bins=60, color="#1f77b4", alpha=0.85)
    ax[1].set_title(f"dose v2.0 (ignition-weighted, norm)\nstd={dose01.std():.3f}",
                    fontsize=11)
    ax[1].set_xlabel("normalized dose")
    fig.suptitle("Yeongdong exposure: OR (saturating) vs expected dose v2.0",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_v0_or_vs_dose.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("그림 저장: %s", out)
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
