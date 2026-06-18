"""Phase-1 MVP end-to-end 실행.

로드 → I(MoE) → S(S_p) → W(시즌통계) → R(p) → 발화점 임계값 →
공간 블록 CV recall@top-k + sanity → submission.csv 저장.

실행:
    .venv/bin/python scripts/run_phase1_mvp.py [--prevalence 0.02] [--daily-weather]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 패키지 루트 임포트 경로 보장.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
    exposure,
    exposure_engine,
    experts,
    hazard,
    io,
    regimes,
    submit,
    validate,
    weather,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase1_mvp")


def main() -> int:
    parser = argparse.ArgumentParser(description="전주 산불 위험 Phase-1 MVP")
    parser.add_argument("--prevalence", type=float, default=0.02,
                        help="운영 예측양성 비율(임계값)")
    parser.add_argument("--daily-weather", action="store_true",
                        help="W 를 일별엔진(관측소 일별 FWI 고위험일)로 계산")
    parser.add_argument("--smoke-exposure", action="store_true",
                        help="exposure 폴백 스모크(작은 M·S) 실행 + S 증강")
    parser.add_argument("--exposure", action="store_true",
                        help="풍하 노출 실연동(Rust 커널, 전 전주). p_exposure 산출+방향 sanity. "
                             "decision 블렌드는 --exposure-alpha 로 제어(기본 0=결정엔 미반영).")
    parser.add_argument("--exposure-alpha", type=float, default=0.0,
                        help="S 블렌드 가중(0=정적S만→결정 미반영, 1=노출만). "
                             "현 발화앵커(인간발화 위주)에선 >0 이 공간CV recall 을 낮춤(보고서 참고).")
    args = parser.parse_args()
    np.random.seed(config.SEED)

    logger.info("=== 1. 데이터 로드 ===")
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)

    logger.info("=== 2. 체제 soft 게이트 (MoE) ===")
    gate, regime_order = regimes.compute_gate(master)

    logger.info("=== 3. 발화 성향 I(p) = MoE 물리식 ===")
    I, per_expert = experts.ignition_propensity(master, gate, regime_order)

    logger.info("=== 4. 확산/노출 취약 S(p) = S_p ===")
    S = master["S_p"].to_numpy().astype(np.float64)
    S = np.clip(S, 0.0, 1.0)

    p_exposure = None
    S_static = S.copy()
    if args.exposure:
        logger.info("--- exposure 실연동(Rust 풍하 확산) ---")
        station_daily = io.load_station_daily()
        aws_daily = io.load_aws_daily()
        p_exposure = exposure_engine.compute_exposure(
            master, I, gate, regime_order,
            station_daily=station_daily, aws_daily=aws_daily,
        )
        S = hazard.blend_exposure(S_static, p_exposure, alpha=args.exposure_alpha)
        logger.info("S 증강(풍하 노출 블렌드 alpha=%.2f) 적용", args.exposure_alpha)
    elif args.smoke_exposure:
        logger.info("--- exposure 폴백 스모크 ---")
        p_exposure = _smoke_exposure(master, pole_xy)
        S = hazard.blend_exposure(S_static, p_exposure, alpha=args.exposure_alpha)
        logger.info("S 증강(노출 블렌드) 적용")

    logger.info("=== 5. 그날 격자 기상 W ===")
    if args.daily_weather:
        station_daily = io.load_station_daily()
        stations = io.load_stations()
        W = weather.daily_high_danger_days(master, station_daily, stations)
    else:
        W = weather.season_weather(master, source="features")

    logger.info("=== 6. 시즌 위험 R(p) = I×S×W ===")
    R = hazard.season_risk(I, S, W)

    logger.info("=== 7. 발화점 앵커 매핑 + 보정 ===")
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_to_pole = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    # p_cal: 단조 보정 확률(해석/보고용). ties 가 많아 순위·임계엔 연속 R 사용.
    p_cal = calibrate.calibrate_probability(R, fire_to_pole)
    logger.info("p_cal 고유값=%d (ties 많음 → 순위는 연속 R 로)",
                int(np.unique(p_cal).size))

    logger.info("=== 8. F1 민감도 곡선 (양성비율 가정별) ===")
    f1_curve = calibrate.f1_sensitivity(R, fire_to_pole)
    for rho, m in f1_curve.items():
        logger.info("  prevalence=%.3f | thr=%.4g pred_pos=%d recall=%.3f f1_proxy=%.3f",
                    rho, m["threshold"], int(m["predicted_positive"]),
                    m["recall"], m["f1_proxy"])

    logger.info("=== 9. 임계값 결정 → decision (연속 R 순위 기반) ===")
    thr, decision = calibrate.decide_threshold(R, args.prevalence)

    logger.info("=== 10. 공간 블록 CV recall@top-k (연속 R) ===")
    blocks = validate.spatial_blocks(pole_xy)
    cv = validate.spatial_cv_recall(R, fire_to_pole, blocks)
    for k, m in cv.items():
        logger.info("  recall@top-%.0f%%: %.3f ± %.3f (folds w/ pos=%d)",
                    100 * k, m["mean"], m["std"], int(m["n_folds_with_pos"]))

    if p_exposure is not None:
        logger.info("=== 10b. exposure 추가 전/후 공간CV recall 비교 ===")
        # 정적S(=alpha0) 와 노출블렌드(alpha=0.5 참고치) 를 항상 비교 보고.
        R_static = hazard.season_risk(I, S_static, W)
        S_blend = hazard.blend_exposure(S_static, p_exposure, alpha=0.5)
        R_blend = hazard.season_risk(I, S_blend, W)
        cv_static = validate.spatial_cv_recall(R_static, fire_to_pole, blocks)
        cv_blend = validate.spatial_cv_recall(R_blend, fire_to_pole, blocks)
        for k in config.TOPK_FOR_RECALL:
            logger.info("  top-%.0f%%: 정적S=%.3f → 노출블렌드(a=0.5)=%.3f (사용 a=%.2f)",
                        100 * k, cv_static[k]["mean"], cv_blend[k]["mean"],
                        args.exposure_alpha)
        # 방향 sanity: 풍하 노출이 실제 2019 고성 burn-scar(서→동)와 맞는가.
        dw = validate.exposure_downwind_goseong(pole_xy, p_exposure)
        logger.info("  고성2019 풍하방향: 노출전주 %d/%d east_frac=%.2f mean_dlon=%+.4f도 "
                    "(실제흉터 W→E ~6.7km — east_frac>0.5 면 일치)",
                    int(dw["n_exposed"]), int(dw["n_local"]),
                    dw["east_frac"], dw["mean_dlon_deg"])

    logger.info("=== 11. 무작위 vs 공간 격차 (낙관편향 진단) ===")
    gap = validate.random_vs_spatial_gap(R, fire_to_pole, blocks, k=0.05)
    logger.info("  random=%.3f spatial=%.3f gap=%.3f",
                gap["random_recall"], gap["spatial_recall"], gap["gap"])

    logger.info("=== 12. sanity: 2019 고성 ===")
    sn = validate.sanity_goseong_2019(pole_xy, R)
    logger.info("  고성인근 전주=%d 평균백분위=%.3f top10%%비율=%.3f",
                int(sn["n_poles"]), sn["mean_pctile"], sn["frac_top10"])

    logger.info("=== 13. 제출 CSV 작성 ===")
    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]
    # 불확실성 프록시: extrap_flag/관측소 거리로 ±폭(해석용).
    unc = _uncertainty_band(master, p_cal)
    sub = submit.build_submission(
        pole_id=master["pole_id"].to_numpy(),
        lon=pole_xy[:, 0], lat=pole_xy[:, 1],
        decision=decision, risk_score=p_cal,
        regime=regime_lbl, p_exposure=p_exposure,
        unc_lo=unc[0], unc_hi=unc[1],
    )
    path = submit.write_submission(sub, name="submission.csv")

    logger.info("=== 완료 ===")
    logger.info("제출: %s | 행수=%d | 양성=%d (%.3f%%)",
                path, sub.height, int(sub["decision"].sum()),
                100.0 * sub["decision"].mean())
    return 0


def _uncertainty_band(master, p_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """관측외삽(extrap_flag)·관측소 거리로 단순 불확실성 밴드(해석용)."""
    nn_km = master["nn_station_km"].to_numpy().astype(np.float64)
    extrap = master["extrap_flag"].to_numpy().astype(np.float64) if "extrap_flag" in master.columns else np.zeros(len(p_cal))
    nn_norm = np.clip(nn_km / (np.nanmax(nn_km) + 1e-9), 0, 1)
    width = 0.05 + 0.15 * nn_norm + 0.10 * extrap
    lo = np.clip(p_cal - width, 0.0, 1.0)
    hi = np.clip(p_cal + width, 0.0, 1.0)
    return lo, hi


def _smoke_exposure(master, pole_xy: np.ndarray) -> np.ndarray:
    """exposure 폴백 스모크: 고위험 상위 소수 발화원 + 소수 풍향 표집.

    CONTRACT 대로 exposure 커널에 평면 km 좌표를 넘긴다(lonlat_to_km).
    """
    rng = np.random.default_rng(config.SEED)
    pole_km = exposure_engine.lonlat_to_km(pole_xy)
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
    southness = (master["mu_southness"].to_numpy().astype(np.float64) * 2.0 - 1.0)
    # 발화 후보: powerline 근접 상위 200개(설비발화 프록시).
    dpow = master["dist_to_powerline"].to_numpy().astype(np.float64)
    ignition_idx = np.argsort(dpow)[:200].astype(np.uint32)
    n_sims = 16
    wind_dir = rng.uniform(0, 360, size=n_sims)
    wind_speed = rng.uniform(3, 12, size=n_sims)
    logger.info("exposure 스모크: rust=%s M=%d S=%d",
                exposure.has_rust(), ignition_idx.size, n_sims)
    return exposure.simulate_exposure(
        pole_km, ignition_idx, wind_dir, wind_speed, fuel, southness,
    )


if __name__ == "__main__":
    raise SystemExit(main())
