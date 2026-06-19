"""Phase-1 MVP / Phase-4 통합 end-to-end 실행.

로드 → I(MoE) → S(S_p) → W(결합: 시즌극값×일별ISI/FFMC) → R(p) [×계층배율] →
발화점 임계값 → 공간 블록 CV recall@top-k(단계별) + 고성 sanity → submission.csv.

Phase-4 통합(기본 동작):
  - W = weather.blended_weather (전주고유 시즌 극값 + 일별 ISI/FFMC 동역학 블렌드).
    --legacy-season-weather / --phase4-daily-weather 로 단일 성분 비교 가능.
  - 계층 EB 지역배율(hierarchy.regional_multiplier)을 누수안전 공간CV 로 평가하고,
    recall 을 올리거나 중립이면 decision 에 채택(R×배율), 떨어뜨리면 off(산출·분석만).
    --multiplier {auto,on,off} 로 강제 가능(기본 auto = 정직한 CV 결정).

실행:
    .venv/bin/python scripts/run_phase1_mvp.py [--prevalence 0.02]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# 패키지 루트 임포트 경로 보장.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
    exposure,
    exposure_engine,
    experts,
    hazard,
    hierarchy,
    io,
    posterior,
    regimes,
    risk_index,
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
    parser.add_argument("--alloc-mode", choices=("auto", "global", "regime_equal",
                        "regime_count", "regime_anchor"), default="auto",
                        help="decision 양성 배분: auto=공간CV recall 가 전역 대비 개선/중립인 "
                             "체제별 방식 채택(아니면 전역). 나머지는 강제. 기본 auto.")
    parser.add_argument("--submission-variants", action="store_true",
                        help="PREVALENCE_GRID π 별 제출 변형 CSV(submission_p{π}.csv) 생성.")
    parser.add_argument("--daily-weather", action="store_true",
                        help="W 를 일별엔진(관측소 일별 FWI 고위험일)로 계산(구 폴백)")
    parser.add_argument("--legacy-season-weather", action="store_true",
                        help="W 를 전주 시즌통계(fwi_q90 등) 단일 성분으로 계산(Phase-4 이전 기본).")
    parser.add_argument("--phase4-daily-weather", action="store_true",
                        help="W 를 일별 ISI/FFMC 동역학 단일 성분으로 계산(블렌드 비채택; 진단용).")
    parser.add_argument("--multiplier", choices=("auto", "on", "off"), default="auto",
                        help="계층 EB 지역배율 결정 채택: auto=누수안전 공간CV 가 recall 을 "
                             "올리거나 중립이면 채택(아니면 off). on/off=강제. 기본 auto.")
    parser.add_argument("--smoke-exposure", action="store_true",
                        help="exposure 폴백 스모크(작은 M·S) 실행 + S 증강")
    parser.add_argument("--exposure", action="store_true",
                        help="풍하 노출 실연동(Rust 커널, 전 전주). p_exposure 산출+방향 sanity. "
                             "decision 블렌드는 --exposure-alpha 로 제어(기본 0=결정엔 미반영).")
    parser.add_argument("--exposure-alpha", type=float, default=0.0,
                        help="S 블렌드 가중(0=정적S만→결정 미반영, 1=노출만). "
                             "현 발화앵커(인간발화 위주)에선 >0 이 공간CV recall 을 낮춤(보고서 참고).")
    # ── Phase-5: 사후분포·커버리지 ────────────────────────────────────────
    parser.add_argument("--posterior", choices=("on", "off"), default="on",
                        help="베이지안 계층 사후분포 + MC 전파(전주별 risk_lo/hi credible) "
                             "+ per-regime conformal 커버리지 검증. 기본 on(Phase-5 채택).")
    parser.add_argument("--posterior-draws", type=int, default=200,
                        help="사후예측 MC draw 수(기본 200).")
    parser.add_argument("--posterior-spatial", choices=("poisson_gamma", "bym"),
                        default="poisson_gamma",
                        help="지역율 사후 백엔드: poisson_gamma=독립 켤레(기본, 공간상관 없음) / "
                             "bym=BYM2 공간 CAR(이웃 borrow, INLA식 Laplace 근사). "
                             "bym 은 recall 레버가 아니라 사후 공간정밀도 확장(커버리지·recall 유지).")
    parser.add_argument("--conformal-alpha", type=float, default=0.10,
                        help="conformal/credible 목표 miscoverage(기본 0.10 → 명목 90%%).")
    parser.add_argument("--decision-from", choices=("alloc", "conformal"), default="alloc",
                        help="제출 decision 산출: alloc=배분 컷(기본, F1 예산제어) / "
                             "conformal=per-regime conformal 임계(커버리지 보장 집합).")
    parser.add_argument("--no-exposure-default", action="store_true",
                        help="posterior 경로에서 기본 exposure 산출을 끈다(p_exposure/ops_priority 미산출).")
    args = parser.parse_args()
    # Phase-5 기본: 사후/커버리지 경로에서 p_exposure 복구(항목1)·ops_priority(항목2)를
    # 위해 exposure 를 기본 산출한다(--exposure 명시 없이도). --no-exposure-default 로 차단.
    if args.posterior == "on" and not args.no_exposure_default and not args.smoke_exposure:
        args.exposure = True
    np.random.seed(config.SEED)

    logger.info("=== 1. 데이터 로드 ===")
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)

    logger.info("=== 2. 체제 soft 게이트 (MoE) ===")
    gate, regime_order = regimes.compute_gate(master)
    # 전주별 우세 체제(gate argmax) — 체제별 임계값/배분에 쓰는 단일 라벨.
    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]

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
    # 기존 시즌통계 W 는 항상 계산(단계별 비교 기준). 일별·블렌드는 데이터 필요 시 로드.
    W_season = weather.season_weather(master, source="features")
    W_daily = None
    if args.legacy_season_weather:
        W = W_season
        logger.info("W = 전주 시즌통계(fwi_q90 등; Phase-4 이전 기본)")
    elif args.daily_weather:
        station_daily = io.load_station_daily()
        stations = io.load_stations()
        W = weather.daily_high_danger_days(master, station_daily, stations)
        logger.info("W = 일별 FWI 고위험일 빈도(구 폴백)")
    elif args.phase4_daily_weather:
        station_daily = io.load_station_daily()
        stations = io.load_stations()
        W = weather.daily_isi_weather(master, station_daily, stations)
        W_daily = W
        logger.info("W = 일별 ISI/FFMC 동역학 단일(블렌드 비채택; 진단)")
    else:
        station_daily = io.load_station_daily()
        stations = io.load_stations()
        W_daily = weather.daily_isi_weather(master, station_daily, stations)
        W = weather.blended_weather(master, station_daily, stations)
        logger.info("W = 결합(시즌극값 %.2f × 일별ISI/FFMC %.2f; Phase-4 기본)",
                    config.W_BLEND_SEASON_WEIGHT, 1.0 - config.W_BLEND_SEASON_WEIGHT)

    logger.info("=== 6. 시즌 위험 R(p) = I×S×W ===")
    R = hazard.season_risk(I, S, W)

    logger.info("=== 7. 발화점 앵커 매핑 + 공간 블록 ===")
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_to_pole = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)

    logger.info("=== 7a. 단계별 공간CV recall: baseline → +일별W → +블렌드W ===")
    # baseline(시즌W) → +일별ISI/FFMC W → 최종 W(블렌드). 결정 W 효과를 정직하게 본다.
    R_seasonW = hazard.season_risk(I, S, W_season)
    cv_season = validate.spatial_cv_recall(R_seasonW, fire_to_pole, blocks)
    cv_blendW = validate.spatial_cv_recall(R, fire_to_pole, blocks)
    cv_dailyW = None
    if W_daily is not None:
        R_dailyW = hazard.season_risk(I, S, W_daily)
        cv_dailyW = validate.spatial_cv_recall(R_dailyW, fire_to_pole, blocks)
    for k in config.TOPK_FOR_RECALL:
        mid = f"{cv_dailyW[k]['mean']:.3f}" if cv_dailyW else "  -  "
        logger.info("  top-%.0f%%: baseline(시즌W)=%.3f → +일별W=%s → 결정W=%.3f",
                    100 * k, cv_season[k]["mean"], mid, cv_blendW[k]["mean"])

    logger.info("=== 7b. 계층 EB 지역배율: 누수안전 공간CV + 채택 결정 ===")
    # 누수안전: fold 별 train 발화점만으로 배율 재계산(test 발화점 미사용).
    cv_mult = validate.spatial_cv_recall_multiplier(
        R, master, fire_to_pole, blocks)
    # 채택 규칙(exposure alpha=0 철학과 동일): recall 가중평균이 오르거나 중립이면 채택.
    score_base = float(np.mean([cv_blendW[k]["mean"] for k in config.TOPK_FOR_RECALL]))
    score_mult = float(np.mean([cv_mult[k]["mean"] for k in config.TOPK_FOR_RECALL]))
    cv_improves = score_mult >= score_base - 1e-6
    for k in config.TOPK_FOR_RECALL:
        logger.info("  top-%.0f%%: W만=%.3f → W×배율(누수안전)=%.3f (Δ=%+.3f)",
                    100 * k, cv_blendW[k]["mean"], cv_mult[k]["mean"],
                    cv_mult[k]["mean"] - cv_blendW[k]["mean"])
    if args.multiplier == "on":
        adopt_mult = True
    elif args.multiplier == "off":
        adopt_mult = False
    else:  # auto
        adopt_mult = cv_improves
    logger.info("  배율 점수(recall 평균): W만=%.4f W×배율=%.4f → 채택=%s (mode=%s)",
                score_base, score_mult, adopt_mult, args.multiplier)

    # 물리 결합 위험(배율 전) — Phase-5 사후 전파의 base_risk(I·S·W) 로 보존.
    R_physics = R.copy()

    # 전 데이터 배율(제출용): 누수 무관(hold-out 없음) → 전체 발화점을 지역 앵커로 사용.
    mult_full = hierarchy.regional_multiplier(
        master, positives.select(["lon", "lat"]), normalize="mean")
    if adopt_mult:
        R = np.clip(R * mult_full, 0.0, 1.0)
        logger.info("계층 EB 배율 decision 에 채택(R ← R×배율, 전역평균≈1 스케일보존)")
    else:
        logger.info("계층 EB 배율 decision 미반영(off; 산출/분석으로만). mult_full 은 보고만.")

    logger.info("=== 8. 발화점 보정(p_cal) ===")
    # p_cal: 단조 보정 확률(해석/보고용). ties 가 많아 순위·임계엔 연속 R 사용.
    p_cal = calibrate.calibrate_probability(R, fire_to_pole)
    logger.info("p_cal 고유값=%d (ties 많음 → 순위는 연속 R 로)",
                int(np.unique(p_cal).size))

    logger.info("=== 8b. F1 민감도 곡선 (양성비율 가정별, 전역 단일컷) ===")
    f1_curve = calibrate.f1_sensitivity(R, fire_to_pole)
    for rho, m in f1_curve.items():
        logger.info("  prevalence=%.3f | thr=%.4g pred_pos=%d recall=%.3f f1_proxy=%.3f",
                    rho, m["threshold"], int(m["predicted_positive"]),
                    m["recall"], m["f1_proxy"])

    logger.info("=== 8c. 배분방식별 공간CV recall@top-k (전역 vs 체제별 ⒝⒞⒟) ===")
    # ⒝ regime_equal 은 ⒞ regime_count 와 수학적 동일이라 둘 다 표기(프레이밍 명시).
    alloc_modes = (calibrate.ALLOC_GLOBAL, calibrate.ALLOC_EQUAL,
                   calibrate.ALLOC_COUNT, calibrate.ALLOC_ANCHOR)
    cv_alloc = validate.spatial_cv_recall_allocation(
        R, fire_to_pole, regime_lbl, blocks,
        prevalence=args.prevalence, modes=alloc_modes)
    alloc_score: dict[str, float] = {}
    for mode in alloc_modes:
        sc = float(np.mean([cv_alloc[mode][k]["mean"] for k in config.TOPK_FOR_RECALL]))
        alloc_score[mode] = sc
        cells = " ".join(f"top{int(100*k)}%={cv_alloc[mode][k]['mean']:.3f}"
                         for k in config.TOPK_FOR_RECALL)
        logger.info("  %-13s | %s | 평균=%.4f", mode, cells, sc)

    # 채택 규칙(정직): 체제별 best 가 전역 대비 recall 평균 개선/중립이면 채택, 악화면 전역.
    g_score = alloc_score[calibrate.ALLOC_GLOBAL]
    regime_best_mode = max(
        (calibrate.ALLOC_COUNT, calibrate.ALLOC_ANCHOR),
        key=lambda mm: alloc_score[mm])
    regime_improves = alloc_score[regime_best_mode] >= g_score - 1e-6
    if args.alloc_mode == "auto":
        adopt_mode = regime_best_mode if regime_improves else calibrate.ALLOC_GLOBAL
    else:
        adopt_mode = args.alloc_mode
    logger.info("  배분 채택: best체제별=%s(평균=%.4f) vs 전역(평균=%.4f) → 채택=%s (mode=%s)",
                regime_best_mode, alloc_score[regime_best_mode], g_score,
                adopt_mode, args.alloc_mode)

    logger.info("=== 8d. F1 민감도: π별 전역 vs 체제별 (proxy-F1/recall/precision) ===")
    # 체제별 F1 비교는 채택 배분(전역이면 count 프레이밍으로 비교만) 기준.
    f1_alloc_mode = adopt_mode if adopt_mode != calibrate.ALLOC_GLOBAL else calibrate.ALLOC_COUNT
    anchor_full = calibrate.regime_anchor_count(regime_lbl, fire_to_pole)
    f1_cmp = calibrate.f1_sensitivity_compare(
        R, fire_to_pole, regime_lbl, alloc_mode=f1_alloc_mode,
        anchor_count=anchor_full)
    for rho in config.PREVALENCE_GRID:
        g, rg = f1_cmp[rho]["global"], f1_cmp[rho]["regime"]
        logger.info("  π=%.3f | 전역 f1=%.3f rec=%.3f prec=%.3f || 체제별 f1=%.3f rec=%.3f prec=%.3f",
                    rho, g["f1_proxy"], g["recall"], g["precision_proxy"],
                    rg["f1_proxy"], rg["recall"], rg["precision_proxy"])

    logger.info("=== 9. 임계값 결정 → decision (채택 배분=%s) ===", adopt_mode)
    if adopt_mode == calibrate.ALLOC_GLOBAL:
        thr, decision = calibrate.decide_threshold(R, args.prevalence)
    else:
        anchor = anchor_full if adopt_mode == calibrate.ALLOC_ANCHOR else None
        _, decision = calibrate.decide_threshold_per_regime(
            R, regime_lbl, args.prevalence, adopt_mode, anchor_count=anchor)

    logger.info("=== 10. 최종 decision R 공간 블록 CV recall@top-k (전역 기준 분리력) ===")
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

    # ── Phase-5: 베이지안 사후분포 + MC 전파 + 커버리지 검증 ──────────────────
    risk_lo = risk_hi = None
    ops_priority = None
    post_artifact: dict = {}
    cov_conf = cov_cred = None
    sgg_post_tbl = None
    if args.posterior == "on":
        backend = args.posterior_spatial
        backend_ko = ("Poisson-Gamma 켤레(독립)" if backend == "poisson_gamma"
                      else "BYM2 공간 CAR(이웃 borrow·INLA Laplace)")
        logger.info("=== 12a. 베이지안 계층 사후분포(%s) + MC 전파 ===", backend_ko)
        # base_risk = 물리 I·S·W(배율 전). 사후 지역율 draw 를 곱해 전주별 credible.
        positives_xy = positives.select(["lon", "lat"])
        post_pg = posterior.regional_rate_posterior(master, positives_xy)
        if backend == "bym":
            post_full = posterior.bym2_spatial_posterior(master, positives_xy)
            adj = post_full.adjacency
            logger.info("  격자 인접: 노드=%d 평균이웃=%.2f 고립=%d 연결성분=%d | τ*=%.1f φ*=%.2f",
                        adj.n_grids, adj.mean_neighbors, int(adj.isolated.sum()),
                        adj.n_components, post_full.tau, post_full.phi)
        else:
            post_full = post_pg
        q_lo = args.conformal_alpha / 2.0
        q_hi = 1.0 - args.conformal_alpha / 2.0
        prop = posterior.propagate_risk_posterior(
            R_physics, post_full, n_draws=args.posterior_draws,
            q_lo=q_lo, q_hi=q_hi, weight_bootstrap=True,
            weight_bootstrap_cv=0.10)
        risk_lo, risk_hi = prop["risk_lo"], prop["risk_hi"]
        logger.info("  전주별 credible(%.0f%%): 평균폭(hi-lo)=%.4g p50_mean=%.4g",
                    100 * (q_hi - q_lo), float(np.mean(risk_hi - risk_lo)),
                    float(np.median(prop["risk_mean"])))

        # zero-event 시군 사후 폭(불확실성) 진단 — 고성·인제·화천·태백 등.
        watch = ("고성", "인제", "화천", "태백")
        if backend == "bym":
            sgg_post_tbl = posterior.spatial_sgg_rate_table(post_full)
            logger.info("  시군 BYM 사후(cv 상위=불확실):")
        else:
            sgg_post_tbl = posterior.sgg_rate_posterior_table(post_full)
            logger.info("  시군 사후폭(변동계수 cv 상위 = 불확실; zero-event 수렴+넓음):")
        for row in sgg_post_tbl.iter_rows(named=True):
            tag = " ★zero-event예상" if row["sgg"] in watch else ""
            logger.info("    %-4s cv=%.3f mult_mean=%.3f rel_sd=%.3f%s",
                        row["sgg"], row["cv"], row["mult_mean"], row["rel_sd"], tag)

        # BYM vs Poisson-Gamma borrow 비교(항목③) — zero-event 시군 사후가 이웃에서
        # 어떻게 달라지나(수치) + 전역 매끄러움(Moran's I).
        bym_compare = None
        if backend == "bym":
            logger.info("=== 12a-2. BYM vs Poisson-Gamma: zero-event 시군 borrow 효과 ===")
            bym_compare = validate.compare_bym_vs_poisson_gamma(
                post_pg, post_full, watch_sgg=watch)
            for row in bym_compare["sgg"]:
                tag = " ★" if row["sgg"] in watch else ""
                logger.info("    %-4s | PG mult=%.3f → BYM mult=%.3f (Δ=%+.3f, borrow=%.3f)%s",
                            row["sgg"], row["pg_mult"], row["bym_mult"],
                            row["delta_mult"], row["borrow"], tag)
            mi = bym_compare["moran"]
            logger.info("  전역 매끄러움 Moran's I: PG=%.3f → BYM=%.3f (BYM 이 더 매끄러우면↑)",
                        mi["pg"], mi["bym"])

        logger.info("=== 12b. per-regime conformal 임계(누수안전 fold train) ===")
        conf_thr_full = calibrate.conformal_threshold_per_regime(
            R, fire_to_pole, regime_lbl, alpha=args.conformal_alpha)

        logger.info("=== 12c. 커버리지 검증: 명목 %.0f%% vs 홀드아웃 발화점 실측(체제별) ===",
                    100 * (1 - args.conformal_alpha))
        cov_conf = validate.coverage_conformal_holdout(
            R, fire_to_pole, regime_lbl, blocks, alpha=args.conformal_alpha)
        cov_cred = validate.coverage_credible_holdout(
            R_physics, master, fire_to_pole, regime_lbl, blocks,
            n_draws=args.posterior_draws, q_lo=q_lo, q_hi=q_hi,
            backend=backend)
        for k in ("all", *config.REGIMES):
            cc = cov_conf.get(k, {})
            cr = cov_cred.get(k, {})
            verdict = ""
            if cc and np.isfinite(cc.get("empirical", float("nan"))):
                d = cc["empirical"] - cc["nominal"]
                verdict = ("과커버" if d > 0.03 else "소커버" if d < -0.03 else "합격≈명목")
            logger.info("  %-10s | conformal 실측=%.3f conf합격=%s | credible 실측=%.3f (명목 %.2f, n=%d)",
                        k, cc.get("empirical", float("nan")), verdict,
                        cr.get("empirical", float("nan")), cc.get("nominal", float("nan")),
                        int(cc.get("n_test", 0)))

        # 운영 플래그(항목2): 영동 풍하 고노출 → ops_priority.
        ops_priority = submit.ops_priority_flag(regime_lbl, p_exposure)

        post_artifact = {
            "posterior_spatial_backend": backend,
            "conformal_alpha": args.conformal_alpha,
            "posterior_draws": args.posterior_draws,
            "credible_mean_width": float(np.mean(risk_hi - risk_lo)),
            "coverage_conformal": cov_conf,
            "coverage_credible": cov_cred,
            "sgg_posterior_table": sgg_post_tbl.to_dicts() if sgg_post_tbl is not None else [],
            "ops_priority_count": int(ops_priority.sum()) if ops_priority is not None else 0,
        }
        if bym_compare is not None:
            post_artifact["bym_vs_poisson_gamma"] = bym_compare
            post_artifact["bym_tau"] = float(post_full.tau)
            post_artifact["bym_phi"] = float(post_full.phi)
            post_artifact["grid_adjacency"] = {
                "n_grids": int(post_full.adjacency.n_grids),
                "mean_neighbors": float(post_full.adjacency.mean_neighbors),
                "n_isolated": int(post_full.adjacency.isolated.sum()),
                "n_components": int(post_full.adjacency.n_components),
            }

    # conformal decision 옵션(커버리지 보장 집합으로 제출 decision 대체).
    if args.posterior == "on" and args.decision_from == "conformal":
        decision = calibrate.conformal_decision(R, regime_lbl, conf_thr_full)
        logger.info("decision ← per-regime conformal 집합(양성=%d, %.3f%%)",
                    int(decision.sum()), 100.0 * decision.mean())
        adopt_mode = f"conformal(α={args.conformal_alpha})"

    logger.info("=== 13. 제출 CSV 작성 (채택 배분=%s) ===", adopt_mode)
    # 불확실성 프록시: extrap_flag/관측소 거리로 ±폭(해석용, 하위호환 보존).
    unc = _uncertainty_band(master, p_cal)
    pole_id_arr = master["pole_id"].to_numpy()
    # 표시용 백분위 — decision 과 **동일한 R** 로 1회 계산(완전 정렬). 글로벌 + 체제내.
    #   raw risk(p_cal)은 절대값이 작아(≈0.01) 정성 전달이 약함 → 0~100 지수 동봉.
    #   배분/conformal decision 은 체제 내 임계라 risk_pctile_regime 과 1:1 정렬됨.
    risk_pct = np.round(risk_index.risk_percentile(R), 2)
    risk_pct_reg = np.round(risk_index.risk_percentile_by_group(R, regime_lbl), 2)
    _PCT_COLS = [pl.Series("risk_pctile", risk_pct),
                 pl.Series("risk_pctile_regime", risk_pct_reg)]
    sub = submit.build_submission(
        pole_id=pole_id_arr,
        lon=pole_xy[:, 0], lat=pole_xy[:, 1],
        decision=decision, risk_score=p_cal,
        regime=regime_lbl, p_exposure=p_exposure,
        risk_lo=risk_lo, risk_hi=risk_hi, ops_priority=ops_priority,
        unc_lo=unc[0], unc_hi=unc[1],
    ).with_columns(_PCT_COLS)
    path = submit.write_submission(sub, name="submission.csv")

    # 체제별 decision 분포(정성 보고).
    logger.info("=== 13a. 제출 decision 분포(체제별) ===")
    regime_dist: dict[str, dict[str, float]] = {}
    for r in regime_order:
        rm = regime_lbl == r
        n_r = int(rm.sum())
        pos_r = int(decision[rm].sum())
        share = 100.0 * pos_r / max(int(decision.sum()), 1)
        regime_dist[r] = dict(n_poles=float(n_r), n_positive=float(pos_r),
                              positive_rate_pct=100.0 * pos_r / max(n_r, 1),
                              share_of_positives_pct=share)
        logger.info("  %s: 전주=%d 양성=%d (체제내 %.3f%% / 전체양성의 %.1f%%)",
                    r, n_r, pos_r, 100.0 * pos_r / max(n_r, 1), share)

    # π별 제출 변형 CSV(정답비율 미지 대비) — 채택 배분과 동일 방식.
    if args.submission_variants:
        logger.info("=== 13b. π별 제출 변형 CSV 생성 ===")
        for rho in config.PREVALENCE_GRID:
            if adopt_mode == calibrate.ALLOC_GLOBAL:
                _, dec_v = calibrate.decide_threshold(R, rho)
            else:
                anchor = anchor_full if adopt_mode == calibrate.ALLOC_ANCHOR else None
                _, dec_v = calibrate.decide_threshold_per_regime(
                    R, regime_lbl, rho, adopt_mode, anchor_count=anchor)
            sub_v = submit.build_submission(
                pole_id=pole_id_arr, lon=pole_xy[:, 0], lat=pole_xy[:, 1],
                decision=dec_v, risk_score=p_cal, regime=regime_lbl,
                p_exposure=p_exposure, risk_lo=risk_lo, risk_hi=risk_hi,
                ops_priority=ops_priority, unc_lo=unc[0], unc_hi=unc[1],
            ).with_columns(_PCT_COLS)
            vname = f"submission_p{rho:g}.csv"
            submit.write_submission(sub_v, name=vname)
            logger.info("  π=%.3f → %s (양성=%d)", rho, vname, int(dec_v.sum()))

    # 분석 산출물(표/JSON) 저장 — 시각화는 다음 에이전트가 그림.
    logger.info("=== 13c. 분석 산출물 저장(outputs/) ===")
    config.OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "prevalence": args.prevalence,
        "total_budget": int(decision.sum()),
        "adopt_alloc_mode": adopt_mode,
        "alloc_cv_recall": {
            mode: {f"top{k}": cv_alloc[mode][k] for k in config.TOPK_FOR_RECALL}
            for mode in alloc_modes
        },
        "alloc_score_mean_recall": alloc_score,
        "regime_anchor_count_full": anchor_full,
        "regime_decision_distribution": regime_dist,
        "f1_sensitivity_global_vs_regime": {
            f"{rho}": f1_cmp[rho] for rho in config.PREVALENCE_GRID
        },
        "f1_sensitivity_global_curve": {
            f"{rho}": f1_curve[rho] for rho in config.PREVALENCE_GRID
        },
        "final_decision_cv_recall": {f"top{k}": cv[k] for k in config.TOPK_FOR_RECALL},
        "random_vs_spatial_gap": gap,
        "goseong_sanity": sn,
        "multiplier_adopted": bool(adopt_mult),
        "posterior_coverage": post_artifact,
    }
    art_path = config.OUT / "regime_threshold_analysis.json"
    with open(art_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2)
    logger.info("  저장: %s", art_path)

    logger.info("=== 완료 ===")
    logger.info("제출: %s | 행수=%d | 양성=%d (%.3f%%) | 배분=%s | 배율채택=%s",
                path, sub.height, int(sub["decision"].sum()),
                100.0 * sub["decision"].mean(), adopt_mode, adopt_mult)
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
