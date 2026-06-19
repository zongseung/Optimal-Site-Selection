"""Phase-3 end-to-end 실행 + 검증 + 제출 재생성.

Phase-3 두 개선을 단계별로 측정·결정한다.
  ① 토지피복 발화신호: I 에 landcover 성분 추가(이미 config·experts 반영).
  ② 조건부 풍하노출: α_p = α_base × 영동게이트 × 양간강도 로 영동·고양간에만 노출 블렌드.

단계별 공간블록CV recall@top-{1,2,5,10}% 를 **전체 + 영동 부분집합**으로 측정:
  - before(Phase2 6성분 튜닝값)
  - +토지피복(Phase3 7성분 튜닝값, config.EXPERT_WEIGHTS)
  - +조건부노출(α_base 그리드 CV 선택)
무작위-공간 격차(낙관편향)·2019 고성 풍하 sanity 도 보고.

결정 규칙(정직): 조건부노출이 전체 공간CV recall 을 올리면 α_base 채택, 아니면 0
(decision 미반영, p_exposure 는 제출 컬럼·영동 분석으로만).

실행:
    .venv/bin/python scripts/run_phase3.py [--prevalence 0.02]
        [--alpha-grid 0.1,0.2,0.3,0.5] [--no-exposure(스킵, 빠른 가중확인)]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
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
logger = logging.getLogger("run_phase3")

# Phase-2 튜닝 6성분 가중(landcover 추가 전) — "before" 단계 비교 기준(불변 참조).
#   출처: outputs/tuned_weights.json(Phase2) / config 주석. landcover=0 으로 7성분 표현.
PHASE2_WEIGHTS_6 = {
    config.REGIME_YEONGDONG: dict(forest=0.3509, road=0.0000, powerline=0.1003,
                                  fwi=0.0977, yanggan=0.0501, fuel=0.4010, landcover=0.0000),
    config.REGIME_YEONGSEO:  dict(forest=0.3684, road=0.2105, powerline=0.3684,
                                  fwi=0.0000, yanggan=0.0526, fuel=0.0000, landcover=0.0000),
    config.REGIME_MOUNTAIN:  dict(forest=0.0000, road=0.1579, powerline=0.4737,
                                  fwi=0.1579, yanggan=0.2105, fuel=0.0000, landcover=0.0000),
}

KS = config.TOPK_FOR_RECALL


def _ignition_from_weights(
    feats: dict[str, np.ndarray], gate: np.ndarray, regime_order: list[str],
    weights: dict[str, dict[str, float]],
) -> np.ndarray:
    """주어진 가중(체제별 dict)으로 MoE 발화성향 I 를 계산(before 비교용)."""
    per_expert = {}
    for r in regime_order:
        w = weights[r]
        wsum = sum(w.values())
        sc = np.zeros(next(iter(feats.values())).shape[0], dtype=np.float64)
        for key, weight in w.items():
            sc += weight * feats[key]
        per_expert[r] = sc / wsum
    expert_mat = np.stack([per_expert[r] for r in regime_order], axis=1)
    I = np.clip(np.sum(gate * expert_mat, axis=1), 0.0, 1.0)
    return I


def _fmt_cv(cv: dict[float, dict[str, float]]) -> str:
    return " ".join(f"top{int(k*100)}={cv[k]['mean']:.4f}" for k in KS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-3 토지피복 + 조건부 풍하노출")
    ap.add_argument("--prevalence", type=float, default=0.02,
                    help="운영 예측양성 비율(임계값)")
    ap.add_argument("--alpha-grid", type=str, default="0.1,0.2,0.3,0.5",
                    help="조건부 α_base 후보(쉼표구분). CV recall 로 선택.")
    ap.add_argument("--no-exposure", action="store_true",
                    help="exposure 시뮬 스킵(가중만 빠르게 확인; 조건부노출 미측정).")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    logger.info("=== 1. 데이터 로드 ===")
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    yanggan = master["yanggan_days"].to_numpy().astype(np.float64)

    logger.info("=== 2. 체제 soft 게이트 (MoE) ===")
    gate, regime_order = regimes.compute_gate(master)
    yd_col = regime_order.index(config.REGIME_YEONGDONG)
    # 영동 부분집합: 영동 게이트가 우세(argmax)인 전주.
    yeongdong_mask = gate.argmax(axis=1) == yd_col
    logger.info("영동 우세 전주: %d (%.2f%%)", int(yeongdong_mask.sum()),
                100.0 * yeongdong_mask.mean())

    logger.info("=== 3. 발화 성향 I(p) ===")
    feats = experts.build_ignition_features(master)
    I_before = _ignition_from_weights(feats, gate, regime_order, PHASE2_WEIGHTS_6)
    I_lc, _ = experts.ignition_propensity(master, gate, regime_order)  # config(7성분)

    logger.info("=== 4. 정적 S, W ===")
    S_static = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    W = weather.season_weather(master, source="features")

    # 검증 앵커·블록.
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)

    def cv_full(R: np.ndarray) -> dict[float, dict[str, float]]:
        return validate.spatial_cv_recall(R, f2p, blocks, ks=KS)

    def cv_yd(R: np.ndarray) -> dict[float, dict[str, float]]:
        return validate.spatial_cv_recall(R, f2p, blocks, ks=KS,
                                          subset_mask=yeongdong_mask)

    logger.info("=== 5. 단계별 공간CV recall (정적 S, alpha=0) ===")
    R_before = hazard.season_risk(I_before, S_static, W)
    R_lc = hazard.season_risk(I_lc, S_static, W)
    cv_before_full, cv_before_yd = cv_full(R_before), cv_yd(R_before)
    cv_lc_full, cv_lc_yd = cv_full(R_lc), cv_yd(R_lc)
    logger.info("[before Phase2]  전체: %s", _fmt_cv(cv_before_full))
    logger.info("[before Phase2]  영동: %s", _fmt_cv(cv_before_yd))
    logger.info("[+토지피복]      전체: %s", _fmt_cv(cv_lc_full))
    logger.info("[+토지피복]      영동: %s", _fmt_cv(cv_lc_yd))

    # 낙관편향(전체) before/lc.
    gap_before = validate.random_vs_spatial_gap(R_before, f2p, blocks, k=0.05)
    gap_lc = validate.random_vs_spatial_gap(R_lc, f2p, blocks, k=0.05)
    logger.info("무작위-공간 격차@5%%: before=%.4f +토지피복=%.4f (음수=낙관편향 없음)",
                gap_before["gap"], gap_lc["gap"])

    # 기본: 조건부노출 미채택(alpha=0). 채택되면 갱신.
    adopted_alpha = 0.0
    p_exposure = None
    R_final = R_lc

    if not args.no_exposure:
        logger.info("=== 6. 풍하 노출 P(노출) 실연동 ===")
        station_daily = io.load_station_daily()
        aws_daily = io.load_aws_daily()
        p_exposure = exposure_engine.compute_exposure(
            master, I_lc, gate, regime_order,
            station_daily=station_daily, aws_daily=aws_daily,
        )

        logger.info("=== 6b. 2019 고성 풍하 sanity ===")
        dw = validate.exposure_downwind_goseong(pole_xy, p_exposure)
        logger.info("고성2019 풍하: 노출전주 %d/%d east_frac=%.2f mean_dlon=%+.4f도 "
                    "(실제흉터 W→E ~6.7km; east_frac>0.5 면 일치)",
                    int(dw["n_exposed"]), int(dw["n_local"]),
                    dw["east_frac"], dw["mean_dlon_deg"])

        logger.info("=== 7. 조건부노출 α_base 그리드 CV 선택 ===")
        alpha_grid = [float(a) for a in args.alpha_grid.split(",")]
        base5_full = cv_lc_full[0.05]["mean"]
        base5_yd = cv_lc_yd[0.05]["mean"]
        logger.info("기준(+토지피복) top5%%: 전체=%.4f 영동=%.4f", base5_full, base5_yd)
        best_alpha, best_full5 = 0.0, base5_full
        for a in alpha_grid:
            alpha_p = hazard.conditional_alpha(
                gate, regime_order, yanggan, alpha_base=a,
                yeongdong_regime=config.REGIME_YEONGDONG)
            S_cond = hazard.blend_exposure_conditional(S_static, p_exposure, alpha_p)
            R_cond = hazard.season_risk(I_lc, S_cond, W)
            cvf, cvy = cv_full(R_cond), cv_yd(R_cond)
            logger.info("  α_base=%.2f | 전체 %s | 영동 %s | α_p>0=%.2f%% max=%.3f",
                        a, _fmt_cv(cvf), _fmt_cv(cvy),
                        100.0 * (alpha_p > 0).mean(), float(alpha_p.max()))
            if cvf[0.05]["mean"] > best_full5 + 1e-9:
                best_full5, best_alpha = cvf[0.05]["mean"], a

        logger.info("=== 8. 조건부노출 채택 결정(정직) ===")
        if best_alpha > 0.0:
            adopted_alpha = best_alpha
            alpha_p = hazard.conditional_alpha(
                gate, regime_order, yanggan, alpha_base=adopted_alpha,
                yeongdong_regime=config.REGIME_YEONGDONG)
            S_final = hazard.blend_exposure_conditional(S_static, p_exposure, alpha_p)
            R_final = hazard.season_risk(I_lc, S_final, W)
            logger.info("채택: α_base=%.2f (전체 top5%% %.4f→%.4f). decision 반영.",
                        adopted_alpha, base5_full, best_full5)
        else:
            logger.info("미채택: 어떤 α_base 도 전체 top5%% CV recall 을 개선 못함 "
                        "(%.4f). alpha=0 유지 — p_exposure 는 제출 컬럼·영동 분석으로만.",
                        base5_full)

    logger.info("=== 9. 최종 단계 공간CV recall (decision 사용 R) ===")
    cv_final_full, cv_final_yd = cv_full(R_final), cv_yd(R_final)
    logger.info("[최종 채택]      전체: %s", _fmt_cv(cv_final_full))
    logger.info("[최종 채택]      영동: %s", _fmt_cv(cv_final_yd))

    logger.info("=== 10. 임계값 결정 → decision ===")
    p_cal = calibrate.calibrate_probability(R_final, f2p)
    thr, decision = calibrate.decide_threshold(R_final, args.prevalence)

    logger.info("=== 11. sanity: 2019 고성(위험 백분위) ===")
    sn = validate.sanity_goseong_2019(pole_xy, R_final)
    logger.info("고성인근 전주=%d 평균백분위=%.3f top10%%비율=%.3f",
                int(sn["n_poles"]), sn["mean_pctile"], sn["frac_top10"])

    logger.info("=== 12. 제출 CSV 작성 ===")
    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]
    unc = _uncertainty_band(master, p_cal)
    sub = submit.build_submission(
        pole_id=master["pole_id"].to_numpy(),
        lon=pole_xy[:, 0], lat=pole_xy[:, 1],
        decision=decision, risk_score=p_cal,
        regime=regime_lbl,
        p_exposure=p_exposure if p_exposure is not None else None,
        unc_lo=unc[0], unc_hi=unc[1],
    )
    path = submit.write_submission(sub, name="submission.csv")

    logger.info("=== 완료 ===")
    logger.info("제출: %s | 행수=%d | 양성=%d (%.3f%%) | 조건부노출 α_base=%.2f",
                path, sub.height, int(sub["decision"].sum()),
                100.0 * sub["decision"].mean(), adopted_alpha)
    return 0


def _uncertainty_band(master, p_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """관측외삽(extrap_flag)·관측소 거리로 단순 불확실성 밴드(해석용)."""
    nn_km = master["nn_station_km"].to_numpy().astype(np.float64)
    extrap = (master["extrap_flag"].to_numpy().astype(np.float64)
              if "extrap_flag" in master.columns else np.zeros(len(p_cal)))
    nn_norm = np.clip(nn_km / (np.nanmax(nn_km) + 1e-9), 0, 1)
    width = 0.05 + 0.15 * nn_norm + 0.10 * extrap
    lo = np.clip(p_cal - width, 0.0, 1.0)
    hi = np.clip(p_cal + width, 0.0, 1.0)
    return lo, hi


if __name__ == "__main__":
    raise SystemExit(main())
