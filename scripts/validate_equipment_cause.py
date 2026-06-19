"""실험 B — 설비·전기 원인 화재 앵커 분리 검증(CLI).

설계 근거(claudedocs/기획서_ablation_설비원인검증_20260619.md §4): 우리 novelty
("전주=전력설비 자산을 발화원으로 모델")의 효과는 인간발화가 아니라 **설비·전기 원인
화재**에서 나타나야 한다. 928 발화점을 resn(자유텍스트)으로 원인 분류한 뒤, 원인별로
(1) 분포·매핑수, (2) pooled recall@top-k + 부트스트랩 95% CI, (3) 위험 백분위 분포,
(4) 2019 고성 특고압 아크 named-case 를 보고한다.

정직성(필수): 설비(grid_electric) n≈18 로 매우 희소하다. pooled recall(train/test
미분리)은 절대 성능이 아니라 **상대 비교**(설비 vs 인간; 후속 aware vs blind)용이다.
n 을 어디서나 함께 보고하고, 애매한 grid_electric 분류는 콘솔에 명시해 수기 검수를 돕는다.

실행:
    .venv/bin/python scripts/validate_equipment_cause.py \
        --bootstrap 2000 --radius-km 1.0 \
        --dump-cause-csv outputs/fire_cause_labeled.csv \
        --out outputs/equipment_cause_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
    experts,
    fire_cause,
    geo,
    io,
    regimes,
    weather,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_equipment_cause")

KS = config.TOPK_FOR_RECALL

# 수기 검수 힌트: 키워드 우선순위상 grid_electric 으로 가지만 실제론 작업 스파크인
# 애매 패턴(예 "벌채...전선줄 스파크"). 콘솔에서 플래그해 오케스트레이터 검토를 돕는다.
AMBIGUOUS_GRID_HINTS = ("벌채", "예초", "용접", "그라인더", "모노레일", "스파크")


def _quartiles(vals: list[float]) -> dict[str, float]:
    """리스트의 중앙값·Q1·Q3·n(빈 리스트면 nan)."""
    if not vals:
        return dict(n=0.0, median=float("nan"), q1=float("nan"), q3=float("nan"))
    a = np.asarray(vals, dtype=np.float64)
    return dict(
        n=float(a.size),
        median=float(np.median(a)),
        q1=float(np.percentile(a, 25)),
        q3=float(np.percentile(a, 75)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 B — 설비원인 화재 앵커 분리 검증")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="부트스트랩 리샘플 수(설비·인간 recall CI).")
    ap.add_argument("--radius-km", type=float, default=1.0,
                    help="발화점→전주 매핑 허용 반경(km).")
    ap.add_argument("--dump-cause-csv", type=str,
                    default="outputs/fire_cause_labeled.csv",
                    help="원인 라벨 CSV(수기 검수용) 경로.")
    ap.add_argument("--out", type=str,
                    default="outputs/equipment_cause_validation.json",
                    help="검증 결과 JSON 경로.")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    logger.info("=== 1. 데이터 로드 ===")
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    resn = positives["resn"].to_list()

    logger.info("=== 2. 현행 config 기준 위험 R = I·S·W 구성 ===")
    gate, regime_order = regimes.compute_gate(master)
    I, _ = experts.ignition_propensity(master, gate, regime_order)
    W = weather.season_weather(master, source="features")
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    R = np.clip(I * S * W, 0.0, 1.0)
    logger.info("R: mean=%.5f p50=%.5f p99=%.5f",
                float(R.mean()), float(np.median(R)), float(np.quantile(R, 0.99)))

    logger.info("=== 3. 원인 분류(resn → 5범주) ===")
    cause = fire_cause.classify_cause(resn)
    cause_str = cause.astype(str)

    # 원인 라벨 CSV 덤프(수기 검수). [resn, cause, lon, lat, occu_year, sgg_cd].
    occu_year = positives["occu_year"].to_list()
    sgg_cd = positives["sgg_cd"].to_list() if "sgg_cd" in positives.columns \
        else [None] * len(resn)
    labeled = pl.DataFrame({
        "resn": [None if r is None else str(r) for r in resn],
        "cause": list(cause_str),
        "lon": fire_xy[:, 0],
        "lat": fire_xy[:, 1],
        "occu_year": occu_year,
        "sgg_cd": [None if s is None else str(s) for s in sgg_cd],
    })
    csv_path = Path(args.dump_cause_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_csv(csv_path)
    logger.info("원인 라벨 CSV 저장: %s (행 %d)", csv_path, labeled.height)

    # 설비(grid_electric) 원문 전수 콘솔 출력 — 오케스트레이터 눈검수용.
    grid_mask = cause_str == "grid_electric"
    grid_rows = []
    logger.info("--- grid_electric 원문 전수(n=%d) ---", int(grid_mask.sum()))
    for idx in np.nonzero(grid_mask)[0]:
        text = resn[idx]
        ambiguous = bool(text) and any(h in str(text) for h in AMBIGUOUS_GRID_HINTS)
        flag = "  <== 애매(작업스파크 가능, 수기검토)" if ambiguous else ""
        logger.info("  [%d] %r (sgg=%s, yr=%s)%s",
                    int(idx), text, sgg_cd[idx], occu_year[idx], flag)
        grid_rows.append(dict(idx=int(idx), resn=None if text is None else str(text),
                              sgg_cd=None if sgg_cd[idx] is None else str(sgg_cd[idx]),
                              occu_year=int(occu_year[idx]) if occu_year[idx] is not None else None,
                              ambiguous=ambiguous))
    n_ambiguous = sum(1 for r in grid_rows if r["ambiguous"])
    if n_ambiguous:
        logger.warning("grid_electric 중 애매(수기 downgrade 후보) %d 건 — 위 플래그 참고",
                       n_ambiguous)

    logger.info("=== 4. 원인 분포 + 반경 내 매핑수 ===")
    rec_by_cause = fire_cause.recall_by_cause(
        R, pole_xy, fire_xy, cause, ks=KS, radius_km=args.radius_km)
    counts = {grp: dict(n_fires=v["n_fires"], n_mapped=v["n_mapped"])
              for grp, v in rec_by_cause.items()}
    for grp in config.CAUSE_PRIORITY:
        if grp in counts:
            logger.info("  %-13s n=%-4d 매핑(<%.1fkm)=%d",
                        grp, counts[grp]["n_fires"], args.radius_km,
                        counts[grp]["n_mapped"])

    logger.info("=== 5. 원인별 pooled recall@top-k (전역 R 랭킹; 상대 비교용) ===")
    recall_table: dict[str, dict[str, float]] = {}
    for grp in config.CAUSE_PRIORITY:
        if grp not in rec_by_cause:
            continue
        row = rec_by_cause[grp]
        recall_table[grp] = {f"recall@{k}": float(row[k]) for k in KS}
        logger.info("  %-13s (n_map=%d) %s", grp, row["n_mapped"],
                    " ".join(f"top{int(k*100)}%={row[k]:.4f}" for k in KS))

    logger.info("=== 6. 부트스트랩 95%% CI — grid_electric, human ===")
    boot: dict[str, dict] = {}
    for grp in ("grid_electric", "human"):
        sel = cause_str == grp
        sub_xy = fire_xy[sel]
        ci = fire_cause.bootstrap_recall_ci(
            R, pole_xy, sub_xy, B=args.bootstrap, ks=KS,
            radius_km=args.radius_km, seed=config.SEED)
        boot[grp] = {f"{k}": ci[k] for k in KS}
        logger.info("  [%s] (B=%d)", grp, args.bootstrap)
        for k in KS:
            c = ci[k]
            logger.info("    top%-3d%% mean=%.4f  95%%CI=[%.4f, %.4f]",
                        int(k * 100), c["mean"], c["ci_lo"], c["ci_hi"])

    logger.info("=== 7. 원인별 위험 백분위 분포(median, Q1–Q3) ===")
    pctile_lists = fire_cause.risk_percentile_by_cause(
        R, pole_xy, fire_xy, cause, radius_km=args.radius_km)
    pctile_summary: dict[str, dict[str, float]] = {}
    for grp in config.CAUSE_PRIORITY:
        if grp not in pctile_lists:
            continue
        q = _quartiles(pctile_lists[grp])
        pctile_summary[grp] = q
        logger.info("  %-13s n=%-4d median=%.3f (Q1=%.3f, Q3=%.3f)",
                    grp, int(q["n"]), q["median"], q["q1"], q["q3"])

    logger.info("=== 8. Named-case — 2019 고성 특고압 전선 아크 ===")
    goseong_case = _goseong_named_case(resn, fire_xy, pole_xy, R, args.radius_km)
    if goseong_case.get("found"):
        logger.info("  resn=%r", goseong_case["resn"])
        logger.info("  매핑 전주 idx=%s (거리=%.3fkm) → 전역 위험 백분위=%.4f",
                    goseong_case["pole_idx"], goseong_case["dist_km"],
                    goseong_case["global_pctile"])
        it = goseong_case["in_top"]
        logger.info("  top-1%%=%s top-2%%=%s top-5%%=%s top-10%%=%s",
                    it.get("0.01"), it.get("0.02"), it.get("0.05"), it.get("0.1"))
    else:
        logger.warning("  특고압 named-case 미발견(또는 반경 밖 미매핑).")

    # 설비 vs 인간 방향성 요약 로깅(메커니즘 정합 1차 신호).
    if "grid_electric" in recall_table and "human" in recall_table:
        logger.info("=== 9. 방향성(설비 vs 인간) ===")
        for k in KS:
            ge = recall_table["grid_electric"][f"recall@{k}"]
            hu = recall_table["human"][f"recall@{k}"]
            logger.info("  top%-3d%% grid=%.4f human=%.4f Δ(grid−human)=%+.4f",
                        int(k * 100), ge, hu, ge - hu)

    logger.info("=== 10. JSON 저장 ===")
    result = dict(
        meta=dict(
            n_fires=int(len(resn)),
            radius_km=float(args.radius_km),
            bootstrap_B=int(args.bootstrap),
            seed=int(config.SEED),
            ks=[float(k) for k in KS],
            honest_note=(
                "grid_electric n is tiny (~18). pooled recall has no train/test "
                "split — interpret RELATIVELY (grid vs human; later aware vs blind), "
                "NOT as absolute performance. n is reported everywhere."
            ),
        ),
        counts=counts,
        recall_by_cause=recall_table,
        bootstrap_ci=boot,
        risk_percentile=pctile_summary,
        goseong_named_case=goseong_case,
        grid_electric_rows=grid_rows,
        n_ambiguous_grid=int(n_ambiguous),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("결과 저장: %s", out_path)

    logger.info("=== 완료 ===")
    return 0


def _goseong_named_case(
    resn: list[str | None],
    fire_xy: np.ndarray,
    pole_xy: np.ndarray,
    R: np.ndarray,
    radius_km: float,
) -> dict:
    """resn 에 '특고압' 포함 발화점을 찾아 최근접 전주의 전역 위험 백분위·top-k 진입 보고.

    2019 고성 특고압 아크(법적 입증 발화원)는 단일 사례지만 정성 가치가 크다.
    """
    hit = [i for i, t in enumerate(resn) if t is not None and "특고압" in str(t)]
    if not hit:
        return dict(found=False)
    i = hit[0]
    # 최근접 전주(반경 무관 최근접; 거리도 함께 보고).
    fx, fy = geo.point_to_km(float(fire_xy[i, 0]), float(fire_xy[i, 1]))
    pole_km = geo.lonlat_to_km(pole_xy)
    d2 = (pole_km[:, 0] - fx) ** 2 + (pole_km[:, 1] - fy) ** 2
    j = int(d2.argmin())
    dist_km = float(np.sqrt(d2[j]))

    n = R.shape[0]
    order = np.argsort(R)
    pctile = np.empty(n, dtype=np.float64)
    pctile[order] = np.linspace(0.0, 1.0, n)
    global_pctile = float(pctile[j])

    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(-R)] = np.arange(n)
    in_top = {f"{k}": bool(rank[j] < max(1, int(np.ceil(k * n)))) for k in KS}

    return dict(
        found=True,
        resn=str(resn[i]),
        fire_lon=float(fire_xy[i, 0]),
        fire_lat=float(fire_xy[i, 1]),
        pole_idx=j,
        dist_km=dist_km,
        mapped_within_radius=bool(dist_km <= radius_km),
        global_pctile=global_pctile,
        in_top=in_top,
    )


if __name__ == "__main__":
    raise SystemExit(main())
