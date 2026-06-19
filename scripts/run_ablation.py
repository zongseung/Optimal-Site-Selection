"""실험 A — asset-aware vs asset-blind LOGO ablation 실행기(end-to-end).

각 암(blind/aware/plus)을 **같은 예산**으로 처음부터 재튜닝하고(S·W·게이트·앵커·
블록 고정), 고정 위험 R 을 만든 뒤 앵커 부트스트랩으로 Δrecall(arm−blind) 95% CI 를
낸다. 읽기 쉬운 표를 콘솔에 찍고, 병합(A×B)에서 재사용 가능한 JSON 을 저장한다.

JSON 구조(병합 재사용):
  provenance       : 방법·예산·seed·앵커수·S/W 고정 명시·사용 W.
  arms[<arm>]      : feature_set·weights_dict·cv_recall·score·random_vs_spatial_gap.
                     (병합은 weights_dict 를 읽어 build_risk 로 R 재구성 가능)
  bootstrap_delta  : {arm: {k: {delta_mean, ci_lo, ci_hi}}} (blind 외, vs blind).
  H_A              : 어떤 k 에서든 CI 하한>0 이면 True.

실행:
    .venv/bin/python scripts/run_ablation.py --sets blind aware plus \
        --n-random 2000 --bootstrap 2000 --workers 50 --out outputs/ablation_asset.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import ablation, config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ablation")

# 콘솔 표·JSON 키용 정렬된 top-k 지점.
KS = config.TOPK_FOR_RECALL


def _fmt(x: float) -> str:
    """nan 안전 4자리 포맷."""
    return "  nan " if x != x else f"{x:.4f}"


def _print_table(arms: dict[str, dict], boot: dict[str, dict]) -> None:
    """암 × recall@top-k + Δ vs blind + 95% CI 콘솔 표."""
    head = "피처셋       " + "  ".join(f"@{int(k*100):>2d}%" for k in KS)
    print("\n" + "=" * 78)
    print("실험 A — asset ablation: 공간CV recall@top-k (mean)")
    print("=" * 78)
    print(head)
    print("-" * 78)
    name = {"blind": "asset-blind", "aware": "asset-aware", "plus": "asset-plus "}
    for arm in arms:
        cv = arms[arm]["cv_recall"]
        row = f"{name.get(arm, arm):<12s}" + "  ".join(_fmt(cv[k]) for k in KS)
        print(row)
    # Δ vs blind + CI.
    print("\nΔrecall vs asset-blind  [95% 부트스트랩 CI]")
    print("-" * 78)
    for arm, per_k in boot.items():
        print(f"  {name.get(arm, arm).strip()} − blind:")
        for k in KS:
            d = per_k[k]
            star = " *" if d["ci_lo"] > 0 else ""
            print(f"    @{int(k*100):>2d}%: Δ={d['delta_mean']:+.4f}  "
                  f"CI[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]{star}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 A asset ablation 실행")
    ap.add_argument("--sets", nargs="+", default=["blind", "aware", "plus"],
                    help="재튜닝할 암(별칭). 'blind' 는 부트스트랩 기준이라 권장 포함.")
    ap.add_argument("--n-random", type=int, default=2000,
                    help="암별 LHS 랜덤 후보 수(공정성: 모든 암 동일).")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Δrecall 앵커 부트스트랩 반복 수.")
    ap.add_argument("--ca-passes", type=int, default=3, help="좌표상승 패스 수.")
    ap.add_argument("--quant-step", type=float, default=0.05,
                    help="simplex 양자화 격자.")
    ap.add_argument("--workers", type=int, default=config.N_THREADS,
                    help="병렬 워커 수(per-arm fresh pool).")
    ap.add_argument("--out", type=str, default="outputs/ablation_asset.json",
                    help="결과 JSON 경로.")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    t0 = time.time()

    logger.info("=== 공유 상태 준비(S·W·게이트·앵커·블록 고정) ===")
    shared = ablation.load_ablation_shared()
    n_mapped = int((shared["fire_to_pole"] >= 0).sum())

    # 1) 암별 재튜닝(같은 예산).
    arms: dict[str, dict] = {}
    R_by_arm: dict[str, np.ndarray] = {}
    for arm in args.sets:
        res = ablation.tune_for_featureset(
            shared, arm, n_random=args.n_random, ca_passes=args.ca_passes,
            quant_step=args.quant_step, workers=args.workers, seed=config.SEED)
        arms[arm] = res
        R_by_arm[arm] = ablation.build_risk(shared, res["feature_set"], res["weights"])

    # 2) 낙관편향 진단(암별 random-vs-spatial gap @5%).
    gaps = ablation.random_vs_spatial_by_arm(shared, R_by_arm, k=0.05)
    for arm in arms:
        arms[arm]["random_vs_spatial_gap"] = gaps[arm]

    # 3) Δrecall 부트스트랩 CI(blind 기준; blind 미포함 시 건너뜀).
    boot: dict[str, dict] = {}
    h_a = False
    h_a_detail: dict[str, list[str]] = {}
    if "blind" in R_by_arm:
        boot = ablation.bootstrap_delta_recall(
            shared, R_by_arm, B=args.bootstrap, ks=KS, seed=config.SEED)
        for arm, per_k in boot.items():
            sig = [f"top{int(k*100)}" for k in KS if per_k[k]["ci_lo"] > 0]
            if sig:
                h_a = True
                h_a_detail[arm] = sig
    else:
        logger.warning("'blind' 암 미포함 → Δrecall 부트스트랩 생략")

    # 4) 콘솔 표.
    _print_table(arms, boot)
    if "blind" in R_by_arm:
        verdict = ("H_A 채택(CI 하한>0 at " + str(h_a_detail) + ")") if h_a \
            else "H_A 기각/미확정(어떤 k 에서도 CI 하한 ≤ 0 — 전체로는 차이 없음)"
        print(f"\n판정: {verdict}")

    # 5) JSON 저장(병합 재사용).
    out = {
        "provenance": {
            "experiment": "A — asset-aware vs asset-blind LOGO ablation",
            "method": "per-arm LHS(Dirichlet) random search + coordinate ascent (fair re-tune)",
            "n_random": int(args.n_random),
            "ca_passes": int(args.ca_passes),
            "quant_step": float(args.quant_step),
            "bootstrap_B": int(args.bootstrap),
            "objective": "weighted spatial-block CV recall@top-k (holdout)",
            "objective_weights": {str(k): v for k, v in ablation.OBJ_W.items()},
            "spatial_cv_block_km": config.SPATIAL_CV_BLOCK_KM,
            "topk_for_recall": [float(k) for k in KS],
            "seed": config.SEED,
            "n_positives_mapped": n_mapped,
            "note": ("S(spread)·W(weather)·gate·anchor·blocks held FIXED across all "
                     "arms; only feature_set & re-tuned weights differ (asset enters I only)."),
            "weather_source": "season_weather(source='features')",
            "feature_sets": {a: list(arms[a]["feature_set"]) for a in arms},
        },
        "arms": {
            a: {
                "feature_set": list(arms[a]["feature_set"]),
                "weights_dict": arms[a]["weights_dict"],
                "cv_recall": {f"top{int(k*100)}": round(arms[a]["cv_recall"][k], 5)
                              for k in KS},
                "score": round(arms[a]["score"], 5),
                "random_vs_spatial_gap": {kk: round(vv, 5)
                                          for kk, vv in arms[a]["random_vs_spatial_gap"].items()},
            }
            for a in arms
        },
        "bootstrap_delta_recall": {
            arm: {f"top{int(k*100)}": {kk: round(vv, 5) for kk, vv in per_k[k].items()}
                  for k in KS}
            for arm, per_k in boot.items()
        },
        "H_A": {"holds": bool(h_a), "significant_ks": h_a_detail},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("결과 저장: %s", out_path)
    logger.info("=== 완료 (%.1fs) ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
