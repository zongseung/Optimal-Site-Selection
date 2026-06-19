"""A×B 병합 — asset-aware vs asset-blind 의 원인별(설비 vs 인간) recall 교차표.

기획서_ablation_설비원인검증 §5(핵심 분석): asset 피처의 이득이 **설비원인 화재에
집중**되고 인간발화엔 없으면, 그건 우연이 아니라 메커니즘이 맞다는 증거다.

실험 A(scripts/run_ablation.py, outputs/ablation_asset.json)의 암별 튜닝가중을 reload 해
캐노니컬 빌더 ablation.build_risk 로 R_blind·R_aware 를 재구성하고, 실험 B(pfire/
fire_cause.py)의 원인 분류·recall_by_cause 로 원인 그룹별 recall 을 잰 뒤 Δ(aware−blind)
를 교차표로 만든다. 설비 grid n≈18 로 희소 → 점추정 + 그룹 부트스트랩(방향만).

실행:
    .venv/bin/python scripts/run_ablation_by_cause.py \
        --ablation-json outputs/ablation_asset.json \
        --out outputs/ablation_by_cause.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import ablation, config, fire_cause, io  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ablation_by_cause")

ARM_FEATURESET = {
    "blind": config.FEATURE_SET_ASSET_BLIND,
    "aware": config.FEATURE_SET_ASSET_AWARE,
    "plus": config.FEATURE_SET_ASSET_PLUS,
}


def _weights_array(weights_dict: dict, feature_set: tuple[str, ...],
                   regime_order: list[str]) -> np.ndarray:
    """JSON weights_dict({regime:{feat:w}}) → (3, d) 가중행렬(regime_order·feature_set 순)."""
    w = np.zeros((len(regime_order), len(feature_set)), dtype=np.float64)
    for i, r in enumerate(regime_order):
        row = weights_dict[r]
        for j, k in enumerate(feature_set):
            w[i, j] = float(row.get(k, 0.0))
    # 행-simplex 안전 정규화(반올림 잔차 보정).
    s = w.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return w / s


def _bootstrap_delta_by_cause(
    R_blind: np.ndarray, R_aware: np.ndarray, pole_xy: np.ndarray,
    fire_xy: np.ndarray, cause: np.ndarray, grp: str,
    ks: tuple[float, ...], radius_km: float, B: int, seed: int,
) -> dict[str, dict[str, float]]:
    """원인 그룹 grp 의 Δrecall(aware−blind) 부트스트랩 95% CI.

    그룹 발화점을 전주에 1회 매핑한 뒤, 매핑된 양성 전주를 복원추출(B회)해 두 R 의
    전역순위로 recall 차를 다시 계산한다(재정렬 없음). n 희소 → 방향만.
    """
    from pfire.calibrate import assign_poles_to_fires

    sel = np.asarray([str(c) == grp for c in cause], dtype=bool)
    f2p = assign_poles_to_fires(pole_xy, fire_xy[sel], radius_km=radius_km)
    pos = np.unique(f2p[f2p >= 0])
    rank_b = np.argsort(np.argsort(-R_blind))   # 0=최고위험
    rank_a = np.argsort(np.argsort(-R_aware))
    n = R_blind.shape[0]
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        cut = max(1, int(np.ceil(k * n)))
        if pos.size == 0:
            out[f"top{int(k*100)}"] = dict(delta=float("nan"),
                                           ci_lo=float("nan"), ci_hi=float("nan"))
            continue
        deltas = np.empty(B)
        for b in range(B):
            samp = rng.choice(pos, size=pos.size, replace=True)
            ra = float((rank_a[samp] < cut).mean())
            rb = float((rank_b[samp] < cut).mean())
            deltas[b] = ra - rb
        base = float((rank_a[pos] < cut).mean() - (rank_b[pos] < cut).mean())
        out[f"top{int(k*100)}"] = dict(
            delta=base,
            ci_lo=float(np.percentile(deltas, 2.5)),
            ci_hi=float(np.percentile(deltas, 97.5)),
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A×B 병합: 원인별 asset ablation")
    ap.add_argument("--ablation-json", default="outputs/ablation_asset.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--radius-km", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/ablation_by_cause.json")
    args = ap.parse_args()

    logger.info("=== 공유 상태 로드 + 암별 가중 reload ===")
    shared = ablation.load_ablation_shared()
    regime_order = list(shared["regime_order"])
    abl = json.load(open(args.ablation_json, encoding="utf-8"))

    pole_xy = shared["pole_xy"]
    positives = io.load_positives()
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    resn = positives["resn"].to_list()
    cause = fire_cause.classify_cause(resn)

    ks = config.TOPK_FOR_RECALL
    R = {}
    for arm in ("blind", "aware"):
        fs = ARM_FEATURESET[arm]
        w = _weights_array(abl["arms"][arm]["weights_dict"], fs, regime_order)
        R[arm] = ablation.build_risk(shared, fs, w)
        logger.info("R[%s] 재구성: mean=%.4g p99=%.4g", arm,
                    float(R[arm].mean()), float(np.quantile(R[arm], 0.99)))

    logger.info("=== 원인별 recall (blind vs aware) ===")
    rec_blind = fire_cause.recall_by_cause(R["blind"], pole_xy, fire_xy, cause,
                                           ks=ks, radius_km=args.radius_km)
    rec_aware = fire_cause.recall_by_cause(R["aware"], pole_xy, fire_xy, cause,
                                           ks=ks, radius_km=args.radius_km)

    causes = ("grid_electric", "work_spark", "natural", "human", "unknown")
    cross: dict[str, dict] = {}
    logger.info("%-14s | %-6s | %s", "cause", "n_map",
                "  ".join(f"Δ@{int(k*100)}%(blind→aware)" for k in ks))
    for grp in causes:
        rb = rec_blind.get(grp, {})
        ra = rec_aware.get(grp, {})
        n_map = int(rb.get("n_mapped", 0))
        row = {"n_fires": int(rb.get("n_fires", 0)), "n_mapped": n_map,
               "blind": {}, "aware": {}, "delta": {}}
        cells = []
        for k in ks:
            b = float(rb.get(k, float("nan")))
            a = float(ra.get(k, float("nan")))
            row["blind"][f"top{int(k*100)}"] = b
            row["aware"][f"top{int(k*100)}"] = a
            row["delta"][f"top{int(k*100)}"] = a - b
            cells.append(f"{b:.3f}→{a:.3f}(Δ{a-b:+.3f})")
        cross[grp] = row
        logger.info("%-14s | n=%-4d | %s", grp, n_map, "  ".join(cells))

    # 핵심 가설 검정: 설비(grid) Δ 가 인간(human) Δ 보다 큰가 + grid Δ 부트스트랩.
    logger.info("=== 핵심: grid_electric Δrecall 부트스트랩 95%% CI (n 희소→방향만) ===")
    grid_boot = _bootstrap_delta_by_cause(
        R["blind"], R["aware"], pole_xy, fire_xy, cause, "grid_electric",
        ks, args.radius_km, args.bootstrap, config.SEED)
    human_boot = _bootstrap_delta_by_cause(
        R["blind"], R["aware"], pole_xy, fire_xy, cause, "human",
        ks, args.radius_km, args.bootstrap, config.SEED)
    for k in ks:
        kk = f"top{int(k*100)}"
        g = grid_boot[kk]
        h = human_boot[kk]
        interaction = g["delta"] - h["delta"]
        logger.info("  %-5s | grid Δ=%+.3f [%+.3f,%+.3f] | human Δ=%+.3f | 상호작용(grid−human)=%+.3f",
                    kk, g["delta"], g["ci_lo"], g["ci_hi"], h["delta"], interaction)

    artifact = {
        "provenance": {
            "experiment": "A×B merge — asset ablation × fire cause",
            "ablation_json": args.ablation_json,
            "bootstrap_B": args.bootstrap,
            "radius_km": args.radius_km,
            "note": "pooled recall(상대비교 전용) · grid n≈18 희소 → 방향/메커니즘만, "
                    "통계적 유의 주장 금지. S·W 고정, asset 은 I 에만.",
        },
        "cross_table": cross,
        "grid_delta_bootstrap": grid_boot,
        "human_delta_bootstrap": human_boot,
        "interaction_grid_minus_human": {
            f"top{int(k*100)}": grid_boot[f"top{int(k*100)}"]["delta"]
            - human_boot[f"top{int(k*100)}"]["delta"] for k in ks
        },
    }
    config.OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    logger.info("저장: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
