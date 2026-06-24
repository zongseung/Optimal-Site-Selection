"""Phase-2 EXPERT_WEIGHTS 튜닝 — 공간블록CV 발화점 recall 목적함수.

설계 근거: Phase-1 의 EXPERT_WEIGHTS 는 EDA 눈대중 초기값이라 공간CV
발화점 recall 이 낮았다(top-5%≈0.077). 여기서는 체제별 7항 가중(landcover 포함)을
**단순체(simplex)** 로 두고, **랜덤/라틴하이퍼큐브 탐색 + 좌표상승**으로
**공간 블록 CV recall@top-k(홀드아웃)** 를 최대화한다.

과적합 방지: 발화점 ~700개뿐 → ① 평가는 항상 공간CV(train recall 미사용),
② 무작위분할 vs 공간분할 격차가 ≈0 인지 확인(낙관편향 감시), ③ 미세과적합
회피를 위해 simplex 를 결정성 그리드 격자(round)로 양자화하고 안정 영역 선택,
④ 결정성(config.SEED).

목적함수: 가중평균 CV recall (top-5% 중심, 1/2/5/10% 함께; 가중치는 OBJ_W).
입력 I 는 gate·feature 가 고정이므로 후보별 평가는
I = Σ_r gate[:,r]·(Σ_j w[r,j]·feat_j) 의 배치 행렬곱으로 빠르게 계산.

실행:
    .venv/bin/python scripts/tune_weights.py [--n-random 4000] [--workers 60]
산출:
    outputs/tuned_weights.json (provenance: 탐색법·후보수·CV점수·전/후)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
    experts,
    io,
    regimes,
    validate,
    weather,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_weights")

# 전문가 7항 순서(config.EXPERT_WEIGHTS 의 키와 동일 순서로 고정; Phase3 landcover 추가).
FEAT_KEYS: tuple[str, ...] = ("forest", "road", "powerline", "fwi", "yanggan", "fuel", "landcover")
# 전문가 항 수(simplex 차원) — FEAT_KEYS 길이에서 단일 진실로 파생.
N_FEATS: int = len(FEAT_KEYS)
# 목적함수 top-k 가중(top-5% 중심). 합=1 로 정규화해 사용.
OBJ_W: dict[float, float] = {0.01: 0.15, 0.02: 0.25, 0.05: 0.40, 0.10: 0.20}

# 워커 프로세스 전역(포크/스폰 공유 상태). 큰 배열 재전송 방지.
_G: dict[str, object] = {}


def _build_shared() -> dict[str, object]:
    """튜닝에 필요한 고정 상태를 1회 계산(피처·게이트·W·S·앵커·블록)."""
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    gate, regime_order = regimes.compute_gate(master)  # (N,3)
    feats = experts.build_ignition_features(master)
    feat_mat = np.stack([feats[k] for k in FEAT_KEYS], axis=1)  # (N, N_FEATS)
    W = weather.season_weather(master, source="features")       # (N,)
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)
    return dict(
        gate=gate, regime_order=list(regime_order), feat_mat=feat_mat,
        W=W, S=S, f2p=f2p, blocks=blocks, pole_xy=pole_xy,
        regime_lbl=gate.argmax(axis=1).astype(np.int32),  # 전주 우세체제 idx
        n_reg=len(regime_order),
        within_regime=False,  # main 에서 --within-regime 시 True 로 덮어씀
    )


def _risk_from_weights(w: np.ndarray, shared: dict[str, object]) -> np.ndarray:
    """가중 행렬 w (3, N_FEATS, 행=체제 simplex) → R(p)=I×S×W.

    I = Σ_r gate[:,r]·expert_r, expert_r = Σ_j w[r,j]·feat_j (행합=1 이므로
    expert ∈[0,1]). gate 행합=1 → I∈[0,1].
    """
    gate = shared["gate"]            # (N,3)
    feat_mat = shared["feat_mat"]    # (N, N_FEATS)
    experts_mat = feat_mat @ w.T     # (N,3) 각 체제 전문가 점수
    I = np.einsum("nr,nr->n", gate, experts_mat)  # (N,)
    I = np.clip(I, 0.0, 1.0)
    R = I * shared["S"] * shared["W"]
    return np.clip(R, 0.0, 1.0)


def _objective(w: np.ndarray, shared: dict[str, object]) -> dict[str, float]:
    """공간CV recall 가중평균(목적값) + 각 top-k recall 반환.

    within_regime=True 면 레짐별 within recall(각 레짐 안에서 top-k% 가 그 레짐
    발화점을 회수)의 **동일가중 평균**을 목적으로 한다 → between-regime 기저율
    지름길을 제거하고 within 판별을 직접 최적화(레짐 크기 무관 동일 비중).
    """
    R = _risk_from_weights(w, shared)
    ks = tuple(OBJ_W.keys())
    wsum = sum(OBJ_W.values())
    if shared.get("within_regime"):
        regime_lbl = shared["regime_lbl"]
        per_k = {k: [] for k in OBJ_W}
        for r in range(int(shared["n_reg"])):
            mask = regime_lbl == r
            cv = validate.spatial_cv_recall(R, shared["f2p"], shared["blocks"],
                                            ks=ks, subset_mask=mask)
            for k in OBJ_W:
                m = cv[k]["mean"]
                per_k[k].append(0.0 if (m != m) else m)  # NaN(발화無) → 0(보수)
        out = {f"recall@{k}": float(np.mean(per_k[k])) for k in OBJ_W}
        score = sum(OBJ_W[k] * np.mean(per_k[k]) for k in OBJ_W) / wsum
        out["score"] = float(score)
        return out
    cv = validate.spatial_cv_recall(R, shared["f2p"], shared["blocks"], ks=ks)
    score = sum(OBJ_W[k] * cv[k]["mean"] for k in OBJ_W) / wsum
    out = {f"recall@{k}": cv[k]["mean"] for k in OBJ_W}
    out["score"] = float(score)
    return out


def _sample_simplex_lhs(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """라틴하이퍼큐브 → 지수변환 → 정규화로 simplex(합=1) n개 표집.

    LHS 로 각 차원 층화 → exponential 변환(Dirichlet 유사)로 simplex 균등 표집에
    가깝게. 결정성은 rng(config.SEED).
    """
    # LHS in [0,1)^dim
    lhs = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        lhs[:, j] = (perm + rng.random(n)) / n
    # 균등(0,1) → 지수: -log(u) → Dirichlet(1,...,1)
    g = -np.log(np.clip(lhs, 1e-12, 1.0))
    return g / g.sum(axis=1, keepdims=True)


def _quantize(w: np.ndarray, step: float = 0.05) -> np.ndarray:
    """simplex 가중을 step 격자로 양자화 후 재정규화(미세과적합 회피·해석성).

    마지막 축(=전문가 N_FEATS 항)이 simplex 라고 보고 그 축으로 정규화한다.
    (3D (M,3,F) / 2D (3,F) / 1D (F,) 모두 last-axis 정규화로 동작.)
    """
    q = np.round(w / step) * step
    q = np.clip(q, 0.0, None)
    s = q.sum(axis=-1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return q / s


# ── 워커 ──────────────────────────────────────────────────────────────────
def _worker_init(shared: dict[str, object]) -> None:
    _G.update(shared)


def _worker_eval(w_flat: np.ndarray) -> float:
    """후보 1개 평가(워커). w_flat=(n_reg*N_FEATS,) → (n_reg, N_FEATS)."""
    n_reg = len(_G["regime_order"])
    w = w_flat.reshape(n_reg, N_FEATS)
    return _objective(w, _G)["score"]


def _eval_batch(cands: np.ndarray, shared: dict[str, object],
                workers: int) -> np.ndarray:
    """후보 배열(M, 3, N_FEATS) 병렬 평가 → score (M,)."""
    flat = cands.reshape(cands.shape[0], -1)
    if workers <= 1:
        _G.update(shared)
        return np.array([_worker_eval(f) for f in flat])
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init,
                             initargs=(shared,)) as ex:
        scores = list(ex.map(_worker_eval, flat, chunksize=4))
    return np.array(scores)


def _coordinate_ascent(w0: np.ndarray, shared: dict[str, object],
                       workers: int, passes: int = 2,
                       step: float = 0.05) -> tuple[np.ndarray, float]:
    """좌표상승: 체제별·항별로 step 만큼 가중을 옮겨(남는 항 비례감소) 개선 탐색.

    각 (체제 r, 항 j) 에 대해 w[r,j] 를 ±step 이동하고 나머지 항을 비례
    재정규화한 후보를 만들어 평가, 최선을 채택. passes 회 반복.
    """
    w = w0.copy()
    best = _objective(w, shared)["score"]
    for p in range(passes):
        improved = False
        cands: list[np.ndarray] = []
        meta: list[tuple[int, int, float]] = []
        for r in range(w.shape[0]):
            for j in range(N_FEATS):
                for delta in (step, -step):
                    cand = w.copy()
                    nv = cand[r, j] + delta
                    if nv < -1e-9 or nv > 1 + 1e-9:
                        continue
                    cand[r, j] = nv
                    cand[r] = np.clip(cand[r], 0.0, None)
                    s = cand[r].sum()
                    if s < 1e-9:
                        continue
                    cand[r] = cand[r] / s
                    cands.append(cand)
                    meta.append((r, j, delta))
        if not cands:
            break
        scores = _eval_batch(np.stack(cands), shared, workers)
        bidx = int(scores.argmax())
        if scores[bidx] > best + 1e-9:
            w = cands[bidx]
            best = float(scores[bidx])
            improved = True
            logger.info("  좌표상승 pass%d: 개선 → score=%.5f (r=%d j=%d d=%+.2f)",
                        p + 1, best, meta[bidx][0], meta[bidx][1], meta[bidx][2])
        if not improved:
            logger.info("  좌표상승 pass%d: 개선 없음 → 종료", p + 1)
            break
    return w, best


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2 EXPERT_WEIGHTS 튜닝")
    ap.add_argument("--n-random", type=int, default=4000,
                    help="랜덤/LHS 후보 수(체제별 simplex 결합)")
    ap.add_argument("--workers", type=int, default=60, help="병렬 워커 수")
    ap.add_argument("--ca-passes", type=int, default=2, help="좌표상승 패스")
    ap.add_argument("--quant-step", type=float, default=0.05,
                    help="simplex 양자화 격자(해석성·미세과적합 회피)")
    ap.add_argument("--within-regime", action="store_true",
                    help="목적함수를 레짐별 within recall 동일가중 평균으로(between 지름길 제거)")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    logger.info("=== 공유 상태 준비 ===")
    t0 = time.time()
    shared = _build_shared()
    shared["within_regime"] = bool(args.within_regime)
    logger.info("준비 %.1fs | 목적함수=%s", time.time() - t0,
                "within-regime(레짐별 동일가중)" if args.within_regime else "global recall")

    # 0) baseline (현 config) 평가
    w_init = np.array([[config.EXPERT_WEIGHTS[r][k] for k in FEAT_KEYS]
                       for r in shared["regime_order"]], dtype=np.float64)
    # config 초기값은 이미 합=1 이지만 안전 정규화.
    w_init = w_init / w_init.sum(axis=1, keepdims=True)
    base = _objective(w_init, shared)
    logger.info("baseline(현 config) score=%.5f recall@5%%=%.4f",
                base["score"], base["recall@0.05"])

    # 1) 랜덤/LHS 탐색 — 체제별 독립 simplex 표집 후 결합
    logger.info("=== 랜덤/LHS 탐색 N=%d ===", args.n_random)
    n = args.n_random
    n_reg = len(shared["regime_order"])
    # 체제별 독립 simplex 표집 후 결합 (n, n_reg, N_FEATS) — 레짐 수 일반화(4레짐 회랑 포함).
    cands = np.stack([_sample_simplex_lhs(rng, n, N_FEATS) for _ in range(n_reg)], axis=1)
    cands = _quantize(cands, step=args.quant_step)
    t1 = time.time()
    scores = _eval_batch(cands, shared, args.workers)
    logger.info("탐색 평가 %.1fs (후보 %d)", time.time() - t1, n)

    order = np.argsort(-scores)
    best_idx = int(order[0])
    w_best = cands[best_idx]
    logger.info("랜덤 최선 score=%.5f (baseline %.5f)", scores[best_idx], base["score"])
    # 상위 후보 안정성: 상위 1% 평균 가중(안정 영역 선택; 미세 단일후보 과신 회피).
    top_n = max(5, n // 100)
    top_w = cands[order[:top_n]].mean(axis=0)
    top_w = top_w / top_w.sum(axis=1, keepdims=True)
    s_topmean = _objective(top_w, shared)["score"]
    logger.info("상위 %d 평균가중 score=%.5f", top_n, s_topmean)
    # 좌표상승 시작점 후보: 랜덤최선·상위평균·현 config baseline 중 최고점.
    # baseline 도 후보에 넣어 "좋은 시작점 주변"을 명시적으로 더 탐색한다
    # (랜덤탐색이 baseline 을 못 넘어도 그 이웃의 개선 가능성을 좌표상승이 흡수).
    cand_starts = [
        (w_best, float(scores[best_idx]), "random_best"),
        (top_w, float(s_topmean), "top_mean"),
        (w_init.copy(), float(base["score"]), "config_baseline"),
    ]
    start_w, start_score, start_tag = max(cand_starts, key=lambda t: t[1])
    logger.info("좌표상승 시작점: %s (score=%.5f)", start_tag, start_score)

    # 2) 좌표상승 마무리
    logger.info("=== 좌표상승 (passes=%d, step=%.2f) ===", args.ca_passes, args.quant_step)
    w_ca, score_ca = _coordinate_ascent(
        start_w, shared, args.workers, passes=args.ca_passes, step=args.quant_step)
    # 양자화본도 평가해 더 나은(또는 동률이면 더 단순한 격자) 쪽 채택.
    w_q = _quantize(w_ca[None], step=args.quant_step)[0]
    score_q = _objective(w_q, shared)["score"]
    if score_q >= score_ca - 1e-9:
        w_tuned, final_score = w_q, score_q
    else:
        w_tuned, final_score = w_ca, score_ca
    # baseline-keep 가드(정직): 탐색·좌표상승이 현 config baseline 을 넘지 못하면
    # baseline 을 그대로 채택한다(점수를 해치지 않는다는 원칙; 좋은 시작점 존중).
    if base["score"] >= final_score - 1e-9:
        logger.info("튜닝 결과(%.5f)가 baseline(%.5f) 미초과 → baseline 유지",
                    final_score, base["score"])
        w_tuned, final_score = w_init.copy(), base["score"]
    final = _objective(w_tuned, shared)
    logger.info("최종 튜닝 score=%.5f (ca=%.5f quant=%.5f)",
                final["score"], score_ca, score_q)

    # 3) 낙관편향 감시: 무작위 vs 공간 격차(튜닝 가중에서)
    R_tuned = _risk_from_weights(w_tuned, shared)
    gap = validate.random_vs_spatial_gap(R_tuned, shared["f2p"], shared["blocks"], k=0.05)
    R_base = _risk_from_weights(w_init, shared)
    gap_base = validate.random_vs_spatial_gap(R_base, shared["f2p"], shared["blocks"], k=0.05)
    logger.info("격차(random-spatial)@5%%: tuned=%.4f base=%.4f", gap["gap"], gap_base["gap"])

    # 전·후 전체 top-k CV recall
    def _full_cv(w):
        R = _risk_from_weights(w, shared)
        cv = validate.spatial_cv_recall(R, shared["f2p"], shared["blocks"])
        return {f"top{int(k*100)}": round(float(cv[k]["mean"]), 4) for k in config.TOPK_FOR_RECALL}

    cv_before = _full_cv(w_init)
    cv_after = _full_cv(w_tuned)
    logger.info("CV recall before=%s after=%s", cv_before, cv_after)

    # 4) provenance 기록
    regime_order = shared["regime_order"]
    tuned_dict = {
        r: {k: round(float(w_tuned[i, j]), 4) for j, k in enumerate(FEAT_KEYS)}
        for i, r in enumerate(regime_order)
    }
    before_dict = {
        r: {k: float(config.EXPERT_WEIGHTS[r][k]) for k in FEAT_KEYS}
        for r in regime_order
    }
    out = {
        "provenance": {
            "method": "LHS(Dirichlet) random search + coordinate ascent",
            "n_random_candidates": int(args.n_random),
            "ca_passes": int(args.ca_passes),
            "quant_step": float(args.quant_step),
            "objective": "weighted spatial-block CV recall@top-k (holdout)",
            "objective_weights": {str(k): v for k, v in OBJ_W.items()},
            "spatial_cv_block_km": config.SPATIAL_CV_BLOCK_KM,
            "n_positives_mapped": int((shared["f2p"] >= 0).sum()),
            "seed": config.SEED,
        },
        "feat_keys": list(FEAT_KEYS),
        "regime_order": regime_order,
        "score_before": round(base["score"], 5),
        "score_after": round(final["score"], 5),
        "cv_recall_before": cv_before,
        "cv_recall_after": cv_after,
        "random_vs_spatial_gap_after": {k: round(float(v), 4) for k, v in gap.items()},
        "random_vs_spatial_gap_before": {k: round(float(v), 4) for k, v in gap_base.items()},
        "weights_before": before_dict,
        "weights_tuned": tuned_dict,
    }
    config.OUT.mkdir(parents=True, exist_ok=True)
    fname = "tuned_weights_within.json" if args.within_regime else "tuned_weights.json"
    path = config.OUT / fname
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("튜닝 가중 기록: %s", path)
    logger.info("=== 완료 === before=%.5f after=%.5f", base["score"], final["score"])
    print(json.dumps(tuned_dict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
