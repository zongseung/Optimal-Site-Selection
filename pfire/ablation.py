"""실험 A — asset-aware vs asset-blind LOGO ablation 핵심 로직.

설계 근거(claudedocs/기획서_ablation_설비원인검증_20260619.md §3): 우리 novelty
("전주=전력설비 자산을 발화원으로 모델")이 실제 발화점 회수에 도움이 되는지를
공정한 Leave-One-Group-Out(LOGO) ablation 으로 측정한다. 핵심은 **공정성**:

  - asset 피처(powerline·substation)는 발화 항 I(p)에만 들어간다. 따라서 S(확산)·
    W(기상)·게이트·앵커·블록을 **모든 암(arm)에 동일·고정**하고, 오직 feature_set 과
    그에 맞춰 처음부터 재튜닝한 가중 w 만 다르게 둔다 → asset 의 순효과가 깨끗이 분리.
  - 각 암은 **같은 예산**(n_random·ca_passes·quant_step·seed)으로 재튜닝한다. 가중만
    0 으로 두는 것은 "asset-blind 가 최선을 다하지 못한" 불공정 비교라 금지.

캐노니컬 위험 빌더(build_risk)는 후속 A×B 병합에서 재사용되므로 자기완결·결정적으로
둔다. 튜닝은 scripts/tune_weights.py 의 LHS(Dirichlet)+좌표상승 패턴을 차원에
무관하게 재사용하되, 병렬 평가는 **자기완결 워커**(per-arm 공유상태 + n_feats 를
initargs 로 직접 전달)로 구현한다 — tune_weights 의 모듈전역 N_FEATS=7 을 몽키패치하면
스폰된 자식이 기본 7 로 재import 해 d≠7 후보를 오변형(misreshape)하는 함정을 피한다.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from pfire import calibrate, config, experts, io, regimes, validate, weather

# tune_weights 의 차원무관 헬퍼만 재사용(dim 을 인자로 받음 / last-axis 정규화).
from scripts.tune_weights import _quantize, _sample_simplex_lhs

logger = logging.getLogger(__name__)

# feature_set 별칭 → experts.build_ignition_features 키 튜플(config 단일진실 참조).
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "blind": config.FEATURE_SET_ASSET_BLIND,
    "aware": config.FEATURE_SET_ASSET_AWARE,
    "plus": config.FEATURE_SET_ASSET_PLUS,
}

# 목적함수 top-k 가중(top-5% 중심; tune_weights.OBJ_W 와 동일). 합으로 정규화해 사용.
OBJ_W: dict[float, float] = {0.01: 0.15, 0.02: 0.25, 0.05: 0.40, 0.10: 0.20}

# 워커 프로세스 전역(per-arm 공유상태 보관). 큰 배열 재전송 방지.
_G: dict[str, object] = {}


# ── 공유 상태 ────────────────────────────────────────────────────────────────
def load_ablation_shared() -> dict:
    """모든 암에 공통·고정인 상태를 1회 계산한다.

    피처 dict·게이트·S·W·앵커(fire_to_pole)·공간블록을 만들어 둔다. asset 의 순효과를
    분리하기 위해 S·W·게이트·앵커·블록은 여기서 고정되고 암마다 바뀌지 않는다.

    Returns
    -------
    dict
        master, gate(N,3), regime_order(list[str]), feats(dict[str,(N,)]),
        S(N,), W(N,), pole_xy(N,2), fire_to_pole(M,), blocks(N,).
    """
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    gate, regime_order = regimes.compute_gate(master)  # (N,3)
    feats = experts.build_ignition_features(master)    # dict[str, (N,)]
    # asset 은 I 에만 들어가므로 W·S 는 asset 무관 — 모든 암 고정.
    W = weather.season_weather(master, source="features")  # (N,)
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_to_pole = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)
    logger.info("ablation shared 준비: N=%d 앵커매핑=%d 피처키=%s",
                master.height, int((fire_to_pole >= 0).sum()), sorted(feats.keys()))
    return dict(
        master=master, gate=gate, regime_order=list(regime_order), feats=feats,
        S=S, W=W, pole_xy=pole_xy, fire_to_pole=fire_to_pole, blocks=blocks,
    )


def _resolve_feature_set(feature_set: str | tuple[str, ...],
                         feats: dict[str, np.ndarray]) -> tuple[str, ...]:
    """별칭('blind'|'aware'|'plus') 또는 키 튜플 → 사용 가능한 키 튜플.

    합성 master 가 dist_to_substation 을 안 가져 'substation' 피처가 없을 수 있다.
    그 경우 'plus' 는 'aware' 로 폴백한다(요청 키가 없으면 KeyError 로 명시).
    """
    if isinstance(feature_set, str):
        if feature_set not in FEATURE_SETS:
            raise KeyError(f"알 수 없는 feature_set 별칭: {feature_set!r}")
        keys = FEATURE_SETS[feature_set]
    else:
        keys = tuple(feature_set)
    missing = [k for k in keys if k not in feats]
    if missing:
        # 'substation' 만 없는 합성 입력은 사용 가능한 키로 정렬(하위호환). 그 외 결측은 오류.
        if set(missing) <= {"substation"}:
            keys = tuple(k for k in keys if k in feats)
            logger.warning("피처 %s 부재 → feature_set 에서 제외(합성입력 폴백)", missing)
        else:
            raise KeyError(f"feature_set 에 없는 피처: {missing}")
    return keys


# ── 캐노니컬 위험 빌더(A×B 재사용) ────────────────────────────────────────────
def build_risk(shared: dict, feature_set: str | tuple[str, ...],
               weights: np.ndarray) -> np.ndarray:
    """캐노니컬 위험 R = clip(I·S·W, 0, 1). **A×B 병합에서 재사용되는 단일 빌더.**

    asset 피처는 I 에만 들어가므로 S·W 는 모든 암에서 동일(shared 에서 고정 주입).
    feature_set 과 가중 weights 만 암마다 다르다.

    빌드 식(스펙 고정)::

        feat_mat   = stack([feats[k] for k in feature_set], axis=1)   # (N, d)
        experts_mat = feat_mat @ w.T                                  # w (3,d) 행-simplex
        I = clip(einsum("nr,nr->n", gate, experts_mat), 0, 1)
        R = clip(I * S * W, 0, 1)

    Parameters
    ----------
    shared : dict
        load_ablation_shared 출력(gate, feats, S, W 사용).
    feature_set : str or tuple[str]
        'blind'|'aware'|'plus' 별칭 또는 experts 피처키 튜플(순서=가중 열 순서).
    weights : numpy.ndarray, shape (3, d)
        체제별 행-simplex 가중(행합=1, d=feature_set 길이).

    Returns
    -------
    numpy.ndarray, shape (N,)
        위험 R ∈ [0,1].

    Raises
    ------
    ValueError
        weights 열 수가 feature_set 길이와 다를 때.
    """
    feats = shared["feats"]
    keys = _resolve_feature_set(feature_set, feats)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != 3 or w.shape[1] != len(keys):
        raise ValueError(
            f"weights 형상 {w.shape} != (3, {len(keys)}) (feature_set={keys})")
    feat_mat = np.stack([feats[k] for k in keys], axis=1)  # (N, d)
    experts_mat = feat_mat @ w.T                           # (N, 3)
    I = np.clip(np.einsum("nr,nr->n", shared["gate"], experts_mat), 0.0, 1.0)
    R = np.clip(I * shared["S"] * shared["W"], 0.0, 1.0)
    return R


# ── 자기완결 병렬 평가(차원무관) ──────────────────────────────────────────────
def _objective_from_R(R: np.ndarray, shared: dict) -> dict[str, float]:
    """위험 R 의 공간CV recall 가중평균(목적값) + 각 top-k recall."""
    cv = validate.spatial_cv_recall(R, shared["fire_to_pole"], shared["blocks"],
                                    ks=tuple(OBJ_W.keys()))
    wsum = sum(OBJ_W.values())
    score = sum(OBJ_W[k] * cv[k]["mean"] for k in OBJ_W) / wsum
    out = {f"recall@{k}": cv[k]["mean"] for k in OBJ_W}
    out["score"] = float(score)
    return out


def _worker_init(state: dict) -> None:
    """워커 전역에 per-arm 공유상태(feat_mat·gate·S·W·앵커·블록·n_feats) 적재."""
    _G.clear()
    _G.update(state)


def _worker_eval(w_flat: np.ndarray) -> float:
    """후보 1개 평가(워커). 차원은 **공유상태의 n_feats**로 재구성(모듈전역 금지).

    핵심 함정 회피: candidate 를 (3, n_feats) 로 reshape 할 때 tune_weights 의
    모듈전역 N_FEATS(=7) 가 아니라 _G["n_feats"] 를 쓴다 — 스폰된 자식이 기본 7 로
    misreshape 하는 것을 막는다.
    """
    d = int(_G["n_feats"])
    w = np.asarray(w_flat, dtype=np.float64).reshape(3, d)
    feat_mat = _G["feat_mat"]          # (N, d)
    experts_mat = feat_mat @ w.T       # (N, 3)
    I = np.clip(np.einsum("nr,nr->n", _G["gate"], experts_mat), 0.0, 1.0)
    R = np.clip(I * _G["S"] * _G["W"], 0.0, 1.0)
    return _objective_from_R(R, _G)["score"]


def _make_worker_state(shared: dict, feat_mat: np.ndarray) -> dict:
    """워커로 보낼 per-arm 경량 상태(필요 배열 + n_feats)."""
    return dict(
        feat_mat=feat_mat, gate=shared["gate"], S=shared["S"], W=shared["W"],
        fire_to_pole=shared["fire_to_pole"], blocks=shared["blocks"],
        n_feats=int(feat_mat.shape[1]),
    )


def _eval_batch(cands: np.ndarray, state: dict, workers: int) -> np.ndarray:
    """후보 배열 (M, 3, d) 병렬 평가 → score (M,). per-arm fresh pool.

    workers<=1 이면 동일 프로세스에서 직렬 평가(테스트·디버그용).
    """
    flat = cands.reshape(cands.shape[0], -1)
    if workers <= 1:
        _worker_init(state)
        return np.array([_worker_eval(f) for f in flat])
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init,
                             initargs=(state,)) as ex:
        scores = list(ex.map(_worker_eval, flat, chunksize=4))
    return np.array(scores)


def _coordinate_ascent(w0: np.ndarray, state: dict, workers: int,
                       passes: int, step: float) -> tuple[np.ndarray, float]:
    """좌표상승: (체제 r, 항 j) 의 가중을 ±step 옮기고 비례재정규화한 후보를 평가.

    각 패스마다 모든 (r,j,±step) 후보를 한 번에 평가해 최선을 채택, 개선 없으면 종료.
    차원 d 는 state["n_feats"] 에서 파생(모듈전역 미사용).
    """
    d = int(state["n_feats"])
    w = w0.copy()
    best = float(_eval_batch(w[None], state, 1)[0])
    for p in range(passes):
        cands: list[np.ndarray] = []
        meta: list[tuple[int, int, float]] = []
        for r in range(3):
            for j in range(d):
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
        scores = _eval_batch(np.stack(cands), state, workers)
        bidx = int(scores.argmax())
        if scores[bidx] > best + 1e-9:
            w = cands[bidx]
            best = float(scores[bidx])
            logger.info("  좌표상승 pass%d: 개선 → score=%.5f (r=%d j=%d d=%+.2f)",
                        p + 1, best, meta[bidx][0], meta[bidx][1], meta[bidx][2])
        else:
            logger.info("  좌표상승 pass%d: 개선 없음 → 종료", p + 1)
            break
    return w, best


# ── 암(arm) 튜닝 ──────────────────────────────────────────────────────────────
def tune_for_featureset(
    shared: dict,
    feature_set: str | tuple[str, ...],
    n_random: int = 2000,
    ca_passes: int = 3,
    quant_step: float = 0.05,
    workers: int = config.N_THREADS,
    seed: int = config.SEED,
) -> dict:
    """주어진 feature_set 만으로 체제별 가중을 공간CV recall 목적함수로 재튜닝.

    LHS(Dirichlet) 랜덤탐색 + 좌표상승. 모든 암에 동일 예산(n_random·ca_passes·
    quant_step·seed)을 써 공정성을 보장한다. S·W·게이트·앵커·블록은 shared 고정.

    Parameters
    ----------
    shared : dict
        load_ablation_shared 출력.
    feature_set : str or tuple[str]
        'blind'|'aware'|'plus' 또는 피처키 튜플.
    n_random : int
        LHS 랜덤 후보 수(체제별 simplex 결합).
    ca_passes : int
        좌표상승 패스 수.
    quant_step : float
        simplex 양자화 격자(해석성·미세과적합 회피).
    workers : int
        병렬 워커 수(per-arm fresh pool).
    seed : int
        결정성 시드.

    Returns
    -------
    dict
        weights : numpy.ndarray (3, d) — 튜닝된 행-simplex 가중.
        weights_dict : {regime: {feat: w}} — 사람이 읽는 형태.
        cv_recall : {k: mean} — config.TOPK_FOR_RECALL 각 지점 공간CV recall.
        score : float — 목적값(OBJ_W 가중 recall).
        feature_set : tuple[str] — 실제 사용 키(폴백 반영).
    """
    feats = shared["feats"]
    keys = _resolve_feature_set(feature_set, feats)
    d = len(keys)
    feat_mat = np.stack([feats[k] for k in keys], axis=1).astype(np.float64)  # (N,d)
    state = _make_worker_state(shared, feat_mat)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    logger.info("[arm %s] 재튜닝 시작 d=%d keys=%s n_random=%d",
                feature_set, d, keys, n_random)

    # 1) LHS(Dirichlet) 랜덤탐색 — 체제별 독립 simplex 표집 후 결합.
    s_yd = _sample_simplex_lhs(rng, n_random, d)
    s_ys = _sample_simplex_lhs(rng, n_random, d)
    s_mt = _sample_simplex_lhs(rng, n_random, d)
    cands = np.stack([s_yd, s_ys, s_mt], axis=1)   # (n,3,d)
    cands = _quantize(cands, step=quant_step)
    scores = _eval_batch(cands, state, workers)
    order = np.argsort(-scores)
    best_idx = int(order[0])
    w_best = cands[best_idx]
    # 상위 1% 평균가중(안정 영역; 단일후보 과신 회피).
    top_n = max(5, n_random // 100)
    top_w = cands[order[:top_n]].mean(axis=0)
    top_w = top_w / top_w.sum(axis=1, keepdims=True)
    s_topmean = float(_eval_batch(top_w[None], state, 1)[0])
    logger.info("[arm %s] 랜덤최선=%.5f 상위%d평균=%.5f (%.1fs)",
                feature_set, scores[best_idx], top_n, s_topmean, time.time() - t0)

    # 좌표상승 시작점: 랜덤최선 vs 상위평균 중 더 좋은 쪽.
    if s_topmean > scores[best_idx]:
        start_w, start_score = top_w, s_topmean
    else:
        start_w, start_score = w_best, float(scores[best_idx])

    # 2) 좌표상승 마무리.
    w_ca, score_ca = _coordinate_ascent(start_w, state, workers,
                                        passes=ca_passes, step=quant_step)
    # 양자화본도 평가해 더 낫거나(또는 동률이면 더 단순한 격자) 쪽 채택.
    w_q = _quantize(w_ca[None], step=quant_step)[0]
    score_q = float(_eval_batch(w_q[None], state, 1)[0])
    if score_q >= score_ca - 1e-9:
        w_tuned, final_score = w_q, score_q
    else:
        w_tuned, final_score = w_ca, score_ca

    # 3) 최종 가중의 전체 top-k 공간CV recall.
    R = build_risk(shared, keys, w_tuned)
    cv = validate.spatial_cv_recall(R, shared["fire_to_pole"], shared["blocks"],
                                    ks=config.TOPK_FOR_RECALL)
    cv_recall = {k: float(cv[k]["mean"]) for k in config.TOPK_FOR_RECALL}

    regime_order = shared["regime_order"]
    weights_dict = {
        r: {keys[j]: round(float(w_tuned[i, j]), 4) for j in range(d)}
        for i, r in enumerate(regime_order)
    }
    logger.info("[arm %s] 완료 score=%.5f cv=%s (%.1fs)",
                feature_set, final_score,
                {f"top{int(k*100)}": round(v, 4) for k, v in cv_recall.items()},
                time.time() - t0)
    return dict(
        weights=w_tuned, weights_dict=weights_dict, cv_recall=cv_recall,
        score=float(final_score), feature_set=tuple(keys),
    )


# ── Δrecall 부트스트랩 CI ─────────────────────────────────────────────────────
def _ranks_desc(R: np.ndarray) -> np.ndarray:
    """위험 R 내림차순 0-기반 순위(0=최고위험). 부트스트랩 전 1회만 계산."""
    order = np.argsort(-R)
    rank = np.empty(R.shape[0], dtype=np.int64)
    rank[order] = np.arange(R.shape[0])
    return rank


def _recall_from_ranks(rank: np.ndarray, pos_idx: np.ndarray,
                       n: int, ks: tuple[float, ...]) -> dict[float, float]:
    """미리 계산한 순위로 양성집합 recall@top-k (argsort 재계산 없음)."""
    res: dict[float, float] = {}
    pr = rank[pos_idx]
    for k in ks:
        cut = max(1, int(np.ceil(k * n)))
        res[k] = float((pr < cut).mean())
    return res


def bootstrap_delta_recall(
    shared: dict,
    R_by_arm: dict[str, np.ndarray],
    B: int = 2000,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    seed: int = config.SEED,
) -> dict[str, dict[float, dict[str, float]]]:
    """앵커 부트스트랩으로 Δrecall(arm − blind) 95% 백분위 CI.

    매핑된 고유 앵커 전주를 복원추출(B회)해, 각 리샘플에서 암별 recall@top-k 와
    Δ = recall_arm − recall_blind 를 재계산한다. **최적화**: 각 암의 위험 순위를
    1회만 계산해 두고, 리샘플은 양성 전주 집합만 바꾸므로 1.39M argsort 를
    반복하지 않는다. 'blind' 암은 기준이라 결과 dict 에서 제외한다.

    Parameters
    ----------
    shared : dict
        load_ablation_shared 출력(fire_to_pole 사용).
    R_by_arm : dict[str, numpy.ndarray]
        암이름 → 고정 위험 R(N,). 'blind' 키 필수(기준).
    B : int
        부트스트랩 반복 수.
    ks : tuple[float]
        top-k 지점.
    seed : int
        결정성 시드(np.random.default_rng).

    Returns
    -------
    dict[str, dict]
        {arm: {k: {delta_mean, ci_lo, ci_hi}}} — blind 외 각 암, vs blind.

    Raises
    ------
    KeyError
        R_by_arm 에 'blind' 가 없을 때.
    """
    if "blind" not in R_by_arm:
        raise KeyError("bootstrap_delta_recall: R_by_arm 에 'blind' 기준 암 필요")
    n = next(iter(R_by_arm.values())).shape[0]
    f2p = shared["fire_to_pole"]
    pos_unique = np.unique(f2p[f2p >= 0])  # 고유 매핑 앵커 전주.
    m = pos_unique.shape[0]

    # 암별 순위 1회 사전계산.
    ranks = {arm: _ranks_desc(R) for arm, R in R_by_arm.items()}
    other_arms = [a for a in R_by_arm if a != "blind"]

    rng = np.random.default_rng(seed)
    # 리샘플마다 Δ 누적: arm → k → list[Δ].
    deltas: dict[str, dict[float, list[float]]] = {
        a: {k: [] for k in ks} for a in other_arms
    }
    for _ in range(B):
        samp = pos_unique[rng.integers(0, m, size=m)]  # 복원추출(중복 허용).
        rec_blind = _recall_from_ranks(ranks["blind"], samp, n, ks)
        for a in other_arms:
            rec_a = _recall_from_ranks(ranks[a], samp, n, ks)
            for k in ks:
                deltas[a][k].append(rec_a[k] - rec_blind[k])

    out: dict[str, dict[float, dict[str, float]]] = {}
    for a in other_arms:
        out[a] = {}
        for k in ks:
            arr = np.asarray(deltas[a][k], dtype=np.float64)
            out[a][k] = dict(
                delta_mean=float(arr.mean()),
                ci_lo=float(np.percentile(arr, 2.5)),
                ci_hi=float(np.percentile(arr, 97.5)),
            )
    return out


def random_vs_spatial_by_arm(
    shared: dict,
    R_by_arm: dict[str, np.ndarray],
    k: float = 0.05,
) -> dict[str, dict[str, float]]:
    """암별 무작위-공간 분할 recall 격차(낙관편향 진단). gap≈0/음수면 건전."""
    out: dict[str, dict[str, float]] = {}
    for arm, R in R_by_arm.items():
        g = validate.random_vs_spatial_gap(R, shared["fire_to_pole"],
                                           shared["blocks"], k=k)
        out[arm] = {kk: float(vv) for kk, vv in g.items()}
    return out
