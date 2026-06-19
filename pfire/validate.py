"""검증 — 공간 블록 교차검증 + 발화점 recall + sanity.

설계 근거: 산불 위험은 공간 자기상관이 강하다. 무작위 분할은 인접 전주가
train/test 에 함께 들어가 낙관 편향을 준다. config.SPATIAL_CV_BLOCK_KM 격자로
GroupKFold(공간 블록)하여 "다른 지역에서도 발화점 상위에 드는가"를 본다.
sanity: 2019 고성(≈128.50,38.21) 주변·설비/전기원인 화재가 고위험 상위인지 확인.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config, geo, hierarchy
from pfire.calibrate import recall_at_topk

logger = logging.getLogger(__name__)


def _fold_assignment(blocks: np.ndarray, n_folds: int) -> np.ndarray:
    """공간 블록 → 폴드 id 결정적 배정(spatial_cv_recall 과 동일 규칙).

    config.SEED 로 블록을 셔플 후 라운드로빈으로 폴드에 배정한다. spatial_cv_recall
    과 누수안전 배율 CV 가 **동일 폴드 분할**을 쓰도록 한 곳에서 만든다.

    Parameters
    ----------
    blocks : numpy.ndarray, shape (N,)
        전주별 공간 블록 id.
    n_folds : int
        폴드 수.

    Returns
    -------
    numpy.ndarray, shape (N,)
        전주별 폴드 id(0..n_folds-1).
    """
    uniq = np.unique(blocks)
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(uniq)
    fold_of_block = {b: i % n_folds for i, b in enumerate(uniq)}
    return np.array([fold_of_block[b] for b in blocks])


def spatial_blocks(
    pole_xy: np.ndarray, block_km: float = config.SPATIAL_CV_BLOCK_KM
) -> np.ndarray:
    """전주를 공간 격자 블록 id 로 라벨링(평면근사 km).

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    block_km : float
        블록 한 변(km).

    Returns
    -------
    numpy.ndarray, shape (N,)
        블록 id(int).
    """
    pole_km = geo.lonlat_to_km(pole_xy)
    x_km, y_km = pole_km[:, 0], pole_km[:, 1]
    ix = np.floor((x_km - x_km.min()) / block_km).astype(np.int64)
    iy = np.floor((y_km - y_km.min()) / block_km).astype(np.int64)
    nx = ix.max() + 1
    block = iy * nx + ix
    logger.info("공간 블록: %d개 (블록 %.0fkm)", int(np.unique(block).size), block_km)
    return block


def spatial_cv_recall(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    blocks: np.ndarray,
    n_folds: int = 5,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    subset_mask: np.ndarray | None = None,
) -> dict[float, dict[str, float]]:
    """공간 블록 GroupKFold recall@top-k.

    블록을 폴드로 나눠, 각 폴드(=한 지역 묶음)를 hold-out 으로 두고 그
    지역 내에서 위험 상위 k% 가 발화점을 회수하는지 측정한다. 폴드별
    recall 의 평균±표준편차를 반환(공간 일반화 성능).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    positive_pole_idx : numpy.ndarray
        발화점-전주 인덱스.
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    n_folds : int
        폴드 수.
    ks : tuple[float]
        top-k 지점.
    subset_mask : numpy.ndarray or None
        (N,) bool. 주어지면 각 폴드 내에서 이 부분집합 전주만으로 상위 k% 랭킹·
        recall 을 측정한다(예: 영동 부분집합 — "영동 안에서 top-k% 가 영동 발화점을
        회수하는가"). None 이면 전체.

    Returns
    -------
    dict[float, dict]
        k → {mean, std, n_folds_with_pos}.
    """
    pos_mask = np.zeros(risk.shape[0], dtype=bool)
    pos_mask[positive_pole_idx[positive_pole_idx >= 0]] = True

    fold = _fold_assignment(blocks, n_folds)

    per_k: dict[float, list[float]] = {k: [] for k in ks}
    for f in range(n_folds):
        sel = fold == f
        if subset_mask is not None:
            sel = sel & subset_mask
        if sel.sum() == 0:
            continue
        risk_f = risk[sel]
        pos_idx_f = np.nonzero(pos_mask[sel])[0]
        if pos_idx_f.size == 0:
            continue
        rec = recall_at_topk(risk_f, pos_idx_f, ks)
        for k in ks:
            per_k[k].append(rec[k])

    out: dict[float, dict[str, float]] = {}
    for k in ks:
        arr = np.array(per_k[k], dtype=np.float64)
        out[k] = dict(
            mean=float(arr.mean()) if arr.size else float("nan"),
            std=float(arr.std()) if arr.size else float("nan"),
            n_folds_with_pos=float(arr.size),
        )
    return out


def spatial_cv_recall_multiplier(
    base_risk: np.ndarray,
    master: pl.DataFrame,
    fire_to_pole: np.ndarray,
    blocks: np.ndarray,
    n_folds: int = 5,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    sgg_col: str = "sgg",
    grid_col: str = "grid_id",
    anchor_radius_km: float = 1.0,
) -> dict[float, dict[str, float]]:
    """누수안전 계층 EB 배율 적용 공간블록 CV recall@top-k.

    각 폴드(=hold-out 지역 묶음)마다, **train 폴드 전주에 귀속된 발화점만**으로
    hierarchy.regional_multiplier 를 재계산해 base_risk 에 곱한 뒤, test 폴드 내에서
    상위 k% 가 발화점을 회수하는지 측정한다. test 폴드 발화점은 배율 추정에서 제외되어
    배율로 새지 않는다(누수 차단). 발화점은 per-pole 라벨이 아니라 지역 집계 앵커로만
    쓰이므로, train 발화점만으로 추정한 지역 배율은 test 전주의 사전(prior) 보정자다.

    Parameters
    ----------
    base_risk : numpy.ndarray, shape (N,)
        배율 적용 전 위험(R=I·S·W). 배율은 여기에 곱해진다.
    master : polars.DataFrame
        전주 마스터(lon, lat, sgg_col, grid_col 포함). 행 순서 = base_risk 순서.
    fire_to_pole : numpy.ndarray, shape (M,)
        각 발화점이 매핑된 전주 인덱스(calibrate.assign_poles_to_fires 출력; -1=미매핑).
        발화점의 fold 귀속을 정하는 데 쓴다(매핑 전주의 fold = 그 발화점의 fold).
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    n_folds : int
        폴드 수.
    ks : tuple[float]
        top-k 지점.
    sgg_col, grid_col : str
        계층 배율 컬럼명.
    anchor_radius_km : float
        배율 내부 발화점→전주 귀속 반경(km).

    Returns
    -------
    dict[float, dict]
        k → {mean, std, n_folds_with_pos}. (배율 적용 후 recall)
    """
    n = base_risk.shape[0]
    pos_pole = fire_to_pole[fire_to_pole >= 0]
    pos_mask = np.zeros(n, dtype=bool)
    pos_mask[pos_pole] = True

    fold = _fold_assignment(blocks, n_folds)
    # 각 발화점의 fold = 그 발화점이 매핑된 전주의 fold(미매핑 발화점은 -1).
    fire_fold = np.full(fire_to_pole.shape[0], -1, dtype=np.int64)
    mapped = fire_to_pole >= 0
    fire_fold[mapped] = fold[fire_to_pole[mapped]]

    fire_lonlat = (
        master.select(["lon", "lat"]).to_numpy().astype(np.float64)[fire_to_pole[mapped]]
        if mapped.any() else np.empty((0, 2), dtype=np.float64)
    )
    # 발화점 좌표는 매핑 전주 좌표로 근사(1km 앵커라 동일 지역 — 배율은 지역 집계만 씀).
    mapped_fold = fire_fold[mapped]

    per_k: dict[float, list[float]] = {k: [] for k in ks}
    for f in range(n_folds):
        sel = fold == f
        if sel.sum() == 0:
            continue
        pos_idx_f = np.nonzero(pos_mask[sel])[0]
        if pos_idx_f.size == 0:
            continue
        # train 폴드(=test 폴드 f 제외) 발화점만으로 배율 재계산.
        train_mask = mapped_fold != f
        train_fires = pl.DataFrame(
            {"lon": fire_lonlat[train_mask, 0], "lat": fire_lonlat[train_mask, 1]},
            schema={"lon": pl.Float64, "lat": pl.Float64},
        )
        mult = hierarchy.regional_multiplier(
            master, train_fires, sgg_col=sgg_col, grid_col=grid_col,
            anchor_radius_km=anchor_radius_km, normalize="mean",
        )
        risk_f = (base_risk * mult)[sel]
        rec = recall_at_topk(risk_f, pos_idx_f, ks)
        for k in ks:
            per_k[k].append(rec[k])

    out: dict[float, dict[str, float]] = {}
    for k in ks:
        arr = np.array(per_k[k], dtype=np.float64)
        out[k] = dict(
            mean=float(arr.mean()) if arr.size else float("nan"),
            std=float(arr.std()) if arr.size else float("nan"),
            n_folds_with_pos=float(arr.size),
        )
    return out


def sanity_goseong_2019(
    pole_xy: np.ndarray,
    risk: np.ndarray,
    center: tuple[float, float] = config.GOSEONG_2019_LONLAT,
    radius_km: float = 5.0,
) -> dict[str, float]:
    """2019 고성 발화 인근 전주가 고위험 상위인지 점검.

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    center : tuple[float, float]
        고성 발화점 중심(lon, lat).
    radius_km : float
        인근 반경.

    Returns
    -------
    dict[str, float]
        n_poles, mean_pctile(인근 전주 위험 백분위 평균), frac_top10.
    """
    cx, cy = geo.point_to_km(center[0], center[1])
    pole_km = geo.lonlat_to_km(pole_xy)
    x, y = pole_km[:, 0], pole_km[:, 1]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    near = d <= radius_km
    if near.sum() == 0:
        logger.warning("고성 인근 전주 0개 — sanity 불가")
        return dict(n_poles=0.0, mean_pctile=float("nan"), frac_top10=float("nan"))
    # 위험 백분위
    order = np.argsort(risk)
    pctile = np.empty_like(risk)
    pctile[order] = np.linspace(0, 1, risk.shape[0])
    mean_pctile = float(pctile[near].mean())
    thr10 = np.quantile(risk, 0.90)
    frac_top10 = float((risk[near] >= thr10).mean())
    logger.info("sanity 고성2019: 인근 %d전주 평균백분위=%.3f top10%%비율=%.3f",
                int(near.sum()), mean_pctile, frac_top10)
    return dict(n_poles=float(near.sum()), mean_pctile=mean_pctile,
                frac_top10=frac_top10)


def exposure_downwind_goseong(
    pole_xy: np.ndarray,
    p_exposure: np.ndarray,
    ignition: tuple[float, float] = config.GOSEONG_2019_LONLAT,
    radius_km: float = 6.0,
    thr: float = 0.05,
) -> dict[str, float]:
    """2019 고성 sanity: 풍하 노출이 실제 burn-scar 방향(서→동)으로 신장됐나.

    고성 발화점 인근(radius_km) 전주 중 노출확률 thr 이상인 전주들이 발화점
    **동쪽**(풍하; 양간지풍 서풍)에 치우치는지(east_frac>0.5, 평균 Δlon>0)
    확인한다. 실제 2019 흉터는 발화점에서 동쪽으로 ~6.7km 신장(W→E).

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    p_exposure : numpy.ndarray, shape (N,)
        풍하 노출확률.
    ignition : tuple[float, float]
        고성 발화점(lon, lat).
    radius_km : float
        인근 반경.
    thr : float
        노출 임계(이상이면 "노출됨").

    Returns
    -------
    dict[str, float]
        n_local, n_exposed, east_frac(동쪽비율), mean_dlon_deg(평균 경도차).
    """
    ix, iy = geo.point_to_km(ignition[0], ignition[1])
    pole_km = geo.lonlat_to_km(pole_xy)
    x, y = pole_km[:, 0], pole_km[:, 1]
    d = np.sqrt((x - ix) ** 2 + (y - iy) ** 2)
    local = d <= radius_km
    exposed = local & (p_exposure >= thr)
    if exposed.sum() == 0:
        logger.warning("고성 인근 노출>thr 전주 0 — 방향 sanity 불가")
        return dict(n_local=float(local.sum()), n_exposed=0.0,
                    east_frac=float("nan"), mean_dlon_deg=float("nan"))
    dlon = pole_xy[exposed, 0] - ignition[0]
    east_frac = float((pole_xy[exposed, 0] > ignition[0]).mean())
    mean_dlon = float(dlon.mean())
    logger.info("고성2019 풍하방향: 노출전주 %d/%d east_frac=%.2f mean_dlon=%+.4f도",
                int(exposed.sum()), int(local.sum()), east_frac, mean_dlon)
    return dict(n_local=float(local.sum()), n_exposed=float(exposed.sum()),
                east_frac=east_frac, mean_dlon_deg=mean_dlon)


def compare_weather_recall(
    R_a: np.ndarray,
    R_b: np.ndarray,
    positive_pole_idx: np.ndarray,
    blocks: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    n_folds: int = 5,
) -> dict[float, dict[str, float]]:
    """두 위험맵(예: 기존 시즌 W vs 일별-ISI W)의 공간CV recall@top-k 비교.

    같은 발화점 앵커·공간 블록·폴드 분할에서 R_a, R_b 의 recall@top-k 평균을
    각각 구하고 격차(b−a)를 함께 반환한다. Phase-4 일별 ISI W 의 효과를 동일
    조건에서 정량 비교하기 위한 진단(스크립트 로깅용).

    Parameters
    ----------
    R_a, R_b : numpy.ndarray, shape (N,)
        비교할 두 위험점수(예: 기존 W·일별 ISI W 로 만든 R).
    positive_pole_idx : numpy.ndarray
        발화점-전주 인덱스(공통 앵커).
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    label_a, label_b : str
        로깅 라벨.
    ks : tuple[float]
        top-k 지점.
    n_folds : int
        폴드 수.

    Returns
    -------
    dict[float, dict]
        k → {recall_a, recall_b, gap}. gap = recall_b − recall_a.
    """
    cv_a = spatial_cv_recall(R_a, positive_pole_idx, blocks, n_folds, ks)
    cv_b = spatial_cv_recall(R_b, positive_pole_idx, blocks, n_folds, ks)
    out: dict[float, dict[str, float]] = {}
    for k in ks:
        ra, rb = cv_a[k]["mean"], cv_b[k]["mean"]
        out[k] = dict(recall_a=ra, recall_b=rb, gap=float(rb - ra))
        logger.info("W비교 top-%.0f%%: %s=%.3f → %s=%.3f (Δ=%+.3f)",
                    100 * k, label_a, ra, label_b, rb, rb - ra)
    return out


def spatial_cv_recall_allocation(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    regime_labels: np.ndarray,
    blocks: np.ndarray,
    prevalence: float = 0.02,
    modes: tuple[str, ...] | None = None,
    n_folds: int = 5,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
) -> dict[str, dict[float, dict[str, float]]]:
    """배분 방식별(전역 vs 체제별 ⒝⒞⒟) 공간블록 CV recall@top-k.

    각 폴드(=hold-out 지역묶음)에서, 전체 위험맵 ``risk`` 와 ``regime_labels`` 로
    decision 을 만들고(예산=ceil(prevalence·N_test)), 그 폴드 안에서 위험 상위 k% 가
    발화점을 회수하는지 측정한다. 단, recall@top-k 자체는 결정 임계와 무관하게
    "위험 순위"의 분리력 지표다 — 배분 방식은 **체제별 순위 재배열**(체제 내 상대순위로
    decision 을 정하므로 양성집합이 달라짐)을 통해 recall 에 영향을 준다.

    핵심: recall@top-k 는 risk 순위만 본다. 따라서 배분 비교를 위해, 각 배분 방식이
    만든 **decision 을 0/1 점수로 쓰되 동점은 risk 로 분해**한 "유효 순위"로 recall 을
    잰다(decision=1 전주가 항상 0 보다 위 → top-(예산)% 가 곧 양성집합). 이렇게 하면
    "이 배분이 고른 양성집합이 발화점을 얼마나 회수하나"를 top-k 곡선으로 비교할 수 있다.

    누수안전(ALLOC_ANCHOR): 앵커밀도 가중은 **train 폴드 발화점만**으로 계산한다
    (test 폴드 발화점 미사용).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    positive_pole_idx : numpy.ndarray
        발화점→전주 인덱스(>=0).
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    prevalence : float
        총 예산 비율(전역 동일).
    modes : tuple[str] or None
        비교할 배분 방식. None 이면 (global, regime_count, regime_anchor).
        regime_equal 은 regime_count 와 수학적 동일이라 기본에서 생략(원하면 명시).
    n_folds : int
        폴드 수.
    ks : tuple[float]
        top-k 지점.

    Returns
    -------
    dict[str, dict]
        mode → {k → {mean, std, n_folds_with_pos}}.
    """
    from pfire import calibrate

    if modes is None:
        modes = (calibrate.ALLOC_GLOBAL, calibrate.ALLOC_COUNT, calibrate.ALLOC_ANCHOR)

    n = risk.shape[0]
    pos_mask = np.zeros(n, dtype=bool)
    pos_mask[positive_pole_idx[positive_pole_idx >= 0]] = True
    fold = _fold_assignment(blocks, n_folds)

    out: dict[str, dict[float, dict[str, float]]] = {}
    for mode in modes:
        per_k: dict[float, list[float]] = {k: [] for k in ks}
        for f in range(n_folds):
            sel = fold == f
            if sel.sum() == 0:
                continue
            pos_idx_f = np.nonzero(pos_mask[sel])[0]
            if pos_idx_f.size == 0:
                continue
            risk_f = risk[sel]
            lbl_f = regime_labels[sel]
            if mode == calibrate.ALLOC_GLOBAL:
                _, dec_f = calibrate.decide_threshold(risk_f, prevalence)
            else:
                anchor = None
                if mode == calibrate.ALLOC_ANCHOR:
                    # 누수안전: train 폴드(=현재 fold 제외) 발화점만으로 앵커밀도.
                    train_pos = np.nonzero(pos_mask & (fold != f))[0]
                    anchor = calibrate.regime_anchor_count(regime_labels, train_pos)
                _, dec_f = calibrate.decide_threshold_per_regime(
                    risk_f, lbl_f, prevalence, mode, anchor_count=anchor)
            # decision 을 1차 점수, risk 를 동점분해(tie_break)로 한 유효 순위 → recall.
            eff = dec_f.astype(np.float64)
            rank = calibrate._stable_rank_desc(eff, tie_break=risk_f)
            n_f = risk_f.shape[0]
            for k in ks:
                cut = max(1, int(np.ceil(k * n_f)))
                in_top = rank[pos_idx_f] < cut
                per_k[k].append(float(in_top.mean()))
        out[mode] = {}
        for k in ks:
            arr = np.array(per_k[k], dtype=np.float64)
            out[mode][k] = dict(
                mean=float(arr.mean()) if arr.size else float("nan"),
                std=float(arr.std()) if arr.size else float("nan"),
                n_folds_with_pos=float(arr.size),
            )
    return out


def random_vs_spatial_gap(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    blocks: np.ndarray,
    k: float = 0.05,
    n_folds: int = 5,
) -> dict[str, float]:
    """무작위 분할 대비 공간 블록 분할의 recall 격차(낙관편향 진단).

    Returns
    -------
    dict[str, float]
        random_recall, spatial_recall, gap.
    """
    pos_mask = np.zeros(risk.shape[0], dtype=bool)
    pos_mask[positive_pole_idx[positive_pole_idx >= 0]] = True
    rng = np.random.default_rng(config.SEED)

    # 무작위 폴드
    rand_fold = rng.integers(0, n_folds, size=risk.shape[0])
    rand_recalls = []
    for f in range(n_folds):
        sel = rand_fold == f
        pidx = np.nonzero(pos_mask[sel])[0]
        if pidx.size:
            rand_recalls.append(recall_at_topk(risk[sel], pidx, (k,))[k])
    rand_recall = float(np.mean(rand_recalls)) if rand_recalls else float("nan")

    spatial = spatial_cv_recall(risk, positive_pole_idx, blocks, n_folds, (k,))
    spatial_recall = spatial[k]["mean"]
    return dict(random_recall=rand_recall, spatial_recall=spatial_recall,
                gap=float(rand_recall - spatial_recall))


# ──────────────────────────────────────────────────────────────────────────
# 커버리지 검증 — 명목(1−α) vs 홀드아웃 발화점 실측 포함율 (Phase-5)
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(기획서_phase5 §5 핵심검증): "명목 커버리지 ↔ 실측 커버리지 일치"가
# 합격 기준이다. 공간 블록 fold 마다 train 발화점만으로 conformal 임계(체제별)를
# 추정하고, hold-out(test) 발화점이 그 위험집합에 드는 비율(실측 커버리지)을 잰다.
# test 발화점은 임계 추정에 미사용 → 누수안전. 명목 (1−α) ≈ 실측이면 보장 성립.


def coverage_conformal_holdout(
    risk: np.ndarray,
    fire_to_pole: np.ndarray,
    regime_labels: np.ndarray,
    blocks: np.ndarray,
    alpha: float = 0.10,
    n_folds: int = 5,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, dict[str, float]]:
    """공간 블록 누수안전 per-regime conformal 실측 커버리지(명목 1−α 대조).

    각 fold f 를 hold-out 으로 두고, **train fold(f 제외) 발화점만**으로
    calibrate.conformal_threshold_per_regime 를 추정한 뒤, fold f 발화점이 위험집합
    (risk ≥ τ_r)에 드는 비율(실측 커버리지)을 체제별로 모은다. 폴드 평균을 반환한다.
    명목 (1−α) 에 수렴하면 conformal 커버리지 보장이 실측으로 성립(합격).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수(decision 산출과 동일 점수).
    fire_to_pole : numpy.ndarray, shape (M,)
        각 발화점→전주 인덱스(calibrate.assign_poles_to_fires; -1=미매핑).
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    alpha : float
        목표 miscoverage(기본 0.10 → 명목 90% 커버리지).
    n_folds : int
        폴드 수.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[str, dict]
        {'all': {nominal, empirical, n_test, n_folds}, <regime>: {...}}. empirical 은
        폴드 평균 커버리지(발화점 가중), nominal=1−α.
    """
    from pfire import calibrate

    n = risk.shape[0]
    fold = _fold_assignment(blocks, n_folds)
    fire_fold = np.full(fire_to_pole.shape[0], -1, dtype=np.int64)
    mapped = fire_to_pole >= 0
    fire_fold[mapped] = fold[fire_to_pole[mapped]]

    keys = ["all", *regimes]
    # 폴드별 (covered, n) 누적 — 발화점 가중 평균(폴드 크기 편차 보정).
    acc: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    for f in range(n_folds):
        train_pole = fire_to_pole[(fire_fold != f) & mapped]
        test_pole = fire_to_pole[(fire_fold == f) & mapped]
        if test_pole.size == 0:
            continue
        thr = calibrate.conformal_threshold_per_regime(
            risk, train_pole, regime_labels, alpha=alpha, regimes=regimes)
        cov = calibrate.conformal_coverage(
            risk, test_pole, regime_labels, thr, regimes=regimes)
        for k in keys:
            c = cov.get(k)
            if c is not None and c["n"] > 0:
                acc[k].append((c["covered"], c["n"]))

    out: dict[str, dict[str, float]] = {}
    for k in keys:
        if acc[k]:
            cov_sum = sum(c for c, _ in acc[k])
            n_sum = sum(nn for _, nn in acc[k])
            emp = float(cov_sum / n_sum) if n_sum > 0 else float("nan")
        else:
            emp, n_sum = float("nan"), 0.0
        out[k] = dict(nominal=float(1 - alpha), empirical=emp,
                      n_test=float(n_sum), n_folds=float(len(acc[k])))
    logger.info("conformal 커버리지(명목 %.0f%%): all 실측=%.3f (n=%d)",
                100 * (1 - alpha), out["all"]["empirical"], int(out["all"]["n_test"]))
    return out


def coverage_credible_holdout(
    base_risk: np.ndarray,
    master: pl.DataFrame,
    fire_to_pole: np.ndarray,
    regime_labels: np.ndarray,
    blocks: np.ndarray,
    n_draws: int = 200,
    n_folds: int = 5,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    sgg_col: str = "sgg",
    grid_col: str = "grid_id",
    anchor_radius_km: float = 1.0,
    regimes: tuple[str, ...] = config.REGIMES,
    backend: str = "poisson_gamma",
) -> dict[str, dict[str, float]]:
    """공간 블록 누수안전 베이지안 credible interval 실측 커버리지.

    각 fold f 마다 **train fold 발화점만**으로 지역율 사후(backend='poisson_gamma'면
    posterior.regional_rate_posterior, 'bym'이면 posterior.bym2_spatial_posterior)를
    추정하고 base_risk 에 전파해 전주별 [risk_lo, risk_hi](명목 q_hi−q_lo, 기본 90%)
    를 만든 뒤, hold-out 발화점에 매핑된 전주에서 점추정 risk_mean 이 그 구간에 드는
    비율(자기일관 커버리지)을 잰다. credible 은 모델 오설정 시 under-cover 할 수 있어
    conformal 과 함께 보고한다. **백엔드 교체로 커버리지가 악화되지 않아야** 한다(누수안전).

    Parameters
    ----------
    base_risk : numpy.ndarray, shape (N,)
        물리 결합 위험 R=I·S·W(·exposure), [0,1]. 사후 전파의 본체.
    master : polars.DataFrame
        전주 마스터(lon, lat, sgg_col, grid_col 포함). 행 순서 = base_risk 순서.
    fire_to_pole : numpy.ndarray, shape (M,)
        발화점→전주 인덱스(-1=미매핑).
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    blocks : numpy.ndarray, shape (N,)
        공간 블록 id.
    n_draws : int
        사후예측 MC draw 수.
    n_folds : int
        폴드 수.
    q_lo, q_hi : float
        credible 분위(기본 0.05/0.95 → 90%).
    sgg_col, grid_col : str
        계층 사후 컬럼명.
    anchor_radius_km : float
        발화점→전주 귀속 반경(km).
    regimes : tuple[str]
        체제 순서.
    backend : {'poisson_gamma', 'bym'}
        지역율 사후 백엔드. 'poisson_gamma'=독립 켤레(기본), 'bym'=BYM2 공간 CAR.

    Returns
    -------
    dict[str, dict]
        {'all': {nominal, empirical, n_test, n_folds}, <regime>: {...}}.
    """
    from pfire import calibrate, posterior

    fold = _fold_assignment(blocks, n_folds)
    fire_fold = np.full(fire_to_pole.shape[0], -1, dtype=np.int64)
    mapped = fire_to_pole >= 0
    fire_fold[mapped] = fold[fire_to_pole[mapped]]
    fire_lonlat = master.select(["lon", "lat"]).to_numpy().astype(np.float64)

    keys = ["all", *regimes]
    acc: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    nominal = float(q_hi - q_lo)
    for f in range(n_folds):
        train_idx = fire_to_pole[(fire_fold != f) & mapped]
        test_idx = fire_to_pole[(fire_fold == f) & mapped]
        if test_idx.size == 0:
            continue
        train_fires = pl.DataFrame(
            {"lon": fire_lonlat[train_idx, 0], "lat": fire_lonlat[train_idx, 1]},
            schema={"lon": pl.Float64, "lat": pl.Float64},
        )
        if backend == "bym":
            post = posterior.bym2_spatial_posterior(
                master, train_fires, sgg_col=sgg_col, grid_col=grid_col,
                anchor_radius_km=anchor_radius_km)
        else:
            post = posterior.regional_rate_posterior(
                master, train_fires, sgg_col=sgg_col, grid_col=grid_col,
                anchor_radius_km=anchor_radius_km)
        # 메모리·시간 절약: 사후예측을 hold-out 발화점 전주(소수)만 계산한다.
        # (격자 표집은 전체 격자 기준 같은 seed → 결정성 보존.)
        test_unique = np.unique(test_idx)
        prop = posterior.propagate_risk_posterior(
            base_risk, post, n_draws=n_draws, q_lo=q_lo, q_hi=q_hi,
            pole_subset=test_unique)
        # subset 결과를 전체 인덱스 공간으로 매핑해 empirical_coverage 재사용.
        lo_full = np.zeros(base_risk.shape[0])
        hi_full = np.ones(base_risk.shape[0])
        mean_full = np.zeros(base_risk.shape[0])
        lo_full[test_unique] = prop["risk_lo"]
        hi_full[test_unique] = prop["risk_hi"]
        mean_full[test_unique] = prop["risk_mean"]
        cov = calibrate.empirical_coverage(
            lo_full, hi_full, test_idx, mean_full,
            regime_labels=regime_labels, regimes=regimes)
        for k in keys:
            c = cov.get(k)
            if c is not None and c["n"] > 0:
                acc[k].append((c["covered"], c["n"]))

    out: dict[str, dict[str, float]] = {}
    for k in keys:
        if acc[k]:
            cov_sum = sum(c for c, _ in acc[k])
            n_sum = sum(nn for _, nn in acc[k])
            emp = float(cov_sum / n_sum) if n_sum > 0 else float("nan")
        else:
            emp, n_sum = float("nan"), 0.0
        out[k] = dict(nominal=nominal, empirical=emp,
                      n_test=float(n_sum), n_folds=float(len(acc[k])))
    logger.info("credible 커버리지(명목 %.0f%%): all 실측=%.3f (n=%d)",
                100 * nominal, out["all"]["empirical"], int(out["all"]["n_test"]))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Phase-5b: BYM2 vs Poisson-Gamma 비교 (이웃 borrow 수치 + 전역 매끄러움)
# ──────────────────────────────────────────────────────────────────────────


def _morans_i(values: np.ndarray, W) -> float:
    """격자값 벡터의 Moran's I 공간자기상관(인접행렬 W 가중) — 매끄러움 진단.

    I = (G/ΣW) · (Σ_ij W_ij·(z_i−z̄)(z_j−z̄)) / Σ_i(z_i−z̄)². +1 강한 공간평활(이웃 유사),
    0 무상관, −1 체커보드. BYM 평활이 PG(독립)보다 I 를 높이면 "전역 매끄러움↑"이다.

    Parameters
    ----------
    values : numpy.ndarray, shape (G,)
        격자별 값(예: 로그상대위험 또는 상대배율).
    W : scipy.sparse matrix, shape (G, G)
        대칭 인접행렬(0/1).

    Returns
    -------
    float
        Moran's I. W 합이 0(간선 없음)이거나 분산 0 이면 nan.
    """
    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    denom = float(np.sum(z * z))
    s0 = float(W.sum())
    if denom <= 0 or s0 <= 0:
        return float("nan")
    num = float(z.dot(W.dot(z)))
    G = z.shape[0]
    return (G / s0) * (num / denom)


def compare_bym_vs_poisson_gamma(
    pg_post,
    bym_post,
    watch_sgg: tuple[str, ...] = ("고성", "인제", "화천", "태백"),
) -> dict:
    """BYM2 공간 CAR 사후를 독립 Poisson-Gamma 사후와 비교(이웃 borrow 수치 + Moran's I).

    같은 발화 앵커로 추정한 두 격자 사후의 시군별 상대배율을 나란히 놓아, **zero-event
    시군(고성·인제·화천·태백)이 이웃에서 강도를 빌려(borrow) PG(부모로만 수렴) 대비 어떻게
    달라지는지** 수치로 보고한다. borrow = |BYM mult − PG mult| 로 "이웃 정보로 추정이
    얼마나 움직였나"(=borrow 크기)를 잡는다. PG 가 부모로 붕괴(예: 0.02)한 zero-event
    시군이 이웃 강도로 끌려 올라오면(예: 0.34) borrow 가 크다. 또한 격자 로그상대위험의
    Moran's I 로 전역 매끄러움(이웃 유사도)을 비교한다.

    Parameters
    ----------
    pg_post : posterior.RatePosterior
        독립 Poisson-Gamma 사후(regional_rate_posterior 출력).
    bym_post : posterior.SpatialRatePosterior
        BYM2 공간 CAR 사후(bym2_spatial_posterior 출력). adjacency 보유.
    watch_sgg : tuple[str]
        zero-event 관심 시군.

    Returns
    -------
    dict
        {'sgg': [{sgg, pg_mult, bym_mult, delta_mult, borrow, is_watch}...],
         'moran': {'pg': float, 'bym': float},
         'borrow_mean_watch': float}.
        sgg 리스트는 |delta_mult| 내림차순 정렬.
    """
    # 격자 단위 정렬(공통 키 = BYM 격자 순서; PG·BYM 같은 격자집합 가정).
    grids = list(bym_post.grid_mu.keys())
    pg_g = max(pg_post.global_rate, 1e-12)
    # 격자별 상대배율: PG=α/β/전역율, BYM=exp(μ+s²/2).
    pg_mult_g = np.array(
        [(pg_post.grid_alpha.get(g, 0.0) / max(pg_post.grid_beta.get(g, 1.0), 1e-12)) / pg_g
         for g in grids])
    bym_mult_g = np.array(
        [float(np.exp(bym_post.grid_mu[g] + 0.5 * bym_post.grid_s2[g])) for g in grids])
    # 로그상대위험(0=전역율): PG=log(mult), BYM=μ. Moran's I 비교용.
    pg_logrr = np.log(np.maximum(pg_mult_g, 1e-12))
    bym_logrr = np.array([bym_post.grid_mu[g] for g in grids])
    W = bym_post.adjacency.W
    moran = {"pg": _morans_i(pg_logrr, W), "bym": _morans_i(bym_logrr, W)}

    # 시군 집계(격자 → 대표 시군; 단순 평균 배율).
    sgg_of = bym_post.sgg_of_grid
    by_sgg: dict[str, list[int]] = {}
    for i, g in enumerate(grids):
        by_sgg.setdefault(sgg_of[g], []).append(i)
    rows = []
    borrow_watch = []
    for sgg, idx in by_sgg.items():
        pg_m = float(np.mean(pg_mult_g[idx]))
        bym_m = float(np.mean(bym_mult_g[idx]))
        delta = bym_m - pg_m
        borrow = abs(bym_m - pg_m)  # 이웃 정보로 추정이 움직인 크기.
        is_watch = sgg in watch_sgg
        if is_watch:
            borrow_watch.append(borrow)
        rows.append(dict(sgg=sgg, pg_mult=pg_m, bym_mult=bym_m,
                         delta_mult=float(delta), borrow=float(borrow),
                         is_watch=bool(is_watch)))
    rows.sort(key=lambda r: abs(r["delta_mult"]), reverse=True)
    borrow_mean = float(np.mean(borrow_watch)) if borrow_watch else float("nan")
    logger.info("BYM vs PG: Moran's I PG=%.3f → BYM=%.3f | zero-event 평균 borrow=%.3f",
                moran["pg"], moran["bym"], borrow_mean)
    return {"sgg": rows, "moran": moran, "borrow_mean_watch": borrow_mean}
