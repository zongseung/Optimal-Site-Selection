"""검증 — 공간 블록 교차검증 + 발화점 recall + sanity.

설계 근거: 산불 위험은 공간 자기상관이 강하다. 무작위 분할은 인접 전주가
train/test 에 함께 들어가 낙관 편향을 준다. config.SPATIAL_CV_BLOCK_KM 격자로
GroupKFold(공간 블록)하여 "다른 지역에서도 발화점 상위에 드는가"를 본다.
sanity: 2019 고성(≈128.50,38.21) 주변·설비/전기원인 화재가 고위험 상위인지 확인.
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config, geo
from pfire.calibrate import recall_at_topk

logger = logging.getLogger(__name__)


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

    Returns
    -------
    dict[float, dict]
        k → {mean, std, n_folds_with_pos}.
    """
    pos_mask = np.zeros(risk.shape[0], dtype=bool)
    pos_mask[positive_pole_idx[positive_pole_idx >= 0]] = True

    uniq = np.unique(blocks)
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(uniq)
    fold_of_block = {b: i % n_folds for i, b in enumerate(uniq)}
    fold = np.array([fold_of_block[b] for b in blocks])

    per_k: dict[float, list[float]] = {k: [] for k in ks}
    for f in range(n_folds):
        sel = fold == f
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
