"""발화점 확산 샘플 데모 — "불이 여기 나면 어디로 번져 위험한가".

이미 있는 풍하 확산 커널(pfire.exposure.simulate_exposure, Rust)을 그대로 써서,
발화점 하나에서 바람을 타고 번지는 **전주별 노출확률**을 몬테카를로로 모사하고
지도(PNG)로 그린다. "시뮬레이터로 확산 위험을 보여줄 수 있다"는 것을 보이는 샘플.

R=I·S·W(시즌 위험분류)와는 별개다 — 이건 **이벤트형**(특정 발화점) 확산 시연이다.

실행:
    # 기본(양간지풍 회랑 발화점, 서풍)
    python scripts/spread_demo.py

    # 발화점·바람 지정
    python scripts/spread_demo.py --lon 128.52 --lat 38.28 --wind-deg 270 --wind-speed 9
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import config, exposure, exposure_engine, io  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("spread_demo")


def simulate_from_point(
    master, lon0: float, lat0: float,
    *, wind_deg: float = 270.0, wind_spread: float = 25.0,
    wind_speed: float = 8.0, n_sims: int = 96, seed: int = config.SEED,
) -> tuple[np.ndarray, int, np.ndarray]:
    """발화점(lon0,lat0)에서 풍하 확산 노출확률을 모사.

    Returns
    -------
    p_exposure : (N,) 전주별 노출확률
    ign_idx    : 발화점에 매핑된 전주 인덱스
    wind_dir   : (n_sims,) 표집된 풍향(진단용)
    """
    lon = master["lon"].to_numpy().astype(np.float64)
    lat = master["lat"].to_numpy().astype(np.float64)
    pole_xy = np.column_stack([lon, lat])
    pole_km = exposure_engine.lonlat_to_km(pole_xy)  # 커널 계약: 평면 km 좌표
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
    southness = master["mu_southness"].to_numpy().astype(np.float64) * 2.0 - 1.0

    # 발화점 → 최근접 전주(cos 위도 보정 평면근사).
    cos_lat = np.cos(np.deg2rad(config.LAT0_DEG))
    d2 = ((lon - lon0) * cos_lat) ** 2 + (lat - lat0) ** 2
    ign_idx = int(d2.argmin())
    logger.info("발화점 (%.4f, %.4f) → 최근접 전주 #%d (%.4f, %.4f)",
                lon0, lat0, ign_idx, lon[ign_idx], lat[ign_idx])

    # 바람 표집: 지정 방향(서풍 등) 주위 정규 표집(양간지풍 서풍 prior).
    rng = np.random.default_rng(seed)
    wind_dir = (rng.normal(wind_deg, wind_spread, size=n_sims)) % 360.0
    wind_spd = np.clip(rng.normal(wind_speed, wind_speed * 0.25, size=n_sims), 1.0, None)

    logger.info("확산 모사: 발화원 1 · 바람 %d회(방향 %.0f°±%.0f, 풍속 %.1f m/s) · 반경 %.1fkm",
                n_sims, wind_deg, wind_spread, wind_speed, config.SPREAD_MAX_DIST_KM)
    p = exposure.simulate_exposure(
        pole_km, np.array([ign_idx], dtype=np.uint32),
        wind_dir, wind_spd, fuel, southness, seed=seed)
    n_exp = int((p > 0.01).sum())
    logger.info("노출(>1%%) 전주 %d본 · 최대확률 %.3f", n_exp, float(p.max()))
    return p, ign_idx, wind_dir


def render_png(master, p: np.ndarray, ign_idx: int, wind_deg: float,
               out_png: Path, *, thresh: float = 0.01, pad_km: float = 6.0) -> Path:
    """확산 노출확률을 지도(PNG)로 — 발화점★ + 풍하 번짐 히트 + 바람 화살표."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scripts.make_figures import setup_korean_font
        setup_korean_font()
    except Exception:
        pass

    lon = master["lon"].to_numpy().astype(np.float64)
    lat = master["lat"].to_numpy().astype(np.float64)
    lon0, lat0 = float(lon[ign_idx]), float(lat[ign_idx])
    cos_lat = np.cos(np.deg2rad(lat0))
    dlon = pad_km / (111.0 * cos_lat)
    dlat = pad_km / 111.0

    # 표시 창(발화점 주변).
    box = ((lon > lon0 - dlon) & (lon < lon0 + dlon)
           & (lat > lat0 - dlat) & (lat < lat0 + dlat))
    exp = p > thresh

    fig, ax = plt.subplots(figsize=(9, 8))
    # 배경 전주(창 안, 미노출) — 옅은 회색.
    bg = box & ~exp
    ax.scatter(lon[bg], lat[bg], s=3, c="#dcdce2", alpha=0.5, linewidths=0)
    # 노출 전주 — 확률 컬러(Reds).
    m = box & exp
    sc = ax.scatter(lon[m], lat[m], s=10, c=p[m], cmap="YlOrRd",
                    vmin=0.0, vmax=max(0.2, float(p[m].max()) if m.any() else 0.2),
                    alpha=0.9, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("노출확률 P(불이 여기까지 번짐)")
    # 발화점 ★.
    ax.scatter([lon0], [lat0], s=260, marker="*", c="#111", zorder=5,
               edgecolors="white", linewidths=1.2, label="발화점")
    # 바람 화살표(풍하 방향 = from+180) — 발화점에서 번지는 방향.
    downwind = np.deg2rad((wind_deg + 180.0) % 360.0)
    ax.annotate("", xy=(lon0 + 0.55 * dlon * np.sin(downwind),
                        lat0 + 0.55 * dlat * np.cos(downwind)),
                xytext=(lon0, lat0),
                arrowprops=dict(arrowstyle="-|>", color="#1f6fb4", lw=2.2))
    ax.text(lon0 + 0.58 * dlon * np.sin(downwind),
            lat0 + 0.58 * dlat * np.cos(downwind), "풍하", color="#1f6fb4",
            fontsize=10, fontweight="bold")

    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlim(lon0 - dlon, lon0 + dlon)
    ax.set_ylim(lat0 - dlat, lat0 + dlat)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    n_exp = int(m.sum())
    ax.set_title(
        f"발화점 확산 위험 시연 — 불이 나면 바람 타고 번지는 노출\n"
        f"발화점 ({lon0:.3f}, {lat0:.3f}) · 서풍 {wind_deg:.0f}° · "
        f"노출 전주 {n_exp:,}본 (반경 {config.SPREAD_MAX_DIST_KM:.0f}km)",
        fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("확산 지도 저장: %s", out_png)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description="발화점 확산 샘플 데모")
    parser.add_argument("--lon", type=float, default=128.52, help="발화점 경도")
    parser.add_argument("--lat", type=float, default=38.28, help="발화점 위도(기본 고성 회랑)")
    parser.add_argument("--wind-deg", type=float, default=270.0, help="풍향(from-방향, 270=서풍)")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="풍속 m/s")
    parser.add_argument("--n-sims", type=int, default=96, help="바람 MC 표집 수")
    parser.add_argument("--out", type=str, default="outputs/predict/spread_demo.png")
    args = parser.parse_args()

    logger.info("=== 전주 로드 ===")
    master = io.load_master()

    logger.info("=== 확산 모사(기존 커널 활용) ===")
    p, ign_idx, _ = simulate_from_point(
        master, args.lon, args.lat,
        wind_deg=args.wind_deg, wind_speed=args.wind_speed, n_sims=args.n_sims)

    logger.info("=== 확산 지도 ===")
    render_png(master, p, ign_idx, args.wind_deg, Path(args.out))
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
