"""데이터(기상)만 넣으면 위험 예상도를 만드는 예측 프로그램.

강원 전주 1,387,831본의 **정적 취약(I·S)**은 그대로 두고, ``--weather`` 폴더로
넘긴 **새 기상**(fwi_station_daily.parquet [, aws_obs_daily.parquet])으로 W 를
재계산해 전주별 시즌 위험 ``R = I·S·W`` 를 산출한다. 산출물:

  - ``risk.csv``       — pole_id, lon, lat, regime, risk_score, risk_pctile, decision
  - ``risk_map.html``  — Leaflet 인터랙티브 위험 예상도(격자 집계, 새 의존성 없음)

설계: W 계산부(``weather.blended_weather``·``exposure_v2``)가 이미 기상 프레임을
인자로 받게 파라미터화되어 있어, 새 parquet 을 읽어 넘기기만 하면 R 이 갱신된다.
정적 취약(I·S)은 전주 인프라·지형이라 기상과 무관하게 고정한다.

주의: 기본 모드에서 새 기상은 W 의 **일별 성분(1−0.82)** 과 풍하 노출에 반영되고,
시즌 극값 성분(0.82)은 전주 피처에 사전계산된 기후값을 유지한다. 즉 "이번 기간
기상이 기후 기준 위험을 어떻게 흔드는가"를 본다(전주별 시즌 성분까지 입력 기상으로
갱신하려면 최근접 관측소 보간 단계가 추가로 필요 — README 참고).

사용:
    # 기본 기상(used_dataset)으로 산출
    python scripts/predict.py --out outputs/predict

    # 새 기상 폴더를 넣어 예상도 갱신
    python scripts/predict.py --weather <기상폴더> --out outputs/predict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# 패키지 루트 임포트 경로 보장(스크립트 직접 실행 대비).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import (  # noqa: E402
    calibrate,
    config,
    experts,
    hazard,
    io,
    regimes,
    risk_index,
)
from pfire import weather as weather_mod  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


# ──────────────────────────────────────────────────────────────────────────
# 기상 로드 — --weather 폴더 우선, 없으면 config 기본(used_dataset).
# ──────────────────────────────────────────────────────────────────────────
def _load_weather(weather_dir: Path | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(station_daily, aws_daily) 프레임을 로드한다.

    weather_dir 이 주어지면 그 안의 fwi_station_daily.parquet / aws_obs_daily.parquet
    을 우선 사용하고, 없는 파일은 기본(used_dataset)으로 폴백한다. 어느 쪽을 썼는지
    로깅한다(silent 금지).
    """
    if weather_dir is None:
        logger.info("기상: 기본(used_dataset) 사용")
        return io.load_station_daily(), io.load_aws_daily()

    weather_dir = Path(weather_dir)
    if not weather_dir.exists():
        raise FileNotFoundError(f"--weather 폴더 없음: {weather_dir}")

    f_station = weather_dir / "fwi_station_daily.parquet"
    f_aws = weather_dir / "aws_obs_daily.parquet"

    if f_station.exists():
        station_daily = pl.read_parquet(f_station)
        logger.info("기상: station_daily ← %s (rows=%d)", f_station, station_daily.height)
    else:
        station_daily = io.load_station_daily()
        logger.warning("기상: %s 없음 → 기본 station_daily 폴백", f_station)

    if f_aws.exists():
        aws_daily = pl.read_parquet(f_aws)
        if "month" not in aws_daily.columns and "obs_date" in aws_daily.columns:
            aws_daily = aws_daily.with_columns(pl.col("obs_date").dt.month().alias("month"))
        logger.info("기상: aws_daily ← %s (rows=%d)", f_aws, aws_daily.height)
    else:
        aws_daily = io.load_aws_daily()
        logger.warning("기상: %s 없음 → 기본 aws_daily 폴백", f_aws)

    return station_daily, aws_daily


# ──────────────────────────────────────────────────────────────────────────
# 위험도 계산 R = I·S·W  (+ 풍하 노출 v2.2 결합)
# ──────────────────────────────────────────────────────────────────────────
def compute_risk(
    weather_dir: Path | None,
    *,
    use_exposure_v2: bool = True,
) -> dict:
    """전주별 위험도를 계산해 결과 dict 로 반환.

    Returns
    -------
    dict
        keys: master, R, regime_lbl, risk_pctile, risk_pctile_regime.
    """
    logger.info("=== 1. 전주 마스터 로드(정적 취약 I·S 기반) ===")
    master = io.load_master()
    stations = io.load_stations()

    logger.info("=== 2. 체제 soft 게이트(MoE) ===")
    gate, regime_order = regimes.compute_gate(master)
    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]

    logger.info("=== 3. 발화 성향 I(p) ===")
    I, _ = experts.ignition_propensity(master, gate, regime_order)

    logger.info("=== 4. 확산/노출 취약 S(p) ===")
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)

    logger.info("=== 5. 기상 W(p) — 입력 기상으로 재계산 ===")
    station_daily, aws_daily = _load_weather(weather_dir)
    W = weather_mod.blended_weather(master, station_daily, stations)

    logger.info("=== 6. 시즌 위험 R(p) = I·S·W ===")
    R = hazard.season_risk(I, S, W)

    if use_exposure_v2 and config.EXPOSURE_V2_BLEND_W > 0:
        try:
            from pfire import exposure_v2 as _expv2
            dose01 = _expv2.compute_dose01(
                master, I, gate, regime_order,
                station_daily=station_daily, aws_daily=aws_daily,
            )
            R = _expv2.blend_into_risk(R, dose01, w=config.EXPOSURE_V2_BLEND_W)
            logger.info("풍하 노출 v2.2 결합(w=%.2f) → R 갱신", config.EXPOSURE_V2_BLEND_W)
        except Exception as e:  # scipy 미설치 등 — 안전 skip(R 유지)
            logger.warning("exposure_v2 결합 skip(%s) — 정적 R 유지", e)

    risk_pctile = np.round(risk_index.risk_percentile(R), 2)
    risk_pctile_regime = np.round(
        risk_index.risk_percentile_by_group(R, regime_lbl), 2)

    return dict(
        master=master, R=R, regime_lbl=regime_lbl,
        risk_pctile=risk_pctile, risk_pctile_regime=risk_pctile_regime,
    )


# ──────────────────────────────────────────────────────────────────────────
# 격자 집계 + Leaflet HTML 위험 예상도
# ──────────────────────────────────────────────────────────────────────────
def _grid_aggregate(
    lon: np.ndarray, lat: np.ndarray, pctile: np.ndarray,
    decision: np.ndarray, regime_lbl: np.ndarray, grid_deg: float,
) -> pl.DataFrame:
    """전주를 grid_deg 격자로 집계 — 셀별 평균 위험백분위·전주수·고위험수·우세체제."""
    ci = np.floor(lon / grid_deg).astype(np.int64)
    cj = np.floor(lat / grid_deg).astype(np.int64)
    df = pl.DataFrame({
        "ci": ci, "cj": cj, "pctile": pctile,
        "decision": decision.astype(np.int64), "regime": regime_lbl,
    })
    # 우세 체제 = 셀 내 최빈 regime.
    dom = (
        df.group_by(["ci", "cj", "regime"]).agg(pl.len().alias("n"))
          .sort(["ci", "cj", "n"], descending=[False, False, True])
          .group_by(["ci", "cj"], maintain_order=True).first()
          .select(["ci", "cj", pl.col("regime").alias("dom_regime")])
    )
    agg = (
        df.group_by(["ci", "cj"]).agg(
            pl.col("pctile").mean().alias("risk"),
            pl.col("pctile").max().alias("risk_max"),
            pl.len().alias("n_poles"),
            pl.col("decision").sum().alias("n_high"),
        )
        .join(dom, on=["ci", "cj"], how="left")
    )
    return agg


REG_KO = {"yeongdong": "영동", "corridor": "양간지풍 회랑",
          "yeongseo": "영서", "mountain": "산간"}


def cells_from_agg(agg: pl.DataFrame, grid_deg: float) -> list[dict]:
    """격자 집계 프레임 → 지도 셀 리스트(bbox·색상·위험·전주수·우세체제).

    셀 색은 matplotlib YlOrRd 로 위험 백분위(0..100)에 매핑한다. 웹앱·HTML 공용.
    """
    from matplotlib import colormaps, colors as mcolors

    cmap = colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)
    cells = []
    for row in agg.iter_rows(named=True):
        r0 = row["cj"] * grid_deg
        c0 = row["ci"] * grid_deg
        risk = float(row["risk"])
        cells.append({
            "b": [round(r0, 5), round(c0, 5),
                  round(r0 + grid_deg, 5), round(c0 + grid_deg, 5)],
            "c": mcolors.to_hex(cmap(norm(risk))),
            "r": round(risk, 1),
            "rx": round(float(row["risk_max"]), 1),
            "n": int(row["n_poles"]),
            "h": int(row["n_high"]),
            "g": REG_KO.get(row["dom_regime"], row["dom_regime"] or "-"),
        })
    return cells


def legend_stops() -> list[dict]:
    """범례 컬러바 눈금(0..100) → [{v, c(hex)}]."""
    from matplotlib import colormaps, colors as mcolors
    cmap = colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)
    return [{"v": v, "c": mcolors.to_hex(cmap(norm(v)))}
            for v in (0, 20, 40, 60, 80, 100)]


def write_html_map(
    agg: pl.DataFrame, center: tuple[float, float],
    n_high_total: int, n_total: int, out_html: Path,
    *, grid_deg: float = 0.02, prevalence: float = 0.02,
    weather_label: str = "기본(used_dataset)",
) -> Path:
    """격자 집계 위험도를 자립형 Leaflet HTML(위험 예상도)로 저장.

    matplotlib 컬러맵(YlOrRd)으로 셀 색을 Python 에서 precompute 해 인라인 JSON 으로
    임베드한다. 지도 타일·Leaflet 라이브러리는 표준 CDN(unpkg/openstreetmap)에서
    로드한다(로컬에서 브라우저로 열면 인터넷 필요).
    """
    cells = cells_from_agg(agg, grid_deg)
    lat0, lon0 = center

    payload = {
        "cells": cells,
        "center": [lat0, lon0],
        "grid_deg": grid_deg,
        "n_cells": len(cells),
        "n_high_total": n_high_total,
        "n_total": n_total,
        "prevalence": prevalence,
        "weather_label": weather_label,
        "legend": legend_stops(),
    }

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    logger.info("위험 예상도 지도: %s (격자 %d셀, %.3f°)", out_html, len(cells), grid_deg)
    return out_html


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>강원 전주 산불위험 예상도</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,'Malgun Gothic',sans-serif}
  #map{position:absolute;top:0;bottom:0;left:0;right:0}
  .panel{position:absolute;z-index:1000;background:rgba(255,255,255,.92);
    border-radius:10px;padding:10px 13px;box-shadow:0 1px 6px rgba(0,0,0,.25);font-size:13px}
  #hdr{top:12px;left:12px;max-width:320px}
  #hdr h1{margin:0 0 4px;font-size:16px}
  #hdr .sub{color:#555;font-size:12px;line-height:1.45}
  #legend{bottom:20px;left:12px}
  #legend .bar{display:flex;height:12px;width:200px;border-radius:3px;overflow:hidden;margin:6px 0 2px}
  #legend .bar span{flex:1}
  #legend .ticks{display:flex;justify-content:space-between;width:200px;color:#555;font-size:11px}
  .leaflet-popup-content{font-size:13px;line-height:1.5}
</style>
</head>
<body>
<div id="map"></div>
<div id="hdr" class="panel">
  <h1>🔥 강원 전주 산불위험 예상도</h1>
  <div class="sub" id="sub"></div>
</div>
<div id="legend" class="panel">
  <b>위험 백분위</b>
  <div class="bar" id="lbar"></div>
  <div class="ticks"><span>0<br>낮음</span><span>50</span><span>100<br>높음</span></div>
</div>
<script>
const D = __PAYLOAD__;
const map = L.map('map').setView(D.center, 9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:18, attribution:'© OpenStreetMap'}).addTo(map);

const layer = L.layerGroup().addTo(map);
for(const c of D.cells){
  const rect = L.rectangle([[c.b[0],c.b[1]],[c.b[2],c.b[3]]],
    {color:c.c, weight:0, fillColor:c.c, fillOpacity:0.62});
  rect.bindPopup(
    `<b>격자 위험 백분위</b><br>`+
    `평균 <b>${c.r}</b> · 최고 ${c.rx}<br>`+
    `전주 ${c.n.toLocaleString()}본 · 고위험 ${c.h.toLocaleString()}본<br>`+
    `우세 체제: ${c.g}`);
  layer.addLayer(rect);
}

document.getElementById('sub').innerHTML =
  `기상: <b>${D.weather_label}</b><br>`+
  `전주 ${D.n_total.toLocaleString()}본 · 고위험(상위 ${(D.prevalence*100).toFixed(1)}%) `+
  `<b>${D.n_high_total.toLocaleString()}</b>본<br>`+
  `격자 ${D.n_cells.toLocaleString()}셀 (${D.grid_deg}°≈${Math.round(D.grid_deg*111)}km) · 셀 클릭 시 상세`;

const bar = document.getElementById('lbar');
for(const s of D.legend){ const el=document.createElement('span'); el.style.background=s.c; bar.appendChild(el); }
</script>
</body>
</html>
"""


def write_png_preview(
    agg: pl.DataFrame, out_png: Path,
    *, grid_deg: float, weather_label: str,
    n_high_total: int, n_total: int, prevalence: float,
) -> Path:
    """격자 위험도를 정적 PNG(위험 예상도)로 저장 — 인터넷·브라우저 없이 열림.

    보고서 삽입·빠른 확인용. HTML 지도와 동일 격자 집계·컬러맵(YlOrRd)을 쓴다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:  # 한글 폰트(있으면) — 없으면 조용히 통과.
        from scripts.make_figures import setup_korean_font
        setup_korean_font()
    except Exception:
        pass

    ci = agg["ci"].to_numpy()
    cj = agg["cj"].to_numpy()
    risk = agg["risk"].to_numpy().astype(np.float64)
    ci0, cj0 = int(ci.min()), int(cj.min())
    nci = int(ci.max()) - ci0 + 1
    ncj = int(cj.max()) - cj0 + 1
    grid = np.full((ncj, nci), np.nan)
    grid[cj - cj0, ci - ci0] = risk
    lon_edges = (np.arange(ci0, ci0 + nci + 1)) * grid_deg
    lat_edges = (np.arange(cj0, cj0 + ncj + 1)) * grid_deg

    lat_mid = float(np.median(lat_edges))
    fig, ax = plt.subplots(figsize=(9, 8))
    pcm = ax.pcolormesh(lon_edges, lat_edges, grid, cmap="YlOrRd",
                        vmin=0.0, vmax=100.0, shading="flat")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))  # 위경도 종횡비 보정
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.82)
    cbar.set_label("위험 백분위 (0=낮음 · 100=높음)")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_title(
        f"강원 전주 산불위험 예상도  ·  기상: {weather_label}\n"
        f"전주 {n_total:,}본 · 고위험(상위 {prevalence * 100:.1f}%) "
        f"{n_high_total:,}본 · 격자 {grid_deg}°(≈{round(grid_deg * 111)}km)",
        fontsize=11)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("위험 예상도 PNG: %s", out_png)
    return out_png


# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="기상 데이터로 전주 산불위험 예상도 생성")
    parser.add_argument("--weather", type=str, default=None,
                        help="새 기상 폴더(fwi_station_daily.parquet, aws_obs_daily.parquet). "
                             "생략 시 기본 used_dataset 기상 사용.")
    parser.add_argument("--out", type=str, default="outputs/predict",
                        help="산출 폴더(risk.csv, risk_map.html). 기본 outputs/predict.")
    parser.add_argument("--prevalence", type=float, default=0.02,
                        help="고위험(decision=1) 예측양성 비율(전역 상위 컷). 기본 0.02.")
    parser.add_argument("--grid-deg", type=float, default=0.02,
                        help="지도 격자 크기(도). 기본 0.02(≈2km).")
    parser.add_argument("--no-exposure-v2", action="store_true",
                        help="풍하 노출 v2.2 결합을 끈다(빠름; R=I·S·W 만).")
    parser.add_argument("--no-map", action="store_true",
                        help="HTML 지도 생략(risk.csv 만 산출).")
    args = parser.parse_args()

    np.random.seed(config.SEED)
    weather_dir = Path(args.weather) if args.weather else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_risk(weather_dir, use_exposure_v2=not args.no_exposure_v2)
    master = result["master"]
    R = result["R"]

    logger.info("=== 7. 고위험 결정(상위 %.1f%%) ===", 100 * args.prevalence)
    _, decision = calibrate.decide_threshold(R, args.prevalence)

    logger.info("=== 8. risk.csv 작성 ===")
    out_csv = out_dir / "risk.csv"
    pl.DataFrame({
        "pole_id": master["pole_id"].to_numpy(),
        "lon": master["lon"].to_numpy(),
        "lat": master["lat"].to_numpy(),
        "regime": result["regime_lbl"],
        "risk_score": np.round(R, 6),
        "risk_pctile": result["risk_pctile"],
        "risk_pctile_regime": result["risk_pctile_regime"],
        "decision": decision.astype(np.int64),
    }).write_csv(out_csv)
    logger.info("  저장: %s (행수=%d, 고위험=%d)",
                out_csv, master.height, int(decision.sum()))

    weather_label = ("기본(used_dataset)" if weather_dir is None
                     else str(weather_dir))
    if not args.no_map:
        logger.info("=== 9. 위험 예상도 지도 작성(HTML + PNG) ===")
        lon = master["lon"].to_numpy().astype(np.float64)
        lat = master["lat"].to_numpy().astype(np.float64)
        agg = _grid_aggregate(lon, lat, result["risk_pctile"], decision,
                              result["regime_lbl"], args.grid_deg)
        center = (float(np.median(lat)), float(np.median(lon)))
        n_high_total = int(decision.sum())
        n_total = int(decision.shape[0])
        write_html_map(agg, center, n_high_total, n_total,
                       out_dir / "risk_map.html", grid_deg=args.grid_deg,
                       prevalence=args.prevalence, weather_label=weather_label)
        write_png_preview(agg, out_dir / "risk_map.png",
                          grid_deg=args.grid_deg, weather_label=weather_label,
                          n_high_total=n_high_total, n_total=n_total,
                          prevalence=args.prevalence)

    logger.info("=== 완료 ===")
    logger.info("산출: %s%s", out_csv,
                "" if args.no_map else
                f" , {out_dir / 'risk_map.html'} , {out_dir / 'risk_map.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
