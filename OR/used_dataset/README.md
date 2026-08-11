# used_dataset — EDA 사용 데이터 (결과물·예보·FIRMS 제외)

스냅샷 2026-06-18. **EDA.ipynb 산불 데이터=safemap 전용** (FIRMS 미사용). 결과물·예보(fwi_grid)·FIRMS 제외.
✅EDA=노트북 직접 사용 / 모델입력=모델 파이프라인용(EDA 미사용, 참고 보존).

| 폴더 | 파일 | 크기 | 행수 | 구분 |
|---|---|---|---|---|
| admin | `admin_gangwon.csv` | 0.1MB | 42 | ✅EDA |
| burn | `burn_goseong_2019_dnbr.geojson` | 0.2MB | - | ✅EDA |
| burn | `burn_goseong_terrain_sample.parquet` | 0.0MB | 4,000 | ✅EDA |
| fire | `safemap_fire_landcover.parquet` | 0.1MB | 916 | ✅EDA |
| fire | `safemap_fire_terrain.parquet` | 0.0MB | 186 | ✅EDA |
| fire | `safemap_fire_weather.parquet` | 0.0MB | 186 | ✅EDA |
| fire | `safemap_positives.parquet` | 0.1MB | 928 | ✅EDA |
| fire | `safemap_positives_declustered.parquet` | 0.0MB | 928 | 모델입력 |
| poles | `gangwon_poles_4326.csv` | 65.5MB | 1,387,831 | ✅EDA |
| poles | `pole_features.parquet` | 99.8MB | 1,387,831 | ✅EDA |
| poles | `pole_kfs.parquet` | 6.1MB | 1,387,831 | 모델입력 |
| poles | `pole_landcover_filled.parquet` | 7.3MB | 1,387,831 | ✅EDA |
| poles | `pole_power.parquet` | 23.4MB | 1,387,831 | 모델입력 |
| poles | `pole_sgg.parquet` | 1.5MB | 1,387,831 | ✅EDA |
| poles | `pole_static_overlay.parquet` | 81.7MB | 1,387,831 | ✅EDA |
| poles | `s2_features.parquet` | 41.1MB | 1,387,831 | ✅EDA |
| poles | `training_labels.parquet` | 29.1MB | 1,387,831 | 모델입력 |
| weather | `aws_obs_daily.parquet` | 4.3MB | 377,822 | ✅EDA |
| weather | `aws_stations_coords_elev.csv` | 0.0MB | 109 | ✅EDA |
| weather | `fwi_station_daily.parquet` | 21.6MB | 377,822 | 모델입력 |
| weather | `kma_aws_coords_gangwon.csv` | 0.0MB | 94 | 모델입력 |
| weather | `pole_fwi_obs.parquet` | 28.5MB | 1,387,831 | 모델입력 |
| weather | `pole_weather_obs.parquet` | 12.2MB | 1,387,831 | ✅EDA |

**합계 423MB · 23개**

## 제외
- 결과물 pole_risk_*·submission_*
- 예보 fwi_grid_daily/season
- **FIRMS firms_positives(+declustered)** — EDA 산불분석은 safemap만(FIRMS는 노트북 카탈로그/주석에만)
- 외부 DEM·DB원본·구버전