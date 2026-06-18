"""pfire 전역 설정 — 단일 진실(single source of truth).

모든 경로·상수·체제(regime) 정의·시즌·시드를 여기 한 곳에 둔다.
다른 모듈은 절대 경로/상수를 하드코딩하지 말고 여기서 import 한다.

설계 근거(요약): 라벨이 없는 전주 산불 위험을 "물리식이 본체 + 체제별 전문가(MoE)로
가중 보정 + 발화점은 검증·임계값 앵커" 로 추정한다. 자세한 내용은
claudedocs/기획서_전주산불위험_조기경보_20260618.md 참고.
"""
from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # OR-project/
DATA = ROOT / "used_dataset"
POLES = DATA / "poles"
FIRE = DATA / "fire"
WEATHER = DATA / "weather"
ADMIN = DATA / "admin"
BURN = DATA / "burn"
OUT = ROOT / "outputs"
SUBMISSIONS = OUT / "submissions"

# 데이터 파일 (EDA=노트북 직접사용 자산 우선)
F_POLE_FEATURES = POLES / "pole_features.parquet"        # 지형·연료·FWI 통계
F_POLE_POWER = POLES / "pole_power.parquet"              # dist_to_powerline/substation
F_POLE_STATIC = POLES / "pole_static_overlay.parquet"    # mu_*, S_p (확산취약 사전계산)
F_POLE_KFS = POLES / "pole_kfs.parquet"                  # kfs_fires (산림청 화재 카운트)
F_POLE_SGG = POLES / "pole_sgg.parquet"                  # pole_id → 시군
F_POLE_LANDCOVER = POLES / "pole_landcover_filled.parquet"
F_POLE_FWI_OBS = WEATHER / "pole_fwi_obs.parquet"        # 관측기반 FWI(전주 보간)
F_TRAINING_LABELS = POLES / "training_labels.parquet"    # 약지도(신뢰 낮음 → 피처/검증 보조만)
F_SAFEMAP_POSITIVES = FIRE / "safemap_positives.parquet" # 발화점 928 (검증·임계값 앵커)
F_FWI_STATION_DAILY = WEATHER / "fwi_station_daily.parquet"  # 관측소 일별 FWI/ISI/풍속
F_AWS_OBS_DAILY = WEATHER / "aws_obs_daily.parquet"      # 관측소 일별 풍향(wd)/풍속 — exposure 풍향 표집
F_STATIONS = WEATHER / "aws_stations_coords_elev.csv"    # 관측소 109 좌표
F_ADMIN = ADMIN / "admin_gangwon.csv"                    # 행정경계 wkt

# ──────────────────────────────────────────────────────────────────────────
# 데이터 규모 불변(invariant)
# ──────────────────────────────────────────────────────────────────────────
# 강원 가상 전주 수(고정). io.load_master / submit.validate_submission 가 강제 검증.
N_POLES_EXPECTED = 1_387_831

# ──────────────────────────────────────────────────────────────────────────
# 시즌 / 재현성
# ──────────────────────────────────────────────────────────────────────────
SEASON_MONTHS = (2, 3, 4, 5)     # 산불조심기간(산림청, 매년 2~5월); 일별 h(p,t) 시즌집계 대상
SEED = 20260618

# 좌표계: 입력 EPSG:4326(위경도). 거리/확산 계산은 강원 중심 위도 기준 근사 평면(km).
LAT0_DEG = 37.7                  # 강원 대표 위도(평면 근사 cos 보정용)
EARTH_R_KM = 6371.0

# ──────────────────────────────────────────────────────────────────────────
# 체제(regime) 정의 — MoE 전문가. 데이터로 확인: 영동 양간일 2.33·FWI 9.2 vs 영서 0.3·5.24
#   게이트는 regimes.py에서 피처(경도·고도·양간일) 기반 soft 로 계산.
#   아래 시군 매핑은 앵커/초기화/검증용(하드 분할 아님).
# ──────────────────────────────────────────────────────────────────────────
REGIME_YEONGDONG = "yeongdong"   # 영동 해안: 양간지풍·강풍·송전선발화(2019 고성)·wind-driven 확산
REGIME_YEONGSEO = "yeongseo"     # 영서 내륙: 농경 소각·입산자 실화(인간발화)
REGIME_MOUNTAIN = "mountain"     # 고지 산간: 연료·산림 지배, FWI 낮음

REGIMES = (REGIME_YEONGDONG, REGIME_YEONGSEO, REGIME_MOUNTAIN)

SGG_TO_REGIME = {
    # 영동 해안
    "강릉": REGIME_YEONGDONG, "동해": REGIME_YEONGDONG, "삼척": REGIME_YEONGDONG,
    "속초": REGIME_YEONGDONG, "양양": REGIME_YEONGDONG, "고성": REGIME_YEONGDONG,
    # 영서 내륙
    "춘천": REGIME_YEONGSEO, "원주": REGIME_YEONGSEO, "홍천": REGIME_YEONGSEO,
    "횡성": REGIME_YEONGSEO, "화천": REGIME_YEONGSEO, "영월": REGIME_YEONGSEO,
    # 고지 산간 (평균고도 >570m)
    "태백": REGIME_MOUNTAIN, "평창": REGIME_MOUNTAIN, "정선": REGIME_MOUNTAIN,
    "인제": REGIME_MOUNTAIN,
}

# 체제별 물리식 전문가 가중치 — Phase2 튜닝값(공간블록CV 발화점 recall 최적화).
#   항목: 발화기하(forest/road/powerline) + 기상(fwi/yanggan) + 연료(fuel/southness)
#   각 전문가는 같은 피처를 쓰되 가중을 다르게 → "어디서 왜 위험한가"가 체제마다 다름.
#   튜닝: LHS(Dirichlet) 랜덤탐색 5000후보 + 좌표상승(scripts/tune_weights.py, step=0.05,
#   목적=공간CV recall@top-{1,2,5,10}% 가중평균, simplex 제약). 결과 provenance 는
#   outputs/tuned_weights.json. 공간CV recall@top5% 0.077→0.100, top10% 0.146→0.171,
#   무작위-공간 격차 ≈0 유지(낙관편향 없음). 5개 폴드시드 재현 std≈0.005~0.008(안정).
#   초기값(EDA 눈대중, before)은 아래 주석에 보존.
EXPERT_WEIGHTS = {
    # before: forest=0.20, road=0.10, powerline=0.20, fwi=0.25, yanggan=0.15, fuel=0.10
    REGIME_YEONGDONG: dict(forest=0.3509, road=0.0000, powerline=0.1003, fwi=0.0977, yanggan=0.0501, fuel=0.4010),
    # before: forest=0.25, road=0.30, powerline=0.05, fwi=0.20, yanggan=0.00, fuel=0.20
    REGIME_YEONGSEO:  dict(forest=0.3684, road=0.2105, powerline=0.3684, fwi=0.0000, yanggan=0.0526, fuel=0.0000),
    # before: forest=0.35, road=0.10, powerline=0.05, fwi=0.10, yanggan=0.00, fuel=0.40
    REGIME_MOUNTAIN:  dict(forest=0.0000, road=0.1579, powerline=0.4737, fwi=0.1579, yanggan=0.2105, fuel=0.0000),
}

# ──────────────────────────────────────────────────────────────────────────
# MC 풍하 확산(exposure, Rust 커널) 파라미터
# ──────────────────────────────────────────────────────────────────────────
SPREAD_MAX_DIST_KM = 5.0         # 확산 컷오프 반경(2019 고성 스케일 참고)
SPREAD_N_SIMS = 256              # 발화원당 MC 시뮬 수
SPREAD_LENGTH_SCALE_KM = 0.8     # 거리 감쇠 스케일 L0
SPREAD_WIND_ANISO = 1.5          # 풍하 신장 계수 α (바람세기×방향정렬)
SPREAD_SOUTHNESS_BETA = 0.3      # 남서사면 확산 보정 β (EDA: SW aspect 일관 신호)

# ──────────────────────────────────────────────────────────────────────────
# 병렬/성능
# ──────────────────────────────────────────────────────────────────────────
# Rust exposure 커널(rayon)의 스레드 상한. pfire/__init__.py 가 패키지 import 시점에
# RAYON_NUM_THREADS 로 적용한다(rayon 전역풀 초기화=첫 simulate_exposure 호출 전이라야
# 반영됨). 전체 64코어 중 50으로 제한해 다른 작업용 여유를 남긴다.
N_THREADS = 50

# ──────────────────────────────────────────────────────────────────────────
# 임계값 / 평가
# ──────────────────────────────────────────────────────────────────────────
# F1 정답 양성비율 미지 → 민감도 곡선으로 보고. 기본 운영 시나리오 양성비율 후보.
PREVALENCE_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
SPATIAL_CV_BLOCK_KM = 10.0       # 공간 블록 교차검증 블록 크기
TOPK_FOR_RECALL = (0.01, 0.02, 0.05, 0.10)  # recall@top-k 평가 지점

# 2019 고성 산불 발화점(lon, lat) — sanity/방향성 검증 앵커(흉터 W→E ~6.7km).
GOSEONG_2019_LONLAT = (128.50, 38.21)
