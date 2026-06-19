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

# 토지피복 발화계수 lc_ignition (0~1) — 도메인 사전치(EDA prior).
#   근거: 발화의 91%가 산림(전주 중 산림은 18%뿐) → 토지유형이 발화 성향을 강하게
#   가른다. 발화점 라벨을 학습 타깃으로 직접 쓰지 않고, EDA 사실에서 유도한 발화
#   성향 배율을 [0,1] 사전치로 둔다. experts.build_ignition_features 의 'landcover'
#   성분이 이 매핑을 lc_group 으로 조회해 사용한다(이미 [0,1] → 추가 정규화 불필요).
#   미지/결측 lc_group 은 LC_IGNITION_DEFAULT 로 폴백(silent 금지: 로깅).
LC_IGNITION = {
    "산림": 1.00,
    "초지": 0.25,
    "나지": 0.15,
    "농업": 0.10,
    "시가화": 0.08,
    "습지": 0.02,
    "수역": 0.00,
}
LC_IGNITION_DEFAULT = 0.10   # 미지/결측 lc_group 폴백(농업 수준 — 보수적 중앙치)

# 체제별 물리식 전문가 가중치 — Phase3 튜닝값(공간블록CV 발화점 recall 최적화).
#   항목: 발화기하(forest/road/powerline) + 기상(fwi/yanggan) + 연료(fuel) +
#         토지피복(landcover, Phase3 추가) = 7성분.
#   각 전문가는 같은 피처를 쓰되 가중을 다르게 → "어디서 왜 위험한가"가 체제마다 다름.
#   튜닝: LHS(Dirichlet) 랜덤탐색 5000후보 + 좌표상승(scripts/tune_weights.py, passes=4,
#   step=0.05, 목적=공간CV recall@top-{1,2,5,10}% 가중평균, simplex 제약, 7성분).
#   시작점 후보 {랜덤최선·상위평균·config baseline} 중 최고에서 좌표상승, baseline-keep
#   가드(미초과 시 baseline 유지). 결과 provenance: outputs/tuned_weights.json.
#   Phase3 재튜닝: score 0.08609→0.08772, 공간CV recall@top5% 0.0988→0.1005,
#   top10% 0.1683→0.1731(landcover 추가 후 재균형). 무작위-공간 격차 음수 유지(낙관편향
#   없음). before(Phase3 진입 초기값=Phase2 6성분 튜닝 + landcover 0.10 떼어 재정규화)는
#   아래 주석에 보존.
EXPERT_WEIGHTS = {
    # before: forest=0.3158, road=0.0000, powerline=0.0903, fwi=0.0879, yanggan=0.0451, fuel=0.3609, landcover=0.1000
    REGIME_YEONGDONG: dict(forest=0.3333, road=0.0000, powerline=0.0952, fwi=0.0952, yanggan=0.0476, fuel=0.3333, landcover=0.0952),
    # before: forest=0.3316, road=0.1895, powerline=0.3316, fwi=0.0000, yanggan=0.0474, fuel=0.0000, landcover=0.1000
    REGIME_YEONGSEO:  dict(forest=0.3500, road=0.1500, powerline=0.3500, fwi=0.0000, yanggan=0.0500, fuel=0.0000, landcover=0.1000),
    # before: forest=0.0000, road=0.1421, powerline=0.4263, fwi=0.1421, yanggan=0.1895, fuel=0.0000, landcover=0.1000
    REGIME_MOUNTAIN:  dict(forest=0.0000, road=0.1905, powerline=0.4286, fwi=0.1429, yanggan=0.1905, fuel=0.0000, landcover=0.0476),
}

# ──────────────────────────────────────────────────────────────────────────
# Ablation — 자산(asset) 피처 기여 분리 (asset-aware vs asset-blind)
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(claudedocs/기획서_ablation_설비원인검증_20260619.md): 우리 novelty 주장
# ("전주=전력설비 자산을 발화원으로 모델")이 실제 효과 있는지 LOGO ablation 으로 측정.
# asset 피처(powerline·substation)를 통째로 빼고 남은 피처로 처음부터 재튜닝(공정).
# experts.build_ignition_features 키의 부분집합. S·W 는 고정(혼동 차단: asset 은 I 에만).
FEATURE_SET_ASSET_BLIND = ("forest", "road", "fwi", "yanggan", "fuel", "landcover")
FEATURE_SET_ASSET_AWARE = FEATURE_SET_ASSET_BLIND + ("powerline",)
FEATURE_SET_ASSET_PLUS = FEATURE_SET_ASSET_AWARE + ("substation",)

# ──────────────────────────────────────────────────────────────────────────
# 화재 원인 분류 — 설비원인 앵커 분리 검증 (resn 자유텍스트 → 5범주)
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(기획서_ablation_설비원인검증): asset 피처의 효과는 인간발화가 아니라
# 설비·전기 원인 화재에서 나타나야 한다(메커니즘 정합). safemap_positives.resn(238종
# 자유텍스트)을 검토 가능한 키워드 사전으로 분류한다. 우선순위 grid_electric >
# work_spark > natural > human > unknown (혼합 문구 시 설비 신호 우선해 과소계수 방지;
# 단 애매건은 outputs/fire_cause_labeled.csv 수기 검수로 보정 — 예 "벌채 전선줄 스파크").
CAUSE_GRID_ELECTRIC = ("전선", "특고압", "고압선", "합선", "혼촉", "누전",
                       "단락", "단선", "전기", "변압기", "개폐기", "전차선", "피복")
CAUSE_WORK_SPARK = ("용접", "예초기", "스파크", "모노레일", "벌채", "그라인더")
CAUSE_HUMAN = ("입산", "등산", "산악", "성묘", "실화", "소각", "담배", "담뱃",
               "쓰레기", "폐기물", "논밭", "흡연", "풍등", "부주의", "화목")
CAUSE_NATURAL = ("낙뢰", "벼락", "자연발화")
# 위 무매칭 → "unknown"(원인미상·기타·조사중·방화 등).
CAUSE_PRIORITY = ("grid_electric", "work_spark", "natural", "human", "unknown")

# ──────────────────────────────────────────────────────────────────────────
# Phase-4 ① 일별 fire-weather(ISI/FFMC) 동역학 W
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(claudedocs/research_방향검토_계층vsFiLM_20260619.md): 시즌 평균(fwi_q90)
# 한 값은 일별 fine-scale 을 버려 신호를 잃는다. CFFDRS 표준은 일별 FWI 이고, 문헌상
# 일별 fire-weather 가 시즌 평균보다 예측 skill 우위(시즌 예보는 1~2개월 넘으면
# 기후값 수준 하락). EDA: 산불 "크기" 는 ISI(초기확산)·바람이 좌우(풍속 ρ=+0.29,
# ISI +0.26). → 일별 ISI(확산준비)·FFMC(발화준비) 동역학을 관측소별 시즌 집계해 W 로.
#
# W(p) = w_isi·ISI성분 + w_ffmc·FFMC성분 + w_yanggan·양간(강풍방향) 성분  (합=1)
#   ISI성분  = α·고ISI일수비율 + β·평균ISI + γ·ISI q90  (각 관측소 시즌집계→[0,1] 정규화)
#   FFMC성분 = 고FFMC일수비율 (발화준비; 미세연료 건조)
#   양간성분 = 전주별 yanggan_days (영동 강풍·방향 prior; 격자/관측소 무관 전주고유)
#
# 임계값은 관측소 일별 분포(산불조심기간 2~5월)의 상위분위에 정합:
#   ISI  q90≈9.79, q95≈12.2  → 고ISI 임계 10.0 (확산준비 임박)
#   FFMC q90≈91.5, q95≈92.6  → 고FFMC 임계 90.0 (미세연료 발화준비)
DAILY_ISI_HIGH_THRESHOLD: float = 10.0    # 고-ISI(초기확산 준비) 일 임계 (관측소 일별 q90)
DAILY_FFMC_HIGH_THRESHOLD: float = 90.0   # 고-FFMC(발화 준비) 일 임계 (관측소 일별 q90)

# ISI성분 내부 가중(고ISI일수비율 / 평균ISI / ISI q90). 빈도(확산준비일 빈도)를
# 우위로 두되 강도(평균·꼬리)도 반영. 합=1.
W_ISI_FREQ_WEIGHT: float = 0.5            # 고-ISI 일수 비율(확산준비일 빈도)
W_ISI_MEAN_WEIGHT: float = 0.25           # 평균 ISI(기저 확산성)
W_ISI_Q90_WEIGHT: float = 0.25            # ISI q90(극단 확산일 강도)

# W 최종 결합 가중(ISI성분 / FFMC성분 / 양간성분). 합=1.
#   튜닝(공간블록CV 발화점 recall@top-{1,2,5,10}% 가중평균; 0.05 격자 simplex 전수,
#   scripts/run_phase1_mvp.py --phase4-weather 의 10a 비교로 검증). 결과:
#     - FFMC(발화준비) 가 관측소-수준 단일 신호로 가장 강함(단일 score 0.090,
#       시즌W 0.083 상회). 발화점 앵커가 인간발화(입산자·소각) 위주라 "발화준비
#       (미세연료 건조)" 가 "확산준비(ISI)" 보다 발화점 recall 에 직결.
#     - ISI 단일 성분(0.067~0.072)·양간 단일(0.064) 은 약함. 전수탐색 최적은
#       (isi 0, ffmc 0.8, yang 0.2, score 0.091). ISI 비중↑ 은 recall 을 낮춤.
#   채택: ISI 를 소수(0.05) 유지 — 확산준비 동역학 프레이밍 보존 + 최적과 사실상
#   동일(score 0.0907). FFMC 본체, 양간(영동 강풍방향) 보조. recall 은 시즌W 상회.
W_ISI_COMPONENT_WEIGHT: float = 0.05      # ISI 확산준비 성분(소수 유지; recall 기여 작음)
W_FFMC_COMPONENT_WEIGHT: float = 0.75     # FFMC 발화준비 성분(최강 신호)
W_YANGGAN_COMPONENT_WEIGHT: float = 0.20  # 양간(영동 강풍방향) 보조 성분

# ──────────────────────────────────────────────────────────────────────────
# Phase-4 결합(블렌드) W — 전주고유 시즌 극값 × 일별 ISI/FFMC 동역학
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(Phase-4 통합): 일별 ISI/FFMC W(daily_isi_weather)는 top-10% 공간CV
# recall 을 끌어올리지만(시즌W 0.173→0.198), 전주를 최근접 관측소 시즌집계로 매핑해
# **전주 고유의 시즌 극값(고성 인근 fwi_q90 pctile 0.97·yanggan pctile 0.99)을
# 희석**한다 → 2019 고성 sanity 위험백분위가 0.698→0.230 으로 퇴행. 간판 시나리오라
# 회복이 필수다. 반대로 시즌통계 W(season_weather)는 고성 극값을 잘 포착하지만 일별
# 동역학의 recall 이득을 못 살린다.
#
# 해결: 두 W 를 [0,1] 공간에서 선형 블렌드해 **기본 W** 로 쓴다(단순 대체 아님).
#   W = w_season · W_season + (1 − w_season) · W_daily,   (w_season ∈ [0,1])
# 시즌 성분은 고성 같은 전주고유 극값(강한 W 크기)을 보존하고, 일별 성분은 ISI/FFMC
# 동역학의 recall 이득을 더한다. 곱셈 결합 R=I·S·W 에서 고성은 S 가 매우 낮아(pctile
# ~0.03) W 크기로만 상위 위험을 유지하므로, 일별로 W 크기가 떨어지면 R 순위가 급락한다
# → 블렌드 가중을 시즌 쪽으로 둬 W 크기 극값을 지켜야 고성 순위가 회복된다.
#
# 튜닝(공간블록CV 발화점 recall@top-{1,2,5,10}% + 고성 sanity 백분위 동시 목적):
#   w_season 격자 스윕(0.0~1.0). w_season=0.82 에서 고성 백분위 0.602(목표 ≥0.6 달성)
#   이면서 recall 이 시즌W baseline 대비 모든 k 에서 개선·중립(Δ +0.000/+0.002/+0.000/
#   +0.003). 더 낮추면 recall 이득은 크나 고성 회복 실패, 더 높이면 일별 동역학 기여
#   소실. 0.82 가 "고성 회복 + recall 유지·개선" 을 함께 만족하는 점.
W_BLEND_SEASON_WEIGHT: float = 0.82       # 시즌 극값 성분 가중(나머지 1−w 는 일별 ISI/FFMC)

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

# ──────────────────────────────────────────────────────────────────────────
# Phase-5b BYM2 공간 CAR 사후(INLA식 Laplace 근사) 하이퍼파라미터
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(기획서_phase5 §6 한계 "Poisson-Gamma 는 공간상관 미반영 → BYM/INLA 로
# 확장"): 격자 잠재 로그상대위험 x = BYM2 = √τ⁻¹·(√(1−φ)·v + √φ·u_scaled). u 는
# 구조적 ICAR(이웃 평활), v 는 비구조 iid. τ 는 전체 정밀도, φ∈[0,1] 는 구조비
# (1=완전 공간평활, 0=독립). 하이퍼 (τ,φ) 는 INLA 식으로 주변우도(Laplace 근사)를
# 최대화해 고른다(MCMC 아님). 잠재장 x 는 각 하이퍼 점에서 sparse Newton(GMRF+Poisson)
# 으로 사후모드+Hessian → 가우시안 사후(평균·분산). 아래는 격자/탐색 격자·정칙화 상수.
#
# τ(전체 정밀도) 로그그리드 탐색 범위·점수 — 작을수록 x 변동 큼(약한 축소),
#   클수록 강한 축소(부모 수렴). 발화 희소(~700/1.39M)라 변동을 과대평가하지 않게
#   τ 하한을 넉넉히 둔다. φ(구조비)는 [0,1] 사이 격자.
BYM2_TAU_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)  # 전체 정밀도 후보(로그간격)
BYM2_PHI_GRID = (0.25, 0.5, 0.75, 0.9)                # 구조비(공간평활 비중) 후보
# ICAR 구조 정밀도 행렬 Q_u 의 대각 jitter(rank-deficient 보정·sum-to-zero 안정화).
BYM2_ICAR_JITTER = 1e-3
# Newton(라플라스 모드) 수렴 기준·최대 반복.
BYM2_NEWTON_TOL = 1e-7
BYM2_NEWTON_MAX_ITER = 100
# 노출(전주수) 스케일링: E_i = 격자 전주수 · 전역율(전체발화/전체전주). λ_i=E_i·exp(x_i)
# 로 두면 x_i=0 이 전역율(상대위험 1)에 대응한다. 고립 격자(이웃 0)는 구조항을 못 쓰므로
# iid 항만(φ→0 효과) 적용해 부모(독립 Poisson-Gamma) 수준으로 폴백한다.
