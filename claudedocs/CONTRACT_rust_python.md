# Rust ↔ Python 인터페이스 계약 (두 에이전트 공통)

> 이 문서는 **변경 금지 계약**이다. Python(Agent 1)과 Rust(Agent 2)는 이 시그니처에만 의존한다.
> Rust 커널이 없어도 Python은 `exposure.py`의 **순수 numpy 폴백**으로 end-to-end 돌아가야 한다.

## 모듈/빌드
- Rust 크레이트: `rust/pfire_kernels/` → maturin/PyO3 로 **`pfire_kernels`** 파이썬 모듈 빌드.
- 설치: `cd rust/pfire_kernels && maturin develop --release` (venv: `/home/dlwhdtmd/OR-project/.venv`).
- Python에서: `import pfire_kernels` (실패 시 `pfire/exposure.py`가 numpy 폴백 사용).

## 단일 진입 함수

```python
def simulate_exposure(
    pole_xy: np.ndarray,        # (N, 2) float64 — 전주 좌표, 평면 근사 km (config.LAT0_DEG 기준)
    ignition_idx: np.ndarray,   # (M,) uint32 — pole_xy 내 "발화 후보" 전주 인덱스
    wind_dir_deg: np.ndarray,   # (S,) float64 — MC 표집 풍향(기상학적 from-방향, 도). S=시뮬 수
    wind_speed: np.ndarray,     # (S,) float64 — MC 표집 풍속 (m/s)
    fuel: np.ndarray,           # (N,) float64 — 전주별 연료/가연성 [0,1]
    southness: np.ndarray,      # (N,) float64 — 남서사면 정렬도 [-1,1]
    max_dist_km: float,         # 확산 컷오프 반경 (config.SPREAD_MAX_DIST_KM)
    length_scale_km: float,     # 거리 감쇠 L0 (config.SPREAD_LENGTH_SCALE_KM)
    wind_aniso: float,          # 풍하 신장 α (config.SPREAD_WIND_ANISO)
    southness_beta: float,      # 남서사면 보정 β (config.SPREAD_SOUTHNESS_BETA)
    seed: int,                  # 재현성 시드 (config.SEED)
) -> np.ndarray:                # (N,) float64 — 전주별 P(노출) ∈ [0,1]
    ...
```

## 의미론(semantics) — 비등방 풍하 확산 프록시
시뮬 s마다 풍하방향 θ_s = wind_dir_deg[s] + 180° (from→to). 각 발화원 g에서 반경 max_dist_km 내 전주 p에 대해:
```
d   = ||g - p||                      (km, 평면근사)
φ   = angle(g → p)                   (도)
align = cos(φ - θ_s)                 풍하 정렬도 [-1,1]
L   = length_scale_km * (1 + wind_aniso * max(0, align) * (wind_speed[s] / 5.0))   풍하로 신장
reach_prob = exp(-d / L) * clip(fuel[p],0,1) * (1 + southness_beta * southness[p])
reached(p) |= (rng.uniform() < reach_prob)         시뮬 s에서 p가 불에 닿았나
```
전주별 P(노출) = (S 시뮬 중 reached 된 비율). 발화원 여러 개면 시뮬당 OR 결합.

## 병렬/성능 요구
- **rayon**로 시뮬(S) 또는 발화원(M) 축 병렬화. 64코어 활용.
- N≈1.38M, M≈수천, S=256 에서 수십 초 내 목표. 공간 인덱스(격자 버킷/KD-tree)로 반경 질의 가속.
- 결정성: 같은 seed → 같은 결과(시뮬별 독립 시드 = seed + s).

## 폴백(필수)
`pfire/exposure.py`는 `try: import pfire_kernels` 실패 시 동일 시그니처의 **numpy 벡터화 폴백**을 제공.
폴백은 느려도 되고(작은 M·S로 스모크), 결과 형상·의미가 같아야 한다. Rust 설치 시 자동으로 Rust 사용.
