# exposure_v2 — 변별 가능한 풍하 노출 재설계

**작성** 2026-06-19 · 격리 실험(다른 세션의 `pfire/` 미접촉). 검증 후 통합.

## 0. 문제 진단 (왜 포화되나)

현 커널(CONTRACT):
```
P(노출)[p] = 1 − ∏_g (1 − reach_g→p)          # 발화원 g 들에 대한 OR
reach = exp(−d/L)·fuel·(1+β·south),  L = L0·(1+α·max(0,align)·ws/5)
```
- 발화후보(I 상위 2%, ~8000개) 다수 + 영동 양간 서풍 → **거의 모든 영동 전주가 *어떤* 발화원의 풍하 5km 안** → OR이 1로 포화.
- 실측: 영동 P(노출) 평균 **0.90**, p90 **1.0** → **영동 *안에서* 변별력 ≈ 0.**
- 본질: 이 식은 "근처에 발화원이 *있나*(=어디나 yes)"를 잼. "주변 대비 *얼마나* 노출"이 아님.

## 1. 연구 합성 — 비포화·변별 노출의 정석 4가지

### (A) OR 확률 → **기대 도즈(누적)** — 포화의 근본 해결
- 번 확률(BP)은 "burn 횟수 / 시뮬 횟수"라 고활동 지역서 1로 포화. **Ager source–sink**는 대신 *"그 지점으로 번져 오는 *기대 화재 수*"* 로 정의 → **누적이라 포화 안 함, 본질적으로 변별적.**
- 근거: Ager, Vaillant, Finney & Preisler (2012) "Analyzing wildfire exposure and source–sink relationships." *Forest Ecol. Manage.* 267:271–283. [DOI 10.1016/j.foreco.2011.11.021](https://doi.org/10.1016/j.foreco.2011.11.021) · [USFS](https://research.fs.usda.gov/treesearch/41721)
- **함의**: `P = 1−∏(1−reach)` (OR) → `dose = Σ_g w_g·reach_g→p` (가중합). 누적이라 안 포화.

### (B) **발화원을 발화확률로 가중** — "있을 법한 불에 대한 노출"
- Ager 2012 source–sink: 노출 = *어디서 불이 실제로 나서 번지나*의 함수. 모든 후보를 동일 취급하지 않음.
- **함의**: `w_g = I_g` (발화 성향) → `dose(p)=Σ_g I_g·reach_g→p`. 후보 균일취급(포화 원인) 제거.

### (C) **비등방 커널을 날카롭게** — 좁고 긴 풍하 회랑
- **Anderson (1983)** *Predicting Wind-Driven Wildland Fire Size and Shape*, USFS RP INT-305 ([FRAMES](https://www.frames.gov/catalog/27010)): 길이/폭비(LWR)를 풍속의 함수로 — 운영식(FARSITE/ELMFIRE):
  `LWR = min(0.936·e^{0.2566·U} + 0.461·e^{−0.1548·U} − 0.397, 8)`, U=중풍속(mph). 무풍→1(원형), 강풍→길고 좁음(상한 8).
- **Richards (1990)** "An elliptical growth model of forest fire fronts." *IJNME* 30(6):1163–1179 ([DOI 10.1002/nme.1620300606](https://doi.org/10.1002/nme.1620300606)): 각 점을 풍향 정렬 **타원 wavelet**로 — FARSITE/FlamMap 화염형 모델.
- 한국 foehn: *Windstorms in the Lee of the Taebaek Mountains* (Atmosphere 2020, 11(4):431, [DOI 10.3390/atmos11040431](https://doi.org/10.3390/atmos11040431)) — 양간풍은 지형 협곡으로 **국지 plume**으로 집중(균일장 아님). + 일반 foehn(Santa Ana/Diablo) 회랑 집중 ([arXiv:2007.01439](https://arxiv.org/pdf/2007.01439)).
- **함의**: 넓은 부채꼴 → **LWR(U) 타원**(풍향 정렬). 강풍일수록 좁고 긴 회랑 → 발화원당 닿는 전주 ↓ → 포화 완화 + 실제 회랑 집중.

### (D) **국지 baseline 정규화** — 포화된 장을 상대화
- **Getis–Ord Gi\*** (Getis & Ord 1992; Ord & Getis 1995, [Wikipedia](https://en.wikipedia.org/wiki/Getis%E2%80%93Ord_statistics)) · **LISA/Local Moran** (Anselin 1995): 값을 **이웃 대비**로 잰 국지 hot-spot. + 질병지도 상대위험(SMR: 관측/국지기대).
- **함의**: `dose` 산출 후 **국지 이웃 평균으로 나눔**(상대 노출) → "영동은 다 높음" 배경 제거 → 회랑/깔때기가 *주변 대비* 튀게.

### (E) (옵션·대공사) **최소이동시간(MTT) 도착시간 그라데이션**
- **Finney (2002)** "Fire growth using minimum travel time methods." *Can. J. For. Res.* 32(8):1420–1424 ([DOI 10.1139/x02-068](https://doi.org/10.1139/x02-068)): 노드망 최소이동시간 → **연속 화재 도착시간** 면. 병렬화 쉬움(우리 Rust와 궁합).
- **함의**: 독립 발화원 커널 → **풍/경사 비등방 cost-distance 도착시간**. 도착 빠를수록 노출↑. 가장 원리적이나 구현 큼.

## 2. 단계별 재설계 (exposure_v2)

| 단계 | 변경 | 노력 | 기대 |
|---|---|---|---|
| **v2.0** | OR → **발화가중 기대도즈** `dose=Σ_g I_g·reach` | 작음(합으로 교체) | 포화 제거 핵심 |
| **v2.1** | 커널을 **LWR(U) 타원**으로(풍향정렬) | 중 | 회랑 집중·발화원당 과닿음↓ |
| **v2.2** | **국지 baseline 정규화**(이웃평균/Gi\*) | 중 | 영동 배경 제거→상대 변별 |
| **v2.3** | (옵션) **MTT 도착시간** cost-distance | 큼 | 가장 원리적 그라데이션 |

**핵심 식 (v2.0~2.2):**
```
dose(p)  = Σ_g  I_g · exp(−[(d∥/L)² + (d⊥/W)²]^½) · fuel_p · (1+β·south_p)   # 타원, L/W=LWR(U)
expo(p)  = dose(p) / local_mean(dose; 반경 r)          # 국지 상대화(배경 제거)
expo01   = 백분위 / 로그-minmax 로 [0,1]                # 위험 결합용 스케일
```

## 3. 검증 계획 (누수안전, 기존 하네스 재사용)

1. **포화 해제 확인**: 영동 *내* expo 의 std/IQR 가 현재(≈0)보다 유의하게 ↑ (변별 생김).
2. **회랑 sanity**: 2019 고성 east_frac≈1 유지 + **회랑 따라 expo 가 *변화*** (1.0 평탄 아님).
3. **결정 기여**: `R=1−(1−R발화)(1−w·expo01)` 누수안전 공간CV — 포화 풀린 expo가 recall 을 올리나(또는 최소 안 무너지나) + **2019/2022 burn-scar 겹침** 개선.
4. **정직 캡션**: KEPCO 라벨 검증 불가 → 물리·ops + 민감도. 도즈는 *프록시*(실제 시뮬레이터 아님).

## 4. 구현 메모
- 먼저 **numpy 프로토타입**(다운샘플/영동 부분집합으로 빠르게) → 변별·CV 확인 후 **Rust 커널 포팅**(`pfire_kernels`와 동형, 합·타원·이웃평균 추가).
- 입력은 `exposure_engine` 준비물(평면km 좌표·발화후보·바람표집) 재사용. `I_g` 가중은 발화후보의 I 값.

## 5. v2.0 결과 (실측, 2026-06-19)

프로토타입 `proto_v0_dose.py`(영동 부분집합) · `proto_v0_fulln_cv.py`(전 전주) · `make_flagship_fig.py`.

**① 포화 해제 — 성공.** 영동 *내* 변별(IQR):
| | 옛 OR | v2.0 dose |
|---|---|---|
| 영동 IQR | 0.012~0.027 | **0.166** (≈6~10배) |
| 분포 | 82%가 ~1.0 벽 | 매끈한 우편향 그라데이션 |

**② 그러나 decision recall — 안 오름.** 전 전주 누수안전 공간CV:
| 변형 | 평균 recall | 영동 top2% share |
|---|---|---|
| baseline R_발화 | 0.0832 | 0.950 |
| OR dose w=0.25 | 0.0836(+0.0004) | 0.971 |
| OR dose w=0.5/1.0 | 0.0802 / 0.0753 ⬇ | 0.986 / 1.000 |

→ dose는 (물리적으로 옳게) **영동 회랑을 더 밀어올려** top을 영동에 더 쏠리게 하는데, **발화 앵커(인간발화)는 거기 적어** recall 안 오름. 방향 sanity는 완벽(2019 고성 east_frac=1.00).

**③ 결론(v2.0까지):** v2.0 iso dose 는 영동 회랑을 더 밀어올려 recall 무기여(영동 쏠림). **단 이 결론은 v2.2(국지정규화)가 뒤집음 → §6.** v2.0 dose 도 노출 *표현/ops* 는 포화 binary → 변별 그라데이션으로 크게 개선. flagship `fig_flagship_exposure.png`, 산출 `outputs/pole_dose_v2.parquet`.

## 6. v2.1 타원 + v2.2 국지정규화 — 확정 결과 (2026-06-19)

`proto_v1_ellipse.py`(타원·국지정규화) · `proto_v2_confirm.py`(w스윕×5seed). 산출 `outputs/pole_dose_v2_ellipse.parquet`, 그림 `fig_v1_corridor.png`.

| 변형 | 영동 IQR | 비고 |
|---|---|---|
| v2.0 iso | 0.166 | — |
| v2.1 LWR 타원 | 0.171 | 미미(타원만으론 거의 무변화) |
| **v2.2 타원+국지정규화** | **0.290** | 변별 최강 |

**핵심 — v2.2 가 recall 을 *확정* 으로 올림** (5 fold-seed mean±std):
| w | recall | Δ vs baseline | 판정 |
|---|---|---|---|
| 0.00 | 0.0788±0.0027 | — | 기준 |
| 0.25 | 0.0836±0.0039 | +0.0049 | 신호 |
| **0.50** | **0.0859±0.0031** | **+0.0071** | **신호**(>합성std 0.004) |
| 0.75 / 1.0 | 0.0820 / 0.0794 | +0.003 / +0.001 | 잡음내(과대주입) |

- w=0.5: top5% 0.1005→**0.1066**, top10% 0.1731→**0.1807**. **무작위-공간 격차 baseline −0.017 → +v2.2 +0.004 (둘 다 ≈0, 낙관편향 없음).**
- **왜**: 국지정규화(Getis–Ord식 상대노출)가 영동 배경을 걷어내 → 노출이 발화앵커가 있는 *모든* 지역 회랑을 부각 → recall↑. **"확산을 결정에 넣어야 한다"가 올바른 공식(de-sat+국지정규화)으로 데이터로 입증.**
- **한계**: +0.0071은 *proxy* recall(인간발화 앵커) 기준·절대값 modest. KEPCO 실타깃 미지. 단 5seed 일관+편향0+방향정확(east 0.98)+원리적.

**채택**: `R = 1−(1−R발화)(1−0.5·dose_v2.2)` (w=0.5). `integrate_v2blend.py` 가 standalone 으로 적용·검증(파이프라인 `run_phase1_mvp` merge 는 다른 세션 멈춘 뒤).



## 부록. 출처 검증 flag
- 1차 검증: Anderson 1983(식 ELMFIRE 기술문서 대조), Richards 1990(DOI/ADS), Ager 2012(DOI/USFS), Finney 2002(DOI 10.1139/x02-068, CJFR 32(8):1420), Getis–Ord(정의).
- flag: Atmosphere 2020(양간 foehn) 본문 403 — 제목/DOI/권호만 확인, 저자 미검증. LWR 계보 Alexander 1985 는 FARSITE 문서 경유 2차.
