# 연구 강점 문헌 조사 — 전력설비(전주) 산불위험 조기경보

**작성** 2026-06-19 · **목적** 우리 설계의 각 기둥을 뒷받침하는 최신 논문을 재조사하고, "우리 연구의 강점"을 인용으로 입증. **방법** 7개 축 병렬 웹조사(WebSearch+WebFetch), 각 출처 URL 검증.

> 모든 인용은 실재·검증된 것. WAF/쿠키월 차단으로 본문 직접 fetch가 막힌 일부는 Crossref/arXiv 메타데이터로 교차검증했고 아래에 **flag**로 표시. 수치는 출처에 충실하게 인용.

---

## 한눈에 — 우리의 강점 8가지와 근거

| # | 우리 설계 | 강점(왜 강한가) | 핵심 근거 |
|---|---|---|---|
| 1 | 발화점을 학습이 아닌 **검증·임계 앵커**로만 사용 (presence-only/PU) | "기록 없음 ≠ 위험 없음"을 정면 처리. 거짓 음성 안 만듦 | Kiryo 2017(nnPU), Phillips 2009, 산사태 PU 2024 |
| 2 | **물리식 하이브리드** I×S×W | 순수지수 대비 +40%, 순수ML 대비 +9% F1, **해석가능** | Li 2024 (F1 0.846) |
| 3 | 위험 = **발화 × 확산 × 기상** 곱셈분해 | 산불위험 표준 독트린(likelihood×intensity=hazard) | Scott 2013, USFS WRC |
| 4 | **지역 전문가 혼합(MoE)** soft 게이트 | 전역단일/개별모형의 원리적 중간(partial pooling) | Jacobs 1991, Shazeer 2017, Gelman&Hill 2007 |
| 5 | **공간 블록 CV** + 무작위-공간 격차 보고 | 무작위CV는 성능을 >50%→0까지 부풀림. 우리는 격차≈0 | Ploton 2020, Roberts 2017 |
| 6 | **풍하 MC 노출**(앙상블 연소확률) | 2개 대륙 운영표준(FSim·Burn-P3)과 동일 패러다임 | Finney 2011, Rothermel 1972 |
| 7 | **양간지풍 방향 prior** | 강원 특이 푄(20.4~27.6 m/s) — 일반 바람이 아닌 결정적 기작 | KOSHAM 2021, 기상관측 |
| 8 | **KEPCO 자산 우선순위화** | 기존 강원 ML은 자산무관(asset-blind); 우리는 전주 결합 = 차별점 | Lee 2025(AUC 0.839, 자산무관), Hennessy 2025 |

---

## 축 1 — Presence-only / PU 학습 (강점 #1)

- **Kiryo, Niu, du Plessis, Sugiyama (2017). "Positive-Unlabeled Learning with Non-Negative Risk Estimator." NeurIPS 2017.** arXiv:1703.00593 · https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html
  - 비편향 PU risk(uPU)는 음수로 발산해 과적합 → **nnPU**가 위험을 비음수로 제약해 딥/유연모델 안정학습. **양성+미분류만으로 충분**, 진짜 음성 불필요.
- **Phillips, Anderson, Schapire (2006). "Maximum entropy modeling of species geographic distributions." Ecological Modelling 190:231–259.** https://www.sciencedirect.com/science/article/pii/S030438000500267X
  - presence-only + 배경표집으로 분포 모델링하는 MaxEnt 정초. 산불 발화점 = 종 출현점과 동일 데이터형.
- **Phillips et al. (2009). "Sample selection bias and presence-only distribution models." Ecological Applications 19(1):181–197.** DOI 10.1890/07-2153.1 · PDF https://www.whoi.edu/cms/files/Phillips_EcolApp_2009_53454.pdf
  - 공간편향이 환경편향으로 전이 → **배경/의사음성 선택이 모델만큼 중요**. 우리의 "음성 조작 금지 + 층화 배경" 정당화.
- **(산사태 PU, 2024). "Enhancing landslide susceptibility mapping using a positive-unlabeled ML approach." Geoenvironmental Disasters.** DOI 10.1186/s40677-024-00281-w
  - "기록 없는 지역이 저위험인 것은 아니다 — 아직 안 났을 뿐" → **PU가 이진분류보다 AUC·recall 향상**. 자연재해 최근접 사례. *flag: 본문 쿠키월, 메타+스니펫 교차검증.*
- **Choi & Chae (2024). "Assessing Wildfire Risk in South Korea ... MaxEnt and SSP." Atmosphere 16(1):5.** https://www.mdpi.com/2073-4433/16/1/5
  - 한국 산불을 MaxEnt presence-only로 모델 — **국내 선례**. *flag: 403, DOI 자릿수(10.3390/atmos16010005) 최종확인 필요.*
- **(2023). "Identifying ... causes of wildfires by maximum entropy ... ignition susceptibility." J. Forestry Research 34(2):355–371.** DOI 10.1007/s11676-022-01502-4
  - 발화점을 presence-only "종분포" 데이터로 다룸 — 산불 문헌 내 직접 선례. *flag: 쿠키월.*
- (보조) **arXiv:2304.09305 (2023)** — 미분류를 음성취급하면 체계적 편향(미분류엔 진짜음성+오라벨양성 혼재) 명시.

## 축 2 — 물리-하이브리드 + 위험 분해 (강점 #2,#3)

- **Li, Zhu, Yuan et al. (2024). "Projecting Large Fires in the Western US With an Interpretable and Accurate Hybrid ML Method." Earth's Future 12(10):e2024EF004588.** DOI 10.1029/2024EF004588
  - 연료 가연성·가용성·인간억제를 **명시적으로 표현**한 하이브리드가 **F1 0.846±0.012, 지수 대비 +40%·ML 대비 +9%**, 해석가능성↑, **물리원리와 정합**. *flag: 본문 WAF, 수치는 Crossref 등재 초록서 인용.*
- **Scott, Thompson, Calkin (2013). "A wildfire risk assessment framework..." USDA RMRS-GTR-315.** https://research.fs.usda.gov/treesearch/56265
  - 위험 = **likelihood × intensity(=hazard)** × 가치자원. 우리 I·S 분리의 독트린 근거.
- **USDA Forest Service — Wildfire Risk to Communities.** https://wildfirerisk.org/understand-risk/
  - "위험 = likelihood × intensity(hazard) + exposure·susceptibility(vulnerability)", 기상·지형·발화 변동. 현재 운영체계가 **I×S×W 구조와 동일** → 우리 곱셈결합이 표준.
- **Singh et al. (2024). "Trending and emerging prospects of physics-based and ML-based wildfire spread models: a review." J. Forestry Research 35:135.** DOI 10.1007/s11676-024-01783-x
  - 순수물리=발화/파라미터 약함, 순수ML=해석성 약함 → **하이브리드 권고**. 우리 철학을 분야리뷰가 지지.
- **Yeo-Chang et al. (2026). "Human Activities and Wildfires..." Fire 9(6):246.** DOI 10.3390/fire9060246
  - 강원·경북: 대부분 인위적, 그러나 **현 한국 예보체계는 인위요인 미반영**(=우리 I항이 메우는 갭). 발화인지 변수로 설명력 ~1.3배. *flag: 저자명 Crossref 일부 깨짐, PDF 대조 필요.*
- **Dorph et al. (2022). "Modelling ignition probability for human- and lightning-caused wildfires in Victoria." NHESS 22:3487–3499.** DOI 10.5194/nhess-22-3487-2022
  - **인프라 근접(도로/주택거리)이 인간발화 최강 예측인자**, 정확도 86.4~90.3%. 우리 forest/road/powerline 피처 직접 검증.

## 축 3 — MoE / 부분풀링 / 공간 비정상성 (강점 #4)

- **Jacobs, Jordan, Nowlan, Hinton (1991). "Adaptive Mixtures of Local Experts." Neural Computation 3(1):79–87.** DOI 10.1162/neco.1991.3.1.79 — MoE 정초(게이트가 전문가에 입력 배분).
- **Shazeer et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer." ICLR 2017.** arXiv:1701.06538 — **학습형 softmax 게이트**로 전문가 가중결합. 우리 soft 게이트의 현대형.
- **Gelman & Hill (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge UP.** DOI 10.1017/CBO9780511790942 — **부분풀링**=완전풀링(1모델)과 무풀링(개별모델)의 데이터기반 중간. 우리 MoE의 통계적 정당화.
- **Brunsdon, Fotheringham, Charlton (1996). "Geographically Weighted Regression..." Geographical Analysis 28(4):281–298.** DOI 10.1111/j.1538-4632.1996.tb00936.x — **공간 비정상성**: 전역단일모델은 공간변동 관계를 못 잡음.
- **Du et al. (2020). "Geographically neural network weighted regression." IJGIS 34(7):1353–1377.** DOI 10.1080/13658816.2019.1707834 — 지리조건화의 **신경망 구현이 GWR보다 우수** = 공유백본+위치조건 가중의 직접 선례.
- **Feng et al. (2026). "Spatial Heterogeneity and Responses of Wildfire Drivers Across Diverse Climatic Regions in China." Remote Sensing 18(7):1007.** DOI 10.3390/rs18071007 — **산불 구동인자가 지역마다 다름** → 영동/영서/산간 전문가 정당화. *flag: 403, DOI는 vol 18(7), '16071007' 오기 주의.*

## 축 4 — 공간 CV + presence-only 지표 (강점 #5)

- **Ploton et al. (2020). "Spatial validation reveals poor predictive performance of large-scale ecological mapping models." Nature Communications 11:4540.** DOI 10.1038/s41467-020-18321-y
  - RF 바이오매스 모델이 **무작위검증선 >50% 설명, 공간검증선 거의 0** → 무작위CV는 과대낙관. 우리 격차≈0이 곧 "이 함정 아님"의 증거.
- **Roberts et al. (2017). "Cross-validation strategies for data with ... spatial ... structure." Ecography 40(8):913–929.** DOI 10.1111/ecog.02881 — 구조적 데이터엔 **블록 CV 필수**.
- **Valavi et al. (2019). "blockCV: An R package..." MEE 10(2):225–232.** DOI 10.1111/2041-210X.13107 — **자기상관 범위로 블록 크기 선택** → 우리 10km 정당화.
- **Sofaer et al. (2019). "The area under the precision–recall curve as a performance metric for rare binary events." MEE 10(4):565–577.** DOI 10.1111/2041-210X.13140 — 희귀·광역엔 **AUC-PR**가 ROC보다 적합(참음성 미포함).
- **Li & Guo (2021). "Plotting ROC and PR curves from presence and background data." Ecology and Evolution 11(15).** DOI 10.1002/ece3.7826 — 음성 없을 때 **PR**가 적합. 우리 PU 세팅에 직결.
- **Hirzel et al. (2006). "Evaluating the ability of habitat suitability models..." Ecological Modelling 199:142–152.** — **Boyce index**(presence-only 전용). *flag: 2006(범위밖); 최근 대체 Ecography 2024 doi 10.1111/ecog.07218.*

## 축 5 — MC 비등방 풍하 확산 (강점 #6,#7)

- **Finney et al. (2011). "A Method for Ensemble Wildland Fire Simulation." Environmental Modeling & Assessment 16(2):153–167.** DOI 10.1007/s10666-010-9241-3 · https://research.fs.usda.gov/treesearch/39311 — **풍향·풍속을 과거에서 표집**+수백~수천 시뮬 → 셀별 연소확률. 우리 노출봉투의 정확한 선례.
- **FSim (USFS).** https://research.fs.usda.gov/firelab/projects/fsim — 수억 가상 화재로 전국 확률·강도맵. "단일footprint 아닌 확률" 패러다임.
- **Burn-P3 / BurnP3+ (Canada).** https://www.canadawildfire.org/burn-p3-english — 확률적 발화+기상 + 결정적 확산 → 연소확률. **2개 대륙 독립 동일 원리**.
- **Rothermel (1972). "A Mathematical Model for Predicting Fire Spread in Wildland Fuels." USFS INT-115.** https://research.fs.usda.gov/firelab/projects/rothermelfirespread — 최대확산방향 + **타원형 풍하 신장**(wavelet). 우리 풍하 신장항의 물리근거.
- **Technosylva × PG&E.** https://technosylva.com/customers/pge/ — **하루 1억+ 시뮬**로 PSPS 결정. *flag: "배전선 발화 68%↓(2022 vs 2021)"는 실재하나 시뮬 단독 인과 아닌 **프로그램 전체 성과** — PG&E 2025 WMP에서 인용 권장.*
- **Kim, Kwak, Kim (2021). "양간지풍 특성을 고려한 동해안 대형 산불의 수치시뮬레이션." 한국방재학회논문집 21(4):39–48.** DOI 10.9798/KOSHAM.2021.21.4.39 — 2019 고성 확산을 양간지풍으로 시뮬, **실제와 근접**. 우리 지역 바람 prior의 한국 검증.

## 축 6 — FiLM / 컨포멀 (로드맵 강점)

- **Perez et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018.** arXiv:1709.07871 — γ·x+β 피처별 아핀 조건화. 지역조건 변조의 정초.
- **Turkoglu et al. (2022). "FiLM-Ensemble: Probabilistic Deep Learning via FiLM." NeurIPS 2022.** arXiv:2206.00050 — **단일망 암묵 앙상블이 명시 딥앙상블에 근접(때론 초과), 메모리 일부**. 지역조건+불확실성 한 메커니즘.
- **Lakshminarayanan et al. (2017). "Simple and Scalable Predictive Uncertainty ... Deep Ensembles." NIPS 2017.** arXiv:1612.01474 — 딥앙상블이 근사베이즈만큼/이상 잘 보정.
- **Ovadia et al. (2019). "Can You Trust Your Model's Uncertainty? ... Under Dataset Shift." NeurIPS 2019.** arXiv:1906.02530 — **분포변화에서 딥앙상블 > MC-dropout**(독립 벤치마크). 우리 지역/계절 변화 세팅에 중요.
- **Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction..."** arXiv:2107.07511 — 분포가정 없는 **유한표본 커버리지 보장**. 임계값을 ad-hoc 아닌 보장형으로.
- **Dayan (2026). "Conformal Risk Control for ... Wildfire Evacuation Mapping."** arXiv:2603.22331 — 산불에 컨포멀 위험통제(FNR≤0.05). *flag: 매우 최신·단독저자·미심사 → 보조 사례로만, 정착 근거는 Angelopoulos&Bates.*

## 축 7 — 한국/KEPCO 현실 근거 (강점 #8)

- **2019 고성-속초 산불 — KEPCO 책임 판결.** 춘천지법 속초지원(2023-04-20): KEPCO가 피해자 64명에 **87억원** 배상, 산림 1,260ha 소실, 원인 = **전주의 개폐기/전선 아크 + 강풍**. (Korea Times 보도) *flag: 구체부품은 "개폐기(switch box)" vs 2차매체 "고압선 아크" 상이 — "전력설비 아크(전선/개폐기)"로 표기, "변압기" 단정 금지.*
- **양간지풍.** 양양~간성 서풍 푄("화풍"), 태백 협곡 가속·건조. 2019.4.4 고성-속초 **순간최대 미시령 27.6 m/s·속초 20.4 m/s**. (매일신문 2019-04-05 등)
- **산림청 산불통계연보 (e-나라지표).** https://www.index.go.kr/unity/potal/main/EachDtlPageDetail.do?idx_cd=1309 — 봄 산불조심기간(2.1–5.15) **건수 66%·피해면적 99%**, 3월 31%. 원인: **입산자 실화 15% + 논밭/쓰레기 소각 19%**. → 우리 2~5월 시즌창 공식 근거.
- **Lee, Choi, Han, Lee (2025). "Year-round daily wildfire prediction ... Gangwon State." Scientific Reports.** https://www.nature.com/articles/s41598-025-15508-5 — 강원 일별 산불 ML, **최고 AUC 0.839(Extra Trees)**, 산불일 <5%(SMOTE), 63% 인위. **자산무관(asset-blind)** — KEPCO 전주 근접 미모델 = **우리 차별점**. PU/불균형 정당화.
- **Hennessy & Chester (2025). "Electric utility vulnerability to wildfires ... in California." Environ. Res.: Infrastructure & Sustainability 5(1):015019.** DOI 10.1088/2634-4505/adb90a — 송전선 17%·변전소 19%·발전소 21%가 고위험지. **자산 공간 우선순위화는 인정된 과제**, 위험은 소수자산 집중.
- **Camp Fire (2018, CAL FIRE).** PG&E 송전선 하드웨어 고장→아크(5,000~10,000°F), **85명 사망, 1.8만 건물**. 고성과 동일 인과사슬(노후 설비+강풍→아크). (NPR/Wikipedia)
- **(보조) "Spatial Prediction of Forest Fire ... Korea's Eastern Coast." Forests 17(2):281.** https://www.mdpi.com/1999-4907/17/2/281 — 한국 동해안 ML에 **인간근접** 변수. *flag: 403, 수치 미검증.*

---

## 종합 — 강점 서사 한 문단

> 우리 접근은 임시방편이 아니라 **여러 분야의 표준이 교차하는 지점**에 있다. (1) 발화점만 있는 데이터는 종분포·PU 학습의 정립된 문제이며(Kiryo 2017, Phillips 2009), 자연재해에서 PU가 이진분류를 능가한다(산사태 2024). (2) 물리지식을 명시한 하이브리드가 지수·순수ML을 능가하고 더 해석가능하다(Li 2024: F1 0.846, +40%/+9%). (3) 위험=발화×확산×기상은 美 산불위험 독트린과 동형(Scott 2013, USFS). (4) 지역 전문가 혼합은 MoE 30년 계보(Jacobs 1991→Shazeer 2017)와 부분풀링(Gelman&Hill)·공간비정상성(Brunsdon 1996→Du 2020)의 합류이며, 산불 구동인자는 실제로 지역의존적이다(Feng 2026). (5) 무작위CV는 성능을 >50%→0까지 부풀리므로(Ploton 2020) 공간블록CV가 필수이고(Roberts 2017), 우리 무작위-공간 격차≈0이 그 함정에 빠지지 않았음을 증명한다. (6) 풍하 MC 노출은 美·加 운영표준과 동일 패러다임이며(Finney 2011, Burn-P3) 타원형 풍하신장의 물리근거가 있다(Rothermel 1972). (7) 양간지풍은 일반 바람이 아니라 강원 특이 푄(20.4~27.6 m/s)으로 한국 시뮬에서 검증됐다(KOSHAM 2021). (8) 결정적으로, 가장 비교가능한 강원 ML(Lee 2025, AUC 0.839)조차 **KEPCO 전주를 모델하지 않는 자산무관** 모델이며, 자산 우선순위화의 가치는 입증됐으나(Hennessy 2025) 강원 전주에 적용된 바 없다 — 그 교집합(PU+물리/양간지풍+KEPCO 자산)이 우리의 실질적 신규성이다.
</content>
