# report/ — 6쪽 보고서 (논문형 .tex)

`report.tex` — 전력설비 산불위험 조기경보(MOSAIC) 보고서. 2단 조판, 한글(kotex), 수식·표·그림 포함.

## 컴파일

LaTeX 엔진이 이 환경엔 없어 **Overleaf** 또는 로컬 TeX(XeLaTeX 권장)에서 빌드하세요.

```bash
# 권장 (한글 폰트 안정)
xelatex report.tex && xelatex report.tex      # 2회 = 참조/목차 갱신

# 또는
pdflatex report.tex && pdflatex report.tex     # kotex 한글
```
- Overleaf: 메뉴 → Compiler = **XeLaTeX**, `report.tex` 업로드 + 아래 그림 경로 유지.

## 그림 경로

`\graphicspath` 가 다음을 잡습니다(상대경로, repo 루트 기준 `report/`에서):
```
../outputs/figures/        # fig1_risk_map, fig5_goseong_2019, fig9_coverage ...
../outputs/figures/eda/    # 04_regime_boxplots ...(EDA.ipynb 산출)
../exposure_v2/            # fig_flagship_exposure ...
```
Overleaf 업로드 시엔 그림들을 같은 폴더 구조로 올리거나 `\graphicspath` 를 평면 경로로 바꾸세요.

## 현재 본문에 배치된 그림
| 위치 | 파일 | 내용 |
|---|---|---|
| 그림1 | `eda/04_regime_boxplots.png` | 체제별 변수 분포(EDA) |
| 그림2 | `fig1_risk_map.png` | 강원 위험지도(2단 폭) |
| 그림3 | `fig_flagship_exposure.png` | OR 포화 vs v2.2 변별 노출 |
| 그림4 | `fig9_coverage.png` | 체제별 커버리지 |
| 그림5 | `fig5_goseong_2019.png` | 2019 고성 케이스 |

## EDA 그림 추가/교체 (필요시)

`outputs/figures/eda/` 에 더 있습니다 — `\includegraphics{파일명}` 으로 교체/추가:
`01_distributions` · `02_corr_heatmap` · `03_separability_aucpr` · `03_signal_map` ·
`05_isw_contribution` · `05_regime_topshare` · `06_morans_i` · `07_uncertainty` · `08_weaklabel_corr`.

모델 그림도 `fig2_regime_alloc`(편중완화) · `fig3_recall_topk` · `fig4_f1_sensitivity` ·
`fig6_ignition_decomp`(위험 분해) · `fig7/8`(시군 서브플롯) 사용 가능.

> 6쪽 기준으로 그림 수를 조절하세요(현재 5개 + 표 3개). 더 넣으면 늘어납니다.
</content>
