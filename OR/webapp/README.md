# 산불위험 예상도 웹앱

데이터(발화점·고위험 비율)를 **웹페이지에서 입력하면 "어디가 위험한지"** 를 지도로
보여주는 로컬 웹앱. 무거운 물리 위험(`R = I·S·W`)은 시작 시 한 번 계산·캐시하고,
입력은 지역배율·배분만 다시 적용하므로 초 단위로 반응한다. Python 표준 라이브러리만
쓰며(**추가 설치 불필요**), 지도는 Leaflet+OpenStreetMap(브라우저에서 인터넷 필요).

## 실행

```bash
# OR/ 에서 (가상환경 .venv)
../.venv/bin/python webapp/server.py --port 8642
# → 브라우저에서 http://127.0.0.1:8642 접속
```

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--port` | 8000 | 서비스 포트(점유 시 다른 값으로) |
| `--exposure-v2` | off | `R_base` 에 풍하 노출 v2.2 결합(최초 계산 느림) |
| `--rebuild` | off | 엔진 캐시(`outputs/predict/_engine_cache.parquet`) 강제 재계산 |

첫 실행은 물리 위험을 계산해 캐시로 저장한다(빠른 경로 ~12초). 이후 실행은 캐시를
즉시 로드한다(~3초). 기상을 바꾸려면 `--rebuild` 로 재계산한다(입력 기상 교체는
`scripts/predict.py --weather <폴더>` 참조).

## 화면에서 하는 일

- **① 고위험 비율(%)** 슬라이더 — 전체 전주 중 상위 몇 %를 "고위험(1)"으로 볼지.
  움직이면 붉은 테두리(선정 격자)가 즉시 다시 그려진다.
- **② 발화점 CSV 업로드** — 과거/가상 발화점을 넣으면 ⑴ 지역 위험배율(계층 EB)을
  재추정하고 ⑵ **체제앵커 배분** 선택 시 고위험 예산을 발화밀도 비례로 나눠, 실제
  불이 났던 지역으로 위험 판정이 이동한다. 파란 점으로 지도에 표시된다.
- **③ 불 나면 번짐(확산 시뮬)** — "🔥 발화점 찍기 모드"를 켜고 **지도를 클릭하면**,
  그 지점에 불이 났다고 보고 기존 풍하 확산 커널(`pfire.exposure.simulate_exposure`)로
  바람 타고 번지는 **노출확률(반경 5km)** 을 지도에 덮어 그린다. 풍향·풍속 슬라이더로
  바람을 바꾸고(서풍 270°→동쪽으로 번짐), 발화점🔥과 풍하 화살표가 함께 표시된다.
  클릭 응답은 1초 미만(발화원 1개라 반경 내만 계산). CLI 버전은 `scripts/spread_demo.py`.
- **결과 요약** — 고위험 전주 수와 4개 체제(영동/회랑/영서/산간)별 분포.
- **색** = 격자 평균 위험 백분위(YlOrRd), **붉은 테두리** = 고위험 선정 격자.

### 발화점 CSV 형식

헤더에 경도·위도 열이 있으면 된다(대소문자 무관). 없으면 앞 두 열을 `lon,lat` 로 본다.

```csv
lon,lat
128.59,38.21
128.47,37.75
```

인식하는 열 이름: `lon|longitude|lng|x|경도`, `lat|latitude|y|위도`.
강원 범위(경도 120~135, 위도 30~45) 밖 좌표는 무시한다.

## 구성

| 파일 | 역할 |
|---|---|
| `server.py` | 표준 라이브러리 HTTP 서버. `GET /` 페이지, `POST /api/predict` 위험지도 JSON |
| `engine.py` | 물리 위험 캐시 + 입력별 지역배율·배분 재적용(`pfire`·`scripts/predict` 재사용) |
| `index.html` | Leaflet 지도 + 입력 폼(슬라이더·CSV 업로드·배분 토글) |

## API

`POST /api/predict` — 시즌 위험분류 지도. 본문 JSON:

```json
{ "prevalence": 0.02, "alloc": "global|regime_anchor", "fires": [[lon,lat], ...] }
```

응답: `{cells:[{b,c,r,rx,n,h,g}], center, n_total, n_high_total, regime_dist, legend, ...}`
— `cells` 는 격자 셀(bbox·색·평균/최고 위험백분위·전주수·고위험수·우세체제).

`POST /api/spread` — 발화점 확산 노출. 본문 JSON:

```json
{ "lon": 128.52, "lat": 38.28, "wind_deg": 270, "wind_speed": 9 }
```

응답: `{ignition:[lat,lon], downwind_deg, radius_km, cells:[{b,c,p}], n_exposed, max_prob}`
— `cells` 는 노출확률 격자(≈450m), `p` 는 셀 최대 노출확률.

> 참고: 발화점은 per-pole 라벨이 아니라 **지역 앵커**다(README §1 결정 철학). 입력
> 발화점은 위험 자체(`R=I·S·W`)를 바꾸지 않고, 지역배율과 배분 예산에만 반영된다.
