# psycho-stock — 군중심리로 읽는 주식시장

> "시장은 단기적으로는 투표기계, 장기적으로는 저울이다." — 벤저민 그레이엄

주가는 숫자로 표시되지만, 그 숫자를 만드는 건 사람들의 **공포·탐욕·군집 행동·손실 회피**입니다.
이 프로젝트는 가격과 거래량에 남은 대중심리의 흔적을 지표로 만들고,
**지금 시장이 어떤 감정 상태인지 → 심리 사이클의 어느 국면인지 → 그 상태에서 과거엔 어떻게 됐는지**를
확률로 보여 줍니다.

## 구조

```
시세(OHLCV) ──────┐
VIX·뉴스(선택) ────┤
시장 전체 군중(선택)┼─▶ 심리 지표 ─▶ 공포·탐욕 지수(0~100) ─▶ 심리 국면 판별 ─┐
 폭·수급·신용·풋콜  │                                                          │
 검색량·커뮤니티 ───┘                                                          ▼
                       워크포워드 학습/예측 + 데이터 소스별 기여도 (model) ─▶ 리포트
```

| 모듈 | 하는 일 |
|---|---|
| `data.py` | CSV / 야후 파이낸스 로딩, 군중심리 가상 시장 생성기 |
| `indicators.py` | 심리 지표 계산 (아래 표) |
| `sentiment.py` | 공포·탐욕 지수, 월스트리트 심리 사이클 국면 판별 |
| `news.py` | 한국어/영어 뉴스·커뮤니티 헤드라인 감정 점수 |
| `crowd.py` | 시장 전체 군중 데이터: 시장 폭, 투자자 수급, 신용잔고, 풋/콜, 검색량, 커뮤니티 |
| `model.py` | 로지스틱 회귀, 누수 없는 워크포워드 백테스트, 국면별 통계 |
| `report.py`, `cli.py` | 한국어 리포트와 명령행 도구 |

### 심리 지표 — 무엇을 '감정'으로 보는가

| 지표 | 심리학적 의미 |
|---|---|
| 20/60일 모멘텀 | **군집 행동(herding)** — 사람들은 오른 것을 더 산다 |
| 200일선 괴리 | 집단적 낙관/비관이 얼마나 누적됐는가 |
| RSI(14) | 단기 과열(탐욕) / 투매(공포) |
| 변동성 비율 | **공포는 변동성으로 드러난다** (최근 20일 / 1년) |
| 고점 대비 하락폭 | **손실 회피** — 고통의 크기 |
| 방향성 거래량 폭증 | 패닉 매도 vs FOMO 매수 |
| 상승일 비율 | 분위기의 일관성 |
| 항복(capitulation) 신호 | 큰 하락 + 거래량 폭증 + 과매도 = 마지막 투매 |
| 광기(euphoria) 신호 | 큰 상승 + 거래량 폭증 + 과매수 + 신고가 근처 |
| VIX (선택) | 옵션 시장이 매긴 공포의 가격 |
| 뉴스 감정 (선택) | 언론·커뮤니티의 말투 |

### 시장 전체 군중 데이터 (`crowd.py`, 모두 선택)

| 데이터 | 심리학적 의미 | 공포·탐욕 방향 | 받는 방법 |
|---|---|---|---|
| 상승/하락 종목 비율, 신고가-신저가, 50일선 위 종목 비율 | 상승이 소수의 잔치인가, 군중 전체의 확신인가 | 높을수록 탐욕 | `--breadth-tickers` (yfinance) 또는 `breadth.csv` |
| 개인 순매수 강도 | 개인의 추격 매수(FOMO) / 투매 | 높을수록 탐욕 | `--krx-flows KOSPI` (pykrx) 또는 `flows.csv` |
| 외국인·기관 순매수 강도 | 개인과 반대편의 '스마트 머니' | 모델에만 사용 | 위와 같음 |
| 신용잔고 증가율 | 빚내서 투자 = 레버리지 탐욕 | 높을수록 탐욕 | `credit.csv` (금융투자협회 등) |
| 풋/콜 비율 | 하락 대비 보험 수요 | 높을수록 공포 | `putcall.csv` (CBOE 등) |
| 검색어 탐욕-공포 | '주식 추천' vs '주식 폭락' 검색 | 높을수록 탐욕 | `--google-trends KR` (pytrends) 또는 `trends.csv` |
| 커뮤니티 감정 / 게시글 폭증 | 개인 투자자의 말투와 흥분도 | 감정은 탐욕 방향, 폭증은 모델에만 | `--community posts.csv` 또는 `community.csv` |

외부 데이터는 **그 날짜에 실제로 알 수 있었던 값만** 쓰도록 공개 지연을 둡니다
(신용잔고 2영업일, 주간 검색량은 그 주가 끝난 뒤). 테스트가 이를 검증합니다.

리포트의 **[6] 데이터 소스별 기여도**는 가격·거래량만 쓴 모델에 소스를 하나씩 더했을 때
워크포워드 AUC가 얼마나 오르는지 보여 줍니다. 실제 시장에서 어떤 대중심리 데이터가
돈을 들여 모을 가치가 있는지 판단하는 기준입니다.

각 지표를 **과거 1년 중 몇 %ile인지**로 바꿔 평균한 것이 공포·탐욕 지수입니다(0=극단적 공포, 100=극단적 탐욕).
절댓값이 아니라 상대 순위를 쓰기 때문에 종목·시장이 달라도 같은 척도로 비교할 수 있습니다.

### 심리 국면 (월스트리트 심리 사이클 단순화)

`의심 → 낙관 → 환희 → 불안 → 공포 → 공황 → 항복 → 절망→희망 → 의심 …`

추세(200일선 위/아래) × 감정 수준(공포·탐욕) × 감정 변화 방향(10일)으로 판별합니다.

## 설치와 실행

```bash
pip install -e ".[data,dev]"

# 실제 시장 (인터넷 필요)
psycho-stock --ticker SPY --vix                 # S&P500 + VIX
psycho-stock --ticker ^KS11 --vix ^VKOSPI       # 코스피 (VKOSPI 를 못 받으면 자동 제외)
psycho-stock --ticker 005930.KS                 # 삼성전자

# 내 데이터
psycho-stock --csv prices.csv --news examples/news_sample.csv --export result.csv

# 시장 전체 군중 데이터까지 (코스피)
export KRX_ID=... KRX_PW=...                    # pykrx 가 KRX 로그인을 요구할 때
psycho-stock --ticker ^KS11 --vix ^VKOSPI --krx-flows KOSPI --google-trends KR \
             --crowd-dir my_crowd/ --community posts.csv

# 미국: S&P500 일부 종목으로 시장 폭 계산
psycho-stock --ticker SPY --vix --breadth-tickers AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM,UNH,HD --google-trends US

# 인터넷 없이 시연 (가상 시장 + 가상 군중 데이터 전부)
psycho-stock --synthetic
```

`--crowd-dir` 폴더에는 아래 이름의 CSV 를 있는 것만 넣으면 됩니다 (형식 예시: `examples/crowd/`).

| 파일 | 컬럼 |
|---|---|
| `breadth.csv` | `date,advancers,decliners,new_highs,new_lows,pct_above_ma50` 또는 `date,<종목별 종가...>` |
| `flows.csv` | `date,individual,foreign,institution` (순매수 금액) |
| `credit.csv` | `date,credit_balance` |
| `putcall.csv` | `date,put_call` |
| `trends.csv` | `date,greed_search,fear_search` 또는 `date,<검색어별 검색량...>` (공포 검색어는 자동 분류) |
| `community.csv` | `date,text` (게시글 단위) 또는 `date,posts,sentiment` (일별 집계) |

받아 온 데이터는 `crowd.save_crowd_dir(crowd, "my_crowd/")` 로 저장해 두고 다음부터 `--crowd-dir` 로 쓰면 됩니다.

옵션: `--horizon 20`(며칠 뒤를 예측할지), `--threshold 0.55`(이 확률보다 높을 때만 보유), `--start 2005-01-01`,
`--no-compare`(데이터 소스별 기여도 비교 생략).

파이썬에서:

```python
from psycho_stock import fetch_yahoo, analyze
from psycho_stock.crowd import load_crowd_dir
from psycho_stock.report import render

result = analyze(fetch_yahoo("^KS11"), crowd=load_crowd_dir("my_crowd/"), horizon=20)
print(render(result, "SPY"))
result.table[["close", "fear_greed_smooth", "phase"]].tail()
```

## 리포트 읽는 법

1. **지금 시장의 감정** — 공포·탐욕 지수와 1주/1달 전 비교, 현재 국면
2. **감정 구성 요소** — 어떤 지표가 공포/탐욕 쪽으로 쏠렸는지
3. **예측** — N거래일 뒤 더 높을 확률과 그 확률을 움직인 심리 요인
4. **국면별 과거 성적** — "과거에 이 국면이었을 때 이후 20일은 어땠나" (과거 전체 기준, 표본 내 통계)
5. **워크포워드 백테스트** — 학습에 쓰지 않은 미래 구간에서만 평가한 정확도·AUC와
   `심리 모델` / `단순 역발상 규칙` / `단순 보유` 전략 비교 (거래비용 0.1% 반영)
6. **데이터 소스별 기여도** — 추가 데이터 각각이 예측력을 얼마나 올렸는지

## 정직하게: 이 시스템이 할 수 있는 것과 없는 것

- **가상 시장(`--synthetic`)의 높은 성적은 당연한 결과입니다.** 그 시장은 군중심리가 가격을 움직이도록
  일부러 만든 세계라서, "심리가 가격을 지배한다면 이 시스템이 그걸 잡아낸다"는 것을 보여 줄 뿐입니다.
- **실제 시장의 AUC는 보통 0.5~0.6 수준**입니다. 심리 패턴은 분명 존재하지만(모멘텀, 공포 국면 이후의
  높은 기대수익 등은 학계에서도 반복 확인된 현상), 많은 사람이 이미 이를 이용하려 하므로 신호가 약합니다.
- 테스트(`test_random_walk_has_no_edge`)는 **심리와 무관한 랜덤워크에서는 예측력이 0**이 나오는지 확인합니다.
  미래 정보가 새어 들어가면 이 테스트가 실패합니다.
- 결과는 **확률**이지 확신이 아닙니다. 투자 판단의 참고 자료로만 쓰세요.

## 다음 단계 아이디어

- 커뮤니티 게시글 자동 수집기 (종목토론방, 레딧 등 — 각 사이트 약관 확인 필요)
- 사전 대신 언어모델 기반 뉴스 감정 분석
- 여러 종목·지수에 동시 적용해 심리 지표의 일반성 검증

## 테스트

```bash
pytest
```
