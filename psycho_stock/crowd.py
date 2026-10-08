"""시장 전체의 군중심리 데이터.

한 종목의 가격만으로는 '그 종목'의 심리만 보인다. 여기서는 시장 참여자 전체의
행동 흔적을 모은다.

| 데이터 | 심리학적 의미 | 입력 형식 |
|---|---|---|
| 시장 폭(breadth) | 상승이 소수 종목만의 잔치인가, 군중 전체의 확신인가 | 구성 종목 시세 → 자동 계산, 또는 CSV |
| 투자자별 수급 | 개인의 추격 매수(FOMO)/투매, 외국인·기관의 반대 포지션 | KRX(pykrx) 또는 CSV |
| 신용잔고 | 빚내서 투자 = 레버리지 탐욕 | CSV (금융투자협회 등) |
| 풋/콜 비율 | 하락 대비 보험 수요 = 공포 | CSV (CBOE 등) |
| 검색량 | '주식 추천' vs '폭락' 검색 = 대중의 관심과 두려움 | 구글 트렌드(pytrends) 또는 CSV |
| 커뮤니티 | 게시글 폭증(흥분)과 말투(감정) | 게시글 CSV 또는 일별 집계 CSV |

모든 외부 데이터는 '그 날짜에 실제로 알 수 있었던 값'만 쓰도록 공개 지연(lag)을 두고
가격 날짜에 맞춘다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .news import score_text

# 지표 컬럼 → (공포·탐욕 방향, 한국어 이름). 방향 0 은 모델 특성으로만 쓴다.
CROWD_FEATURES: dict[str, tuple[int, str]] = {
    "breadth_ad": (+1, "상승/하락 종목 비율"),
    "breadth_hilo": (+1, "신고가-신저가"),
    "breadth_above_ma50": (+1, "50일선 위 종목 비율"),
    "retail_flow": (+1, "개인 순매수 강도"),
    "foreign_flow": (0, "외국인 순매수 강도"),
    "institution_flow": (0, "기관 순매수 강도"),
    "credit_change": (+1, "신용잔고 증가율(빚투)"),
    "put_call": (-1, "풋/콜 비율(보험 수요)"),
    "search_mood": (+1, "검색어 탐욕-공포"),
    "community_buzz": (0, "커뮤니티 게시글 폭증"),
    "community_sentiment": (+1, "커뮤니티 감정"),
}

# 데이터 소스 → 그 소스에서 나오는 모델 특성 (기여도 비교용)
SOURCE_FEATURES: dict[str, list[str]] = {
    "VIX": ["vix_level"],
    "뉴스": ["news_sentiment_5"],
    "시장 폭": ["breadth_ad", "breadth_hilo", "breadth_above_ma50"],
    "투자자 수급": ["retail_flow", "foreign_flow", "institution_flow"],
    "신용잔고": ["credit_change"],
    "풋/콜": ["put_call"],
    "검색량": ["search_mood"],
    "커뮤니티": ["community_buzz", "community_sentiment"],
}

GREED_TERMS = {
    "KR": ["주식 추천", "주식 계좌 개설", "급등주", "코인 투자"],
    "US": ["stocks to buy", "how to buy stocks", "options trading", "get rich"],
}
FEAR_TERMS = {
    "KR": ["주식 폭락", "경기 침체", "반대매매", "주식 손절"],
    "US": ["stock market crash", "recession", "margin call", "bear market"],
}


@dataclass
class CrowdData:
    """선택적인 시장 전체 데이터 묶음. 있는 것만 채우면 된다."""

    breadth: pd.DataFrame | None = None  # advancers, decliners, new_highs, new_lows, pct_above_ma50 중 일부
    flows: pd.DataFrame | None = None  # individual, foreign, institution (순매수 금액)
    credit: pd.Series | None = None  # 신용잔고
    put_call: pd.Series | None = None  # 풋/콜 비율
    trends: pd.DataFrame | None = None  # greed_search, fear_search (주간이면 주 시작일 기준)
    community: pd.DataFrame | None = None  # posts, sentiment (일별)

    def available(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def __bool__(self) -> bool:
        return bool(self.available())


# ---------------------------------------------------------------- 시장 폭

def breadth_from_constituents(closes: pd.DataFrame, high_window: int = 252) -> pd.DataFrame:
    """구성 종목 종가(열=종목)로 시장 폭 지표를 계산한다."""
    change = closes.diff()
    valid = closes.notna() & closes.shift(1).notna()
    rolling_max = closes.rolling(high_window, min_periods=20).max()
    rolling_min = closes.rolling(high_window, min_periods=20).min()
    ma50 = closes.rolling(50, min_periods=20).mean()
    out = pd.DataFrame(index=closes.index)
    out["advancers"] = ((change > 0) & valid).sum(axis=1)
    out["decliners"] = ((change < 0) & valid).sum(axis=1)
    out["new_highs"] = (closes >= rolling_max).sum(axis=1)
    out["new_lows"] = (closes <= rolling_min).sum(axis=1)
    above = (closes > ma50).where(ma50.notna())
    out["pct_above_ma50"] = above.mean(axis=1)
    return out.iloc[1:]


def fetch_breadth(tickers: list[str], start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """yfinance 로 구성 종목 시세를 받아 시장 폭을 계산한다."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("구성 종목 시세를 받지 못했습니다")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return breadth_from_constituents(closes)


# ---------------------------------------------------------------- 외부 소스

def fetch_krx_flows(market: str = "KOSPI", start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """pykrx 로 시장(KOSPI/KOSDAQ) 또는 종목의 투자자별 순매수 금액을 받는다.

    KRX 정책상 KRX_ID / KRX_PW 환경변수(KRX 회원 계정)가 필요할 수 있다.
    """
    try:
        from pykrx import stock
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError("pykrx 가 필요합니다: pip install pykrx") from exc
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        print("[안내] pykrx 가 KRX 로그인을 요구하면 KRX_ID, KRX_PW 환경변수를 설정하세요.")
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    raw = stock.get_market_trading_value_by_date(start.replace("-", ""), end.replace("-", ""), market)
    if raw is None or raw.empty:
        raise RuntimeError(f"{market} 수급 데이터를 받지 못했습니다")
    flows = raw.rename(columns={"개인": "individual", "외국인합계": "foreign", "기관합계": "institution"})
    flows.index = pd.to_datetime(flows.index)
    return flows[[c for c in ("individual", "foreign", "institution") if c in flows.columns]].astype(float)


def fetch_google_trends(geo: str = "KR", timeframe: str = "today 5-y",
                        greed_terms: list[str] | None = None, fear_terms: list[str] | None = None) -> pd.DataFrame:
    """pytrends 로 탐욕/공포 검색어 묶음의 검색량을 받는다 (주간, 0~100)."""
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError("pytrends 가 필요합니다: pip install pytrends") from exc
    key = "KR" if geo.upper() == "KR" else "US"
    greed = greed_terms or GREED_TERMS[key]
    fear = fear_terms or FEAR_TERMS[key]
    client = TrendReq(hl="ko" if key == "KR" else "en-US", tz=540 if key == "KR" else 300)
    # 한 요청에 최대 5개 검색어이므로 묶음별로 따로 받는다. 묶음 간 비교는 정규화된 비율로만 쓴다.
    frames = {}
    for name, terms in (("greed_search", greed), ("fear_search", fear)):
        client.build_payload(terms[:5], timeframe=timeframe, geo=geo.upper() if geo else "")
        df = client.interest_over_time()
        if df.empty:
            raise RuntimeError(f"구글 트렌드 데이터를 받지 못했습니다: {terms}")
        frames[name] = df.drop(columns="isPartial", errors="ignore").sum(axis=1)
    return pd.DataFrame(frames)


def community_daily(posts: pd.DataFrame, date_column: str = "date", text_column: str = "text") -> pd.DataFrame:
    """게시글 단위(date, text) 데이터를 일별 게시글 수·평균 감정으로 집계한다."""
    df = posts[[date_column, text_column]].dropna().copy()
    df["sentiment"] = df[text_column].map(score_text)
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    return df.groupby(date_column).agg(posts=(text_column, "size"), sentiment=("sentiment", "mean"))


# ---------------------------------------------------------------- CSV 폴더

CROWD_FILES = {
    "breadth": "breadth.csv",  # date,advancers,decliners,new_highs,new_lows,pct_above_ma50  또는 date,<종목1>,<종목2>,... 종가
    "flows": "flows.csv",  # date,individual,foreign,institution
    "credit": "credit.csv",  # date,credit_balance
    "put_call": "putcall.csv",  # date,put_call
    "trends": "trends.csv",  # date,greed_search,fear_search  또는 date,<검색어들...>
    "community": "community.csv",  # date,text  또는 date,posts,sentiment
}


def _read_dated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "날짜", "일자")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    return df


def _classify_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"greed_search", "fear_search"} <= set(df.columns):
        return df[["greed_search", "fear_search"]]
    fear_words = {w for terms in FEAR_TERMS.values() for w in terms} | {"폭락", "crash", "recession", "침체", "손절"}
    fear_cols = [c for c in df.columns if any(w in c.lower() for w in fear_words)]
    greed_cols = [c for c in df.columns if c not in fear_cols]
    if not fear_cols or not greed_cols:
        raise ValueError("trends.csv 는 greed_search,fear_search 컬럼이나 탐욕/공포 검색어 컬럼이 필요합니다")
    return pd.DataFrame({"greed_search": df[greed_cols].sum(axis=1), "fear_search": df[fear_cols].sum(axis=1)})


def load_crowd_dir(directory: str) -> CrowdData:
    """폴더 안의 표준 이름 CSV 들을 읽는다. 없는 파일은 건너뛴다."""
    base = Path(directory)
    crowd = CrowdData()
    for attr, filename in CROWD_FILES.items():
        path = base / filename
        if not path.exists():
            continue
        df = _read_dated(path)
        if attr == "breadth":
            known = {"advancers", "decliners", "new_highs", "new_lows", "pct_above_ma50"}
            crowd.breadth = df[[c for c in df.columns if c in known]] if known & set(df.columns) else breadth_from_constituents(df)
        elif attr == "flows":
            crowd.flows = df.astype(float)
        elif attr == "credit":
            crowd.credit = df.iloc[:, 0].astype(float)
        elif attr == "put_call":
            crowd.put_call = df.iloc[:, 0].astype(float)
        elif attr == "trends":
            crowd.trends = _classify_trend_columns(df.astype(float))
        elif attr == "community":
            if "text" in df.columns:
                crowd.community = community_daily(df.reset_index(), "date", "text")
            else:
                crowd.community = df[["posts", "sentiment"]].astype(float)
    return crowd


# ---------------------------------------------------------------- 지표 계산

def _align(obj: pd.Series | pd.DataFrame, index: pd.DatetimeIndex, lag_bdays: int = 0, lag_days: int = 0):
    """데이터 날짜를 공개 시점으로 미룬 뒤 가격 날짜에 맞춰 직전 값을 쓴다."""
    obj = obj.astype(float)
    obj.index = pd.to_datetime(obj.index).normalize()
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    if lag_bdays:
        obj.index = obj.index + pd.offsets.BDay(lag_bdays)
    if lag_days:
        obj.index = obj.index + pd.DateOffset(days=lag_days)
    target = pd.DatetimeIndex(index).normalize()
    merged = obj.reindex(obj.index.union(target)).ffill(limit=15)
    out = merged.reindex(target)
    out.index = index
    return out


def _flow_strength(net: pd.Series) -> pd.Series:
    """20일 누적 순매수를 평소 거래 규모로 나눈 값 (시장 규모 성장에 무관한 척도)."""
    scale = net.abs().rolling(252, min_periods=40).mean() * 20
    return net.rolling(20, min_periods=10).sum() / scale


def crowd_indicators(crowd: CrowdData, index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)

    if crowd.breadth is not None:
        b = _align(crowd.breadth, index)
        if {"advancers", "decliners"} <= set(b.columns):
            total = (b["advancers"] + b["decliners"]).replace(0, np.nan)
            out["breadth_ad"] = ((b["advancers"] - b["decliners"]) / total).rolling(10, min_periods=3).mean()
            if {"new_highs", "new_lows"} <= set(b.columns):
                out["breadth_hilo"] = ((b["new_highs"] - b["new_lows"]) / total).rolling(10, min_periods=3).mean()
        if "pct_above_ma50" in b.columns:
            out["breadth_above_ma50"] = b["pct_above_ma50"]

    if crowd.flows is not None:
        f = _align(crowd.flows, index)
        for src, dst in (("individual", "retail_flow"), ("foreign", "foreign_flow"), ("institution", "institution_flow")):
            if src in f.columns:
                out[dst] = _flow_strength(f[src])

    if crowd.credit is not None:
        credit = _align(crowd.credit, index, lag_bdays=2)  # 신용잔고는 보통 1~2 영업일 뒤 공개
        out["credit_change"] = credit.pct_change(20, fill_method=None)

    if crowd.put_call is not None:
        out["put_call"] = _align(crowd.put_call, index).rolling(5, min_periods=1).mean()

    if crowd.trends is not None:
        t = crowd.trends
        weekly = len(t) > 1 and pd.Series(t.index).diff().median() >= pd.Timedelta(days=6)
        # 주간 데이터는 '주 시작일' 로 표시되므로 그 주가 끝난 뒤에야 알 수 있다.
        t = _align(t, index, lag_days=7 if weekly else 1)
        out["search_mood"] = (t["greed_search"] - t["fear_search"]) / (t["greed_search"] + t["fear_search"] + 1e-9)

    if crowd.community is not None:
        c = _align(crowd.community, index)
        out["community_buzz"] = c["posts"] / c["posts"].rolling(60, min_periods=10).mean()
        out["community_sentiment"] = c["sentiment"].rolling(5, min_periods=1).mean()

    return out


def save_crowd_dir(crowd: CrowdData, directory: str) -> None:
    """CrowdData 를 load_crowd_dir 이 읽는 형식으로 저장한다 (받은 데이터 캐시용)."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    columns = {"credit": "credit_balance", "put_call": "put_call"}
    for attr, filename in CROWD_FILES.items():
        obj = getattr(crowd, attr)
        if obj is None:
            continue
        frame = obj.to_frame(columns[attr]) if isinstance(obj, pd.Series) else obj
        frame.rename_axis("date").to_csv(base / filename)
