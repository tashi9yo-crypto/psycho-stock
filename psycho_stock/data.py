"""시세 데이터 로딩.

모든 함수는 DatetimeIndex 와 open/high/low/close/volume 소문자 컬럼을 가진
DataFrame 을 반환한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns:
        # 배당/분할 조정 종가를 쓰고 OHLC 도 같은 비율로 맞춘다.
        ratio = df["adj_close"] / df["close"]
        for col in ("open", "high", "low"):
            if col in df.columns:
                df[col] = df[col] * ratio
        df["close"] = df["adj_close"]
        df = df.drop(columns="adj_close")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[REQUIRED_COLUMNS].astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")].dropna(subset=["close"])


def load_csv(path: str, date_column: str = "date") -> pd.DataFrame:
    """date, open, high, low, close, volume 컬럼을 가진 CSV 를 읽는다."""
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    if date_column.lower() not in cols:
        raise ValueError(f"날짜 컬럼 '{date_column}' 이 없습니다")
    raw = raw.set_index(cols[date_column.lower()])
    return _normalize(raw)


def fetch_yahoo(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """yfinance 로 시세를 받는다 (pip install yfinance 필요)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError("yfinance 가 필요합니다: pip install yfinance") from exc
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"{ticker} 데이터를 받지 못했습니다 (네트워크나 티커를 확인하세요)")
    return _normalize(raw)


def fetch_yahoo_close(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """VIX 처럼 종가만 필요한 보조 지표용."""
    return fetch_yahoo(ticker, start, end)["close"].rename(ticker)


def synthetic_market(n_days: int = 3000, seed: int = 7, start: str = "2010-01-04") -> pd.DataFrame:
    """군중심리가 가격을 움직이는 가상 시장의 시세 (자세한 설명은 _simulate)."""
    return _simulate(n_days, seed, start)[0]


def _simulate(n_days: int, seed: int, start: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """군중심리가 가격을 움직이는 가상 시장 (에이전트 기반 모형).

    - 펀더멘털 가치는 랜덤워크로 천천히 움직인다.
    - 추세추종자(군중)의 비중과 낙관/비관 '분위기'가 서로를 강화(양의 피드백)한다.
    - 가격이 가치에서 너무 멀어지면 가치투자자가 되돌림 압력을 준다.
    - 분위기가 한쪽으로 쏠렸다가 갑자기 뒤집히면서 거품과 폭락이 생긴다.

    실제 데이터 없이 시스템 전체를 시연하고 테스트하기 위한 용도다.
    """
    rng = np.random.default_rng(seed)
    log_value = np.log(100.0)
    log_price = log_value
    mood = 0.0  # -1(극단적 비관) ~ +1(극단적 낙관)
    prev_ret = 0.0
    trend = 0.0
    closes, opens, highs, lows, vols, moods, trends = [], [], [], [], [], [], []

    for _ in range(n_days):
        log_value += rng.normal(0.0003, 0.006)
        mispricing = log_price - log_value

        # 분위기: 최근 수익률(추세)에 전염되고, 서로를 따라 하며(herding), 가끔 뒤집힌다.
        herding = 1.6 * mood
        mood_drive = herding + 25.0 * trend - 2.0 * mispricing + rng.normal(0, 0.35)
        mood = 0.92 * mood + 0.08 * np.tanh(mood_drive)

        crowd_demand = 0.010 * mood + 0.30 * trend
        fundamental_demand = -0.025 * mispricing
        vol = 0.006 + 0.012 * abs(mood) ** 2 + 0.3 * abs(prev_ret)
        ret = crowd_demand + fundamental_demand + rng.normal(0, vol)
        ret = float(np.clip(ret, -0.15, 0.15))

        open_ = np.exp(log_price) * np.exp(rng.normal(0, vol / 4))
        log_price += ret
        close = np.exp(log_price)
        span = abs(rng.normal(0, vol)) * close
        high = max(open_, close) + span * 0.5
        low = min(open_, close) - span * 0.5
        volume = 1e6 * (1 + 4 * abs(mood) + 60 * abs(ret)) * np.exp(rng.normal(0, 0.2))

        trend = 0.9 * trend + 0.1 * ret
        prev_ret = ret
        moods.append(mood)
        trends.append(trend)
        closes.append(close)
        opens.append(open_)
        highs.append(high)
        lows.append(low)
        vols.append(volume)

    index = pd.bdate_range(start=start, periods=n_days, name="date")
    prices = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=index,
    )
    return prices, np.array(moods), np.array(trends)


def synthetic_market_with_crowd(n_days: int = 3000, seed: int = 7, start: str = "2010-01-04"):
    """가상 시장의 시세와, 같은 '분위기(mood)'에서 나온 시장 전체 군중 데이터.

    개인 순매수·신용잔고·검색량·커뮤니티는 분위기를 잡음과 함께 따라가고,
    풋/콜 비율은 반대로 움직인다. 구성 종목 40개로 시장 폭도 만든다.
    """
    from .crowd import CrowdData, breadth_from_constituents

    prices, mood, trend = _simulate(n_days, seed, start)
    rng = np.random.default_rng(seed + 1)
    index = prices.index
    market_ret = prices["close"].pct_change().fillna(0).to_numpy()

    betas = rng.uniform(0.6, 1.4, 40)
    idio = rng.normal(0, 0.015, (n_days, 40))
    stock_ret = market_ret[:, None] * betas + idio + 0.002 * mood[:, None]
    closes = pd.DataFrame(50 * np.exp(np.cumsum(np.log1p(np.clip(stock_ret, -0.5, 0.5)), axis=0)), index=index)

    scale = 1e11
    individual = scale * (0.8 * mood + 40 * trend + rng.normal(0, 0.6, n_days))
    foreign = scale * (-0.4 * mood + rng.normal(0, 0.6, n_days))
    flows = pd.DataFrame({"individual": individual, "foreign": foreign, "institution": -(individual + foreign)}, index=index)

    credit = pd.Series(1e13 * np.exp(np.cumsum(0.0015 * mood + rng.normal(0, 0.002, n_days))), index=index)
    put_call = pd.Series(np.clip(0.9 - 0.35 * mood + rng.normal(0, 0.12, n_days), 0.3, 2.0), index=index)

    weekly_mood = pd.Series(mood, index=index).resample("W-SAT").mean()
    weekly_mood.index = weekly_mood.index - pd.DateOffset(days=6)  # 구글 트렌드처럼 주 시작일(일요일) 표기
    k = len(weekly_mood)
    trends = pd.DataFrame({
        "greed_search": np.clip(50 + 35 * weekly_mood.to_numpy() + rng.normal(0, 8, k), 0, 100),
        "fear_search": np.clip(40 - 35 * weekly_mood.to_numpy() + rng.normal(0, 8, k), 0, 100),
    }, index=weekly_mood.index)

    posts = rng.poisson(20 + 120 * np.abs(mood))
    sentiment = np.clip(0.8 * mood + rng.normal(0, 0.25, n_days), -1, 1)
    community = pd.DataFrame({"posts": posts.astype(float), "sentiment": sentiment}, index=index)

    crowd = CrowdData(
        breadth=breadth_from_constituents(closes), flows=flows, credit=credit,
        put_call=put_call, trends=trends, community=community,
    )
    return prices, crowd
