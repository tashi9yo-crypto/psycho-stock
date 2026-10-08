"""가격·거래량에서 군중심리를 읽어내는 지표들.

각 지표는 '사람들이 지금 어떤 감정 상태인가'를 대리(proxy)한다.
모든 계산은 t 시점까지의 데이터만 사용한다 (미래 정보 누수 없음).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """과매수/과매도 — 단기 탐욕·공포의 과열 정도."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def rolling_percentile(series: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """현재 값이 과거 window 기간 중 몇 %ile 인지 (0~100). 오늘 값까지만 사용."""
    min_periods = max(20, window // 4)
    return series.rolling(window, min_periods=min_periods).rank(pct=True) * 100


def compute_indicators(prices: pd.DataFrame, vix: pd.Series | None = None) -> pd.DataFrame:
    """심리 지표 테이블을 만든다.

    컬럼 설명
    - momentum_20 / momentum_60: 군집 행동(herding). 사람들은 오른 것을 더 산다.
    - ma200_gap: 200일선 대비 괴리. 집단적 낙관/비관이 얼마나 쌓였는지.
    - rsi_14: 단기 과열 (탐욕) / 투매 (공포).
    - vol_ratio: 최근 변동성 / 장기 변동성. 공포는 변동성으로 드러난다.
    - drawdown: 1년 고점 대비 하락폭. 손실 회피 → 고통의 크기.
    - volume_surge: 평소 대비 거래량 (부호=방향). 패닉 매도 vs FOMO 매수.
    - up_day_ratio: 최근 20일 중 상승일 비율. 분위기의 일관성.
    - capitulation: 큰 하락 + 거래량 폭증 + 과매도 동시 발생 = 항복 매도.
    - euphoria: 큰 상승 + 거래량 폭증 + 과매수 + 고점 근처 = 광기 매수.
    - vix_level: (선택) 내재 변동성. 옵션 시장의 공포 가격.
    """
    close = prices["close"]
    volume = prices["volume"]
    ret = close.pct_change()

    out = pd.DataFrame(index=prices.index)
    out["return_1d"] = ret
    out["momentum_20"] = close.pct_change(20)
    out["momentum_60"] = close.pct_change(60)
    ma200 = close.rolling(200, min_periods=100).mean()
    out["ma200_gap"] = close / ma200 - 1
    out["rsi_14"] = rsi(close, 14)

    vol_short = ret.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)
    vol_long = ret.rolling(TRADING_DAYS, min_periods=60).std() * np.sqrt(TRADING_DAYS)
    out["volatility_20"] = vol_short
    out["vol_ratio"] = vol_short / vol_long

    rolling_high = close.rolling(TRADING_DAYS, min_periods=20).max()
    out["drawdown"] = close / rolling_high - 1

    avg_volume = volume.rolling(50, min_periods=20).mean()
    surge = volume / avg_volume
    out["volume_surge"] = surge * np.sign(ret.fillna(0))
    out["up_day_ratio"] = (ret > 0).astype(float).rolling(20, min_periods=10).mean()

    ret_std = ret.rolling(60, min_periods=20).std()
    big_down = ret < -2 * ret_std
    big_up = ret > 2 * ret_std
    heavy_volume = surge > 1.8
    out["capitulation"] = (big_down & heavy_volume & (out["rsi_14"] < 35)).astype(int)
    out["euphoria"] = (
        big_up & heavy_volume & (out["rsi_14"] > 70) & (out["drawdown"] > -0.03)
    ).astype(int)

    if vix is not None:
        out["vix_level"] = vix.reindex(prices.index).ffill()

    return out
