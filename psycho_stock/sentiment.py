"""공포·탐욕 지수와 시장 심리 국면(사이클) 판별."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rolling_percentile

# (지표 컬럼, 방향) — 방향 +1 이면 값이 클수록 탐욕, -1 이면 클수록 공포.
FEAR_GREED_COMPONENTS: list[tuple[str, int]] = [
    ("momentum_20", +1),
    ("momentum_60", +1),
    ("ma200_gap", +1),
    ("rsi_14", +1),
    ("vol_ratio", -1),
    ("drawdown", +1),
    ("volume_surge", +1),
    ("up_day_ratio", +1),
    ("vix_level", -1),
]

# 월스트리트 심리 사이클 (Wall Street Cheat Sheet) 를 단순화한 국면.
PHASES = {
    "euphoria": ("환희", "모두가 확신하고 '이번엔 다르다'고 말하는 구간. 위험 관리가 가장 중요하다."),
    "optimism": ("낙관", "상승이 이어지며 자신감이 붙는 구간. 추세를 따라가되 과열을 경계한다."),
    "anxiety": ("불안", "고점 이후 흔들림. '잠깐 조정일 뿐'이라는 부정이 시작되는 구간."),
    "fear": ("공포", "하락 추세가 굳어지며 손실 회피 심리로 매도가 매도를 부르는 구간."),
    "panic": ("공황", "극단적 공포. 감정적 투매가 나오며 역발상 기회가 생기기 시작한다."),
    "capitulation": ("항복", "거래량 폭증과 함께 마지막 투매가 나온 상태. 역사적으로 바닥 근처인 경우가 많다."),
    "despair": ("절망→희망", "하락 추세지만 심리가 바닥에서 회복 중. 대부분은 아직 믿지 않는다."),
    "disbelief": ("의심", "상승 추세로 돌아섰지만 군중은 아직 의심하는 구간. 역사적으로 수익이 좋은 국면."),
    "neutral": ("중립", "뚜렷한 감정 쏠림이 없는 구간."),
}


def fear_greed_index(indicators: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """각 심리 지표를 과거 1년 대비 백분위(0~100)로 바꾼 뒤 평균한다.

    0 에 가까울수록 극단적 공포, 100 에 가까울수록 극단적 탐욕.
    """
    parts = {}
    for col, direction in FEAR_GREED_COMPONENTS:
        if col not in indicators.columns:
            continue
        pct = rolling_percentile(indicators[col], window)
        parts[f"fg_{col}"] = pct if direction > 0 else 100 - pct
    comp = pd.DataFrame(parts, index=indicators.index)
    comp["fear_greed"] = comp.mean(axis=1, skipna=True).where(comp.notna().sum(axis=1) >= 3)
    comp["fear_greed_smooth"] = comp["fear_greed"].ewm(span=5, min_periods=1).mean()
    return comp


def fear_greed_label(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value < 20:
        return "극단적 공포"
    if value < 40:
        return "공포"
    if value <= 60:
        return "중립"
    if value <= 80:
        return "탐욕"
    return "극단적 탐욕"


def classify_phases(indicators: pd.DataFrame, fear_greed: pd.Series) -> pd.Series:
    """추세(가격) × 감정(공포·탐욕) × 감정 변화 방향으로 심리 국면을 판별한다."""
    fg = fear_greed.ewm(span=5, min_periods=1).mean()
    fg_change = fg - fg.shift(10)
    uptrend = indicators["ma200_gap"] > 0
    recent_capitulation = indicators["capitulation"].rolling(10, min_periods=1).max() > 0

    phase = pd.Series("neutral", index=indicators.index, dtype=object)
    up, down = uptrend, ~uptrend

    phase[up & (fg >= 80)] = "euphoria"
    phase[up & (fg >= 55) & (fg < 80) & (fg_change >= 0)] = "optimism"
    phase[up & (fg >= 45) & (fg_change < -10)] = "anxiety"
    phase[up & (fg < 45) & (fg_change >= 0)] = "disbelief"

    phase[down & (fg < 45) & (fg >= 20) & (fg_change < 0)] = "fear"
    phase[down & (fg < 20)] = "panic"
    phase[down & (fg < 45) & (fg_change >= 10)] = "despair"
    phase[down & (fg >= 45) & (fg_change < -10)] = "anxiety"
    phase[recent_capitulation & down & (fg < 35)] = "capitulation"

    phase[fg.isna() | indicators["ma200_gap"].isna()] = np.nan
    return phase
