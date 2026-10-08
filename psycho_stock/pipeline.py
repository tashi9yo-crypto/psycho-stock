"""데이터 → 심리 지표 → 공포·탐욕 → 국면 → 예측까지 한 번에."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import compute_indicators
from .model import BacktestResult, latest_prediction, phase_statistics, walk_forward
from .sentiment import classify_phases, fear_greed_index


def build_table(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    news_sentiment: pd.Series | None = None,
) -> pd.DataFrame:
    ind = compute_indicators(prices, vix=vix)
    fg = fear_greed_index(ind)
    table = prices.join(ind).join(fg)
    table["phase"] = classify_phases(ind, fg["fear_greed"])
    if news_sentiment is not None:
        ns = news_sentiment.copy()
        ns.index = pd.to_datetime(ns.index).normalize()
        table["news_sentiment"] = ns.reindex(table.index.normalize()).to_numpy()
    return table


@dataclass
class Analysis:
    table: pd.DataFrame
    backtest: BacktestResult
    phase_stats: pd.DataFrame
    latest_proba: float
    latest_contrib: pd.Series
    horizon: int


def analyze(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    news_sentiment: pd.Series | None = None,
    horizon: int = 20,
    threshold: float = 0.5,
) -> Analysis:
    table = build_table(prices, vix=vix, news_sentiment=news_sentiment)
    bt = walk_forward(table, horizon=horizon, threshold=threshold)
    stats = phase_statistics(table, horizon=horizon)
    proba, contrib = latest_prediction(table, horizon=horizon)
    return Analysis(table, bt, stats, proba, contrib, horizon)
