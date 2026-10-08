"""심리 지표 → 미래 수익률 방향 예측 모델과 워크포워드 백테스트.

미래 정보 누수를 막기 위해
- 정규화 통계(평균/표준편차)는 학습 구간에서만 계산하고,
- 학습 라벨이 예측 시점 이전에 확정된 행만 학습에 쓰며 (horizon 만큼 purge),
- 예측은 항상 '앞으로' 굴러가며(walk-forward) 재학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sentiment import PHASES

BASE_FEATURES = [
    "fear_greed",
    "fear_greed_change",
    "momentum_20",
    "momentum_60",
    "ma200_gap",
    "rsi_14",
    "vol_ratio",
    "drawdown",
    "volume_surge_5",
    "up_day_ratio",
    "capitulation_recent",
    "euphoria_recent",
]
OPTIONAL_FEATURES = ["vix_level", "news_sentiment_5"]


def build_features(table: pd.DataFrame) -> pd.DataFrame:
    """분석 테이블(지표 + 공포탐욕 + 국면)에서 모델 입력 특성을 만든다."""
    f = pd.DataFrame(index=table.index)
    f["fear_greed"] = table["fear_greed_smooth"]
    f["fear_greed_change"] = table["fear_greed_smooth"] - table["fear_greed_smooth"].shift(10)
    for col in ["momentum_20", "momentum_60", "ma200_gap", "rsi_14", "vol_ratio", "drawdown", "up_day_ratio"]:
        f[col] = table[col]
    f["volume_surge_5"] = table["volume_surge"].rolling(5, min_periods=1).mean()
    f["capitulation_recent"] = table["capitulation"].rolling(10, min_periods=1).max()
    f["euphoria_recent"] = table["euphoria"].rolling(10, min_periods=1).max()
    if "vix_level" in table:
        f["vix_level"] = table["vix_level"]
    if "news_sentiment" in table:
        f["news_sentiment_5"] = table["news_sentiment"].fillna(0).rolling(5, min_periods=1).mean()
    if "phase" in table:
        for key in PHASES:
            f[f"phase_{key}"] = (table["phase"] == key).astype(float)
    return f


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


class LogisticModel:
    """L2 정규화 로지스틱 회귀 (뉴턴법). 계수가 그대로 해석 가능하다."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50):
        self.l2 = l2
        self.max_iter = max_iter
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.std_
        return np.column_stack([np.ones(len(z)), np.nan_to_num(z)])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        a = self._design(x)
        w = np.zeros(a.shape[1])
        penalty = np.full(a.shape[1], self.l2)
        penalty[0] = 0.0  # 절편은 규제하지 않음
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
            grad = a.T @ (p - y) + penalty * w
            hess = (a * (p * (1 - p))[:, None]).T @ a + np.diag(penalty + 1e-9)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.coef_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(self._design(x) @ self.coef_, -30, 30)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def performance(daily_returns: pd.Series, periods: int = 252) -> dict[str, float]:
    r = daily_returns.dropna()
    if r.empty:
        return {"cagr": np.nan, "volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    equity = (1 + r).cumprod()
    years = len(r) / periods
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    sharpe = r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return {"cagr": float(cagr), "volatility": float(vol), "sharpe": float(sharpe), "max_drawdown": float(mdd)}


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    strategies: dict[str, dict[str, float]]
    coefficients: pd.Series
    feature_names: list[str] = field(default_factory=list)


def walk_forward(
    table: pd.DataFrame,
    horizon: int = 20,
    min_train: int = 750,
    retrain_every: int = 60,
    threshold: float = 0.5,
    cost: float = 0.001,
    l2: float = 5.0,
) -> BacktestResult:
    """워크포워드 백테스트.

    t 시점에 '앞으로 horizon 거래일 뒤 가격이 오를 확률'을 예측하고,
    확률이 threshold 보다 높으면 다음 날 보유, 아니면 현금으로 둔다.
    """
    features = build_features(table)
    names = list(features.columns)
    target = forward_return(table["close"], horizon)
    label = (target > 0).astype(float).where(target.notna())
    x_all = features.to_numpy(dtype=float)
    y_all = label.to_numpy()
    usable = ~np.isnan(x_all).all(axis=1) & features[["fear_greed", "ma200_gap"]].notna().all(axis=1).to_numpy()

    n = len(table)
    proba = np.full(n, np.nan)
    model = None
    start = int(np.argmax(usable)) + min_train if usable.any() else n
    for k in range(start, n, retrain_every):
        train_end = k - horizon  # 이 행 이전 라벨만 k 시점에 확정돼 있다
        train_mask = usable[:train_end] & ~np.isnan(y_all[:train_end])
        if train_mask.sum() < 200:
            continue
        model = LogisticModel(l2=l2).fit(x_all[:train_end][train_mask], y_all[:train_end][train_mask])
        stop = min(k + retrain_every, n)
        block = slice(k, stop)
        proba[block] = np.where(usable[block], model.predict_proba(x_all[block]), np.nan)

    pred = pd.DataFrame(
        {"proba_up": proba, "forward_return": target, "label": label, "fear_greed": table["fear_greed_smooth"]},
        index=table.index,
    )
    daily_ret = table["close"].pct_change()
    tested = pred["proba_up"].notna()

    position = (pred["proba_up"] > threshold).astype(float).where(tested)
    model_ret = position.shift(1) * daily_ret - position.diff().abs().shift(1).fillna(0) * cost

    # 비교용 단순 역발상 규칙: 공포(<30)에 사서 탐욕(>70)에 판다.
    fg = pred["fear_greed"]
    contrarian = pd.Series(np.nan, index=fg.index)
    contrarian[fg < 30] = 1.0
    contrarian[fg > 70] = 0.0
    contrarian = contrarian.ffill().fillna(0).where(tested)
    contrarian_ret = contrarian.shift(1) * daily_ret - contrarian.diff().abs().shift(1).fillna(0) * cost

    hold_ret = daily_ret.where(tested.shift(1, fill_value=False))
    pred["position"] = position
    pred["model_return"] = model_ret
    pred["contrarian_return"] = contrarian_ret

    scored = pred[tested & pred["label"].notna()]
    y, p = scored["label"].to_numpy(), scored["proba_up"].to_numpy()
    metrics = {
        "n_predictions": float(len(scored)),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()) if len(y) else np.nan,
        "base_rate_up": float(y.mean()) if len(y) else np.nan,
        "auc": roc_auc(y, p) if len(y) else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
        "exposure": float(position[tested].mean()) if tested.any() else np.nan,
    }
    strategies = {
        "심리 모델": performance(model_ret[tested.shift(1, fill_value=False)]),
        "역발상 규칙(공포 매수/탐욕 매도)": performance(contrarian_ret[tested.shift(1, fill_value=False)]),
        "단순 보유": performance(hold_ret),
    }
    coefs = (
        pd.Series(model.coef_[1:], index=names).sort_values(key=np.abs, ascending=False)
        if model is not None
        else pd.Series(dtype=float)
    )
    return BacktestResult(pred, metrics, strategies, coefs, names)


def phase_statistics(table: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """국면별로 이후 horizon 일 수익률이 어땠는지 (과거 전체 기준 기술통계)."""
    fwd = forward_return(table["close"], horizon)
    df = pd.DataFrame({"phase": table["phase"], "fwd": fwd}).dropna()
    stats = df.groupby("phase")["fwd"].agg(
        days="count", mean_return="mean", median_return="median", win_rate=lambda s: (s > 0).mean()
    )
    stats["name"] = [PHASES[p][0] for p in stats.index]
    return stats.sort_values("mean_return", ascending=False)


def latest_prediction(table: pd.DataFrame, horizon: int = 20, l2: float = 5.0) -> tuple[float, pd.Series]:
    """사용 가능한 모든 과거로 학습해 가장 최근 시점의 상승 확률을 낸다."""
    features = build_features(table)
    target = forward_return(table["close"], horizon)
    label = (target > 0).astype(float).where(target.notna())
    ok = features[["fear_greed", "ma200_gap"]].notna().all(axis=1) & label.notna()
    model = LogisticModel(l2=l2).fit(features[ok].to_numpy(float), label[ok].to_numpy())
    last = features.iloc[[-1]].to_numpy(float)
    contrib = pd.Series(
        model.coef_[1:] * np.nan_to_num((last[0] - model.mean_) / model.std_), index=features.columns
    )
    return float(model.predict_proba(last)[0]), contrib.sort_values(key=np.abs, ascending=False)
