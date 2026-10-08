import numpy as np
import pandas as pd
import pytest

from psycho_stock import analyze, build_table, data, score_text
from psycho_stock.cli import main
from psycho_stock.indicators import compute_indicators
from psycho_stock.model import LogisticModel, roc_auc, walk_forward
from psycho_stock.news import daily_news_sentiment
from psycho_stock.sentiment import PHASES


def random_walk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    idx = pd.bdate_range("2005-01-03", periods=n)
    vol = rng.lognormal(14, 0.3, n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": vol}, index=idx
    )


@pytest.fixture(scope="module")
def synthetic_table():
    return build_table(data.synthetic_market(n_days=1500))


def test_no_lookahead_in_table():
    prices = data.synthetic_market(n_days=1200)
    full = build_table(prices)
    cut = build_table(prices.iloc[:900])
    cols = [c for c in cut.columns if c != "phase"]
    pd.testing.assert_frame_equal(full.iloc[:900][cols], cut[cols])
    assert (full["phase"].iloc[:900].fillna("") == cut["phase"].fillna("")).all()


def test_fear_greed_range_and_phases(synthetic_table):
    fg = synthetic_table["fear_greed"].dropna()
    assert len(fg) > 1000
    assert fg.between(0, 100).all()
    phases = set(synthetic_table["phase"].dropna())
    assert phases <= set(PHASES)
    assert len(phases) >= 5


def test_random_walk_has_no_edge():
    """심리와 무관한 랜덤워크에서 모델이 '예측력'을 보이면 누수가 있다는 뜻."""
    bt = walk_forward(build_table(random_walk()), horizon=20)
    assert 0.40 < bt.metrics["auc"] < 0.60


def test_synthetic_crowd_market_is_predictable(synthetic_table):
    bt = walk_forward(synthetic_table, horizon=20, min_train=500)
    assert bt.metrics["auc"] > 0.6


def test_logistic_model_learns_simple_rule():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = (x[:, 0] > 0).astype(float)
    model = LogisticModel(l2=0.1).fit(x, y)
    assert roc_auc(y, model.predict_proba(x)) > 0.95


def test_capitulation_detected_on_crash():
    prices = random_walk(400)
    prices.iloc[300, prices.columns.get_loc("close")] *= 0.88
    prices.iloc[301:, prices.columns.get_loc("close")] *= 0.88
    prices.iloc[300, prices.columns.get_loc("volume")] *= 5
    for k in range(295, 300):  # 투매 직전 며칠간 하락으로 과매도 상태 만들기
        prices.iloc[k:, prices.columns.get_loc("close")] *= 0.98
    ind = compute_indicators(prices)
    assert ind["capitulation"].iloc[300] == 1


def test_news_scoring():
    assert score_text("코스피 폭락에 투자자 공포 확산") < 0
    assert score_text("반도체 호재로 신고가 돌파") > 0
    assert score_text("Stocks surge to record high") > 0
    assert score_text("Markets did not crash") > 0
    assert score_text("날씨가 맑다") == 0
    news = pd.DataFrame({"date": ["2024-01-02", "2024-01-02", "2024-01-03"],
                         "headline": ["급등", "급락", "폭락 공포"]})
    daily = daily_news_sentiment(news)
    assert daily.loc["2024-01-02"] == 0
    assert daily.loc["2024-01-03"] < 0


def test_analyze_with_vix_and_news():
    prices = data.synthetic_market(n_days=1300)
    vix = (prices["close"].pct_change().rolling(20).std() * 1600).rename("vix")
    news = pd.Series(np.sin(np.arange(len(prices)) / 30), index=prices.index)
    result = analyze(prices, vix=vix, news_sentiment=news)
    assert 0 <= result.latest_proba <= 1
    assert "vix_level" in result.backtest.feature_names
    assert "news_sentiment_5" in result.backtest.feature_names


def test_cli_with_csv(tmp_path, capsys):
    path = tmp_path / "prices.csv"
    data.synthetic_market(n_days=1300).to_csv(path)
    out_path = tmp_path / "out.csv"
    assert main(["--csv", str(path), "--export", str(out_path)]) == 0
    text = capsys.readouterr().out
    assert "공포·탐욕 지수" in text and "워크포워드" in text
    assert "proba_up" in pd.read_csv(out_path).columns
