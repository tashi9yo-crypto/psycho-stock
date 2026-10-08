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


# ------------------------------------------------------------ 시장 전체 군중 데이터

from psycho_stock import crowd as crowd_mod  # noqa: E402
from psycho_stock.data import synthetic_market_with_crowd  # noqa: E402


@pytest.fixture(scope="module")
def crowd_market():
    return synthetic_market_with_crowd(n_days=1300)


def test_breadth_from_constituents():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = np.linspace(10, 20, 60)
    closes = pd.DataFrame({"a": up, "b": up, "c": up[::-1]}, index=idx)
    b = crowd_mod.breadth_from_constituents(closes)
    assert (b["advancers"] == 2).all() and (b["decliners"] == 1).all()
    assert b["new_highs"].iloc[-1] == 2 and b["new_lows"].iloc[-1] == 1


def test_crowd_alignment_respects_publication_lag():
    idx = pd.bdate_range("2024-01-01", periods=40)
    weekly = pd.DataFrame({"greed_search": 10.0, "fear_search": 10.0}, index=pd.date_range("2023-12-31", periods=6, freq="7D"))
    weekly.loc["2024-01-14", "greed_search"] = 90.0  # 1/14(일)~1/20(토) 주
    out = crowd_mod.crowd_indicators(crowd_mod.CrowdData(trends=weekly), idx)
    assert out.loc["2024-01-19", "search_mood"] == 0  # 그 주가 끝나기 전엔 알 수 없다
    assert out.loc["2024-01-22", "search_mood"] > 0.5

    credit = pd.Series(np.arange(1, 41, dtype=float), index=idx)
    credit.iloc[30] = 1000.0
    lagged = crowd_mod._align(credit, idx, lag_bdays=2)
    assert lagged.iloc[32] == 1000.0 and lagged.iloc[31] != 1000.0


def test_crowd_no_lookahead(crowd_market):
    prices, crowd = crowd_market
    full = build_table(prices, crowd=crowd)
    cut_prices = prices.iloc[:1000]
    cut_crowd = crowd_mod.CrowdData(**{
        k: getattr(crowd, k)[getattr(crowd, k).index <= cut_prices.index[-1]] for k in crowd.available()
    })
    cut = build_table(cut_prices, crowd=cut_crowd)
    cols = [c for c in cut.columns if c != "phase"]
    pd.testing.assert_frame_equal(full.iloc[:1000][cols], cut[cols])


def test_crowd_features_enter_index_and_model(crowd_market):
    prices, crowd = crowd_market
    result = analyze(prices, crowd=crowd)
    table = result.table
    for col in crowd_mod.CROWD_FEATURES:
        assert col in table.columns, col
        assert col in result.backtest.feature_names
    assert "fg_retail_flow" in table.columns and "fg_put_call" in table.columns
    assert "fg_foreign_flow" not in table.columns  # 방향 0 은 지수에 넣지 않는다
    assert result.sources is not None
    assert {"가격·거래량만", "+ 투자자 수급", "+ 커뮤니티", "전체"} <= set(result.sources.index)
    # 가상 시장에서 개인 순매수와 풋/콜은 분위기와 같은/반대 방향이어야 한다
    corr = table[["retail_flow", "put_call", "fear_greed"]].corr()
    assert corr.loc["retail_flow", "fear_greed"] > 0
    assert corr.loc["put_call", "fear_greed"] < 0


def test_crowd_dir_roundtrip_and_cli(tmp_path, crowd_market, capsys):
    prices, crowd = crowd_market
    crowd_dir = tmp_path / "crowd"
    crowd_mod.save_crowd_dir(crowd, crowd_dir)
    loaded = crowd_mod.load_crowd_dir(crowd_dir)
    assert set(loaded.available()) == set(crowd.available())

    raw_posts = pd.DataFrame({"date": ["2024-01-02"] * 3 + ["2024-01-03"],
                              "text": ["가즈아 풀매수", "떡상 간다", "물림 손절", "반대매매 공포"]})
    daily = crowd_mod.community_daily(raw_posts)
    assert daily.loc["2024-01-02", "posts"] == 3 and daily.loc["2024-01-03", "sentiment"] < 0

    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)
    assert main(["--csv", str(price_path), "--crowd-dir", str(crowd_dir)]) == 0
    text = capsys.readouterr().out
    assert "시장 전체 군중" in text and "데이터 소스별 기여도" in text


def test_trend_columns_classified_by_keyword(tmp_path):
    path = tmp_path / "trends.csv"
    pd.DataFrame({"date": ["2024-01-07", "2024-01-14"], "주식 추천": [50, 60], "주식 폭락": [10, 80]}).to_csv(path, index=False)
    loaded = crowd_mod.load_crowd_dir(tmp_path)
    assert list(loaded.trends.columns) == ["greed_search", "fear_search"]
    assert loaded.trends["fear_search"].iloc[1] == 80


def test_html_dashboard(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main(["--synthetic", "--no-compare", "--html", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "군중심리 계기판" in html
    assert '"fear_greed"' in html and "__DATA__" not in html
