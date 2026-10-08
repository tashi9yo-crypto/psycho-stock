"""뉴스·커뮤니티 헤드라인의 감정 점수 (사전 기반, 한국어/영어).

정교한 NLP 대신 투자 심리에 특화된 단어 사전을 쓴다. 날짜별 평균 점수를
예측 모델의 추가 특성(news_sentiment)으로 넣을 수 있다.
"""

from __future__ import annotations

import re

import pandas as pd

POSITIVE = {
    # 한국어
    "급등": 2, "폭등": 2, "상승": 1, "반등": 1, "호재": 2, "신고가": 2, "최고치": 2, "랠리": 1,
    "강세": 1, "호황": 2, "기대": 1, "낙관": 1, "돌파": 1, "매수세": 1, "흑자": 1, "개선": 1,
    "대박": 2, "불장": 2, "가즈아": 2, "떡상": 2, "풀매수": 2,
    # 영어
    "surge": 2, "soar": 2, "rally": 1, "gain": 1, "gains": 1, "rise": 1, "rises": 1, "bull": 1,
    "bullish": 2, "record": 1, "high": 1, "beat": 1, "beats": 1, "optimism": 1, "boom": 2,
    "upgrade": 1, "strong": 1, "recovery": 1, "moon": 2,
}
NEGATIVE = {
    "급락": -2, "폭락": -2, "하락": -1, "약세": -1, "악재": -2, "공포": -2, "패닉": -2, "위기": -2,
    "침체": -2, "우려": -1, "불안": -1, "손실": -1, "적자": -1, "부도": -2, "파산": -2, "매도세": -1,
    "투매": -2, "반대매매": -2, "떡락": -2, "물림": -1, "손절": -1, "경고": -1, "쇼크": -2,
    "crash": -2, "plunge": -2, "plunges": -2, "fall": -1, "falls": -1, "drop": -1, "drops": -1,
    "bear": -1, "bearish": -2, "fear": -2, "panic": -2, "selloff": -2, "sell-off": -2,
    "recession": -2, "crisis": -2, "loss": -1, "losses": -1, "downgrade": -1, "warning": -1,
    "default": -2, "bankruptcy": -2, "slump": -2, "worry": -1, "worries": -1,
}
NEGATIONS_EN = {"not", "no", "never"}
NEGATIONS_KO = ("않", "없", "못")
LEXICON = {**POSITIVE, **NEGATIVE}
_TOKEN = re.compile(r"[A-Za-z\-]+|[가-힣]+")


def score_text(text: str) -> float:
    """-1(매우 부정) ~ +1(매우 긍정)."""
    tokens = _TOKEN.findall(str(text).lower())
    total, hits = 0.0, 0
    for i, tok in enumerate(tokens):
        value = LEXICON.get(tok)
        if value is None and "가" <= tok[0] <= "힣":
            # 한국어는 조사가 붙으므로 사전 단어로 시작하는지 확인 (예: "폭락에", "상승세")
            value = next((v for w, v in LEXICON.items() if len(w) >= 2 and tok.startswith(w)), None)
        if value is None:
            continue
        window = tokens[max(0, i - 2): i + 2]
        if any(t in NEGATIONS_EN or t.startswith(NEGATIONS_KO) for t in window if t != tok):
            value = -value
        total += value
        hits += 1
    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, total / (2 * hits)))


def daily_news_sentiment(news: pd.DataFrame, date_column: str = "date", text_column: str = "headline") -> pd.Series:
    """date, headline 컬럼의 DataFrame 을 날짜별 평균 감정 점수로 집계한다."""
    df = news[[date_column, text_column]].dropna().copy()
    df["score"] = df[text_column].map(score_text)
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    return df.groupby(date_column)["score"].mean().rename("news_sentiment")
