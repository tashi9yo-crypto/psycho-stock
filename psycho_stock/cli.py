"""명령행 실행: python -m psycho_stock --ticker SPY --vix"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data
from .news import daily_news_sentiment
from .pipeline import analyze
from .report import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="군중심리 기반 주식시장 분석·예측")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="야후 파이낸스 티커 (예: SPY, ^KS11, 005930.KS)")
    src.add_argument("--csv", help="date,open,high,low,close,volume 형식 CSV 파일")
    src.add_argument("--synthetic", action="store_true", help="군중심리 가상 시장으로 시연")
    parser.add_argument("--start", default="2000-01-01", help="시작일 (ticker 사용 시)")
    parser.add_argument("--vix", nargs="?", const="^VIX", default=None,
                        help="변동성 지수를 공포 지표로 추가 (기본 ^VIX, 한국은 ^VKOSPI 등)")
    parser.add_argument("--news", help="date,headline 컬럼의 뉴스/커뮤니티 CSV")
    parser.add_argument("--horizon", type=int, default=20, help="예측 기간(거래일)")
    parser.add_argument("--threshold", type=float, default=0.5, help="보유 결정 확률 기준")
    parser.add_argument("--export", help="일별 분석 테이블을 CSV 로 저장할 경로")
    args = parser.parse_args(argv)

    if args.synthetic:
        prices, name = data.synthetic_market(), "가상 시장"
    elif args.csv:
        prices, name = data.load_csv(args.csv), args.csv
    else:
        prices, name = data.fetch_yahoo(args.ticker, start=args.start), args.ticker

    vix = None
    if args.vix:
        try:
            vix = data.fetch_yahoo_close(args.vix, start=str(prices.index[0].date()))
        except Exception as exc:  # 보조 지표는 실패해도 분석은 계속한다
            print(f"[경고] {args.vix} 를 받지 못해 제외합니다: {exc}", file=sys.stderr)

    news = daily_news_sentiment(pd.read_csv(args.news)) if args.news else None

    if len(prices) < 1000:
        print(f"[경고] 데이터가 {len(prices)}일뿐입니다. 최소 4년(1000일) 이상을 권장합니다.", file=sys.stderr)

    result = analyze(prices, vix=vix, news_sentiment=news, horizon=args.horizon, threshold=args.threshold)
    print(render(result, name))

    if args.export:
        out = result.table.join(result.backtest.predictions[["proba_up", "position"]])
        out.to_csv(args.export)
        print(f"\n일별 테이블 저장: {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
