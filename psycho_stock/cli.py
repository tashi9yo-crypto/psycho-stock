"""명령행 실행: python -m psycho_stock --ticker SPY --vix"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import crowd as crowd_mod
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
    g = parser.add_argument_group("시장 전체 군중 데이터 (모두 선택)")
    g.add_argument("--crowd-dir", help="breadth/flows/credit/putcall/trends/community.csv 가 든 폴더")
    g.add_argument("--breadth-tickers", help="시장 폭 계산용 구성 종목 (쉼표 구분 또는 한 줄에 하나씩 적은 파일)")
    g.add_argument("--krx-flows", nargs="?", const="KOSPI", help="KRX 투자자별 수급 (KOSPI/KOSDAQ/종목코드, pykrx)")
    g.add_argument("--google-trends", nargs="?", const="KR", help="구글 트렌드 탐욕/공포 검색량 (국가코드 KR/US, pytrends)")
    g.add_argument("--community", help="커뮤니티 게시글 CSV (date,text 또는 date,posts,sentiment)")
    parser.add_argument("--no-compare", action="store_true", help="데이터 소스별 기여도 비교 생략 (빠름)")
    parser.add_argument("--horizon", type=int, default=20, help="예측 기간(거래일)")
    parser.add_argument("--threshold", type=float, default=0.5, help="보유 결정 확률 기준")
    parser.add_argument("--export", help="일별 분석 테이블을 CSV 로 저장할 경로")
    parser.add_argument("--html", help="브라우저로 보는 대시보드를 저장할 경로 (예: report.html)")
    args = parser.parse_args(argv)

    crowd = crowd_mod.CrowdData()
    if args.synthetic:
        prices, crowd = data.synthetic_market_with_crowd()
        name = "가상 시장"
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

    start = str(prices.index[0].date())
    if args.crowd_dir:
        loaded = crowd_mod.load_crowd_dir(args.crowd_dir)
        for key in loaded.available():
            setattr(crowd, key, getattr(loaded, key))
    fetchers = []
    if args.breadth_tickers:
        tickers = _read_list(args.breadth_tickers)
        fetchers.append(("breadth", "시장 폭", lambda: crowd_mod.fetch_breadth(tickers, start=start)))
    if args.krx_flows:
        fetchers.append(("flows", "KRX 수급", lambda: crowd_mod.fetch_krx_flows(args.krx_flows, start=start)))
    if args.google_trends:
        fetchers.append(("trends", "구글 트렌드", lambda: crowd_mod.fetch_google_trends(geo=args.google_trends)))
    for key, label, fetch in fetchers:
        try:
            setattr(crowd, key, fetch())
        except Exception as exc:  # 보조 데이터는 실패해도 분석은 계속한다
            print(f"[경고] {label} 데이터를 받지 못해 제외합니다: {exc}", file=sys.stderr)
    if args.community:
        raw = pd.read_csv(args.community)
        crowd.community = (
            crowd_mod.community_daily(raw) if "text" in raw.columns
            else raw.set_index(pd.to_datetime(raw["date"]))[["posts", "sentiment"]]
        )
    if crowd:
        print(f"[정보] 사용하는 군중 데이터: {', '.join(crowd.available())}", file=sys.stderr)

    news = daily_news_sentiment(pd.read_csv(args.news)) if args.news else None

    if len(prices) < 1000:
        print(f"[경고] 데이터가 {len(prices)}일뿐입니다. 최소 4년(1000일) 이상을 권장합니다.", file=sys.stderr)

    result = analyze(prices, vix=vix, news_sentiment=news, crowd=crowd, horizon=args.horizon,
                     threshold=args.threshold, compare_sources=not args.no_compare)
    print(render(result, name))

    if args.html:
        from pathlib import Path

        from .html_report import render_html

        Path(args.html).write_text(render_html(result, name), encoding="utf-8")
        print(f"\n대시보드 저장: {args.html}  (브라우저로 여세요)")

    if args.export:
        out = result.table.join(result.backtest.predictions[["proba_up", "position"]])
        out.to_csv(args.export)
        print(f"\n일별 테이블 저장: {args.export}")
    return 0


def _read_list(value: str) -> list[str]:
    from pathlib import Path

    path = Path(value)
    text = path.read_text() if path.exists() else value
    return [t.strip() for t in text.replace(",", "\n").splitlines() if t.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
