"""분석 결과를 사람이 읽을 수 있는 한국어 리포트로."""

from __future__ import annotations

import unicodedata

import pandas as pd

from .crowd import CROWD_FEATURES
from .pipeline import Analysis
from .sentiment import PHASES, fear_greed_label

FEATURE_NAMES_KO = {
    "fear_greed": "공포·탐욕 지수",
    "fear_greed_change": "공포·탐욕 변화(10일)",
    "momentum_20": "20일 모멘텀(군집)",
    "momentum_60": "60일 모멘텀(군집)",
    "ma200_gap": "200일선 괴리",
    "rsi_14": "RSI(과열/투매)",
    "vol_ratio": "변동성 급등(공포)",
    "drawdown": "고점 대비 하락(고통)",
    "volume_surge_5": "방향성 거래량 폭증",
    "up_day_ratio": "상승일 비율",
    "capitulation_recent": "최근 항복 매도",
    "euphoria_recent": "최근 광기 매수",
    "vix_level": "VIX(옵션 공포)",
    "news_sentiment_5": "뉴스 감정",
    **{col: name for col, (_, name) in CROWD_FEATURES.items()},
}


def _name(feature: str) -> str:
    if feature.startswith("phase_"):
        key = feature.removeprefix("phase_")
        return f"국면={PHASES.get(key, (key,))[0]}"
    return FEATURE_NAMES_KO.get(feature, feature)


def _pad(text: str, width: int) -> str:
    """한글(전각) 문자를 2칸으로 계산해 왼쪽 정렬."""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - shown)


def _pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:+.1f}%"


def _gauge(value: float, width: int = 30) -> str:
    if pd.isna(value):
        return ""
    filled = int(round(value / 100 * width))
    return "공포 [" + "#" * filled + "-" * (width - filled) + "] 탐욕"


def render(analysis: Analysis, name: str = "") -> str:
    t = analysis.table
    last = t.iloc[-1]
    date = t.index[-1].date()
    fg = last["fear_greed_smooth"]
    phase = last["phase"]
    phase_ko, phase_desc = PHASES.get(phase, ("N/A", "데이터가 부족합니다.")) if isinstance(phase, str) else ("N/A", "데이터가 부족합니다.")
    bt = analysis.backtest
    m = bt.metrics
    h = analysis.horizon

    lines = []
    title = f"군중심리 분석 리포트 {name}".strip()
    lines += [f"=== {title} ({date}) ===", ""]
    lines += ["[1] 지금 시장의 감정", f"  공포·탐욕 지수 : {fg:.0f} / 100  → {fear_greed_label(fg)}", f"  {_gauge(fg)}"]
    week_ago = t["fear_greed_smooth"].iloc[-6] if len(t) > 6 else float("nan")
    month_ago = t["fear_greed_smooth"].iloc[-21] if len(t) > 21 else float("nan")
    lines.append(f"  1주 전 {week_ago:.0f} / 1달 전 {month_ago:.0f}")
    lines += [f"  심리 국면     : {phase_ko}", f"    - {phase_desc}"]
    if last.get("capitulation", 0):
        lines.append("  !! 오늘 항복 매도 신호(큰 하락 + 거래량 폭증 + 과매도) 발생")
    if last.get("euphoria", 0):
        lines.append("  !! 오늘 광기 매수 신호(큰 상승 + 거래량 폭증 + 과매수) 발생")
    lines.append("")

    lines.append("[2] 감정 구성 요소 (0=극단적 공포, 100=극단적 탐욕)")
    components = [c.removeprefix("fg_") for c in t.columns if c.startswith("fg_")]
    groups = [
        ("가격·거래량", [k for k in components if k not in CROWD_FEATURES]),
        ("시장 전체 군중", [k for k in components if k in CROWD_FEATURES]),
    ]
    for title, keys in groups:
        if not keys:
            continue
        lines.append(f"  - {title}")
        for key in keys:
            label = FEATURE_NAMES_KO.get(key, FEATURE_NAMES_KO.get(key + "_5", key))
            value = last[f"fg_{key}"]
            shown = "  N/A" if pd.isna(value) else f"{value:5.0f}"
            lines.append(f"    {_pad(label, 22)} {shown}")
    extra = [FEATURE_NAMES_KO[c] for c in CROWD_FEATURES if c in t.columns and CROWD_FEATURES[c][0] == 0]
    if extra:
        lines.append(f"  (예측 모델에만 쓰는 지표: {', '.join(extra)})")
    lines.append("")

    lines.append(f"[3] 예측: 앞으로 {h}거래일 뒤 가격이 지금보다 높을 확률")
    lines.append(f"  상승 확률 : {analysis.latest_proba * 100:.0f}%")
    lines.append("  확률을 끌어올리거나(+) 내린(-) 주요 심리 요인:")
    for feat, val in analysis.latest_contrib.head(5).items():
        lines.append(f"    {_pad(_name(feat), 24)} {val:+.2f}")
    lines.append("")

    lines.append(f"[4] 국면별 과거 성적 (이후 {h}일 수익률, 과거 전체 기준)")
    lines.append(f"  {_pad('국면', 12)}{'일수':>6}{'평균':>9}{'중앙값':>9}{'승률':>7}")
    for key, row in analysis.phase_stats.iterrows():
        marker = " <- 현재" if key == phase else ""
        lines.append(
            f"  {_pad(row['name'], 12)}{int(row['days']):>6}{_pct(row['mean_return']):>9}"
            f"{_pct(row['median_return']):>9}{row['win_rate'] * 100:>6.0f}%{marker}"
        )
    lines.append("")

    lines.append("[5] 워크포워드 백테스트 (학습에 쓰지 않은 미래 구간에서만 평가)")
    lines.append(
        f"  예측 {int(m['n_predictions'])}회 | 정확도 {m['accuracy'] * 100:.1f}% "
        f"(항상 '상승' 찍기 {m['base_rate_up'] * 100:.1f}%) | AUC {m['auc']:.3f} | 보유 비중 {m['exposure'] * 100:.0f}%"
    )
    lines.append(f"  {_pad('전략', 34)}{'연수익':>7}{'변동성':>7}{'샤프':>6}{'최대낙폭':>6}")
    for strat, perf in bt.strategies.items():
        lines.append(
            f"  {_pad(strat, 34)}{_pct(perf['cagr']):>9}{_pct(perf['volatility']):>9}"
            f"{perf['sharpe']:>7.2f}{_pct(perf['max_drawdown']):>9}"
        )
    lines.append("  AUC 0.5 = 동전 던지기. 0.55 이상이면 의미 있는 신호일 가능성이 있다.")
    lines.append("")

    if analysis.sources is not None:
        lines.append("[6] 데이터 소스별 기여도 (가격·거래량 모델에 하나씩 더했을 때, 워크포워드)")
        lines.append(f"  {_pad('모델', 20)}{'AUC':>7}{'변화':>8}{'샤프':>7}")
        for model_name, row in analysis.sources.iterrows():
            lines.append(f"  {_pad(model_name, 20)}{row['auc']:>7.3f}{row['auc_gain']:>+8.3f}{row['sharpe']:>7.2f}")
        lines.append("  변화가 +0.01 미만이면 그 데이터는 이 시장에서 예측에 거의 도움이 안 된다는 뜻이다.")
        lines.append("")
    lines.append("※ 확률적 참고 지표일 뿐 투자 권유가 아닙니다. 과거 패턴은 반복되지 않을 수 있습니다.")
    return "\n".join(lines)
