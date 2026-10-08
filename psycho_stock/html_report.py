"""분석 결과를 브라우저에서 보는 대시보드(HTML 한 파일)로 만든다.

외부 라이브러리 없이 SVG 로 그리며, 데이터는 페이지 안에 JSON 으로 들어간다.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .crowd import CROWD_FEATURES
from .pipeline import Analysis
from .report import FEATURE_NAMES_KO, _name
from .sentiment import PHASES, fear_greed_label


def _num(x, digits: int = 4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), digits)


def _payload(analysis: Analysis, name: str) -> dict:
    t = analysis.table
    pred = analysis.backtest.predictions
    last = t.iloc[-1]
    phase = last["phase"] if isinstance(last["phase"], str) else None

    tested = pred["proba_up"].notna()
    model_eq = (1 + pred["model_return"].where(tested.shift(1, fill_value=False)).fillna(0)).cumprod()
    contra_eq = (1 + pred["contrarian_return"].where(tested.shift(1, fill_value=False)).fillna(0)).cumprod()
    hold_eq = (1 + t["close"].pct_change().where(tested.shift(1, fill_value=False)).fillna(0)).cumprod()
    first_test = tested.idxmax() if tested.any() else None

    components = []
    for col in [c for c in t.columns if c.startswith("fg_")]:
        key = col.removeprefix("fg_")
        components.append({
            "label": FEATURE_NAMES_KO.get(key, FEATURE_NAMES_KO.get(key + "_5", key)),
            "group": "crowd" if key in CROWD_FEATURES else "price",
            "value": _num(last[col], 1),
        })

    return {
        "name": name,
        "date": str(t.index[-1].date()),
        "horizon": analysis.horizon,
        "latest": {
            "fear_greed": _num(last["fear_greed_smooth"], 1),
            "fear_greed_label": fear_greed_label(last["fear_greed_smooth"]),
            "week_ago": _num(t["fear_greed_smooth"].iloc[-6], 1) if len(t) > 6 else None,
            "month_ago": _num(t["fear_greed_smooth"].iloc[-21], 1) if len(t) > 21 else None,
            "phase": phase,
            "phase_name": PHASES[phase][0] if phase else "N/A",
            "phase_desc": PHASES[phase][1] if phase else "데이터가 부족합니다.",
            "proba": _num(analysis.latest_proba, 3),
            "capitulation": int(last.get("capitulation", 0) or 0),
            "euphoria": int(last.get("euphoria", 0) or 0),
        },
        "drivers": [{"label": _name(k), "value": _num(v, 3)} for k, v in analysis.latest_contrib.head(6).items()],
        "components": components,
        "series": {
            "dates": [str(d.date()) for d in t.index],
            "close": [_num(v, 4) for v in t["close"]],
            "fear_greed": [_num(v, 1) for v in t["fear_greed_smooth"]],
            "proba": [_num(v, 3) for v in pred["proba_up"]],
            "phase": [p if isinstance(p, str) else None for p in t["phase"]],
            "eq_model": [_num(v, 4) if first_test is not None and d >= first_test else None for d, v in model_eq.items()],
            "eq_contrarian": [_num(v, 4) if first_test is not None and d >= first_test else None for d, v in contra_eq.items()],
            "eq_hold": [_num(v, 4) if first_test is not None and d >= first_test else None for d, v in hold_eq.items()],
        },
        "phases": {k: v[0] for k, v in PHASES.items()},
        "phase_stats": [
            {"key": k, "name": r["name"], "days": int(r["days"]), "mean": _num(r["mean_return"]),
             "median": _num(r["median_return"]), "win": _num(r["win_rate"], 3)}
            for k, r in analysis.phase_stats.iterrows()
        ],
        "metrics": {k: _num(v) for k, v in analysis.backtest.metrics.items()},
        "strategies": [{"name": k, **{m: _num(v) for m, v in perf.items()}} for k, perf in analysis.backtest.strategies.items()],
        "sources": None if analysis.sources is None else [
            {"name": k, "auc": _num(r["auc"], 3), "gain": _num(r["auc_gain"], 3), "sharpe": _num(r["sharpe"], 2)}
            for k, r in analysis.sources.iterrows()
        ],
    }


def render_html(analysis: Analysis, name: str = "", standalone: bool = True) -> str:
    """standalone=False 면 <html>/<head>/<body> 없이 본문만 (외부 페이지에 끼워 넣을 때)."""
    data = json.dumps(_payload(analysis, name), ensure_ascii=False).replace("</", "<\\/")
    body = TEMPLATE.replace("__DATA__", data)
    if not standalone:
        return body
    return (
        '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


TEMPLATE = r"""<title>군중심리 계기판</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
/* 레이아웃: 위에 '지금의 감정' 요약 띠, 아래로 x축을 공유하는 세 개의 시계열 패널, 그 뒤 근거 표들 */
:root {
  --bg: #f3f5f7;
  --panel: #ffffff;
  --ink: #14181d;
  --ink-2: #4b5560;
  --ink-3: #7b8591;
  --rule: #dfe4ea;
  --grid: #e9edf1;
  --fear: #c8423f;
  --fear-soft: #f6dedd;
  --fear-faint: #fbefee;
  --greed: #2a6fc4;
  --greed-soft: #dae6f6;
  --greed-faint: #eef3fb;
  --neutral-mid: #eceae6;
  --s1: #2a78d6;
  --s2: #eb6834;
  --s3: #7b8591;
  --good: #1a7f4b;
  --bad: #c8423f;
  --font-ui: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #101317; --panel: #171b21; --ink: #eef1f4; --ink-2: #b3bcc6; --ink-3: #85909c;
    --rule: #2a3038; --grid: #222830;
    --fear: #e66767; --fear-soft: #44262a; --fear-faint: #2a1d21;
    --greed: #5b9be8; --greed-soft: #1f3350; --greed-faint: #18222f; --neutral-mid: #2a2d31;
    --s1: #3987e5; --s2: #d95926; --s3: #85909c; --good: #3fb37a; --bad: #e66767;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #101317; --panel: #171b21; --ink: #eef1f4; --ink-2: #b3bcc6; --ink-3: #85909c;
  --rule: #2a3038; --grid: #222830;
  --fear: #e66767; --fear-soft: #44262a; --fear-faint: #2a1d21;
  --greed: #5b9be8; --greed-soft: #1f3350; --greed-faint: #18222f; --neutral-mid: #2a2d31;
  --s1: #3987e5; --s2: #d95926; --s3: #85909c; --good: #3fb37a; --bad: #e66767;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-ui); font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px; display: grid; gap: 20px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; }
h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance; }
h2 { margin: 0; font-size: 15px; font-weight: 600; text-wrap: balance; }
.meta { color: var(--ink-3); font-size: 13px; font-family: var(--font-data); }
.num { font-family: var(--font-data); font-variant-numeric: tabular-nums; }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 18px; min-width: 0; }
.panel-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 6px 16px; margin-bottom: 10px; }
.hint { color: var(--ink-3); font-size: 13px; }
.notice { font-size: 13px; color: var(--ink-2); background: var(--panel); border: 1px dashed var(--rule); border-radius: 8px; padding: 10px 14px; }

/* 요약 띠 */
.now { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 1fr); gap: 0; padding: 0; }
.now > div { padding: 18px 20px; min-width: 0; }
.now > div + div { border-left: 1px solid var(--rule); }
.label { font-size: 12px; color: var(--ink-3); letter-spacing: 0.04em; }
.big { font-family: var(--font-data); font-size: 44px; line-height: 1.1; font-weight: 500; font-variant-numeric: tabular-nums; }
.big small { font-size: 16px; color: var(--ink-3); font-weight: 400; }
.scale { position: relative; height: 10px; border-radius: 5px; margin: 12px 0 6px;
  background: linear-gradient(90deg, var(--fear) 0%, var(--fear-soft) 30%, var(--neutral-mid) 50%, var(--greed-soft) 70%, var(--greed) 100%); }
.scale .pin { position: absolute; top: -5px; width: 4px; height: 20px; margin-left: -2px; background: var(--ink); border-radius: 2px; box-shadow: 0 0 0 2px var(--panel); }
.scale-legend { display: flex; justify-content: space-between; font-size: 12px; color: var(--ink-3); }
.chip { display: inline-block; font-size: 13px; font-weight: 600; padding: 2px 10px; border-radius: 999px; border: 1px solid currentColor; }
.chip.fear { color: var(--fear); } .chip.greed { color: var(--greed); } .chip.mid { color: var(--ink-2); }
.phase-name { font-size: 26px; font-weight: 700; margin: 4px 0 6px; }
.desc { color: var(--ink-2); font-size: 14px; }
.alert { margin-top: 8px; font-size: 13px; font-weight: 600; color: var(--fear); }

/* 차트 */
.controls { display: flex; gap: 4px; }
.controls button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--rule); background: transparent; color: var(--ink-2); cursor: pointer; }
.controls button[aria-pressed="true"] { background: var(--ink); color: var(--panel); border-color: var(--ink); }
.controls button:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.chart { position: relative; }
.chart svg { display: block; width: 100%; overflow: visible; }
.chart-title { font-size: 13px; color: var(--ink-2); margin: 14px 0 2px; display: flex; gap: 12px; flex-wrap: wrap; align-items: baseline; }
.chart-title b { color: var(--ink); font-weight: 600; }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); white-space: nowrap; z-index: 2; }
.tip .row { display: flex; justify-content: space-between; gap: 14px; }
.tip .k { color: var(--ink-3); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }

/* 하단 그리드 */
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; }
.bars { display: grid; gap: 6px; }
.bar-row { display: grid; grid-template-columns: 9.5em minmax(0, 1fr) 2.6em; gap: 10px; align-items: center; font-size: 13px; }
.bar-row .name { color: var(--ink-2); overflow-wrap: anywhere; }
.bar-track { position: relative; height: 12px; background: var(--grid); border-radius: 3px; }
.bar-track::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--ink-3); }
.bar-fill { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.group-label { font-size: 12px; color: var(--ink-3); letter-spacing: 0.04em; margin-top: 8px; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: right; font-weight: 500; color: var(--ink-3); font-size: 12px; padding: 6px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--grid); font-family: var(--font-data); font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-family: var(--font-ui); }
tr.current td { background: var(--greed-faint); font-weight: 600; }
.pos { color: var(--good); } .neg { color: var(--bad); }
.footnote { font-size: 12px; color: var(--ink-3); margin-top: 10px; }
.kpis { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-bottom: 12px; font-size: 13px; color: var(--ink-2); }
.kpis b { font-family: var(--font-data); color: var(--ink); font-weight: 500; font-size: 15px; }
footer { font-size: 12px; color: var(--ink-3); }

@media (max-width: 760px) {
  .now { grid-template-columns: minmax(0, 1fr); }
  .now > div + div { border-left: 0; border-top: 1px solid var(--rule); }
  .grid2 { grid-template-columns: minmax(0, 1fr); }
  .big { font-size: 38px; }
}
@media (prefers-reduced-motion: no-preference) { .scale .pin { transition: left .4s ease; } }
</style>

<div class="wrap">
  <header>
    <h1 id="title">군중심리 계기판</h1>
    <div class="meta" id="asof"></div>
  </header>
  <div class="notice" id="notice" hidden></div>

  <section class="panel now" aria-label="지금 시장의 감정">
    <div>
      <div class="label">공포·탐욕 지수</div>
      <div class="big" id="fg"></div>
      <div class="scale"><div class="pin" id="pin"></div></div>
      <div class="scale-legend"><span>극단적 공포 0</span><span>50</span><span>100 극단적 탐욕</span></div>
      <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
        <span class="chip" id="fg-label"></span><span class="hint" id="fg-change"></span>
      </div>
    </div>
    <div>
      <div class="label">심리 국면</div>
      <div class="phase-name" id="phase"></div>
      <div class="desc" id="phase-desc"></div>
      <div class="alert" id="alert" hidden></div>
    </div>
    <div>
      <div class="label" id="proba-label"></div>
      <div class="big" id="proba"></div>
      <div class="desc" style="margin-top:6px;">확률을 움직인 요인</div>
      <div class="bars" id="drivers" style="margin-top:6px;"></div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>가격과 감정의 흐름</h2>
      <div class="controls" role="group" aria-label="기간">
        <button type="button" data-range="126">6개월</button>
        <button type="button" data-range="252" aria-pressed="true">1년</button>
        <button type="button" data-range="756">3년</button>
        <button type="button" data-range="0">전체</button>
      </div>
    </div>
    <div class="chart-title"><b>가격</b><span class="hint">선 아래 띠 색 = 그날의 심리 국면 (공포 계열 빨강, 탐욕 계열 파랑)</span></div>
    <div class="chart" id="c-price"></div>
    <div class="chart-title"><b>공포·탐욕 지수</b><span class="hint">20 아래 극단적 공포, 80 위 극단적 탐욕</span></div>
    <div class="chart" id="c-fg"></div>
    <div class="chart-title"><b>상승 확률 (모델)</b><span class="hint" id="proba-hint"></span></div>
    <div class="chart" id="c-proba"></div>
  </section>

  <div class="grid2">
    <section class="panel">
      <div class="panel-head"><h2>감정 구성 요소</h2><span class="hint">가운데 선 = 50 (중립)</span></div>
      <div class="bars" id="components"></div>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>국면별 과거 성적</h2><span class="hint" id="phase-hint"></span></div>
      <div class="scroll"><table id="phase-table"></table></div>
      <div class="footnote">과거 전체 기간의 기술 통계입니다. 미래를 보장하지 않습니다.</div>
    </section>
  </div>

  <section class="panel">
    <div class="panel-head"><h2>백테스트</h2><span class="hint">학습에 쓰지 않은 미래 구간에서만 평가, 거래비용 0.1% 반영</span></div>
    <div class="kpis" id="kpis"></div>
    <div class="legend" id="eq-legend"></div>
    <div class="hint">1원이 몇 원이 됐는지 (로그 눈금: 한 칸 위로 갈 때마다 배수로 커짐)</div>
    <div class="chart" id="c-eq"></div>
    <div class="scroll" style="margin-top:12px;"><table id="strat-table"></table></div>
  </section>

  <section class="panel" id="sources-panel" hidden>
    <div class="panel-head"><h2>데이터 소스별 기여도</h2><span class="hint">가격·거래량 모델에 하나씩 더했을 때의 AUC 변화</span></div>
    <div class="scroll"><table id="sources-table"></table></div>
    <div class="footnote">변화가 +0.01 미만이면 그 데이터는 이 시장에서 예측에 거의 도움이 안 된다는 뜻입니다.</div>
  </section>

  <footer>확률적 참고 지표일 뿐 투자 권유가 아닙니다. 과거 패턴은 반복되지 않을 수 있습니다.</footer>
</div>

<script>
const D = __DATA__;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x, d = 1) => x == null ? "–" : (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
const FEAR_PHASES = new Set(["fear", "panic", "capitulation", "anxiety"]);
const GREED_PHASES = new Set(["euphoria", "optimism"]);
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

// ---------- 요약
const L = D.latest;
$("title").textContent = "군중심리 계기판" + (D.name ? " · " + D.name : "");
$("asof").textContent = D.date + " 기준";
$("fg").innerHTML = (L.fear_greed == null ? "–" : Math.round(L.fear_greed)) + "<small> / 100</small>";
$("pin").style.left = (L.fear_greed ?? 50) + "%";
const fgChip = $("fg-label");
fgChip.textContent = L.fear_greed_label;
fgChip.classList.add(L.fear_greed < 40 ? "fear" : L.fear_greed > 60 ? "greed" : "mid");
const r0 = (x) => x == null ? "–" : Math.round(x);
$("fg-change").textContent = `1주 전 ${r0(L.week_ago)} · 1달 전 ${r0(L.month_ago)}`;
$("phase").textContent = L.phase_name;
$("phase").style.color = FEAR_PHASES.has(L.phase) ? css("--fear") : GREED_PHASES.has(L.phase) ? css("--greed") : "";
$("phase-desc").textContent = L.phase_desc;
if (L.capitulation || L.euphoria) {
  $("alert").hidden = false;
  $("alert").textContent = L.capitulation ? "오늘 항복 매도 신호 발생 (큰 하락 + 거래량 폭증 + 과매도)" : "오늘 광기 매수 신호 발생 (큰 상승 + 거래량 폭증 + 과매수)";
}
$("proba-label").textContent = `${D.horizon}거래일 뒤 더 오를 확률`;
$("proba").innerHTML = (L.proba == null ? "–" : Math.round(L.proba * 100)) + "<small>%</small>";
$("proba-hint").textContent = `${D.horizon}거래일 뒤 가격이 더 높을 확률. 50% 위면 모델은 보유, 아래면 현금`;

function divergingBars(container, rows, { center = 0, span = 1, fmt }) {
  container.innerHTML = "";
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const v = r.value;
    const rel = v == null ? 0 : Math.max(-1, Math.min(1, (v - center) / span));
    const color = rel < 0 ? "var(--fear)" : "var(--greed)";
    const left = rel < 0 ? 50 + rel * 50 : 50;
    row.innerHTML = `<span class="name">${r.label}</span>
      <span class="bar-track" title="${r.label}: ${fmt(v)}"><span class="bar-fill" style="left:${left}%;width:${Math.abs(rel) * 50}%;background:${color}"></span></span>
      <span class="num" style="text-align:right">${fmt(v)}</span>`;
    container.appendChild(row);
  }
}
const maxDriver = Math.max(0.5, ...D.drivers.map((d) => Math.abs(d.value || 0)));
divergingBars($("drivers"), D.drivers, { center: 0, span: maxDriver, fmt: (v) => v == null ? "–" : (v >= 0 ? "+" : "") + v.toFixed(2) });

const comps = $("components");
for (const [group, title] of [["price", "가격·거래량"], ["crowd", "시장 전체 군중"]]) {
  const rows = D.components.filter((c) => c.group === group);
  if (!rows.length) continue;
  const h = document.createElement("div"); h.className = "group-label"; h.textContent = title; comps.appendChild(h);
  const box = document.createElement("div"); box.className = "bars"; comps.appendChild(box);
  divergingBars(box, rows, { center: 50, span: 50, fmt: (v) => v == null ? "–" : Math.round(v) });
}

// ---------- 표
function table(tableEl, head, rows) {
  tableEl.innerHTML = "<thead><tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r) => `<tr class="${r.cls || ""}">` + r.cells.map((c) => `<td class="${c.cls || ""}">${c.v}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
const sign = (x) => x == null ? "" : x >= 0 ? "pos" : "neg";
$("phase-hint").textContent = `이후 ${D.horizon}거래일 수익률`;
table($("phase-table"), ["국면", "일수", "평균", "중앙값", "승률"], D.phase_stats.map((p) => ({
  cls: p.key === L.phase ? "current" : "",
  cells: [{ v: p.name + (p.key === L.phase ? " · 현재" : "") }, { v: p.days }, { v: pct(p.mean), cls: sign(p.mean) },
          { v: pct(p.median), cls: sign(p.median) }, { v: Math.round(p.win * 100) + "%" }],
})));
const M = D.metrics;
$("kpis").innerHTML = [
  ["예측 횟수", M.n_predictions?.toLocaleString()], ["정확도", (M.accuracy * 100).toFixed(1) + "%"],
  ["항상 '상승' 찍기", (M.base_rate_up * 100).toFixed(1) + "%"], ["AUC", M.auc?.toFixed(3)], ["보유 비중", Math.round(M.exposure * 100) + "%"],
].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("") + `<span class="hint">AUC 0.5 = 동전 던지기, 0.55 이상이면 의미 있는 신호일 가능성</span>`;
table($("strat-table"), ["전략", "연수익", "변동성", "샤프", "최대낙폭"], D.strategies.map((s) => ({
  cells: [{ v: s.name }, { v: pct(s.cagr), cls: sign(s.cagr) }, { v: pct(s.volatility) }, { v: s.sharpe?.toFixed(2) ?? "–" }, { v: pct(s.max_drawdown), cls: "neg" }],
})));
if (D.sources) {
  $("sources-panel").hidden = false;
  table($("sources-table"), ["모델", "AUC", "변화", "샤프"], D.sources.map((s) => ({
    cells: [{ v: s.name }, { v: s.auc?.toFixed(3) }, { v: (s.gain >= 0 ? "+" : "") + s.gain.toFixed(3), cls: s.gain >= 0.01 ? "pos" : s.gain < 0 ? "neg" : "" }, { v: s.sharpe?.toFixed(2) }],
  })));
}
if (D.name === "가상 시장") {
  $("notice").hidden = false;
  $("notice").textContent = "시연용 가상 시장 데이터입니다. 이 시장은 군중심리가 가격을 움직이도록 만든 세계라 예측 성적이 실제보다 훨씬 좋게 나옵니다. 실제 시장의 AUC는 보통 0.5~0.6 수준입니다.";
}

// ---------- 시계열 차트 (x축 공유, 호버 동기화)
const S = D.series;
const N = S.dates.length;
let range = 252;
let hoverIdx = null;
const charts = [];

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function makeChart(id, opts) {
  const root = $(id);
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; root.appendChild(tip);
  const chart = { root, tip, opts, svg: null };
  charts.push(chart);
  return chart;
}

function drawChart(c) {
  const { opts, root } = c;
  if (c.svg) c.svg.remove();
  const W = Math.max(280, root.clientWidth), H = opts.height;
  const m = { l: 46, r: 12, t: 8, b: opts.xAxis ? 24 : 6 };
  const start = range ? Math.max(0, N - range) : 0;
  const idx = [...Array(N - start).keys()].map((i) => i + start);
  const x = (i) => m.l + ((i - start) / Math.max(1, N - 1 - start)) * (W - m.l - m.r);
  const tf = opts.log ? (v) => Math.log10(v) : (v) => v;
  let [lo, hi] = opts.domain || (() => {
    const vals = opts.lines.flatMap((ln) => idx.map((i) => ln.values[i])).filter((v) => v != null && (!opts.log || v > 0)).map(tf);
    const a = Math.min(...vals), b = Math.max(...vals), pad = (b - a) * 0.06 || 1;
    return [a - pad, b + pad];
  })();
  const y = (v) => m.t + (1 - (tf(v) - lo) / (hi - lo)) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": opts.aria });
  c.svg = svg; c.x = x; c.y = y; c.start = start; c.W = W; c.m = m;
  root.insertBefore(svg, c.tip);

  for (const b of opts.bands || []) {
    el("rect", { x: m.l, width: W - m.l - m.r, y: y(b[1]), height: y(b[0]) - y(b[1]), fill: css(b[2]) }, svg);
  }
  const ticks = opts.ticks || (opts.log
    ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter((t) => tf(t) >= lo && tf(t) <= hi)
    : niceTicks(lo, hi, 4));
  for (const t of ticks) {
    if (tf(t) < lo || tf(t) > hi) continue;
    el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: css("--grid"), "stroke-width": 1 }, svg);
    const tx = el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: css("--ink-3"), "font-family": css("--font-data") }, svg);
    tx.textContent = opts.tickFmt ? opts.tickFmt(t) : t;
  }
  if (opts.refLine != null) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(opts.refLine), y2: y(opts.refLine), stroke: css("--ink-3"), "stroke-dasharray": "3 3", "stroke-width": 1 }, svg);
  }
  if (opts.phaseStrip) {
    const bw = (W - m.l - m.r) / Math.max(1, N - 1 - start);
    const yb = H - m.b - 6;
    for (const i of idx) {
      const p = S.phase[i];
      if (!p || !(FEAR_PHASES.has(p) || GREED_PHASES.has(p))) continue;
      el("rect", { x: x(i) - bw / 2, y: yb, width: bw + 0.5, height: 6, fill: css(FEAR_PHASES.has(p) ? "--fear" : "--greed"), opacity: 0.75 }, svg);
    }
  }
  for (const ln of opts.lines) {
    let d = "", pen = false;
    for (const i of idx) {
      const v = ln.values[i];
      if (v == null) { pen = false; continue; }
      d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1);
      pen = true;
    }
    el("path", { d, fill: "none", stroke: css(ln.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const lastI = idx.filter((i) => ln.values[i] != null).pop();
    if (lastI != null) el("circle", { cx: x(lastI), cy: y(ln.values[lastI]), r: 4, fill: css(ln.color), stroke: css("--panel"), "stroke-width": 2 }, svg);
  }
  if (opts.xAxis) {
    const n = W < 500 ? 3 : 6;
    for (let k = 0; k <= n; k++) {
      const i = Math.round(start + (k / n) * (N - 1 - start));
      const t = el("text", { x: x(i), y: H - 6, "text-anchor": k === 0 ? "start" : k === n ? "end" : "middle", "font-size": 11, fill: css("--ink-3"), "font-family": css("--font-data") }, svg);
      t.textContent = S.dates[i].slice(0, 7);
    }
  }
  c.cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, svg);
  c.dots = opts.lines.map((ln) => el("circle", { r: 4, fill: css(ln.color), stroke: css("--panel"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  const onMove = (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
    const i = Math.round(start + ((px * (W / rect.width) - m.l) / (W - m.l - m.r)) * (N - 1 - start));
    setHover(Math.max(start, Math.min(N - 1, i)), c);
  };
  hit.addEventListener("mousemove", onMove);
  hit.addEventListener("touchmove", onMove, { passive: true });
  hit.addEventListener("mouseleave", () => setHover(null));
  updateHover(c, null);
}

function updateHover(c, src) {
  const i = hoverIdx;
  if (i == null || i < c.start) {
    c.cross.setAttribute("visibility", "hidden");
    c.dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    c.tip.hidden = true;
    return;
  }
  const xi = c.x(i);
  c.cross.setAttribute("x1", xi); c.cross.setAttribute("x2", xi); c.cross.setAttribute("visibility", "visible");
  c.opts.lines.forEach((ln, k) => {
    const v = ln.values[i];
    const d = c.dots[k];
    if (v == null) { d.setAttribute("visibility", "hidden"); return; }
    d.setAttribute("cx", xi); d.setAttribute("cy", c.y(v)); d.setAttribute("visibility", "visible");
  });
  if (c !== src) { c.tip.hidden = true; return; }
  const rows = c.opts.tip(i).map(([k, v]) => `<div class="row"><span class="k">${k}</span><span class="num">${v}</span></div>`).join("");
  c.tip.innerHTML = `<div class="num" style="margin-bottom:4px">${S.dates[i]}</div>${rows}`;
  c.tip.hidden = false;
  const scale = c.svg.getBoundingClientRect().width / c.W;
  const left = xi * scale;
  const tw = c.tip.offsetWidth;
  c.tip.style.left = (left + 12 + tw > c.root.clientWidth ? left - tw - 12 : left + 12) + "px";
  c.tip.style.top = "4px";
}
function setHover(i, src) { hoverIdx = i; charts.forEach((c) => updateHover(c, src)); }

const phaseName = (i) => S.phase[i] ? D.phases[S.phase[i]] : "–";
const fmtPrice = (v) => v == null ? "–" : v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(2);
makeChart("c-price", {
  height: 190, aria: "가격 추이", phaseStrip: true,
  lines: [{ values: S.close, color: "--ink" }],
  tickFmt: (t) => fmtPrice(t),
  tip: (i) => [["가격", fmtPrice(S.close[i])], ["국면", phaseName(i)], ["공포·탐욕", r0(S.fear_greed[i])]],
});
makeChart("c-fg", {
  height: 150, aria: "공포·탐욕 지수 추이", domain: [0, 100], ticks: [0, 20, 50, 80, 100],
  bands: [[0, 20, "--fear-soft"], [20, 40, "--fear-faint"], [60, 80, "--greed-faint"], [80, 100, "--greed-soft"]],
  lines: [{ values: S.fear_greed, color: "--ink" }],
  tip: (i) => [["공포·탐욕", r0(S.fear_greed[i])], ["국면", phaseName(i)]],
});
makeChart("c-proba", {
  height: 130, aria: "모델 상승 확률 추이", domain: [0, 1], ticks: [0, 0.5, 1], refLine: 0.5, xAxis: true,
  tickFmt: (t) => Math.round(t * 100) + "%",
  lines: [{ values: S.proba, color: "--s1" }],
  tip: (i) => [["상승 확률", S.proba[i] == null ? "–" : Math.round(S.proba[i] * 100) + "%"], ["판단", S.proba[i] == null ? "–" : S.proba[i] > 0.5 ? "보유" : "현금"]],
});

// 백테스트 자산 곡선 (전체 검증 구간, 범위 버튼과 무관)
const EQ = [
  { values: S.eq_model, color: "--s1", label: "심리 모델" },
  { values: S.eq_contrarian, color: "--s2", label: "역발상 규칙" },
  { values: S.eq_hold, color: "--s3", label: "단순 보유" },
];
$("eq-legend").innerHTML = EQ.map((s) => `<span><i style="background:var(${s.color})"></i>${s.label}</span>`).join("");
const eqStart = S.eq_model.findIndex((v) => v != null);
const eqChart = makeChart("c-eq", {
  height: 200, aria: "전략별 자산 곡선 (로그 눈금)", xAxis: true, lines: EQ, refLine: 1, log: true,
  tickFmt: (t) => t + "x",
  tip: (i) => EQ.map((s) => [s.label, s.values[i] == null ? "–" : s.values[i].toFixed(2) + "x"]),
});

function drawAll() {
  for (const c of charts) {
    if (c === eqChart) {
      const saved = range; range = eqStart > 0 ? N - eqStart : 0; drawChart(c); range = saved;
    } else drawChart(c);
  }
}
document.querySelectorAll(".controls button").forEach((b) => b.addEventListener("click", () => {
  range = +b.dataset.range;
  document.querySelectorAll(".controls button").forEach((o) => o.setAttribute("aria-pressed", o === b ? "true" : "false"));
  drawAll();
}));
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 120); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", drawAll);
new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
drawAll();
</script>
"""
