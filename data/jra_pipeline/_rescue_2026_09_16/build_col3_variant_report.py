# -*- coding: utf-8 -*-
"""3連複フォーメーション3列目の基準比較レポート(score5>=30【既存】 vs score4>=30 vs score4>=40)を
生成する。既存の3連複フォーメーション・マーク細分化検証(8090331f)と同じダーク基調のトートボード
デザインを継承。"""
import json, pathlib

SP = pathlib.Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad")
OUT = SP / "col3_variant_report.html"

search = json.loads((SP / "col3_variant_search.json").read_text(encoding="utf-8"))
diag = json.loads((SP / "col3_variant_diagnosis.json").read_text(encoding="utf-8"))

meta = search["meta"]
MARK_LABEL = meta["mark_label"]
VARIANTS = meta["variants"]  # [{name, field, threshold, label}, ...]
POOL_STATS = meta["pool_stats"]

VARIANT_ROWS_JSON = {}
for v in VARIANTS:
    vname = v["name"]
    rows = search["results"][vname]
    VARIANT_ROWS_JSON[vname] = json.dumps([{
        "c1": r["c1"], "c2": r["c2"], "c3": r["c3"],
        "c1l": MARK_LABEL[r["c1"]], "c2l": MARK_LABEL[r["c2"]], "c3l": MARK_LABEL[r["c3"]],
        "all": r["all"], "high": r["high"], "mid": r["mid"], "low": r["low"],
    } for r in rows], ensure_ascii=False)

TITLE = "3連複フォーメーション3列目基準比較"
DESCRIPTION = ("3連複フォーメーション3列目の母集団定義を「スコア5(BOX5モデル予測)≥30【既存】」"
               "「スコア4(BOX4モデル予測)≥30」「スコア4(BOX4モデル予測)≥40」の3通りに変えた場合、"
               "216パターン(単一マーク6状態×3列独立)の的中率・回収率・95%信頼区間がどう変わるかを比較する。")

HTML = r"""<title>__TITLE__</title>
<meta name="description" content="__DESCRIPTION__">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=M+PLUS+1p:wght@400;500;700;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    color-scheme: light;
    --bg: #F3F1E7; --bg-elev: #FBFAF4; --bg-card: #FFFFFF; --bg-sunken: #E8E4D4;
    --ink: #16201A; --ink-muted: #55604F; --ink-faint: #838C79;
    --rule: #D9D4BF; --rule-strong: #C2BC9F;
    --accent: #1F7A4D; --accent-ink: #FFFFFF; --accent-soft: #DCEEE1; --accent-soft-ink: #145536;
    --warn: #9A6A12; --warn-soft: #F6E8CC; --warn-soft-ink: #6B4A0C;
    --bad: #B23A2E; --bad-soft: #F2DCD6; --bad-soft-ink: #7A2A20;
    --tierHigh: #1F7A4D; --tierMid: #9A6A12; --tierLow: #7A5FB2;
    --v0: #2B5FCC; --v1: #A5478A; --v2: #C77A1E;
    --shadow: 0 1px 2px rgba(20,30,20,.08), 0 10px 24px -14px rgba(20,30,20,.28);
    --display: "Oswald", "Yu Gothic", sans-serif;
    --sans: "M PLUS 1p", "Hiragino Sans", "Noto Sans JP", sans-serif;
    --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #0E1210; --bg-elev: #161C18; --bg-card: #1B2320; --bg-sunken: #10140F;
      --ink: #EDF3EE; --ink-muted: #9FB0A6; --ink-faint: #6C7A72;
      --rule: #2A332E; --rule-strong: #3B463E;
      --accent: #4ADE80; --accent-ink: #08210F; --accent-soft: #16341F; --accent-soft-ink: #9FEFB8;
      --warn: #F2C14E; --warn-soft: #3A2E12; --warn-soft-ink: #F2C14E;
      --bad: #F0654A; --bad-soft: #3A1E16; --bad-soft-ink: #F3A08D;
      --tierHigh: #4ADE80; --tierMid: #F2C14E; --tierLow: #B79CF2;
      --v0: #6C9BFF; --v1: #E28FCB; --v2: #F2B15C;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 28px -16px rgba(0,0,0,.6);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0E1210; --bg-elev: #161C18; --bg-card: #1B2320; --bg-sunken: #10140F;
    --ink: #EDF3EE; --ink-muted: #9FB0A6; --ink-faint: #6C7A72;
    --rule: #2A332E; --rule-strong: #3B463E;
    --accent: #4ADE80; --accent-ink: #08210F; --accent-soft: #16341F; --accent-soft-ink: #9FEFB8;
    --warn: #F2C14E; --warn-soft: #3A2E12; --warn-soft-ink: #F2C14E;
    --bad: #F0654A; --bad-soft: #3A1E16; --bad-soft-ink: #F3A08D;
    --tierHigh: #4ADE80; --tierMid: #F2C14E; --tierLow: #B79CF2;
    --v0: #6C9BFF; --v1: #E28FCB; --v2: #F2B15C;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 28px -16px rgba(0,0,0,.6);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.6; -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); }
  .page { max-width: 1220px; margin: 0 auto; padding: 28px 20px 90px; }
  .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent); font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
  }
  .eyebrow::before { content: ""; width: 20px; height: 2px; background: var(--accent); }
  h1 { font-family: var(--display); font-size: 27px; font-weight: 700; margin: 0 0 10px; text-wrap: balance; }
  h2 { font-family: var(--display); font-size: 18px; font-weight: 600; margin: 40px 0 12px; }
  .dek { color: var(--ink-muted); font-size: 14px; max-width: 82ch; margin: 0 0 8px; }
  .cross-link { font-size: 12.5px; margin: 4px 0 18px; }
  .cross-link a { text-decoration: none; }
  .cross-link a:hover { text-decoration: underline; }

  .warnbar {
    background: var(--warn-soft); color: var(--warn-soft-ink); border: 1px solid var(--warn);
    border-radius: 10px; padding: 14px 16px; margin: 8px 0 4px; font-size: 12.5px; line-height: 1.75;
    display: flex; gap: 10px; align-items: flex-start;
  }
  .warnbar .icon { font-size: 16px; flex: none; }
  .warnbar b { color: var(--warn-soft-ink); }

  .variant-legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0; }
  .variant-card { background: var(--bg-card); border: 1px solid var(--rule); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }
  .variant-card h3 { margin: 0 0 6px; font-size: 13px; font-family: var(--display); font-weight: 600; display: flex; align-items: center; gap: 6px; }
  .variant-card .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
  .variant-card .pool-line { font-size: 12px; color: var(--ink-muted); margin: 3px 0; }
  .variant-card .pool-line b { color: var(--ink); font-family: var(--mono); }

  .cmp-table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 12px; box-shadow: var(--shadow); margin: 10px 0; }
  table.cmp { width: 100%; border-collapse: collapse; font-size: 12.5px; background: var(--bg-card); }
  table.cmp th, table.cmp td { padding: 8px 12px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
  table.cmp th:first-child, table.cmp td:first-child { text-align: left; }
  table.cmp thead th { background: var(--bg-elev); font-size: 10.5px; text-transform: uppercase; letter-spacing: .03em; color: var(--ink-faint); font-weight: 700; }
  table.cmp td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  table.cmp tr.is-variant-head td { font-weight: 700; background: var(--bg-elev); }
  table.cmp td.rate.over { color: var(--accent); font-weight: 700; }
  table.cmp td.rate.under { color: var(--ink-faint); }

  .diag-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 12px; margin: 10px 0 24px; }
  .diag-card { background: var(--bg-card); border: 1px solid var(--rule); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }
  .diag-card.is-baseline { border-style: dashed; }
  .diag-head { font-family: var(--display); font-weight: 600; font-size: 13px; margin-bottom: 2px; }
  .diag-cond { font-size: 10.5px; color: var(--ink-faint); margin-bottom: 10px; line-height: 1.6; }
  .diag-pop-row { margin: 8px 0; }
  .diag-pop-label { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--ink-muted); margin-bottom: 3px; }
  .diag-pop-label .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .diag-pop-label .pt { font-family: var(--mono); font-weight: 700; color: var(--ink); margin-left: auto; }
  .ci-bar-wrap { position: relative; height: 14px; background: var(--bg-sunken); border-radius: 4px; }
  .ci-bar-track { position: absolute; top: 5px; left: 0; right: 0; height: 4px; }
  .ci-bar-range { position: absolute; top: 0; height: 4px; background: var(--rule-strong); border-radius: 2px; }
  .ci-bar-point { position: absolute; top: -3px; width: 2px; height: 10px; background: var(--ink-muted); border-radius: 1px; }
  .ci-breakeven { position: absolute; top: -3px; width: 1px; height: 10px; background: var(--bad); }
  .ci-labels { display: flex; justify-content: space-between; font-size: 9px; color: var(--ink-faint); font-family: var(--mono); margin-top: 2px; }

  .variant-tabbar { display: flex; gap: 8px; margin: 20px 0 4px; border-bottom: 1px solid var(--rule); }
  .vtab-btn {
    font-family: var(--sans); font-size: 13px; font-weight: 700; color: var(--ink-muted);
    background: none; border: none; padding: 10px 4px; cursor: pointer; position: relative; top: 1px;
    border-bottom: 2px solid transparent;
  }
  .vtab-btn.is-active { color: var(--accent); border-bottom-color: var(--accent); }
  .vtab-btn:hover { color: var(--accent); }

  .table-controls { display: flex; gap: 10px; align-items: center; margin: 14px 0 8px; flex-wrap: wrap; }
  .table-controls label { font-size: 12px; color: var(--ink-muted); display: flex; gap: 6px; align-items: center; }
  .table-controls select {
    font-family: var(--mono); font-size: 12px; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--rule-strong);
    background: var(--bg-card); color: var(--ink);
  }
  .table-count { font-family: var(--mono); font-size: 11.5px; color: var(--ink-faint); margin-left: auto; }

  .tbl-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 12px; box-shadow: var(--shadow); }
  table.patterns { width: 100%; border-collapse: collapse; font-size: 12px; background: var(--bg-card); }
  table.patterns th {
    position: sticky; top: 0; background: var(--bg-elev); text-align: left; padding: 8px 10px; cursor: pointer;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .03em; color: var(--ink-faint); font-weight: 700;
    border-bottom: 1px solid var(--rule-strong); white-space: nowrap; user-select: none;
  }
  table.patterns th:hover { color: var(--accent); }
  table.patterns th.sorted::after { content: " " attr(data-dir); }
  table.patterns td { padding: 7px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
  table.patterns tbody tr:hover td { background: var(--bg-elev); }
  table.patterns td.num { font-family: var(--mono); text-align: right; font-variant-numeric: tabular-nums; }
  table.patterns td.rate.over { color: var(--accent); font-weight: 700; }
  table.patterns td.rate.under { color: var(--ink-faint); }
  .lowsample-tag { font-size: 9px; border: 1px dashed var(--warn); color: var(--warn); border-radius: 3px; padding: 0 3px; margin-left: 5px; }

  .notes-block { margin-top: 34px; font-size: 12px; color: var(--ink-faint); max-width: 92ch; line-height: 1.8; }
  .notes-block p { margin: 0 0 8px; }
  footer { margin-top: 30px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 11px; color: var(--ink-faint); }
</style>

<div class="page">
  <div class="eyebrow">JRA通常戦・3連複フォーメーション検証</div>
  <h1>3連複フォーメーション3列目基準比較</h1>
  <p class="dek">__DESCRIPTION__</p>
  <div class="cross-link"><a href="https://claude.ai/code/artifact/8090331f-d155-4af7-afe8-99be22feffe7">← 3連複フォーメーション・マーク細分化検証(既存3列目=スコア5≥30版)</a></div>
  <div class="cross-link"><a href="https://claude.ai/code/artifact/627a1a36-6038-450d-87c9-a198ed880db6">← 3連複フォーメーション列別検証ツール(インタラクティブ版)</a></div>

  <div class="warnbar">
    <span class="icon">⚠️</span>
    <span>
      3列目の基準を変えるだけでも、216パターン×4母集団×3基準=2,592通りの結果をこの319レース
      という同一母集団で比較していることに変わりはありません。点推定の回収率が上振れしていても、
      下の「代表候補の頑健性チェック」が示す通りブロックブートストラップ95%信頼区間は
      いずれも損益分岐点100%を跨いでおり、統計的に有意な優位性は確認できていません。
      <b>参考値としてご覧ください。</b>
    </span>
  </div>

  <h2>3列目の3つの基準</h2>
  <div class="variant-legend">__VARIANT_LEGEND__</div>

  <h2>ベースライン(全列マークなし)の比較</h2>
  <p class="dek">1列目・2列目はマークなし(BOX4上位4頭そのまま)、3列目だけを3基準で切り替えた場合の
  的中率・回収率。3列目の母集団が変わると、購入点数(スコア対象頭数)・的中率・回収率がどう動くかを
  直接比較できます。</p>
  <div class="cmp-table-wrap">
    <table class="cmp">
      <thead><tr><th>基準 / 母集団</th><th>購入R</th><th>的中R</th><th>的中率</th><th>回収率</th></tr></thead>
      <tbody>__BASELINE_TABLE__</tbody>
    </table>
  </div>

  <h2>各基準内の回収率トップ候補の比較</h2>
  <p class="dek">各基準・各母集団で216パターン中もっとも回収率が高かった組み合わせ(最低購入20R以上)。
  基準を変えると上位に来るマークの組み合わせ自体も変わる点に注意してください。</p>
  <div class="cmp-table-wrap">
    <table class="cmp">
      <thead><tr><th>基準 / 母集団</th><th>1列目</th><th>2列目</th><th>3列目</th><th>購入R</th><th>的中率</th><th>回収率</th></tr></thead>
      <tbody>__TOP1_TABLE__</tbody>
    </table>
  </div>

  <h2>代表候補の頑健性チェック(ブロックブートストラップ95%CI)</h2>
  <p class="dek">上記のベースラインと回収率トップ候補について、開催日×競馬場ブロック単位の
  ブートストラップで95%信頼区間を算出しました。基準ごとにタブで切り替えられます。</p>
  <div class="variant-tabbar">__DIAG_TABS__</div>
  __DIAG_PANELS__

  <h2>全216パターン一覧</h2>
  <div class="table-controls">
    <label>3列目の基準: <select id="variantFilter">__VARIANT_OPTIONS__</select></label>
    <label>表示する母集団: <select id="popFilter">
      <option value="all" selected>全319R</option>
      <option value="high">高確信度(gap_top2上位)</option>
      <option value="mid">中確信度</option>
      <option value="low">低確信度(gap_top2下位)</option>
    </select></label>
    <label>最低購入レース数: <select id="minBetFilter">
      <option value="0">制限なし</option>
      <option value="20" selected>20R以上(推奨)</option>
      <option value="50">50R以上</option>
      <option value="100">100R以上</option>
    </select></label>
    <span class="table-count" id="rowCount"></span>
  </div>
  <div class="tbl-wrap">
    <table class="patterns" id="patTable">
      <thead><tr>
        <th data-key="c1l" data-type="str">1列目マーク</th>
        <th data-key="c2l" data-type="str">2列目マーク</th>
        <th data-key="c3l" data-type="str">3列目マーク</th>
        <th data-key="n_bet" data-type="num">購入R</th>
        <th data-key="hit_races" data-type="num">的中R</th>
        <th data-key="hit_rate_pct" data-type="num">的中率</th>
        <th data-key="return" data-type="num">総払戻</th>
        <th data-key="return_rate_pct" data-type="num" class="sorted" data-dir="▼">回収率</th>
      </tr></thead>
      <tbody id="patBody"></tbody>
    </table>
  </div>

  <div class="notes-block">
    <p><b>券種・買い方:</b> 3連複フォーメーション固定。1列目・2列目・3列目からそれぞれ1頭ずつ選び
    3頭とも異なる組み合わせを100円ずつ購入(重複除去後の点数)。1列目=BOX4上位4頭+マーク追加、
    2列目=BOX4上位4頭−マーク除外(1列目とは独立集計)。3列目のみ本ページの比較対象で、基準
    「スコア5(BOX5モデル予測)≥30」「スコア4(BOX4モデル予測)≥30」「スコア4(BOX4モデル予測)≥40」
    のプールから該当マーク馬を除外。</p>
    <p><b>マークの意味:</b> 「外し」=単勝回収率10%未満台帳、「複外」=複勝回収率25%未満台帳、
    「W外し」=ワイド回収率10%未満台帳、「馬連外し」=馬連回収率10%未満台帳、「3連複外し」=
    3連複回収率10%未満台帳、それぞれの非冗長パターンに一致した馬。</p>
    <p><b>確信度(gap_top2):</b> BOX5モデルの予測1位と2位のスコア差。三分位で高/中/低に3等分
    (境界値 __TIER_LO_EDGE__ / __TIER_HI_EDGE__、各群 高__TIER_HIGH__R・中__TIER_MID__R・低__TIER_LOW__R)。</p>
    <p><b>母集団:</b> 馬柱データ2と同じ全319レース(2026-07-11〜09-06、通常戦のみ)。3連複の
    払戻データは実際のJRA払戻金額を使用。</p>
  </div>

  <footer>build_col3_variant_report.py — 元データ: col3_variant_search.json(216パターン×4母集団×3基準)</footer>
</div>

<script>
const VARIANT_ROWS = __VARIANT_ROWS_JSON__;
let sortKey = "return_rate_pct", sortDir = -1, minBet = 20, pop = "all", variant = "__DEFAULT_VARIANT__";

function render() {
  const rows = VARIANT_ROWS[variant];
  const filtered = rows.filter(r => r[pop].n_bet >= minBet);
  filtered.sort((a, b) => (a[pop][sortKey] - b[pop][sortKey]) * sortDir);
  const tbody = document.getElementById('patBody');
  tbody.innerHTML = filtered.slice(0, 216).map(r => {
    const p = r[pop];
    const low = p.n_bet < 60;
    const rateCls = p.return_rate_pct > 100 ? 'over' : 'under';
    return `<tr>
      <td>${r.c1l}</td><td>${r.c2l}</td><td>${r.c3l}</td>
      <td class="num">${p.n_bet}${low ? '<span class="lowsample-tag">少標本</span>' : ''}</td>
      <td class="num">${p.hit_races}</td>
      <td class="num">${p.hit_rate_pct.toFixed(1)}%</td>
      <td class="num">¥${p.return.toLocaleString()}</td>
      <td class="num rate ${rateCls}">${p.return_rate_pct.toFixed(1)}%</td>
    </tr>`;
  }).join('');
  document.getElementById('rowCount').textContent = `${filtered.length}件を表示`;
}

document.querySelectorAll('table.patterns th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey === key) { sortDir *= -1; } else { sortKey = key; sortDir = -1; }
    document.querySelectorAll('table.patterns th').forEach(t => { t.classList.remove('sorted'); t.removeAttribute('data-dir'); });
    th.classList.add('sorted');
    th.setAttribute('data-dir', sortDir === -1 ? '▼' : '▲');
    render();
  });
});
document.getElementById('minBetFilter').addEventListener('change', (e) => { minBet = parseInt(e.target.value, 10); render(); });
document.getElementById('popFilter').addEventListener('change', (e) => { pop = e.target.value; render(); });
document.getElementById('variantFilter').addEventListener('change', (e) => { variant = e.target.value; render(); });
render();

document.querySelectorAll('.vtab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.vtab-btn').forEach(b => b.classList.remove('is-active'));
    btn.classList.add('is-active');
    document.querySelectorAll('.diag-panel').forEach(p => { p.hidden = p.dataset.variant !== btn.dataset.variant; });
  });
});
</script>
"""

TIER_DOT = {"all": "var(--ink-muted)", "high": "var(--tierHigh)", "mid": "var(--tierMid)", "low": "var(--tierLow)"}
POP_LABEL = {"all": "全319R", "high": "高確信度", "mid": "中確信度", "low": "低確信度"}
VARIANT_DOT = {"score5_30": "var(--v0)", "score4_30": "var(--v1)", "score4_40": "var(--v2)"}
VARIANT_SHORT = {"score5_30": "スコア5≥30【既存】", "score4_30": "スコア4≥30", "score4_40": "スコア4≥40"}

# ---- 3列目の基準の凡例カード ----
variant_cards = []
for v in VARIANTS:
    vname = v["name"]
    ps = POOL_STATS[vname]
    variant_cards.append(f"""
    <div class="variant-card">
      <h3><span class="dot" style="background:{VARIANT_DOT[vname]}"></span>{v['label']}</h3>
      <p class="pool-line">平均プール頭数: <b>{ps['avg_pool_size']}</b>頭/R(範囲 {ps['min_pool_size']}〜{ps['max_pool_size']}頭)</p>
      <p class="pool-line">プール0頭のレース: <b>{ps['empty_races']}</b> / {ps['n_races']}R</p>
    </div>""")

# ---- ベースライン比較テーブル ----
baseline_rows = []
for v in VARIANTS:
    vname = v["name"]
    cand_list = diag[vname]["candidates"]
    baseline = next(c for c in cand_list if c["c1"] == "none" and c["c2"] == "none" and c["c3"] == "none")
    baseline_rows.append(f'<tr class="is-variant-head"><td colspan="5">{v["label"]}</td></tr>')
    for pop_key in ("all", "high", "mid", "low"):
        p = baseline[pop_key]
        rate_cls = "over" if p["return_rate_pct"] > 100 else "under"
        baseline_rows.append(f"""<tr>
          <td>{POP_LABEL[pop_key]}</td>
          <td class="num">{p['n_bet']}</td>
          <td class="num">{p['hit_races']}</td>
          <td class="num">{p['hit_rate_pct']:.1f}%</td>
          <td class="num rate {rate_cls}">{p['return_rate_pct']:.1f}%</td>
        </tr>""")

# ---- トップ候補比較テーブル ----
top1_rows = []
for v in VARIANTS:
    vname = v["name"]
    cand_list = diag[vname]["candidates"]
    top1_rows.append(f'<tr class="is-variant-head"><td colspan="7">{v["label"]}</td></tr>')
    for c in cand_list:
        if "baseline" in c["label"]:
            continue
        pop_key = c["label"].split("-")[0]
        p = c[pop_key]
        rate_cls = "over" if p["return_rate_pct"] > 100 else "under"
        top1_rows.append(f"""<tr>
          <td>{POP_LABEL[pop_key]}</td>
          <td>{c['c1_label']}</td><td>{c['c2_label']}</td><td>{c['c3_label']}</td>
          <td class="num">{p['n_bet']}</td>
          <td class="num">{p['hit_rate_pct']:.1f}%</td>
          <td class="num rate {rate_cls}">{p['return_rate_pct']:.1f}%</td>
        </tr>""")


def build_diag_cards(cand_list):
    cards = []
    for d in cand_list:
        is_baseline = "baseline" in d["label"]
        cond = f"1列目: {d['c1_label']} / 2列目: {d['c2_label']} / 3列目: {d['c3_label']}"
        pop_rows = []
        for pop_key in ("all", "high", "mid", "low"):
            p = d[pop_key]
            lo, hi, pt = p["ci_lo"], p["ci_hi"], p["return_rate_pct"]
            scale_max = max(hi, 250) * 1.05
            lo_pct = max(0, lo) / scale_max * 100
            hi_pct = min(hi, scale_max) / scale_max * 100
            pt_pct = max(0, min(pt, scale_max)) / scale_max * 100
            be_pct = 100 / scale_max * 100
            pop_rows.append(f"""
            <div class="diag-pop-row">
              <div class="diag-pop-label"><span class="dot" style="background:{TIER_DOT[pop_key]}"></span>
                {POP_LABEL[pop_key]}(n={p['n_bet']}) <span class="pt">{pt:.1f}%</span></div>
              <div class="ci-bar-wrap">
                <div class="ci-bar-track">
                  <div class="ci-bar-range" style="left:{lo_pct:.1f}%;width:{max(0,hi_pct-lo_pct):.1f}%"></div>
                  <div class="ci-bar-point" style="left:{pt_pct:.1f}%"></div>
                  <div class="ci-breakeven" style="left:{be_pct:.1f}%"></div>
                </div>
              </div>
              <div class="ci-labels"><span>{lo:.0f}%</span><span>hit {p['hit_rate_pct']:.0f}%</span><span>{hi:.0f}%</span></div>
            </div>""")
        cards.append(f"""
        <div class="diag-card{' is-baseline' if is_baseline else ''}">
          <div class="diag-head">{d['label']}</div>
          <div class="diag-cond">{cond}</div>
          {"".join(pop_rows)}
        </div>""")
    return "".join(cards)


diag_tabs = []
diag_panels = []
for i, v in enumerate(VARIANTS):
    vname = v["name"]
    active = " is-active" if i == 0 else ""
    diag_tabs.append(f'<button class="vtab-btn{active}" data-variant="{vname}">{VARIANT_SHORT[vname]}</button>')
    hidden_attr = "" if i == 0 else " hidden"
    cards_html = build_diag_cards(diag[vname]["candidates"])
    diag_panels.append(f'<div class="diag-panel diag-grid" data-variant="{vname}"{hidden_attr}>{cards_html}</div>')

variant_options = "".join(
    f'<option value="{v["name"]}"{" selected" if i == 0 else ""}>{v["label"]}</option>'
    for i, v in enumerate(VARIANTS)
)

meta_tier = meta["tier_count"]

HTML = (HTML.replace("__TITLE__", TITLE)
            .replace("__DESCRIPTION__", DESCRIPTION)
            .replace("__VARIANT_LEGEND__", "".join(variant_cards))
            .replace("__BASELINE_TABLE__", "".join(baseline_rows))
            .replace("__TOP1_TABLE__", "".join(top1_rows))
            .replace("__DIAG_TABS__", "".join(diag_tabs))
            .replace("__DIAG_PANELS__", "".join(diag_panels))
            .replace("__VARIANT_OPTIONS__", variant_options)
            .replace("__VARIANT_ROWS_JSON__", json.dumps(VARIANT_ROWS_JSON if False else {v["name"]: json.loads(VARIANT_ROWS_JSON[v["name"]]) for v in VARIANTS}, ensure_ascii=False))
            .replace("__DEFAULT_VARIANT__", VARIANTS[0]["name"])
            .replace("__TIER_HIGH__", str(meta_tier["high"]))
            .replace("__TIER_MID__", str(meta_tier["mid"]))
            .replace("__TIER_LOW__", str(meta_tier["low"]))
            .replace("__TIER_LO_EDGE__", f'{meta["gap_top2_edges"]["lo"]:.3f}')
            .replace("__TIER_HI_EDGE__", f'{meta["gap_top2_edges"]["hi"]:.3f}'))

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
