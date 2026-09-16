# -*- coding: utf-8 -*-
"""3連複フォーメーション「1列目・2列目・3列目」を、それぞれ独立したチェックボックス
(外し/複外/W外し/馬連外し/3連複外し、計5種)で拡張できるインタラクティブ検証ツール。
  1列目 = BOX4上位4頭 ∪ (1列目でチェックしたマークの馬)                [追加方向]
  2列目 = BOX4上位4頭 − (2列目でチェックしたマークの馬)               [除外方向・1列目とは独立]
  3列目 = スコア5≥30の馬プール − (3列目でチェックしたマークの馬)     [除外方向・2列目とは独立]
全319レース分の必要データ(馬番・馬名・BOX4上位4頭か・各マーク・実際の着順・3連複払戻)を
JSONとして埋め込み、クライアントサイドJSで組み合わせ生成・決済・ブロックブートストラップまで
その場で再計算する。"""
import sys, json, pathlib
sys.path.insert(0, r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
import jra_dataset

SP = pathlib.Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad")
OUT = SP / "formation_tool.html"

data = jra_dataset.load(rebuild=False)
races_ds, actual = data["races"], data["actual"]
actual_by_id = {r["race_id"]: actual.get(r["race_id"], {}) for r in races_ds}

data2 = json.loads((SP / "data2_races.json").read_text(encoding="utf-8"))
marks5 = json.loads((SP / "low_hit_marks5.json").read_text(encoding="utf-8"))
TANSHO, FUKU, WIDE = marks5["tansho"], marks5["fuku"], marks5["wide"]
UMAREN, SANRENPUKU = marks5["umaren"], marks5["sanrenpuku"]

races_out = []
for r in data2:
    race_id = r["race_id"]
    block = f"{r['kaisai_date']}_{r['racecourse']}"
    horses = []
    for h in r["horses"]:
        umaban = h.get("umaban")
        key = f"{race_id}/{umaban}"
        rank4 = h.get("rank4")
        fp = h.get("finish_pos")
        try:
            fp_i = int(fp)
        except (TypeError, ValueError):
            fp_i = None
        sc5 = h.get("score5")
        horses.append({
            "u": umaban, "n": h.get("name"),
            "b": bool(rank4 is not None and rank4 <= 4),  # BOX4上位4頭か
            "t": key in TANSHO, "f": key in FUKU, "w": key in WIDE,
            "ur": key in UMAREN, "sr": key in SANRENPUKU,
            "s": (round(sc5 * 100, 1) if sc5 is not None else None),  # スコア5(100点換算)
            "fp": fp_i,
        })
    table = actual_by_id.get(race_id, {}).get("3連複", {})
    pay = {"-".join(str(x) for x in sorted(c)): p for c, p in table.items() if p > 0}
    races_out.append({
        "id": race_id, "d": r["kaisai_date"], "c": r["racecourse"],
        "rn": r.get("race_number"), "name": r.get("race_name"), "blk": block,
        "h": horses, "pay": pay,
    })

RACES_JSON = json.dumps(races_out, ensure_ascii=False, separators=(",", ":"))
print(f"embedded {len(races_out)} races, json size {len(RACES_JSON):,} chars")

TITLE = "3連複フォーメーション列別検証ツール"
DESCRIPTION = ("3連複フォーメーションの1・2・3列目をそれぞれ独立した基準+チェックボックス"
               "(外し/複外/W外し/馬連外し/3連複外し、計5種)で検証できるインタラクティブツール。"
               "1列目=BOX4上位4頭に該当マーク馬を追加、2列目=BOX4上位4頭から同マーク該当馬を除外、"
               "3列目=スコア5(モデル予測)が30以上の馬プールから同マーク該当馬を除外。3列とも起点は独立。")

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
    --col2: #2B5FCC; --col2-soft: #DDE7FA; --col2-soft-ink: #1E3E8C;
    --col3: #A5478A; --col3-soft: #F5E1EF; --col3-soft-ink: #7A3169;
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
      --col2: #7DA2F2; --col2-soft: #182238; --col2-soft-ink: #A9C2F5;
      --col3: #E28FCB; --col3-soft: #301B29; --col3-soft-ink: #EBB6DC;
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
    --col2: #7DA2F2; --col2-soft: #182238; --col2-soft-ink: #A9C2F5;
    --col3: #E28FCB; --col3-soft: #301B29; --col3-soft-ink: #EBB6DC;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 28px -16px rgba(0,0,0,.6);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.6; -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); }
  .page { max-width: 1320px; margin: 0 auto; padding: 28px 20px 90px; }
  .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent); font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
  }
  .eyebrow::before { content: ""; width: 20px; height: 2px; background: var(--accent); }
  h1 { font-family: var(--display); font-size: 27px; font-weight: 700; margin: 0 0 10px; text-wrap: balance; }
  .dek { color: var(--ink-muted); font-size: 14px; max-width: 86ch; margin: 0 0 8px; }
  .cross-link { font-size: 12.5px; margin: 4px 0 18px; }
  .cross-link a { text-decoration: none; }
  .cross-link a:hover { text-decoration: underline; }

  .warnbar {
    background: var(--warn-soft); color: var(--warn-soft-ink); border: 1px solid var(--warn);
    border-radius: 10px; padding: 14px 16px; margin: 8px 0 20px; font-size: 12.5px; line-height: 1.75;
    display: flex; gap: 10px; align-items: flex-start;
  }
  .warnbar .icon { font-size: 16px; flex: none; }

  .col-panel { background: var(--bg-card); border: 1px solid var(--rule); border-radius: 12px; box-shadow: var(--shadow); padding: 16px 18px; margin: 8px 0 20px; }
  .col-row { display: grid; grid-template-columns: 190px 1fr; gap: 14px; align-items: center; padding: 9px 0; border-bottom: 1px dashed var(--rule); }
  .col-row:last-child { border-bottom: none; }
  .col-name { font-family: var(--display); font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px; }
  .col-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
  .col-dot.c1 { background: var(--accent); }
  .col-dot.c2 { background: var(--col2); }
  .col-dot.c3 { background: var(--col3); }
  .col-base { font-size: 10.5px; color: var(--ink-faint); margin-top: 2px; }
  .col-checks { display: flex; gap: 16px; flex-wrap: wrap; }
  .col-checks label { display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; user-select: none; }
  .col-checks input { width: 16px; height: 16px; cursor: pointer; }
  .tag-out { font-size: 9.5px; border: 1px solid var(--ink-faint); color: var(--ink-faint); border-radius: 3px; padding: 0 4px; }

  .summary-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap: 10px; margin: 10px 0 8px; }
  .summary-stat { background: var(--bg-card); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 14px; text-align: center; box-shadow: var(--shadow); }
  .summary-stat .sl { font-size: 9.5px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: .04em; }
  .summary-stat .sv { font-family: var(--mono); font-size: 19px; font-weight: 700; margin-top: 3px; }
  .summary-stat .sv.accent { color: var(--accent); }
  .summary-stat.wide { grid-column: span 2; }
  .ci-note { font-size: 10.5px; color: var(--ink-faint); margin-top: 3px; }
  .calc-note { font-size: 10.5px; color: var(--ink-faint); margin: 0 0 18px; }

  .table-controls { display: flex; gap: 10px; align-items: center; margin: 14px 0 8px; flex-wrap: wrap; }
  .table-controls label { font-size: 12px; color: var(--ink-muted); display: flex; gap: 6px; align-items: center; }
  .table-controls select {
    font-family: var(--mono); font-size: 12px; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--rule-strong);
    background: var(--bg-card); color: var(--ink);
  }
  .table-count { font-family: var(--mono); font-size: 11.5px; color: var(--ink-faint); margin-left: auto; }

  .tbl-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 12px; box-shadow: var(--shadow); max-height: 900px; overflow-y: auto; }
  table.races { width: 100%; border-collapse: collapse; font-size: 12px; background: var(--bg-card); }
  table.races th {
    position: sticky; top: 0; background: var(--bg-elev); text-align: left; padding: 8px 10px; z-index: 2;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .03em; color: var(--ink-faint); font-weight: 700;
    border-bottom: 1px solid var(--rule-strong); white-space: nowrap;
  }
  table.races td { padding: 8px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  table.races td.rname { max-width: 150px; font-size: 11.5px; color: var(--ink-muted); }
  table.races td .sub { font-size: 10px; color: var(--ink-faint); }
  table.races tbody tr:hover td { background: var(--bg-elev); }
  table.races tr.hit td { background: var(--accent-soft); }
  table.races tr.hit:hover td { filter: brightness(0.97); }
  .chips { display: flex; flex-wrap: wrap; gap: 3px; min-width: 110px; max-width: 230px; align-items: center; }
  .h-chip {
    display: inline-flex; align-items: center; justify-content: center; min-width: 20px; height: 18px; padding: 0 3px;
    border-radius: 4px; font-family: var(--mono); font-size: 10px; font-weight: 700;
  }
  .h-chip.c1 { background: var(--accent-soft); color: var(--accent-soft-ink); border: 1px solid var(--accent); }
  .h-chip.c2 { background: var(--col2-soft); color: var(--col2-soft-ink); border: 1px solid var(--col2); }
  .h-chip.c3 { background: var(--col3-soft); color: var(--col3-soft-ink); border: 1px solid var(--col3); }
  .pn { font-size: 10px; color: var(--ink-faint); margin-left: 3px; }
  td.num { font-family: var(--mono); white-space: nowrap; }
  td.result { font-size: 11px; color: var(--ink-muted); white-space: nowrap; }
  td.paycell { font-family: var(--mono); font-weight: 700; white-space: nowrap; }
  td.paycell.hit { color: var(--accent); }
  td.paycell.miss { color: var(--ink-faint); font-weight: 400; }

  .notes-block { margin-top: 30px; font-size: 12px; color: var(--ink-faint); max-width: 92ch; line-height: 1.8; }
  .notes-block p { margin: 0 0 8px; }
  footer { margin-top: 30px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 11px; color: var(--ink-faint); }
</style>

<div class="page">
  <div class="eyebrow">3連複フォーメーション・インタラクティブ検証</div>
  <h1>3連複フォーメーション列別マーク検証ツール</h1>
  <p class="dek">__DESCRIPTION__</p>
  <div class="cross-link"><a href="https://claude.ai/code/artifact/96de8129-f3f9-4676-833c-0a9a5393eafd">← 3連複フォーメーション探索(全1,050パターン一覧)</a></div>
  <div class="cross-link"><a href="https://claude.ai/code/artifact/2e88cb0d-b081-4c68-aa95-e02795206ffe">← 馬柱データ2(元データ)</a></div>

  <div class="warnbar">
    <span class="icon">⚠️</span>
    <span>
      軸に使うBOX4モデル、比較に使う「外し」「複外」「W外し」「馬連外し」「3連複外し」の各マークは、
      いずれもこの319レース(2026-07-11〜09-06)という同一母集団を対象にした網羅的探索で得られたものです。
      チェックボックスを変えるたびに違う「パターン」を試すことになり、多数の組み合わせを試すほど
      選択バイアスで良い数値が出やすくなります。表示される的中率・回収率は<b>参考値</b>としてご覧ください。
    </span>
  </div>

  <div class="col-panel">
    <div class="col-row">
      <div class="col-name"><span class="col-dot c1"></span>1列目</div>
      <div>
        <div class="col-base">基準: BOX4モデル予測上位4頭 + 下記チェックした馬</div>
        <div class="col-checks">
          <label><input type="checkbox" data-col="1" data-mark="t"> 外し</label>
          <label><input type="checkbox" data-col="1" data-mark="f"> 複外</label>
          <label><input type="checkbox" data-col="1" data-mark="w"> W外し</label>
          <label><input type="checkbox" data-col="1" data-mark="ur"> 馬連外し</label>
          <label><input type="checkbox" data-col="1" data-mark="sr"> 3連複外し</label>
        </div>
      </div>
    </div>
    <div class="col-row">
      <div class="col-name"><span class="col-dot c2"></span>2列目</div>
      <div>
        <div class="col-base">基準: BOX4モデル予測上位4頭 <span class="tag-out">マーク不問</span> から、下記チェックした該当マークの馬を除外(1列目とは独立)</div>
        <div class="col-checks">
          <label><input type="checkbox" data-col="2" data-mark="t"> 外しを除外</label>
          <label><input type="checkbox" data-col="2" data-mark="f"> 複外を除外</label>
          <label><input type="checkbox" data-col="2" data-mark="w"> W外しを除外</label>
          <label><input type="checkbox" data-col="2" data-mark="ur"> 馬連外しを除外</label>
          <label><input type="checkbox" data-col="2" data-mark="sr"> 3連複外しを除外</label>
        </div>
      </div>
    </div>
    <div class="col-row">
      <div class="col-name"><span class="col-dot c3"></span>3列目</div>
      <div>
        <div class="col-base">基準: スコア5(モデル予測)が30以上の馬 <span class="tag-out">マーク不問</span> から、下記チェックした該当マークの馬を除外(2列目とは独立)</div>
        <div class="col-checks">
          <label><input type="checkbox" data-col="3" data-mark="t"> 外しを除外</label>
          <label><input type="checkbox" data-col="3" data-mark="f"> 複外を除外</label>
          <label><input type="checkbox" data-col="3" data-mark="w"> W外しを除外</label>
          <label><input type="checkbox" data-col="3" data-mark="ur"> 馬連外しを除外</label>
          <label><input type="checkbox" data-col="3" data-mark="sr"> 3連複外しを除外</label>
        </div>
      </div>
    </div>
  </div>

  <div class="summary-row">
    <div class="summary-stat"><div class="sl">購入レース</div><div class="sv" id="sNBet">-</div></div>
    <div class="summary-stat"><div class="sl">的中レース</div><div class="sv" id="sHit">-</div></div>
    <div class="summary-stat"><div class="sl">的中率</div><div class="sv" id="sHitRate">-</div></div>
    <div class="summary-stat"><div class="sl">総投資</div><div class="sv" id="sStake">-</div></div>
    <div class="summary-stat"><div class="sl">総払戻</div><div class="sv" id="sReturn">-</div></div>
    <div class="summary-stat wide"><div class="sl">回収率(95%CI・ブロックブートストラップ)</div>
      <div class="sv accent" id="sRate">-</div>
      <div class="ci-note" id="sCi">-</div>
    </div>
  </div>
  <p class="calc-note" id="calcNote">計算中...</p>

  <div class="table-controls">
    <label>表示: <select id="hitFilter">
      <option value="all" selected>全レース</option>
      <option value="hit">的中レースのみ</option>
      <option value="miss">不的中レースのみ</option>
    </select></label>
    <span class="table-count" id="rowCount"></span>
  </div>
  <div class="tbl-wrap">
    <table class="races" id="raceTable">
      <thead><tr>
        <th>開催日</th><th>レース名</th><th>1列目</th><th>2列目</th><th>3列目</th>
        <th>購入点数</th><th>結果(着順)</th><th>払戻</th>
      </tr></thead>
      <tbody id="raceBody"></tbody>
    </table>
  </div>

  <div class="notes-block">
    <p><b>列の定義(3列とも起点は独立、互いを含みません):</b> 1列目=BOX4モデル予測上位4頭に、
    チェックした該当マーク馬を<b>追加</b>。2列目=BOX4モデル予測上位4頭(1列目とは別集計)から、
    チェックした該当マーク馬を<b>除外</b>。3列目=スコア5(モデル予測)が30以上の馬プールから、
    チェックした該当マーク馬を<b>除外</b>。<b>1列目はチェックで馬が増え、2・3列目はチェックで
    馬が減る</b>、列によって向きが逆になる点にご注意ください。購入点数は「1列目・2列目・3列目から
    それぞれ1頭ずつ選び3頭とも異なる」全組み合わせの重複除去後の件数、1点100円。</p>
    <p><b>マークの意味:</b> 「外し」=単勝回収率10%未満台帳(的中率0.0%・非冗長83パターン)に一致、
    「複外」=複勝回収率25%未満台帳(的中率5%未満・非冗長54パターン)に一致、「W外し」=ワイド回収率
    10%未満台帳(的中率0.0%・非冗長66パターン)に一致、「馬連外し」=馬連回収率10%未満台帳(的中率0.0%・
    非冗長112パターン)に一致、「3連複外し」=3連複回収率10%未満台帳(的中率0.0%・非冗長193パターン)に
    一致。5種とも本番score/pred_rankには使われていない参考の消し材料です。1列目ではこれらのマーク
    該当馬を追加、2・3列目では除外に使います。</p>
    <p><b>スコア5(3列目の起点):</b> BOX5モデルの予測スコアを100点満点に換算した値。まずこのスコアが
    30以上の馬をマーク不問で全て集め、そこから上記チェックで該当マーク馬を間引きます。</p>
    <p><b>母集団:</b> 馬柱データ2と同じ全319レース(2026-07-11〜09-06、通常戦のみ)。全ての計算は
    このページの読み込み後、ブラウザ内でその場で行われます(サーバー通信なし)。</p>
  </div>

  <footer>build_formation_tool.py — 券種: 3連複フォーメーション固定・1列目はチェックで追加、2・3列目はそれぞれ独立の起点からチェックで除外</footer>
</div>

<script>
const RACES = __RACES_JSON__;
const WEEKDAY_JP = ["月","火","水","木","金","土","日"];
function dateLabel(d) {
  const y=+d.slice(0,4), m=+d.slice(4,6), day=+d.slice(6,8);
  const dt = new Date(y, m-1, day);
  return `${m}/${day}(${WEEKDAY_JP[dt.getDay()===0?6:dt.getDay()-1]})`;
}
function esc(s) {
  return (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

const SCORE3_MIN = 30;  // 3列目の起点プール: スコア5がこの値以上(マーク不問)
const MARK_KEYS = ["t", "f", "w", "ur", "sr"];
const checks = {
  1: {t:false,f:false,w:false,ur:false,sr:false},
  2: {t:false,f:false,w:false,ur:false,sr:false},
  3: {t:false,f:false,w:false,ur:false,sr:false},
};
document.querySelectorAll('input[type=checkbox][data-col]').forEach(cb => {
  cb.addEventListener('change', () => {
    checks[cb.dataset.col][cb.dataset.mark] = cb.checked;
    recompute();
  });
});

function anyMarkChecked(h, col) {
  return MARK_KEYS.some(k => h[k] && checks[col][k]);
}

function colSets(horses) {
  const c1 = new Set(), c2 = new Set(), c3 = new Set();
  // 1列目: BOX4上位4頭に、チェックした該当マーク馬を追加
  for (const h of horses) {
    if (h.b || anyMarkChecked(h, 1)) c1.add(h.u);
  }
  // 2列目: 1列目とは独立。BOX4上位4頭から、チェックした該当マーク馬を除外(絞り込み)
  for (const h of horses) {
    if (!h.b) continue;
    if (anyMarkChecked(h, 2)) continue;
    c2.add(h.u);
  }
  // 3列目: 2列目とは独立。スコア5≥30のプールから、チェックした該当マーク馬を除外(絞り込み)
  for (const h of horses) {
    if (h.s == null || h.s < SCORE3_MIN) continue;
    if (anyMarkChecked(h, 3)) continue;
    c3.add(h.u);
  }
  return [c1, c2, c3];
}

function formationCombos(c1, c2, c3) {
  const combos = new Set();
  const a1 = [...c1], a2 = [...c2], a3 = [...c3];
  for (const x of a1) for (const y of a2) {
    if (y === x) continue;
    for (const z of a3) {
      if (z === x || z === y) continue;
      const key = [x,y,z].sort((p,q)=>p-q).join('-');
      combos.add(key);
    }
  }
  return combos;
}

function horseChip(h, cls) {
  return `<span class="h-chip ${cls}" title="${esc(h.n)}">${h.u}</span>`;
}

let lastResults = [];

function recompute() {
  let nBet=0, hitRaces=0, stakeSum=0, retSum=0;
  const blockStat = {};
  lastResults = [];
  for (const r of RACES) {
    const [c1,c2,c3] = colSets(r.h);
    const combos = formationCombos(c1,c2,c3);
    let stake=0, ret=0, isHit=false;
    if (combos.size>0) {
      stake = combos.size*100;
      for (const k of combos) { const p = r.pay[k]; if (p) ret += p; }
      isHit = ret>0;
      nBet++; stakeSum+=stake; retSum+=ret;
      if (isHit) hitRaces++;
      const bs = blockStat[r.blk] || [0,0];
      bs[0]+=stake; bs[1]+=ret; blockStat[r.blk]=bs;
    }
    const byU = {}; for (const h of r.h) byU[h.u]=h;
    const top3 = r.h.filter(h=>h.fp!=null).sort((a,b)=>a.fp-b.fp).slice(0,3);
    lastResults.push({r, c1, c2, c3, combos, stake, ret, isHit, top3});
  }
  const hitRate = nBet? hitRaces/nBet*100 : 0;
  const returnRate = stakeSum? retSum/stakeSum*100 : 0;

  // ブロックブートストラップ(95%CI、擬似乱数はシード固定のLCGで再現性を持たせる)
  const blocks = Object.keys(blockStat);
  const stakesB = blocks.map(b=>blockStat[b][0]), retsB = blocks.map(b=>blockStat[b][1]);
  let seed = 2026;
  function rnd() { seed = (seed*1664525+1013904223) >>> 0; return seed/4294967296; }
  const nB = blocks.length;
  const bootRates = [];
  for (let i=0;i<3000;i++){
    let s=0,rt=0;
    for (let j=0;j<nB;j++){ const idx=Math.floor(rnd()*nB); s+=stakesB[idx]; rt+=retsB[idx]; }
    bootRates.push(s>0? rt/s*100 : 0);
  }
  bootRates.sort((a,b)=>a-b);
  const ciLo = bootRates[Math.floor(0.025*bootRates.length)];
  const ciHi = bootRates[Math.floor(0.975*bootRates.length)];

  document.getElementById('sNBet').textContent = `${nBet}/319`;
  document.getElementById('sHit').textContent = hitRaces;
  document.getElementById('sHitRate').textContent = hitRate.toFixed(1)+'%';
  document.getElementById('sStake').textContent = '¥'+stakeSum.toLocaleString();
  document.getElementById('sReturn').textContent = '¥'+retSum.toLocaleString();
  document.getElementById('sRate').textContent = returnRate.toFixed(1)+'%';
  document.getElementById('sCi').textContent =
    `[${ciLo.toFixed(1)}%, ${ciHi.toFixed(1)}%]（${nB}ブロック）` +
    (ciLo<100 ? '（損益分岐点100%を含む=非有意）' : '（損益分岐点100%を上回る）');
  document.getElementById('calcNote').textContent =
    `${RACES.length}レース中${nBet}レースで1列目・2列目・3列目とも1頭以上そろい購入対象（カバー率${(nBet/RACES.length*100).toFixed(1)}%）`;

  renderTable();
}

function renderTable() {
  const mode = document.getElementById('hitFilter').value;
  const tbody = document.getElementById('raceBody');
  const rows = [];
  for (const x of lastResults) {
    if (mode==='hit' && !x.isHit) continue;
    if (mode==='miss' && x.isHit) continue;
    const r = x.r;
    if (x.combos.size===0 && mode!=='miss' && mode!=='all') continue;
    const c1h = r.h.filter(h=>x.c1.has(h.u));
    const c2h = r.h.filter(h=>x.c2.has(h.u));
    const c3h = r.h.filter(h=>x.c3.has(h.u));
    const top3txt = x.top3.map(h=>`${h.fp}着${h.u}${esc(h.n)}`).join(' ');
    const hitCls = x.combos.size===0 ? '' : (x.isHit ? 'hit' : 'miss');
    const payTxt = x.combos.size===0 ? '<span class="sub">対象外</span>' :
      (x.isHit ? `的中 ¥${x.ret.toLocaleString()}` : '不的中');
    const c2cell = c2h.length ? c2h.map(h=>horseChip(h,'c2')).join('') : '<span class="sub">(該当なし)</span>';
    const c3cell = c3h.length ? c3h.map(h=>horseChip(h,'c3')).join('') : '<span class="sub">(該当なし)</span>';
    rows.push(`<tr class="${hitCls}">
      <td>${dateLabel(r.d)}<br><span class="sub">${esc(r.c)}${esc(r.rn)}R</span></td>
      <td class="rname">${esc(r.name)}</td>
      <td class="chips">${c1h.map(h=>horseChip(h,'c1')).join('')}<span class="pn">(${c1h.length})</span></td>
      <td class="chips">${c2cell}<span class="pn">(${c2h.length})</span></td>
      <td class="chips">${c3cell}<span class="pn">(${c3h.length})</span></td>
      <td class="num">${x.combos.size}点<br><span class="sub">¥${x.stake.toLocaleString()}</span></td>
      <td class="result">${top3txt}</td>
      <td class="paycell ${hitCls}">${payTxt}</td>
    </tr>`);
  }
  tbody.innerHTML = rows.join('');
  document.getElementById('rowCount').textContent = `${rows.length}件を表示`;
}

document.getElementById('hitFilter').addEventListener('change', renderTable);
recompute();
</script>
"""

HTML = (HTML.replace("__TITLE__", TITLE)
            .replace("__DESCRIPTION__", DESCRIPTION)
            .replace("__RACES_JSON__", RACES_JSON))

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
