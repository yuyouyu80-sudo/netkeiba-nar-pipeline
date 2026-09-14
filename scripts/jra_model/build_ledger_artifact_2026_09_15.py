# -*- coding: utf-8 -*-
"""回収率110%台帳(単勝・複勝・ワイド・馬連・馬単・3連複・3連単)7券種ぶんのインタラクティブ
HTMLを`jra_ledger_search_2026_09_15.py`の出力から生成する。

デザイン・AND再計算エンジンは既存の「3連複回収率110%台帳」(2026-09-07生成、元の生成
スクリプトはリポジトリに存在しないが公開済みArtifactのHTMLから復元)を踏襲。馬レコードの
圧縮(キー短縮・カテゴリ値短縮)は`build_factor_artifact.py`の`_compact_horse_keys`/
`_encode_horses_positionally`をそのまま再利用する(無改造)。
"""
import copy
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
import build_factor_artifact as BFA  # noqa: E402 (_compact_horse_keys等を再利用)
import jra_dataset_wide as JDW  # noqa: E402
import jra_factor_registry as FR  # noqa: E402

import os  # noqa: E402
_DEFAULT_SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
)
SCRATCHPAD = Path(os.environ.get("CLAUDE_SCRATCHPAD", str(_DEFAULT_SCRATCHPAD)))
if not SCRATCHPAD.parent.exists():
    SCRATCHPAD = PROJECT_ROOT / ".artifact_out"
SCRATCHPAD.mkdir(parents=True, exist_ok=True)

SEARCH_JSON = DATA_DIR / "jra_ledger_search_2026_09_15_result.json"
FACTOR_JSON = DATA_DIR / "factor_database.json"

# 券種: (slug, combo_k, ordered, ファイル名, 既存Artifact URL(無ければNone=新規), 索引ページ上の呼称)
BET_META = {
    "単勝": dict(slug="tansho", k=1, ordered=False, file="ledger_tansho.html",
                url="https://claude.ai/code/artifact/4b9bba27-4c29-4942-903f-743cf68db35a"),
    "複勝": dict(slug="fukusho", k=1, ordered=False, file="ledger_fukusho.html",
                url="https://claude.ai/code/artifact/2297a9a1-8cd5-4386-8bae-e04fc2e88a4c"),
    "馬連": dict(slug="umaren", k=2, ordered=False, file="ledger_umaren.html",
                url="https://claude.ai/code/artifact/88c905f2-13cb-4a06-b2b0-cd1790c52fb4"),
    "ワイド": dict(slug="wide", k=2, ordered=False, file="ledger_wide.html",
                 url="https://claude.ai/code/artifact/3e774502-09c8-4c39-8fa7-b22fa90c79df"),
    "馬単": dict(slug="umatan", k=2, ordered=True, file="ledger_umatan.html",
                url=None),
    "3連複": dict(slug="sanrenpuku", k=3, ordered=False, file="ledger_sanrenpuku.html",
                 url="https://claude.ai/code/artifact/5aac2d4a-5cb6-4418-9a1e-fce73f867971"),
    "3連単": dict(slug="sanrentan", k=3, ordered=True, file="ledger_sanrentan.html",
                 url="https://claude.ai/code/artifact/bf303c79-32b6-4588-a57a-a30a8a31a536"),
}

NOTES_BY_BT = {
    "単勝": "単勝は1頭を当てるだけの単純な券種ですが、条件を絞るほど回収率100%超えの多くは"
            "少数の穴馬的中に支えられています。的中率が高くても回収率が低い(堅実だが儲からない)"
            "パターンと、的中率は低いが回収率が高い(たまに大穴が当たって平均を押し上げている)"
            "パターンの両方が混在する点にご注意ください。",
    "複勝": "複勝は3着以内(小頭数レースでは2着以内)に入れば的中するため、単勝より的中率は"
            "高くなりやすい券種です。ただし回収率100%超えのパターンの多くは、やはり一部の"
            "好走(人気薄の3着内入線)に支えられています。",
    "馬連": "馬連は2頭の組番(着順不問)を当てる券種のため、対象馬を絞っても組み合わせ数"
            "(点数)は該当馬数k頭に対しk×(k-1)÷2で増えます。的中率が数%台のパターンも"
            "少なくありません。",
    "ワイド": "ワイドは上位3着以内のうち2頭の組番を当てる(3組が同時的中の対象になりうる)"
             "券種のため、他の組番系券種より的中率は高めに出やすい一方、1点あたりの配当は"
             "小さめです。",
    "馬単": "馬単は2頭の着順(1着→2着)まで当てる券種のため、対象馬を絞っても組み合わせ数"
           "(点数)は該当馬数k頭に対しk×(k-1)(着順ありなので馬連の2倍)で増えます。的中率は"
           "馬連よりさらに低くなりやすい点にご注意ください。",
    "3連複": "3連複は3頭の組番(着順不問)を当てる券種のため、対象馬を絞っても組み合わせ数"
            "(点数)が急増しやすく、的中率が数%台のパターンが大半になります。少数の高配当的中に"
            "支えられている可能性が高い点にご注意ください。",
    "3連単": "3連単は3頭の着順(1着→2着→3着)まで当てる券種のため、組番系の中で最も点数が"
            "膨らみやすく、的中率は1%未満のパターンも珍しくありません。回収率が数百〜数千%に"
            "達するパターンも、実態はごく少数(1〜数レース)の万馬券級的中に支えられています。",
}

CSS_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="description" content="__DESC__">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700;900&display=swap');

  :root {
    color-scheme: light;
    --bg: #ECEEEA;
    --bg-elev: #F7F8F5;
    --bg-card: #FFFFFF;
    --bg-sunken: #E2E4DD;
    --ink: #1A1C1B;
    --ink-muted: #5B5F5A;
    --ink-faint: #7B8078;
    --rule: #D2D5CD;
    --rule-strong: #B7BBB1;
    --accent: #B23A2E;
    --accent-ink: #FFFFFF;
    --accent-soft: #F2DCD6;
    --accent-soft-ink: #7A2A20;
    --warn: #9A6A12;
    --warn-soft: #F6E8CC;
    --warn-soft-ink: #6B4A0C;
    --good: #2E6B47;
    --good-soft: #DCEBE1;
    --good-soft-ink: #1E4A31;
    --shadow: 0 1px 2px rgba(20,22,18,.06), 0 6px 16px -10px rgba(20,22,18,.18);
    --sans: "Zen Kaku Gothic New", "Yu Gothic", "Hiragino Sans", "Noto Sans JP", sans-serif;
    --mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #16181A; --bg-elev: #1C1F21; --bg-card: #202325; --bg-sunken: #101213;
      --ink: #E7E8E3; --ink-muted: #A6ABA1; --ink-faint: #83887E;
      --rule: #2E3230; --rule-strong: #3D423E;
      --accent: #E2604C; --accent-ink: #1A1110; --accent-soft: #3A241F; --accent-soft-ink: #F0B7A9;
      --warn: #D9A94C; --warn-soft: #3A2E12; --warn-soft-ink: #EFCE8C;
      --good: #59B989; --good-soft: #163524; --good-soft-ink: #A9E4C4;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 20px -12px rgba(0,0,0,.5);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #16181A; --bg-elev: #1C1F21; --bg-card: #202325; --bg-sunken: #101213;
    --ink: #E7E8E3; --ink-muted: #A6ABA1; --ink-faint: #83887E;
    --rule: #2E3230; --rule-strong: #3D423E;
    --accent: #E2604C; --accent-ink: #1A1110; --accent-soft: #3A241F; --accent-soft-ink: #F0B7A9;
    --warn: #D9A94C; --warn-soft: #3A2E12; --warn-soft-ink: #EFCE8C;
    --good: #59B989; --good-soft: #163524; --good-soft-ink: #A9E4C4;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 20px -12px rgba(0,0,0,.5);
  }

  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.6; -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); }
  .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  .page { max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }

  .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
  }
  .eyebrow::before { content: ""; width: 18px; height: 1px; background: var(--accent); }
  h1 { font-size: 27px; font-weight: 900; margin: 0 0 8px; letter-spacing: -.01em; text-wrap: balance; }
  .dek { color: var(--ink-muted); font-size: 14px; max-width: 74ch; margin: 0 0 6px; }
  .cross-link { font-size: 12.5px; margin: 6px 0 18px; }

  .stat-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1px;
    background: var(--rule); border: 1px solid var(--rule); border-radius: 12px; overflow: hidden;
    margin-bottom: 18px; box-shadow: var(--shadow);
  }
  .stat {
    background: var(--bg-card); padding: 14px 16px;
  }
  .stat .label { font-size: 10.5px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
  .stat .value { font-family: var(--mono); font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat .value.accent { color: var(--accent); }

  .warnbar {
    background: var(--warn-soft); color: var(--warn-soft-ink); border: 1px solid var(--warn);
    border-radius: 10px; padding: 12px 16px; margin: 0 0 14px; font-size: 13px; line-height: 1.65;
    display: flex; gap: 12px; align-items: flex-start;
  }
  .warnbar .icon { font-size: 16px; flex: none; }

  .rebuild-note {
    background: var(--bg-elev); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 16px;
    margin: 0 0 20px; font-size: 12.5px; line-height: 1.65; color: var(--ink-muted);
  }
  .rebuild-note b { color: var(--ink); }

  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
  .toolbar input[type="search"] {
    flex: 1 1 260px; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--rule-strong);
    background: var(--bg-card); color: var(--ink); font-family: var(--sans); font-size: 13.5px;
  }
  .toolbar .count { font-size: 12px; color: var(--ink-faint); font-family: var(--mono); white-space: nowrap; }
  .toolbar .hint { font-size: 11px; color: var(--ink-faint); }
  .btn-reset {
    font-size: 12px; padding: 6px 12px; border-radius: 7px; border: 1px solid var(--rule-strong);
    background: var(--bg-card); color: var(--ink-muted); cursor: pointer; font-family: var(--sans);
    white-space: nowrap;
  }
  .btn-reset:hover { border-color: var(--accent); color: var(--accent); }
  .sort-explain {
    font-size: 12px; color: var(--ink-muted); background: var(--bg-elev); border: 1px solid var(--rule);
    border-radius: 8px; padding: 8px 12px; margin: -2px 0 12px; line-height: 1.6;
  }
  .sort-explain mark { background: var(--accent-soft); color: var(--accent-soft-ink); padding: 0 3px; border-radius: 3px; }
  tbody tr.tie-row td { background: var(--accent-soft); }
  tbody tr.tie-row:hover td { background: var(--accent-soft); filter: brightness(0.96); }
  tbody tr.tie-start td { border-top: 1px solid var(--accent); }
  tbody tr.is-checked td { background: var(--good-soft); }
  tbody tr.is-checked.tie-row td { background: var(--good-soft); }
  th.col-check, td.col-check { text-align: center !important; width: 34px; }
  td.col-check input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--good); cursor: pointer; }

  .and-panel {
    border: 1px solid var(--rule-strong); border-radius: 10px; background: var(--bg-card);
    box-shadow: var(--shadow); padding: 14px 16px; margin: 0 0 18px;
    position: sticky; top: 10px; z-index: 30;
  }
  .and-panel .and-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .and-panel h2 { font-size: 13.5px; margin: 0; font-weight: 700; }
  .and-panel .btn-clear {
    font-size: 11.5px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--rule-strong);
    background: var(--bg-elev); color: var(--ink-muted); cursor: pointer; font-family: var(--sans);
  }
  .and-panel .btn-clear:hover { border-color: var(--good); color: var(--good); }
  .and-panel .placeholder { font-size: 12.5px; color: var(--ink-faint); }
  .and-panel .selected-list { font-size: 11.5px; color: var(--ink-faint); margin-bottom: 8px; }
  .and-panel .merged-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }
  .and-panel .merged-chip {
    font-size: 11.5px; padding: 3px 9px; border-radius: 999px; background: var(--good-soft);
    color: var(--good-soft-ink); white-space: nowrap;
  }
  .and-panel .merged-chip b { font-weight: 600; }
  .and-panel .merged-chip .or-note { font-size: 9.5px; opacity: .75; margin-left: 3px; }
  .and-panel .merged-chip.muted { background: var(--bg-elev); color: var(--ink-faint); border: 1px dashed var(--rule); }
  .and-panel .and-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 1px;
    background: var(--rule); border: 1px solid var(--rule); border-radius: 10px; overflow: hidden;
  }
  .and-panel .and-stat { background: var(--bg-elev); padding: 10px 14px; }
  .and-panel .and-stat .label { font-size: 10px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 3px; }
  .and-panel .and-stat .value { font-family: var(--mono); font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .and-panel .and-stat .value.accent { color: var(--good); }
  .and-panel .and-note { font-size: 11px; color: var(--ink-faint); margin-top: 10px; line-height: 1.6; }
  .and-panel .and-zero { font-size: 12.5px; color: var(--warn-soft-ink); background: var(--warn-soft); border: 1px solid var(--warn); border-radius: 8px; padding: 8px 12px; }

  .table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--bg-card); box-shadow: var(--shadow); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-faint);
    font-weight: 600; background: var(--bg-sunken); text-align: right; padding: 10px 12px;
    border-bottom: 1px solid var(--rule-strong); cursor: pointer; white-space: nowrap; user-select: none;
  }
  thead th:first-child, thead th.col-cond { text-align: left; }
  thead th .arrow { font-size: 9px; margin-left: 3px; opacity: .5; }
  thead th.sorted .arrow { opacity: 1; color: var(--accent); }
  thead th .sort-order {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; border-radius: 50%; background: var(--accent); color: var(--accent-ink);
    font-size: 9px; font-weight: 700; margin-left: 4px; font-family: var(--mono); vertical-align: middle;
  }
  tbody td { padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--rule); vertical-align: top; }
  tbody td:first-child, td.col-cond { text-align: left; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--bg-elev); }
  .rank { font-family: var(--mono); color: var(--ink-faint); font-size: 12px; }
  .cond-chips { display: flex; flex-wrap: wrap; gap: 4px; max-width: 480px; }
  .cond-chip {
    font-size: 11.5px; padding: 2px 8px; border-radius: 999px; background: var(--bg-sunken);
    color: var(--ink-muted); white-space: nowrap;
  }
  .cond-chip b { color: var(--ink); font-weight: 600; }
  .return-rate { font-family: var(--mono); font-weight: 700; color: var(--accent); font-size: 14px; }
  .hit-rate { font-family: var(--mono); }
  .hit-rate.is-fragile { color: var(--warn-soft-ink); }
  .fragile-badge {
    display: inline-block; font-size: 9.5px; font-weight: 700; color: var(--warn-soft-ink);
    background: var(--warn-soft); border: 1px solid var(--warn); border-radius: 3px; padding: 0 4px;
    margin-left: 5px; vertical-align: middle; white-space: nowrap;
  }
  .no-results { padding: 40px; text-align: center; color: var(--ink-faint); font-size: 13px; }

  .notes-block { margin-top: 22px; font-size: 12px; color: var(--ink-faint); max-width: 84ch; line-height: 1.7; }
  .notes-block p { margin: 0 0 8px; }
  footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 11px; color: var(--ink-faint); }
</style>

<div class="page">
  <div class="eyebrow">JRA予想モデル・探索結果</div>
  <h1>__H1__</h1>
  <p class="dek">
    <a href="https://claude.ai/code/artifact/3fb262c6-079e-44ba-85b8-0bcaa02b2486">JRAファクター検証データベース</a>で、
    __BT__の対象レース数が母集団全体の40%以上、かつ回収率が110%を超える条件の組み合わせをビームサーチで探索し、
    回収率が高い順に全件収録したものです。日次の予想レポートとは別物で、検証済みの結論を示すものではありません。
  </p>
  <div class="cross-link">← <a href="https://claude.ai/code/artifact/1e94de55-9f1b-467b-a2e7-5e17eb436bbc">競馬予想レポート集</a>に戻る</div>

  <div class="stat-row" id="stat-row"></div>

  <div class="warnbar">
    <span class="icon">⚠️</span>
    <span>
      この一覧は自由なファクター探索(全__NATOMS__atom・__NGROUPS__グループ×最大6段の組み合わせ)から回収率110%超のものだけを抜き出したもので、
      <b>強い事後選択バイアス</b>がかかっています。回収率が高くても<b>的中率が低いパターンが多く</b>、
      少数の高配当的中に支えられている可能性が高い点にご注意ください。95%信頼区間を検証していないため、
      「差がある」と解釈しないでください。
    </span>
  </div>

  <div class="rebuild-note">
    <b>2026-09-15再構築</b>: 台帳生成に使ったビームサーチスクリプト自体がリポジトリから失われていたため、
    同じ設計思想(候補ファクター×最大6段・対象レース数40%以上・回収率110%超を全件収録)で探索アルゴリズムを
    新規実装し、2026-09-14時点のファクター検証データベース(レーダー細分化9本を含む__NGROUPS__グループ・
    __NATOMS__atom、旧台帳の169〜210atomから拡大)を使って再探索しました。探索母集団は直近2開催日を
    ホールドアウトとして除いた__SEARCHPOP__レース(全__FULLPOP__レース中)です。馬単は今回追加した新規券種です。
  </div>

  <div class="toolbar">
    <input type="search" id="filter-input" placeholder="条件で絞り込み…(例: 期待値、輸送、コース)">
    <span class="count" id="result-count"></span>
    <span class="hint">まずクリックで並び替え開始・続けてShiftクリックで第2・第3条件を追加</span>
    <button class="btn-reset" id="reset-sort">並び替えをリセット</button>
  </div>

  <div class="and-panel">
    <div class="and-head">
      <h2>選択したパターンをAND演算</h2>
      <button class="btn-clear" id="and-clear" hidden>選択を解除</button>
    </div>
    <div id="and-body">
      <p class="placeholder">表の右端のチェックボックスでパターンを選ぶと、選んだ条件をすべて同時に満たす馬だけに
        絞り込んだ場合の対象R数・的中率・回収率をここで再計算します(グループが重複する場合はそのグループ内でOR)。</p>
    </div>
  </div>

  <p class="sort-explain" id="sort-explain" hidden></p>

  <div class="table-wrap">
    <table id="report-table">
      <thead>
        <tr>
          <th data-key="rank" style="text-align:right">順位<span class="arrow">▾</span></th>
          <th class="col-cond" style="cursor:default">条件</th>
          <th data-key="n_races">対象R数<span class="arrow">▾</span></th>
          <th data-key="n_points">点数<span class="arrow">▾</span></th>
          <th data-key="hit_races">的中数<span class="arrow">▾</span></th>
          <th data-key="hit_rate_pct">的中率<span class="arrow">▾</span></th>
          <th data-key="return_rate_pct" class="sorted">回収率<span class="arrow">▾</span></th>
          <th class="col-check" style="cursor:default">AND</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
    <div class="no-results" id="no-results" hidden>該当する条件がありません</div>
  </div>

  <div class="notes-block" id="notes-block"></div>

  <footer id="footer"></footer>
</div>
"""

JS_TEMPLATE = r"""<script type="application/json" id="data-blob">__DATA_BLOB__</script>
<script type="application/json" id="race-data-blob">__RACE_DATA_BLOB__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data-blob").textContent);
  const meta = DATA.meta;
  const rows = DATA.rows;
  const rowsByRank = new Map(rows.map(r => [r.rank, r]));
  const BET_TYPE = "__BT__";
  const COMBO_K = __K__;
  const ORDERED = __ORDERED__;

  const RACE_DATA = JSON.parse(document.getElementById("race-data-blob").textContent);
  const FACTOR_GROUPS = RACE_DATA.factor_groups;
  (function decompressHorses() {
    const F = RACE_DATA._fields;
    if (!F) return;
    for (const race of RACE_DATA.races) {
      const rc = race.rc || {};
      race.horses = race.horses.map(row => {
        const o = Object.assign({}, rc);
        for (let i = 0; i < F.length; i++) {
          const v = row[i];
          if (v !== "" && v !== null && v !== undefined) o[F[i]] = v;
        }
        return o;
      });
    }
  })();
  const AND_MAIN_RACES = RACE_DATA.races;

  function andCombosK(arr, k, ordered) {
    const out = [];
    const n = arr.length;
    if (!ordered) {
      const combo = [];
      (function rec(start) {
        if (combo.length === k) { out.push(combo.slice()); return; }
        for (let i = start; i < n; i++) { combo.push(arr[i]); rec(i + 1); combo.pop(); }
      })(0);
    } else {
      const used = new Array(n).fill(false);
      const chosen = [];
      (function rec() {
        if (chosen.length === k) { out.push(chosen.slice()); return; }
        for (let i = 0; i < n; i++) {
          if (used[i]) continue;
          used[i] = true; chosen.push(arr[i]);
          rec();
          chosen.pop(); used[i] = false;
        }
      })();
    }
    return out;
  }
  function andEvalOption(horse, group, opt) {
    const v = horse[group.source];
    if (v === null || v === undefined) return false;
    if (group.kind === "rank_le") return v <= opt.params.le;
    if (group.kind === "threshold_ge") return v >= opt.params.ge;
    if (group.kind === "category_in") return opt.params.in.includes(v);
    return false;
  }
  function andHorseQualifies(horse, selectedByGroup) {
    for (const gid in selectedByGroup) {
      const ids = selectedByGroup[gid];
      if (!ids || ids.size === 0) continue;
      const group = FACTOR_GROUPS[gid];
      if (!group) continue;
      const opts = group.options.filter(o => ids.has(o.id));
      if (!opts.some(o => andEvalOption(horse, group, o))) return false;
    }
    return true;
  }
  function andFindPayout(payoutList, combo, ordered) {
    const key = ordered ? [...combo] : [...combo].sort((a, b) => a - b);
    for (const p of payoutList) {
      const pk = ordered ? [...p.combo] : [...p.combo].sort((a, b) => a - b);
      if (pk.length === key.length && pk.every((v, i) => v === key[i])) return p.payout;
    }
    return 0;
  }
  function andAggregate(selectedByGroup) {
    let n_races = 0, n_points = 0, hit_races = 0, stake = 0, ret = 0;
    for (const race of AND_MAIN_RACES) {
      if (!race.offered_bet_types.includes(BET_TYPE)) continue;
      const idx = [];
      race.horses.forEach((h, i) => { if (andHorseQualifies(h, selectedByGroup)) idx.push(i); });
      if (idx.length < COMBO_K) continue;
      const combos = andCombosK(idx.map(i => race.horses[i].umaban), COMBO_K, ORDERED);
      if (!combos.length) continue;
      const payoutList = ((RACE_DATA.payouts[race.race_id] || {})[BET_TYPE]) || [];
      let raceReturn = 0;
      for (const c of combos) raceReturn += andFindPayout(payoutList, c, ORDERED);
      n_races++; n_points += combos.length; stake += combos.length * 100; ret += raceReturn;
      if (raceReturn > 0) hit_races++;
    }
    return {
      n_races, n_points, hit_races, stake, return: ret,
      hit_rate_pct: n_races ? hit_races / n_races * 100 : 0,
      return_rate_pct: stake ? ret / stake * 100 : 0,
    };
  }

  const selectedRanks = new Set();

  function updateAndPanel() {
    const body = document.getElementById("and-body");
    const clearBtn = document.getElementById("and-clear");
    clearBtn.hidden = selectedRanks.size === 0;

    const selectedByGroup = {};
    const ranksSorted = [...selectedRanks].sort((a, b) => a - b);
    for (const rank of ranksSorted) {
      const row = rowsByRank.get(rank);
      if (!row) continue;
      for (const [gid, optId] of Object.entries(row.selected)) {
        if (!selectedByGroup[gid]) selectedByGroup[gid] = new Set();
        selectedByGroup[gid].add(optId);
      }
    }

    const chips = Object.entries(selectedByGroup).map(([gid, ids]) => {
      const group = FACTOR_GROUPS[gid];
      const labels = group.options.filter(o => ids.has(o.id)).map(o => o.label);
      const orNote = ids.size > 1 ? `<span class="or-note">(OR、${ids.size}択)</span>` : "";
      return `<span class="merged-chip"><b>${group.label}</b>: ${labels.join(" / ")}${orNote}</span>`;
    }).join("") || `<span class="merged-chip muted">条件なし(絞り込みなしの全馬がベースライン)</span>`;

    const result = andAggregate(selectedByGroup);

    const selectedListHtml = ranksSorted.length === 0
      ? `未選択(下表のチェックボックスでパターンを選ぶとここに反映されます)`
      : `選択中: ${ranksSorted.map(r => `#${r}`).join(", ")}(${ranksSorted.length}パターン)`;

    let statsHtml;
    if (result.n_races === 0) {
      statsHtml = `<div class="and-zero">この組み合わせを同時に満たす馬が${COMBO_K}頭に届かない、または${BET_TYPE}が発売されていないため、対象レースが0件でした。</div>`;
    } else {
      statsHtml = `
        <div class="and-stats">
          <div class="and-stat"><div class="label">対象R数</div><div class="value">${result.n_races}</div></div>
          <div class="and-stat"><div class="label">点数</div><div class="value">${result.n_points}</div></div>
          <div class="and-stat"><div class="label">的中数</div><div class="value">${result.hit_races}</div></div>
          <div class="and-stat"><div class="label">的中率</div><div class="value">${result.hit_rate_pct.toFixed(1)}%</div></div>
          <div class="and-stat"><div class="label">回収率</div><div class="value accent">${result.return_rate_pct.toFixed(1)}%</div></div>
        </div>
        <p class="and-note">同じグループを複数パターンが異なる値で指定している場合はOR(いずれかを満たせば可)で合成しています。
        全グループ間はAND(すべて同時に満たす必要)です。母集団はこの一覧全体と同じ、直近2開催日を除いた${AND_MAIN_RACES.length}レースです。</p>`;
    }

    body.innerHTML = `<div class="selected-list">${selectedListHtml}</div>
      <div class="merged-chips">${chips}</div>
      ${statsHtml}`;
  }

  document.getElementById("and-clear").addEventListener("click", () => {
    selectedRanks.clear();
    document.querySelectorAll(".pattern-check").forEach(cb => { cb.checked = false; });
    document.querySelectorAll("tr.is-checked").forEach(tr => tr.classList.remove("is-checked"));
    updateAndPanel();
  });

  document.getElementById("stat-row").innerHTML = `
    <div class="stat"><div class="label">券種</div><div class="value">${meta.bet_type}</div></div>
    <div class="stat"><div class="label">母集団</div><div class="value">${meta.total_races}R</div></div>
    <div class="stat"><div class="label">対象R下限(全体の${Math.round(meta.min_race_frac*100)}%)</div><div class="value">${meta.min_races}R</div></div>
    <div class="stat"><div class="label">回収率しきい値</div><div class="value">${meta.return_threshold.toFixed(0)}%超</div></div>
    <div class="stat"><div class="label">該当パターン数</div><div class="value accent">${meta.n_patterns}</div></div>
  `;

  document.getElementById("notes-block").innerHTML = `
    <p><b>母集団:</b> 通常戦${meta.population.normal} / 新馬戦${meta.population.shinba} /
    未勝利戦${meta.population.mishoubi}(計${meta.population.total}レース、
    ${meta.population.date_range[0]}〜${meta.population.date_range[1]}、${meta.population.n_dates}開催日)。
    探索はこのうち直近2開催日をホールドアウトとして除いた${meta.total_races}レースを対象に実施しています。</p>
    <p>__NOTE_TEXT__</p>
  `;

  document.getElementById("footer").textContent =
    `jra_ledger_search_2026_09_15.py / build_ledger_artifact_2026_09_15.py — 生成日時: ${meta.generated_at}`;

  const DEFAULT_SORT = [{ key: "return_rate_pct", dir: -1 }];
  let sortKeys = DEFAULT_SORT.slice();
  let filterText = "";
  let userHasSorted = false;

  const COLUMN_LABELS = {
    rank: "順位", n_races: "対象R数", n_points: "点数", hit_races: "的中数",
    hit_rate_pct: "的中率", return_rate_pct: "回収率",
  };

  function condMatchesFilter(cond, q) {
    return cond.some(c => (c.group + c.value).toLowerCase().includes(q));
  }

  function render() {
    const q = filterText.trim().toLowerCase();
    let filtered = rows.filter(r => !q || condMatchesFilter(r.conditions, q));
    filtered = filtered.slice().sort((a, b) => {
      for (const { key, dir } of sortKeys) {
        const va = a[key], vb = b[key];
        if (va < vb) return -1 * dir;
        if (va > vb) return 1 * dir;
      }
      return 0;
    });

    let tieGroupCount = 0;
    if (sortKeys.length > 1) {
      const primaryKey = sortKeys[0].key;
      let groupStart = 0;
      for (let i = 1; i <= filtered.length; i++) {
        if (i === filtered.length || filtered[i][primaryKey] !== filtered[groupStart][primaryKey]) {
          const size = i - groupStart;
          if (size > 1) {
            tieGroupCount++;
            for (let j = groupStart; j < i; j++) {
              filtered[j]._tieGroup = tieGroupCount;
              filtered[j]._tieStart = (j === groupStart);
            }
          }
          groupStart = i;
        }
      }
    }

    const tbody = document.getElementById("table-body");
    tbody.innerHTML = filtered.map(r => {
      const chips = r.conditions.map(c =>
        `<span class="cond-chip"><b>${c.group}</b>: ${c.value}</span>`).join("");
      const fragile = r.hit_rate_pct < 5;
      const tieCls = r._tieGroup ? ` tie-row${r._tieStart ? " tie-start" : ""}` : "";
      const checkedCls = selectedRanks.has(r.rank) ? " is-checked" : "";
      const rowCls = (tieCls + checkedCls).trim();
      return `<tr${rowCls ? ` class="${rowCls}"` : ""} data-rank="${r.rank}">
        <td class="rank">${r.rank}</td>
        <td class="col-cond"><div class="cond-chips">${chips}</div></td>
        <td class="num">${r.n_races}</td>
        <td class="num">${r.n_points}</td>
        <td class="num">${r.hit_races}</td>
        <td class="num hit-rate${fragile ? " is-fragile" : ""}">${r.hit_rate_pct.toFixed(1)}%${fragile ? '<span class="fragile-badge">的中率5%未満</span>' : ""}</td>
        <td class="num return-rate">${r.return_rate_pct.toFixed(1)}%</td>
        <td class="col-check"><input type="checkbox" class="pattern-check" data-rank="${r.rank}"${selectedRanks.has(r.rank) ? " checked" : ""}></td>
      </tr>`;
    }).join("");

    tbody.querySelectorAll(".pattern-check").forEach(cb => {
      cb.addEventListener("change", () => {
        const rank = Number(cb.dataset.rank);
        if (cb.checked) selectedRanks.add(rank); else selectedRanks.delete(rank);
        cb.closest("tr").classList.toggle("is-checked", cb.checked);
        updateAndPanel();
      });
    });

    document.getElementById("no-results").hidden = filtered.length > 0;
    document.getElementById("result-count").textContent = q
      ? `${filtered.length} / ${rows.length}件表示`
      : `${rows.length}件`;

    const explainEl = document.getElementById("sort-explain");
    if (sortKeys.length > 1) {
      const chain = sortKeys.map((s, i) => `${i + 1}. ${COLUMN_LABELS[s.key]}(${s.dir === 1 ? "昇順" : "降順"})`).join(" → ");
      explainEl.hidden = false;
      explainEl.innerHTML = tieGroupCount > 0
        ? `${chain}。<mark>網掛けした${tieGroupCount}グループ</mark>が「${COLUMN_LABELS[sortKeys[0].key]}が同じ行」で、そこだけ第2条件以降が順序に反映されています(それ以外の行は第1条件だけで順位が確定しています)。`
        : `${chain}。ただし今の絞り込みでは「${COLUMN_LABELS[sortKeys[0].key]}」の値がすべて異なるため、第2条件以降は見た目上どこにも反映されません。`;
    } else {
      explainEl.hidden = true;
    }

    document.querySelectorAll("thead th[data-key]").forEach(th => {
      const key = th.dataset.key;
      const idx = sortKeys.findIndex(s => s.key === key);
      const arrow = th.querySelector(".arrow");
      let orderBadge = th.querySelector(".sort-order");
      th.classList.toggle("sorted", idx !== -1);
      if (idx !== -1) {
        arrow.textContent = sortKeys[idx].dir === 1 ? "▴" : "▾";
        if (sortKeys.length > 1) {
          if (!orderBadge) {
            orderBadge = document.createElement("span");
            orderBadge.className = "sort-order";
            th.appendChild(orderBadge);
          }
          orderBadge.textContent = String(idx + 1);
        } else if (orderBadge) {
          orderBadge.remove();
        }
      } else {
        arrow.textContent = "▾";
        if (orderBadge) orderBadge.remove();
      }
    });
  }

  document.querySelectorAll("thead th[data-key]").forEach(th => {
    th.addEventListener("click", (e) => {
      const key = th.dataset.key;
      const defaultDir = (key === "rank") ? 1 : -1;

      if (!userHasSorted) {
        sortKeys = [{ key, dir: defaultDir }];
        userHasSorted = true;
        render();
        return;
      }

      const idx = sortKeys.findIndex(s => s.key === key);
      if (e.shiftKey) {
        if (idx === -1) {
          sortKeys.push({ key, dir: defaultDir });
        } else {
          sortKeys[idx].dir *= -1;
        }
      } else {
        if (idx === 0 && sortKeys.length === 1) {
          sortKeys[0].dir *= -1;
        } else {
          sortKeys = [{ key, dir: defaultDir }];
        }
      }
      render();
    });
  });

  document.getElementById("reset-sort").addEventListener("click", () => {
    sortKeys = DEFAULT_SORT.slice();
    userHasSorted = false;
    render();
  });

  document.getElementById("filter-input").addEventListener("input", (e) => {
    filterText = e.target.value;
    render();
  });

  render();
  updateAndPanel();
})();
</script>
"""


def build_race_data_blob(races: list, offered_bet_types: dict, actual: dict, bet_type: str) -> dict:
    """AND再計算用の`race-data-blob`を1券種ぶん構築する(build_factor_artifact.pyの
    圧縮ヘルパーを再利用)。races引数は探索母集団(ホールドアウト除外後)のfactor_database
    レコードそのもの。"""
    factor_groups = copy.deepcopy(FR.FACTOR_GROUPS)
    out_races = []
    for r in races:
        horses = [h for h in r["horses"] if h.get("umaban") is not None]
        if not horses:
            continue
        payout_rows = []
        m = actual.get(r["race_id"], {}).get(bet_type, {})
        for combo, payout in m.items():
            combo_seq = [combo] if isinstance(combo, int) else list(combo)
            payout_rows.append({"combo": combo_seq, "payout": int(payout)})
        out_races.append({
            "race_id": r["race_id"], "kaisai_date": r["kaisai_date"],
            "offered_bet_types": offered_bet_types.get(r["race_id"], []),
            "horses": [dict(h) for h in horses],
            "_payout_rows": payout_rows,
        })
    data = {"races": out_races, "factor_groups": factor_groups}
    data = BFA._compact_horse_keys(data)
    data = BFA._encode_horses_positionally(data)
    payouts = {}
    for r in out_races:
        payouts[r["race_id"]] = {bet_type: r.pop("_payout_rows")}
    data["payouts"] = payouts
    return data


def main():
    print("検索結果読み込み中...", flush=True)
    search = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
    fdb = json.loads(FACTOR_JSON.read_text(encoding="utf-8"))
    all_races = fdb["races"]
    holdout_dates = set(search["meta"]["holdout_dates"])
    search_races = [r for r in all_races if r["kaisai_date"] not in holdout_dates]

    print("payouts読み込み中(jra_dataset_wide)...", flush=True)
    wide = JDW.load(rebuild=False)
    actual = wide["actual"]
    offered = wide["offered_bet_types"]

    full_pop = search["meta"]["full_population"]

    for bt, meta_info in BET_META.items():
        print(f"=== {bt} ===", flush=True)
        sec = search["bet_types"][bt]
        rows = sec["rows"]
        data_blob = {
            "meta": {
                "bet_type": bt, "total_races": sec["total_races"],
                "min_race_frac": sec["min_race_frac"], "min_races": sec["min_races"],
                "return_threshold": sec["return_threshold"], "n_patterns": sec["n_patterns"],
                "population": {
                    "normal": full_pop["normal"], "shinba": full_pop["shinba"],
                    "mishoubi": full_pop["mishoubi"], "total": full_pop["total"],
                    "date_range": full_pop["date_range"], "n_dates": full_pop["n_dates"],
                },
                "generated_at": search["meta"]["generated_at"],
            },
            "rows": [{k: v for k, v in row.items() if k != "coverage_pct"} for row in rows],
        }
        race_data_blob = build_race_data_blob(search_races, offered, actual, bt)

        title = f"{bt}回収率110%台帳"
        desc = (f"JRAファクター検証データベースで対象レース数が全体の40%以上、"
                f"回収率が110%を超える{bt}パターンを回収率順に全件収録")
        css_html = (CSS_TEMPLATE
                    .replace("__TITLE__", title).replace("__DESC__", desc)
                    .replace("__H1__", title).replace("__BT__", bt)
                    .replace("__NATOMS__", str(search["meta"]["n_atoms"]))
                    .replace("__NGROUPS__", str(search["meta"]["n_groups"]))
                    .replace("__SEARCHPOP__", str(search["meta"]["search_population"]))
                    .replace("__FULLPOP__", str(full_pop["total"])))
        js_html = (JS_TEMPLATE
                   .replace("__DATA_BLOB__", json.dumps(data_blob, ensure_ascii=False).replace("</script", "<\\/script"))
                   .replace("__RACE_DATA_BLOB__", json.dumps(race_data_blob, ensure_ascii=False).replace("</script", "<\\/script"))
                   .replace("__BT__", bt).replace("__K__", str(meta_info["k"]))
                   .replace("__ORDERED__", "true" if meta_info["ordered"] else "false")
                   .replace("__NOTE_TEXT__", NOTES_BY_BT[bt]))

        html = css_html + "\n" + js_html
        out_path = SCRATCHPAD / meta_info["file"]
        out_path.write_text(html, encoding="utf-8")
        print(f"  wrote {out_path} ({len(html)/1024/1024:.2f}MB)  patterns={sec['n_patterns']}", flush=True)


if __name__ == "__main__":
    main()
