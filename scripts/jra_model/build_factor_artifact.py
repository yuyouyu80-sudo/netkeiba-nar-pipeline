# -*- coding: utf-8 -*-
"""ファクター検証データベースArtifact(インタラクティブHTML)の生成。

`data/jra_pipeline/factor_database.json`(build_factor_dataset.pyの出力)を読み込み、
テンプレートへ埋め込んでscratchpadへ公開用HTMLを書き出す(データパイプラインは永続化、
最終レンダリング結果は既存の日次レポートと同様に都度生成する設計、計画セクション5)。
"""
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
FACTOR_JSON = DATA_DIR / "factor_database.json"

# セッション固有のscratchpad(会話ごとに変わる)。存在しない場合は環境変数
# CLAUDE_SCRATCHPAD、それも無ければリポジトリ直下の一時ディレクトリへフォールバックする
# (2026-09-04: 旧セッションのパスが固定で埋まっていて別セッションから実行できなかったため)。
import os  # noqa: E402

_DEFAULT_SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
)
SCRATCHPAD = Path(os.environ.get("CLAUDE_SCRATCHPAD", str(_DEFAULT_SCRATCHPAD)))
if not SCRATCHPAD.parent.exists():
    SCRATCHPAD = PROJECT_ROOT / ".artifact_out"
SCRATCHPAD.mkdir(parents=True, exist_ok=True)
OUT_HTML = SCRATCHPAD / "factor_database.html"


TEMPLATE = r"""<title>JRAファクター検証データベース</title>
<meta name="description" content="JRAの予想ファクターを自由に組み合わせ、券種別の的中率・回収率を即座に確認する探索ツール">
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
    --pending: #6B5B95;
    --pending-ink: #FFFFFF;
    --pending-soft: #E7E1F0;
    --pending-soft-ink: #4A3D6B;
    --warn: #9A6A12;
    --warn-soft: #F6E8CC;
    --warn-soft-ink: #6B4A0C;
    --score-track: #E2E4DD;
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
      --pending: #A997D6; --pending-ink: #1F1830; --pending-soft: #332B4A; --pending-soft-ink: #D4C8EC;
      --warn: #D9A94C; --warn-soft: #3A2E12; --warn-soft-ink: #EFCE8C;
      --score-track: #2A2D28;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 20px -12px rgba(0,0,0,.5);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #16181A; --bg-elev: #1C1F21; --bg-card: #202325; --bg-sunken: #101213;
    --ink: #E7E8E3; --ink-muted: #A6ABA1; --ink-faint: #83887E;
    --rule: #2E3230; --rule-strong: #3D423E;
    --accent: #E2604C; --accent-ink: #1A1110; --accent-soft: #3A241F; --accent-soft-ink: #F0B7A9;
    --pending: #A997D6; --pending-ink: #1F1830; --pending-soft: #332B4A; --pending-soft-ink: #D4C8EC;
    --warn: #D9A94C; --warn-soft: #3A2E12; --warn-soft-ink: #EFCE8C;
    --score-track: #2A2D28;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 20px -12px rgba(0,0,0,.5);
  }

  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.6; -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); }
  .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  .page { max-width: 1280px; margin: 0 auto; padding: 28px 24px 80px; }

  .masthead { margin-bottom: 16px; }
  .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; display: flex; align-items: center; gap: 8px;
  }
  .eyebrow::before { content: ""; width: 18px; height: 1px; background: var(--accent); }
  h1.title { font-size: 26px; font-weight: 900; margin: 10px 0 6px; letter-spacing: -.01em; }
  .dek { color: var(--ink-muted); font-size: 14px; max-width: 74ch; }
  .cross-link { font-size: 12.5px; margin-top: 8px; }
  .pop-summary {
    margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px 20px;
    font-family: var(--mono); font-size: 12px; color: var(--ink-faint);
  }
  .pop-summary b { color: var(--ink-muted); font-weight: 600; }

  .warnbar {
    background: var(--warn-soft); color: var(--warn-soft-ink); border: 1px solid var(--warn);
    border-radius: 10px; padding: 12px 16px; margin: 16px 0 20px; font-size: 13px; line-height: 1.6;
    display: flex; gap: 12px; align-items: flex-start;
  }
  .warnbar .icon { font-size: 16px; flex: none; }
  .warnbar .count { font-family: var(--mono); font-weight: 700; }

  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 20px; align-items: start; }
  @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }

  .panel {
    background: var(--bg-card); border: 1px solid var(--rule); border-radius: 12px;
    box-shadow: var(--shadow); padding: 16px;
  }
  aside.panel { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }

  .bettype-row { margin-bottom: 14px; }
  .bettype-row label { display: block; font-size: 11.5px; font-weight: 700; color: var(--ink-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .03em; }
  select#bettype {
    width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--rule-strong);
    background: var(--bg-elev); color: var(--ink); font-family: var(--sans); font-size: 13.5px;
  }

  .search-row { margin-bottom: 12px; }
  .search-row input {
    width: 100%; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--rule-strong);
    background: var(--bg-elev); color: var(--ink); font-size: 12.5px;
  }
  .search-feedback { display: block; margin-top: 4px; font-size: 11px; color: var(--ink-faint); min-height: 1.4em; }

  .tier { margin-bottom: 4px; }
  .tier > summary {
    cursor: pointer; font-size: 12.5px; font-weight: 700; color: var(--ink-muted);
    padding: 8px 4px; list-style: none; display: flex; align-items: center; gap: 6px;
  }
  .tier > summary::-webkit-details-marker { display: none; }
  .tier > summary::before { content: "▸"; font-size: 10px; color: var(--ink-faint); transition: transform .15s; }
  .tier[open] > summary::before { transform: rotate(90deg); }

  fieldset.fgroup {
    border: none; border-top: 1px solid var(--rule); margin: 0; padding: 10px 2px 12px;
  }
  fieldset.fgroup legend {
    font-size: 12px; font-weight: 600; color: var(--ink); padding: 0; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  }
  .ref-badge {
    font-size: 9.5px; font-weight: 700; color: var(--pending-soft-ink); background: var(--pending-soft);
    border: 1px solid var(--pending); border-radius: 3px; padding: 0 4px; white-space: nowrap;
  }
  /* 2026-09-03、UI/UXレビュー指摘反映: 「参考・未採用」(検証済みで見送り)と「新規・未検証」
     (まだ判断材料が無い)は意思決定上の意味が違うため、色を分けて区別する。 */
  .new-badge {
    font-size: 9.5px; font-weight: 700; color: var(--warn-soft-ink); background: var(--warn-soft);
    border: 1px solid var(--warn); border-radius: 3px; padding: 0 4px; white-space: nowrap;
  }
  .fgroup .opt-list { display: flex; flex-wrap: wrap; gap: 5px; }
  .fgroup .opt {
    font-size: 12px; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--rule-strong);
    color: var(--ink-muted); cursor: pointer; user-select: none; background: var(--bg-elev);
  }
  .fgroup input { position: absolute; opacity: 0; width: 1px; height: 1px; overflow: hidden; }
  .fgroup input:checked + label .opt { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  .fgroup input:focus-visible + label .opt { outline: 2px solid var(--accent); outline-offset: 1px; }
  .extra-note { font-size: 11px; color: var(--ink-faint); margin-top: 5px; line-height: 1.5; }

  .clear-row { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--rule); display: flex; gap: 8px; }
  .btn {
    font-size: 12px; padding: 6px 12px; border-radius: 7px; border: 1px solid var(--rule-strong);
    background: var(--bg-elev); color: var(--ink-muted); cursor: pointer; font-family: var(--sans);
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }

  .results-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
  .results-head h2 { font-size: 15px; margin: 0; }
  .active-chips { display: flex; flex-wrap: wrap; gap: 5px; font-size: 11.5px; color: var(--ink-faint); }
  .active-chips .chip { background: var(--bg-sunken); border-radius: 999px; padding: 2px 8px; }
  .attempt-mini {
    font-size: 11px; font-weight: 700; color: var(--warn-soft-ink); background: var(--warn-soft);
    border: 1px solid var(--warn); border-radius: 999px; padding: 2px 9px; white-space: nowrap;
    cursor: help;
  }
  .attempt-mini:empty { display: none; }

  .table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--bg-card); }
  table.results { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.results th, table.results td { padding: 9px 11px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
  table.results th:first-child, table.results td:first-child { text-align: left; }
  table.results thead th {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-faint);
    font-weight: 600; background: var(--bg-sunken); text-align: right;
  }
  table.results thead th:first-child { text-align: left; }
  table.results tbody tr:last-child td { border-bottom: none; }
  table.results tbody tr:hover td { background: var(--bg-elev); }
  .rate { font-family: var(--mono); font-weight: 700; }
  .rate.is-plus { color: var(--accent); }
  .rate.is-mid { color: var(--ink); }
  .rate.is-minus { color: var(--ink-faint); }
  .ci { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); }
  .lowsample td { opacity: .5; }
  .lowsample .lbl { font-size: 9.5px; color: var(--warn); font-weight: 700; }
  .na-cell { color: var(--ink-faint); font-size: 11.5px; }
  .frag-note { font-size: 10.5px; color: var(--ink-faint); display: block; margin-top: 2px; }
  .frag-note.is-fragile { color: var(--warn); font-weight: 600; }
  th[title] { cursor: help; text-decoration: underline dotted var(--ink-faint); text-underline-offset: 3px; }

  .holdout-block { margin-top: 18px; padding: 14px 16px; border: 1px dashed var(--rule-strong); border-radius: 10px; }
  .holdout-block h3 { font-size: 12.5px; margin: 0 0 6px; color: var(--ink-muted); }
  .holdout-block p { font-size: 12px; color: var(--ink-faint); margin: 0 0 10px; max-width: 70ch; }
  .holdout-log { margin-top: 10px; font-size: 12.5px; }
  .holdout-log .entry { padding: 6px 0; border-top: 1px solid var(--rule); font-family: var(--mono); }

  .notes-block { margin-top: 22px; font-size: 12px; color: var(--ink-faint); max-width: 84ch; line-height: 1.7; }
  .notes-block p { margin: 0 0 8px; }

  footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 11px; color: var(--ink-faint); }
</style>

<div class="page">
  <div class="masthead">
    <div class="eyebrow">JRA予想モデル・探索ツール</div>
    <h1 class="title">ファクター検証データベース</h1>
    <p class="dek">
      券種とファクターを自由に組み合わせ、的中率・回収率をその場で確認する探索用ツールです。
      日次の予想レポートとは別物で、検証済みの結論を示すものではありません。
    </p>
    <div class="cross-link">← <a href="https://claude.ai/code/artifact/c7416ae0-e986-4dff-9819-60e767f7aff2">馬柱データ予想レポート</a>に戻る</div>
    <div class="pop-summary" id="pop-summary"></div>
  </div>

  <div class="warnbar">
    <span class="icon">⚠️</span>
    <span>
      このツールは自由にファクターを組み合わせられるため、良さそうに見える結果は<b>事後選択バイアス</b>の産物である可能性が高いです。
      95%信頼区間がブレークイーブン(回収率100%)をまたぐ場合は「差がある」と解釈しないでください。
      過去にも同種の自由探索(NAR300パターン探索・JRA BOX4/3モデル・レーダー面積8,892パターン総当たり)が撤回・不採用となっています。
      これまでに条件を <span class="count" id="attempt-count">0</span> 回変更しました
      <span id="attempt-formula"></span>。
    </span>
  </div>

  <div class="layout">
    <aside class="panel" id="filter-panel">
      <div class="bettype-row">
        <label for="bettype">券種</label>
        <select id="bettype">
          <option value="">すべて(8券種+3連複/3連単の買い方追加分)</option>
        </select>
      </div>
      <div class="search-row">
        <input type="search" id="factor-search" placeholder="ファクターを検索…(グループ名・注記・選択肢も対象)">
        <span class="search-feedback" id="search-feedback"></span>
      </div>
      <div id="tiers"></div>
      <div class="clear-row">
        <button class="btn" id="clear-btn">条件をすべて解除</button>
      </div>
    </aside>

    <section>
      <div class="results-head">
        <h2>結果</h2>
        <span class="attempt-mini" id="attempt-mini" title="条件変更のたびに増えます。多いほど「たまたま良い結果」に行き当たる確率が上がります(上部の注意書き参照)"></span>
        <div class="active-chips" id="active-chips"></div>
      </div>
      <div class="table-wrap">
        <table class="results" id="results-table">
          <thead>
            <tr>
              <th>券種</th>
              <th title="条件を満たす馬が1頭以上おり、かつその券種が発売されていたレースの数">対象R数</th>
              <th title="複勝・単勝は該当馬の頭数、組番系券種はレースごとの組み合わせ数(BOX買い)の合計">点数</th>
              <th>的中率</th>
              <th title="1点100円換算、Σ払戻/Σ賭金">回収率</th>
              <th title="ブロック(開催日×競馬場)単位の復元抽出、95%信頼区間">95%CI</th>
            </tr>
          </thead>
          <tbody id="results-body"></tbody>
        </table>
      </div>

      <div class="holdout-block">
        <h3>ホールドアウトで確認</h3>
        <p>
          左の母集団(直近<span id="holdout-n-dates">2</span>開催日を除く)で見つけた条件を、
          除外しておいた直近の開催日に対して<b>1回だけ</b>評価します。行き来して良い数字を探せないよう、
          結果は毎回ログに追記されます(上書きしません)。
          <br><span style="color:var(--warn-soft-ink);">※注意: 各馬の予想スコア順位・期待値自体は
          騎手/厩舎/種牡馬勝率等のシュリンケージpriorsを使って算出されていますが、そのpriorsは
          ホールドアウト日を含む全491レースから計算されています。そのため「モデルが本当に
          未知のデータに対して予測した」という厳密な意味でのホールドアウトではなく、
          あくまで「行き来した探索を1回に強制する」という運用上のガードレールです。</span>
        </p>
        <button class="btn primary" id="holdout-btn">現在の条件でホールドアウト評価する</button>
        <div class="holdout-log" id="holdout-log"></div>
      </div>

      <div class="notes-block" id="notes-block"></div>
    </section>
  </div>

  <footer>
    build_factor_dataset.py / jra_factor_registry.py — 生成日時: <span id="generated-at"></span>
  </footer>
</div>

<script type="application/json" id="data-blob">__FACTOR_DATA_JSON__</script>

<script>
(function () {
  "use strict";

  const DATA = JSON.parse(document.getElementById("data-blob").textContent);

  // 埋め込みJSONは公開時のアップロード量を抑えるため圧縮表現になっている(2026-09-04):
  // 馬レコードはキー名を持たない位置配列、全馬同値のフィールドはレース側の rc へ持ち上げ済み。
  // ここで元の「馬1頭 = 1オブジェクト」へ復元してから以降のロジックへ渡す(以降は無改造)。
  if (DATA._fields) {
    const F = DATA._fields;
    for (const race of DATA.races) {
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
  }
  const BET_TYPES = DATA.bet_types;
  const N_HOLDOUT_DATES = 2;

  // ------------------------------------------------------------- 母集団 / ホールドアウト分割
  const allDates = [...new Set(DATA.races.map(r => r.kaisai_date))].sort();
  const holdoutDates = new Set(allDates.slice(-N_HOLDOUT_DATES));
  const mainRaces = DATA.races.filter(r => !holdoutDates.has(r.kaisai_date));
  const holdoutRaces = DATA.races.filter(r => holdoutDates.has(r.kaisai_date));

  // ------------------------------------------------------------- 組合せ生成(jra_backtest.combos_forの移植)
  function combinations(arr, k) {
    const out = [];
    const n = arr.length;
    if (k > n || k <= 0) return out;
    const idx = Array.from({ length: k }, (_, i) => i);
    while (true) {
      out.push(idx.map(i => arr[i]));
      let i = k - 1;
      while (i >= 0 && idx[i] === n - k + i) i--;
      if (i < 0) break;
      idx[i]++;
      for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
    }
    return out;
  }
  function permutations(arr, k) {
    const out = [];
    const n = arr.length;
    function rec(chosen, remaining) {
      if (chosen.length === k) { out.push(chosen.slice()); return; }
      for (let i = 0; i < remaining.length; i++) {
        const rest = remaining.slice(0, i).concat(remaining.slice(i + 1));
        chosen.push(remaining[i]);
        rec(chosen, rest);
        chosen.pop();
      }
    }
    if (k <= n) rec([], arr);
    return out;
  }
  function combosFor(umabans, wakus) {
    const u = umabans.slice();
    const w = [...new Set(wakus.filter(x => x != null && !Number.isNaN(x)))].sort((a, b) => a - b);
    return {
      "単勝": u.map(x => [x]), "複勝": u.map(x => [x]),
      "枠連": combinations(w, 2), "馬連": combinations(u, 2), "ワイド": combinations(u, 2),
      "馬単": permutations(u, 2), "3連複": combinations(u, 3), "3連単": permutations(u, 3),
    };
  }

  // ------------------------------------------------------------- 3連複・3連単の買い方追加
  // (2026-09-05ユーザー依頼「345BOX・1頭2頭軸・フォーメーションの買い方も追加」)。
  // 通常のBOX(combosFor、対象馬すべての総当たり)はそのまま残し、対象馬を本番予想スコア
  // 順位(pred_rank)で絞り込む・軸に固定する追加の買い方を、別の券種名として選べるように
  // 増設する。軸流し/マルチの用語・組合せ定義はjra_axis_backtest.py(1頭軸流し版、
  // 既存git管理下スクリプト)の規約を2頭軸へ素直に拡張したもの。
  const PRED_RANK_KEY = (DATA.factor_groups.score_rank || {}).source ?? null;
  function rankOf(h) {
    if (PRED_RANK_KEY === null) return null;
    const v = h[PRED_RANK_KEY];
    return (v === undefined || v === null || v === "") ? null : v;
  }
  function topNByRank(picked, n) {
    const ranked = picked.filter(h => rankOf(h) !== null);
    ranked.sort((a, b) => rankOf(a) - rankOf(b));
    return ranked.slice(0, n);
  }
  const EXTRA_GROUPS = [
    { label: "3連複の買い方(BOX以外)",
      types: ["3連複_BOX3", "3連複_BOX4", "3連複_BOX5",
              "3連複_1頭軸流し", "3連複_2頭軸流し", "3連複_フォーメーション"] },
    { label: "3連単の買い方(BOX以外)",
      types: ["3連単_BOX3", "3連単_BOX4", "3連単_BOX5",
              "3連単_1頭軸流し", "3連単_1頭軸マルチ", "3連単_2頭軸流し", "3連単_2頭軸マルチ",
              "3連単_フォーメーション"] },
  ];
  const EXTRA_BET_TYPES = EXTRA_GROUPS.flatMap(g => g.types);
  const EXTRA_BASE = {};
  for (const g of EXTRA_GROUPS) {
    const base = g.types[0].startsWith("3連複") ? "3連複" : "3連単";
    for (const t of g.types) EXTRA_BASE[t] = base;
  }
  // 買い方の説明(結果テーブルの券種セルにtitleツールチップとして表示)。
  const EXTRA_DESC = {
    "BOX3": "本番予想スコア(BOX5モデル)上位3頭のBOX(3頭に満たないレースは対象外)。",
    "BOX4": "同上位4頭のBOX。", "BOX5": "同上位5頭のBOX。",
    "1頭軸流し": "スコア1位の1頭を軸に固定し、残りの対象馬(相手)との組合せを流す。",
    "1頭軸マルチ": "軸1頭がどの着順に来ても的中(軸流しの3倍の点数・賭け金)。",
    "2頭軸流し": "スコア上位2頭を軸(1・2着)に固定し、相手を3着に流す。",
    "2頭軸マルチ": "軸2頭がどちらも3着以内に来れば着順を問わず的中(2頭軸流しの3倍の点数・賭け金)。",
    "フォーメーション": "1着候補=スコア1位、2着候補=1〜3位、3着候補=1〜5位(3連複は着順不問)の3グループから重複馬を除いて組む。",
  };
  function comboKind(betType) {
    const us = betType.indexOf("_");
    return us < 0 ? null : betType.slice(us + 1);
  }
  function combosForVariant(picked, betType) {
    const kind = comboKind(betType);
    const ordered = EXTRA_BASE[betType] === "3連単";
    if (kind === "BOX3" || kind === "BOX4" || kind === "BOX5") {
      const box = topNByRank(picked, Number(kind.slice(3)));
      if (box.length < 3) return [];
      const u = box.map(h => h.umaban);
      return ordered ? permutations(u, 3) : combinations(u, 3);
    }
    if (kind === "1頭軸流し" || kind === "1頭軸マルチ") {
      const axisArr = topNByRank(picked, 1);
      if (axisArr.length < 1) return [];
      const a = axisArr[0].umaban;
      const partners = picked.filter(h => h.umaban !== a).map(h => h.umaban);
      if (partners.length < 2) return [];
      if (!ordered) return combinations(partners, 2).map(([x, y]) => [a, x, y]);
      if (kind === "1頭軸流し") return permutations(partners, 2).map(([x, y]) => [a, x, y]);
      return combinations(partners, 2).flatMap(([x, y]) => permutations([a, x, y], 3));
    }
    if (kind === "2頭軸流し" || kind === "2頭軸マルチ") {
      const axisArr = topNByRank(picked, 2);
      if (axisArr.length < 2) return [];
      const [a1, a2] = axisArr.map(h => h.umaban);
      const partners = picked.filter(h => h.umaban !== a1 && h.umaban !== a2).map(h => h.umaban);
      if (partners.length < 1) return [];
      if (!ordered) return partners.map(x => [a1, a2, x]);
      if (kind === "2頭軸流し") return partners.flatMap(x => [[a1, a2, x], [a2, a1, x]]);
      return partners.flatMap(x => permutations([a1, a2, x], 3));
    }
    if (kind === "フォーメーション") {
      const g1 = picked.filter(h => rankOf(h) !== null && rankOf(h) <= 1);
      const g2 = picked.filter(h => rankOf(h) !== null && rankOf(h) <= 3);
      const g3 = picked.filter(h => rankOf(h) !== null && rankOf(h) <= 5);
      const seen = new Set();
      const out = [];
      for (const a of g1) for (const b of g2) for (const c of g3) {
        if (a.umaban === b.umaban || b.umaban === c.umaban || a.umaban === c.umaban) continue;
        if (ordered) { out.push([a.umaban, b.umaban, c.umaban]); continue; }
        const key = [a.umaban, b.umaban, c.umaban].slice().sort((x, y) => x - y).join(",");
        if (!seen.has(key)) { seen.add(key); out.push([a.umaban, b.umaban, c.umaban]); }
      }
      return out;
    }
    return [];
  }

  // ------------------------------------------------------------- ファクター評価
  function evalOption(horse, group, opt) {
    const v = horse[group.source];
    if (v === null || v === undefined) return false;
    if (group.kind === "rank_le") return v <= opt.params.le;
    if (group.kind === "threshold_ge") return v >= opt.params.ge;
    if (group.kind === "category_in") return opt.params.in.includes(v);
    return false;
  }
  function horseQualifies(horse, selectedByGroup) {
    for (const gid in DATA.factor_groups) {
      const ids = selectedByGroup.get(gid);
      if (!ids || ids.size === 0) continue;
      const group = DATA.factor_groups[gid];
      const opts = group.options.filter(o => ids.has(o.id));
      const ok = opts.some(o => evalOption(horse, group, o));
      if (!ok) return false;
    }
    return true;
  }

  // ------------------------------------------------------------- 決済
  function arraysEqualOrdered(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function findPayout(payoutList, combo, ordered) {
    const key = ordered ? combo : [...combo].sort((a, b) => a - b);
    for (const p of payoutList) {
      const pk = ordered ? p.combo : [...p.combo].sort((a, b) => a - b);
      if (arraysEqualOrdered(key, pk)) return p.payout;
    }
    return 0;
  }
  function settleRace(race, idx, betType) {
    const baseType = EXTRA_BASE[betType] || betType;
    if (!race.offered_bet_types.includes(baseType)) return null;
    const picked = idx.map(i => race.horses[i]);
    const combos = EXTRA_BASE[betType]
      ? combosForVariant(picked, betType)
      : combosFor(picked.map(h => h.umaban), picked.map(h => h.waku))[betType];
    if (combos.length === 0) return null;
    const payoutList = (DATA.payouts[race.race_id] || {})[baseType] || [];
    const ordered = (baseType === "馬単" || baseType === "3連単");
    let ret = 0;
    for (const c of combos) ret += findPayout(payoutList, c, ordered);
    return { stake: combos.length * 100, return: ret, points: combos.length };
  }

  // ------------------------------------------------------------- ブロックブートストラップ
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function blockBootstrap(perRace, n, seed) {
    const byBlock = new Map();
    for (const r of perRace) {
      const agg = byBlock.get(r.block) || { stake: 0, return: 0 };
      agg.stake += r.stake; agg.return += r.return;
      byBlock.set(r.block, agg);
    }
    const blocks = [...byBlock.values()];
    const nb = blocks.length;
    if (nb === 0) return { mean: 0, lo: 0, hi: 0, n_blocks: 0 };
    const rng = mulberry32(seed);
    const out = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      let s = 0, r = 0;
      for (let i = 0; i < nb; i++) {
        const b = blocks[(rng() * nb) | 0];
        s += b.stake; r += b.return;
      }
      out[k] = s > 0 ? (r / s * 100) : 0;
    }
    out.sort((a, b) => a - b);
    const mean = out.reduce((a, b) => a + b, 0) / n;
    const pct = p => out[Math.min(n - 1, Math.max(0, Math.round((p / 100) * (n - 1))))];
    return { mean, lo: pct(2.5), hi: pct(97.5), n_blocks: nb };
  }

  // ------------------------------------------------------------- フラジリティ(上位1件除外)
  function fragility(perRace) {
    if (perRace.length === 0) return null;
    let best = null, bestIdx = -1;
    perRace.forEach((r, i) => { if (best === null || r.return > best) { best = r.return; bestIdx = i; } });
    if (best <= 0) return null;
    const totalStake = perRace.reduce((a, r) => a + r.stake, 0);
    const totalReturn = perRace.reduce((a, r) => a + r.return, 0);
    const dropStake = totalStake - perRace[bestIdx].stake;
    const dropReturn = totalReturn - perRace[bestIdx].return;
    const before = totalStake > 0 ? totalReturn / totalStake * 100 : 0;
    const after = dropStake > 0 ? dropReturn / dropStake * 100 : 0;
    return { before, after, delta: after - before };
  }

  // ------------------------------------------------------------- 集計
  function aggregateOne(races, selectedByGroup, betType) {
    let nRaces = 0, nPoints = 0, hitPoints = 0, totalStake = 0, totalReturn = 0;
    const perRace = [];
    for (const race of races) {
      const idx = [];
      race.horses.forEach((h, i) => { if (horseQualifies(h, selectedByGroup)) idx.push(i); });
      if (idx.length === 0) continue;
      const settled = settleRace(race, idx, betType);
      if (settled === null) continue;
      nRaces++; nPoints += settled.points;
      totalStake += settled.stake; totalReturn += settled.return;
      if (settled.return > 0) hitPoints++;
      perRace.push({ block: race.block_id, stake: settled.stake, return: settled.return });
    }
    const boot = blockBootstrap(perRace, 2000, 11);
    return {
      bet_type: betType, n_races: nRaces, n_points: nPoints,
      hit_rate_pct: nPoints ? hitPoints / nRaces * 100 : 0,
      stake: totalStake, return: totalReturn,
      return_rate_pct: totalStake ? totalReturn / totalStake * 100 : 0,
      ci_lo: boot.lo, ci_hi: boot.hi, n_blocks: boot.n_blocks,
      fragility: fragility(perRace),
    };
  }
  function aggregate(races, selectedByGroup, betTypesToShow) {
    return betTypesToShow.map(bt => aggregateOne(races, selectedByGroup, bt));
  }

  // ------------------------------------------------------------- UI状態
  const selectedByGroup = new Map();
  let attemptCount = 0;
  const MIN_RACES = 15, MIN_BLOCKS = 8;

  // URLフラグメント(#s=...)経由の共有状態(2026-09-05追加、ユーザー依頼: 探索した組合せを
  // 選択済みの状態でページを開けるようにしたい)。#s=encodeURIComponent(JSON.stringify(
  // {selected:{gid:[optId,...]}, bettype:"単勝"})) の形式。存在すればlocalStorageより優先し、
  // 以後の再訪問でも維持されるようlocalStorageへも書き戻す。
  function parseHashState() {
    const h = location.hash;
    if (!h || !h.startsWith("#s=")) return null;
    const obj = JSON.parse(decodeURIComponent(h.slice(3)));
    if (!obj || typeof obj !== "object") return null;
    return obj;
  }
  function loadState() {
    try {
      const fromHash = parseHashState();
      if (fromHash) {
        for (const [gid, ids] of Object.entries(fromHash.selected || {})) {
          if (DATA.factor_groups[gid]) selectedByGroup.set(gid, new Set(ids));
        }
        if (fromHash.bettype) document.getElementById("bettype").value = fromHash.bettype;
        saveState();
        return;
      }
    } catch (e) { /* 壊れたURL共有パラメータは無視してlocalStorageへフォールバック */ }
    try {
      const raw = localStorage.getItem("jra_factor_db_state_v1");
      if (!raw) return;
      const obj = JSON.parse(raw);
      // 2026-09-03、コードレビュー指摘反映: レジストリ側でグループが削除/リネームされた後に
      // 古いlocalStorageが残っていると、存在しないgidがselectedByGroupに残ってactiveChipsText等が
      // 例外で落ちる(白画面化)。DATA.factor_groupsに実在するgidだけを復元する。
      for (const [gid, ids] of Object.entries(obj.selected || {})) {
        if (DATA.factor_groups[gid]) selectedByGroup.set(gid, new Set(ids));
      }
      if (obj.bettype) document.getElementById("bettype").value = obj.bettype;
    } catch (e) { /* ignore corrupt state */ }
  }
  function saveState() {
    try {
      const selected = {};
      for (const [gid, ids] of selectedByGroup.entries()) selected[gid] = [...ids];
      localStorage.setItem("jra_factor_db_state_v1", JSON.stringify({
        selected, bettype: document.getElementById("bettype").value,
      }));
    } catch (e) { /* private mode等で失敗しても致命的ではない */ }
  }

  // ------------------------------------------------------------- パネル構築
  function buildPanel() {
    const betSel = document.getElementById("bettype");
    for (const bt of BET_TYPES) {
      const o = document.createElement("option");
      o.value = bt; o.textContent = bt;
      betSel.appendChild(o);
    }
    for (const g of EXTRA_GROUPS) {
      const grp = document.createElement("optgroup");
      grp.label = g.label;
      for (const bt of g.types) {
        const o = document.createElement("option");
        o.value = bt; o.textContent = comboKind(bt);
        o.title = EXTRA_DESC[comboKind(bt)] || "";
        grp.appendChild(o);
      }
      betSel.appendChild(grp);
    }

    const tiersEl = document.getElementById("tiers");
    DATA.group_tiers.forEach((tier, ti) => {
      const details = document.createElement("details");
      details.className = "tier";
      details.open = true; // 全ファクターを常時表示する(ユーザー依頼、折りたたみ初期状態を廃止)
      const summary = document.createElement("summary");
      summary.textContent = tier.label;
      details.appendChild(summary);

      for (const [gid, group] of Object.entries(DATA.factor_groups)) {
        if ((group.tier || "basic") !== tier.id) continue;
        const fs = document.createElement("fieldset");
        fs.className = "fgroup";
        fs.dataset.label = group.label;
        const legend = document.createElement("legend");
        legend.textContent = group.label;
        // バッジはtier定義側の `badge` を優先し、無ければ従来のハードコード規則にフォールバック
        // する(2026-09-04: タイアを追加するたびにこのif文を書き足さずに済むよう汎用化)。
        if (tier.id === "reference") {
          const badge = document.createElement("span");
          badge.className = "ref-badge"; badge.textContent = tier.badge || "参考・未採用";
          legend.appendChild(badge);
        } else if (tier.badge || tier.id.startsWith("candidate")) {
          const badge = document.createElement("span");
          badge.className = "new-badge"; badge.textContent = tier.badge || "新規・未検証";
          legend.appendChild(badge);
        }
        fs.appendChild(legend);

        const list = document.createElement("div");
        list.className = "opt-list";
        const inputType = group.widget === "radio" ? "radio" : "checkbox";
        const inputName = "grp-" + gid;

        if (group.widget === "radio") {
          const noneId = "opt-none-" + gid;
          list.appendChild(makeOpt(inputType, inputName, noneId, "絞り込みなし", gid, null, true));
        }
        for (const opt of group.options) {
          list.appendChild(makeOpt(inputType, inputName, opt.id, opt.label, gid, opt.id, false));
        }
        fs.appendChild(list);

        if (group.extra_note) {
          const note = document.createElement("div");
          note.className = "extra-note"; note.textContent = group.extra_note;
          fs.appendChild(note);
        }
        details.appendChild(fs);
      }
      tiersEl.appendChild(details);
    });
  }
  function makeOpt(type, name, id, label, gid, optId, isNone) {
    const input = document.createElement("input");
    input.type = type; input.name = name; input.id = id;
    input.checked = isNone;
    input.addEventListener("change", () => onOptionChange(gid, optId, type, input.checked));
    const span = document.createElement("span");
    span.className = "opt"; span.textContent = label;
    const lbl = document.createElement("label");
    lbl.htmlFor = id; lbl.appendChild(span);
    const frag = document.createDocumentFragment();
    frag.appendChild(input); frag.appendChild(lbl);
    const holder = document.createElement("span");
    holder.appendChild(frag);
    return holder;
  }
  function onOptionChange(gid, optId, type, checked) {
    let set = selectedByGroup.get(gid);
    if (!set) { set = new Set(); selectedByGroup.set(gid, set); }
    if (type === "radio") {
      set.clear();
      if (optId !== null) set.add(optId);
    } else {
      if (optId === null) { set.clear(); }
      else if (checked) set.add(optId); else set.delete(optId);
    }
    attemptCount++;
    saveState();
    render();
  }

  document.getElementById("bettype").addEventListener("change", () => { attemptCount++; saveState(); render(); });
  document.getElementById("clear-btn").addEventListener("click", () => {
    selectedByGroup.clear();
    document.querySelectorAll('.fgroup input[type="radio"][id^="opt-none-"]').forEach(el => { el.checked = true; });
    document.querySelectorAll('.fgroup input[type="checkbox"]').forEach(el => { el.checked = false; });
    document.getElementById("bettype").value = "";
    attemptCount++;
    saveState();
    render();
  });
  document.getElementById("factor-search").addEventListener("input", (e) => {
    // 2026-09-03、UI/UXレビュー指摘反映: グループ名だけでなく注記・選択肢文言も検索対象にし、
    // ヒット件数フィードバックと、該当グループが1件も無いtierの折りたたみ(非表示)を行う。
    const q = e.target.value.trim().toLowerCase();
    let hitCount = 0;
    document.querySelectorAll(".fgroup").forEach(fs => {
      const label = fs.dataset.label.toLowerCase();
      const note = (fs.querySelector(".extra-note")?.textContent || "").toLowerCase();
      const optText = [...fs.querySelectorAll(".opt")].map(o => o.textContent).join(" ").toLowerCase();
      const hit = !q || label.includes(q) || note.includes(q) || optText.includes(q);
      fs.style.display = hit ? "" : "none";
      if (hit) hitCount++;
    });
    document.querySelectorAll(".tier").forEach(tierEl => {
      const anyVisible = [...tierEl.querySelectorAll(".fgroup")].some(fs => fs.style.display !== "none");
      tierEl.style.display = (q && !anyVisible) ? "none" : "";
    });
    document.getElementById("search-feedback").textContent = q ? `${hitCount}件ヒット` : "";
  });
  document.getElementById("holdout-btn").addEventListener("click", () => {
    const betType = document.getElementById("bettype").value;
    const btShow = betType ? [betType] : BET_TYPES.concat(EXTRA_BET_TYPES);
    const results = aggregate(holdoutRaces, selectedByGroup, btShow);
    const chips = activeChipsText();
    const log = document.getElementById("holdout-log");
    const entry = document.createElement("div");
    entry.className = "entry";
    const now = new Date().toLocaleTimeString("ja-JP");
    entry.textContent = `[${now}] 条件: ${chips || "(絞り込みなし)"} → ` +
      results.map(r => `${fmtBetType(r.bet_type)} n=${r.n_races} 回収率${r.return_rate_pct.toFixed(1)}% [${r.ci_lo.toFixed(1)},${r.ci_hi.toFixed(1)}]`).join(" / ");
    log.prepend(entry);
  });

  // ------------------------------------------------------------- 描画
  function fmtRateCls(rate) {
    return rate >= 100 ? "is-plus" : (rate >= 80 ? "is-mid" : "is-minus");
  }
  function fmtBetType(bt) {
    const kind = comboKind(bt);
    return kind === null ? bt : `${EXTRA_BASE[bt]}(${kind})`;
  }
  function betTypeCellHtml(bt) {
    const kind = comboKind(bt);
    const title = kind && EXTRA_DESC[kind] ? ` title="${EXTRA_DESC[kind]}"` : "";
    return `<span${title}>${fmtBetType(bt)}</span>`;
  }
  function activeChipsText() {
    const parts = [];
    for (const [gid, ids] of selectedByGroup.entries()) {
      if (!ids || ids.size === 0) continue;
      const group = DATA.factor_groups[gid];
      if (!group) continue; // 2026-09-03: 存在しないgid(削除/リネーム後の古い状態等)を無視
      const labels = group.options.filter(o => ids.has(o.id)).map(o => o.label);
      if (labels.length) parts.push(`${group.label}: ${labels.join("/")}`);
    }
    return parts.join("、");
  }
  function render() {
    const betType = document.getElementById("bettype").value;
    const btShow = betType ? [betType] : BET_TYPES.concat(EXTRA_BET_TYPES);
    const results = aggregate(mainRaces, selectedByGroup, btShow);

    const tbody = document.getElementById("results-body");
    tbody.innerHTML = "";
    for (const r of results) {
      const tr = document.createElement("tr");
      const low = r.n_races < MIN_RACES || r.n_blocks < MIN_BLOCKS;
      if (low) tr.className = "lowsample";
      if (r.n_races === 0) {
        tr.innerHTML = `<td>${betTypeCellHtml(r.bet_type)}</td><td class="num na-cell" colspan="5">対象外(該当馬なし、または券種成立せず)</td>`;
      } else {
        const fragHtml = (r.fragility && r.fragility.before > 0)
          ? `<span class="frag-note${(r.fragility.after < 100 && r.fragility.before >= 100) ? " is-fragile" : ""}">上位1件除外で${r.fragility.after.toFixed(0)}%(${r.fragility.delta >= 0 ? "+" : ""}${r.fragility.delta.toFixed(0)}pt)</span>`
          : "";
        tr.innerHTML = `
          <td>${betTypeCellHtml(r.bet_type)}${low ? '<span class="lbl"> サンプル不足</span>' : ""}</td>
          <td class="num">${r.n_races}</td>
          <td class="num">${r.n_points}</td>
          <td class="num">${r.hit_rate_pct.toFixed(1)}%</td>
          <td class="num rate ${fmtRateCls(r.return_rate_pct)}">${r.return_rate_pct.toFixed(1)}%${fragHtml}</td>
          <td class="num ci">[${r.ci_lo.toFixed(1)}, ${r.ci_hi.toFixed(1)}]<br>n_blocks=${r.n_blocks}</td>`;
      }
      tbody.appendChild(tr);
    }

    const chipsEl = document.getElementById("active-chips");
    const text = activeChipsText();
    chipsEl.innerHTML = text ? text.split("、").map(c => `<span class="chip">${c}</span>`).join("") : '<span class="chip">条件なし(母集団全体)</span>';

    document.getElementById("attempt-count").textContent = attemptCount;
    // 2026-09-03、UI/UXレビュー指摘反映: 上部warnbarはスクロールで画面外に流れるため、
    // 結果テーブル直上にも試行回数を複製表示する。
    document.getElementById("attempt-mini").textContent =
      attemptCount > 0 ? `⚠ 条件変更 ${attemptCount}回目` : "";
    const p = 1 - Math.pow(0.95, attemptCount);
    document.getElementById("attempt-formula").textContent = attemptCount > 0
      ? `(真の効果が無くても少なくとも1つで信頼区間が0を超える確率は約 1-0.95^${attemptCount} ≈ ${(p * 100).toFixed(0)}%)`
      : "";
  }

  // ------------------------------------------------------------- 初期化
  function init() {
    const pop = DATA.population;
    document.getElementById("pop-summary").innerHTML =
      `<span><b>母集団</b> 通常戦${pop.normal.n_races} / 新馬戦${pop.shinba.n_races} / 未勝利戦${pop.mishoubi.n_races}(計${pop.total.n_races}レース)</span>` +
      `<span><b>期間</b> ${pop.date_range[0]}〜${pop.date_range[1]}(${pop.n_dates}開催日・${pop.n_blocks}ブロック)</span>` +
      `<span><b>インタラクティブ対象</b> ${mainRaces.length}レース(直近${N_HOLDOUT_DATES}開催日はホールドアウト用に除外)</span>`;
    document.getElementById("holdout-n-dates").textContent = N_HOLDOUT_DATES;
    document.getElementById("generated-at").textContent = DATA.generated_at;
    const notes = DATA.notes;
    document.getElementById("notes-block").innerHTML =
      `<p><b>データ精度:</b> ${notes.odds_accuracy}</p>` +
      `<p><b>期待値の偏り:</b> ${notes.ev_bias}</p>` +
      `<p><b>モデル信頼性:</b> ${notes.model_reliability}</p>`;

    buildPanel();
    loadState();
    // 既定選択: 何も選ばれていなければ「本番予想スコア1位のみ」を初期値にする
    // (未フィルタ状態は計算コストが大きく、かつ「市場平均を眺めるだけ」で無意味なため)。
    if (selectedByGroup.size === 0) {
      selectedByGroup.set("score_rank", new Set(["score_top1"]));
      const el = document.getElementById("score_top1");
      if (el) el.checked = true;
    } else {
      for (const [gid, ids] of selectedByGroup.entries()) {
        for (const id of ids) { const el = document.getElementById(id); if (el) el.checked = true; }
      }
    }
    render();
  }
  init();
})();
</script>
"""


# JS側が馬レコードのキー名を直接参照している(= 短縮してはいけない)フィールド。
# それ以外のキーはすべて factor_groups[*].source 経由でしか読まれないため、短いコードへ
# 機械的に付け替えてよい(2026-09-04: ファクターを33個追加した結果JSONが12.3MBまで膨らみ、
# 公開リクエストがECONNRESETで失敗したため導入した圧縮ステップ)。
_KEEP_HORSE_KEYS = {"umaban", "waku"}


def _compact_horse_keys(data: dict) -> dict:
    """馬レコードの長いキー名を "a"/"b"/… の短縮コードへ付け替え、factor_groups[*].source も
    同じ辞書で書き換える(参照の対応関係は必ず両方同時に更新される)。値・構造は不変。"""
    horses = [h for r in data["races"] for h in r["horses"]]
    if not horses:
        return data
    keys = [k for k in horses[0].keys() if k not in _KEEP_HORSE_KEYS]

    def code(i):
        s, i = "", i + 1
        while i:
            i, m = divmod(i - 1, 52)
            s = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[m]) + s
        return s

    mapping, used = {}, set(_KEEP_HORSE_KEYS)
    for i, k in enumerate(keys):
        c = code(i)
        while c in used:
            c = code(len(mapping) + len(used))
        mapping[k], _ = c, used.add(c)

    for r in data["races"]:
        r["horses"] = [{mapping.get(k, k): v for k, v in h.items()} for h in r["horses"]]
    for g in data["factor_groups"].values():
        g["source"] = mapping.get(g["source"], g["source"])
    data["_key_map"] = {v: k for k, v in mapping.items()}  # デバッグ用の逆引き(小さい)
    return _compact_category_values(data, mapping)


def _compact_category_values(data: dict, key_map: dict) -> dict:
    """category_in グループのソース列に入る**文字列**値を短縮コードへ置換する。

    JS側の判定は `opt.params.in.includes(v)` の等値比較だけなので、馬レコードの値と
    options の params.in を同じ辞書で同時に書き換えれば意味は完全に保たれる
    ("insufficient_history" のような長いセンチネル文字列が6,443行ぶん繰り返されるのを潰す)。
    int/bool の選択肢(枠番・年齢・開催後半フラグ等)はそのまま残す。
    """
    cat_sources = {g["source"] for g in data["factor_groups"].values()
                   if g["kind"] == "category_in"}
    if not cat_sources:
        return data
    horses = [h for r in data["races"] for h in r["horses"]]

    vocab = {src: {} for src in cat_sources}
    for h in horses:
        for src in cat_sources:
            v = h.get(src)
            if isinstance(v, str) and v not in vocab[src]:
                vocab[src][v] = None
    # options 側にしか出てこない値(実データ0件の選択肢)も辞書に含める
    for g in data["factor_groups"].values():
        if g["kind"] != "category_in":
            continue
        for opt in g["options"]:
            for v in opt["params"]["in"]:
                if isinstance(v, str) and v not in vocab[g["source"]]:
                    vocab[g["source"]][v] = None
    for src, vs in vocab.items():
        for i, v in enumerate(list(vs)):
            vs[v] = str(i)

    for r in data["races"]:
        for h in r["horses"]:
            for src in cat_sources:
                v = h.get(src)
                if isinstance(v, str):
                    h[src] = vocab[src][v]
    for g in data["factor_groups"].values():
        if g["kind"] != "category_in":
            continue
        vs = vocab[g["source"]]
        for opt in g["options"]:
            opt["params"]["in"] = [vs.get(v, v) if isinstance(v, str) else v
                                   for v in opt["params"]["in"]]
    # 逆引き(短縮キー -> {コード: 元の値})。key_mapは呼び出し側で既にsourceへ適用済みなので
    # ここでのsourceは短縮キーそのもの。
    data["_value_map"] = {s: {c: v for v, c in vs.items()} for s, vs in vocab.items()}
    return data


def _encode_horses_positionally(data: dict) -> dict:
    """馬レコードを {キー: 値} の辞書から**位置配列**へ変換し、キー名の繰り返しを消す。

    さらに「どのレースでも全馬同値」になるフィールド(サーフェス・コース形態・馬場・想定ペース・
    レース種別など)は自動検出してレース側の1個の辞書へ持ち上げる。復元はテンプレートJSの
    ブートストラップ(JSON.parse直後)が行い、以降のロジックは従来どおり馬オブジェクトを見る。
    欠損は `""`(空文字)で表す — 実データの値は数値か短縮コード(空文字にはならない)なので
    衝突しない。
    """
    all_keys, seen = [], set()
    for r in data["races"]:
        for h in r["horses"]:
            for k in h:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    race_const = []
    for k in all_keys:
        if all(len({h.get(k) for h in r["horses"]}) <= 1 for r in data["races"]):
            race_const.append(k)
    race_const_set = set(race_const)
    fields = [k for k in all_keys if k not in race_const_set]

    for r in data["races"]:
        first = r["horses"][0] if r["horses"] else {}
        r["rc"] = {k: first[k] for k in race_const if first.get(k) is not None}
        r["horses"] = [[h.get(k, "") if h.get(k) is not None else "" for k in fields]
                       for h in r["horses"]]
    data["_fields"] = fields
    return data


def main():
    data = json.loads(FACTOR_JSON.read_text(encoding="utf-8"))
    before = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    data = _compact_horse_keys(data)
    # null のフィールドはキーごと落とす(JS側 evalOption は undefined も null と同様に
    # 「非該当」として扱うため意味は変わらない)。horse_name はテンプレートJSが一切参照して
    # いない表示用の残骸なので同時に除去する。いずれも公開時のアップロード量削減が目的。
    _drop_key = data["_key_map"] and next(
        (k for k, v in data["_key_map"].items() if v == "horse_name"), None)
    for r in data["races"]:
        r["horses"] = [{k: v for k, v in h.items() if v is not None and k != _drop_key}
                       for h in r["horses"]]
    data = _encode_horses_positionally(data)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    print(f"キー短縮: {before:,} -> {len(payload):,} bytes")
    # </script>混入対策(JSON埋め込みの定石)
    payload_escaped = payload.replace("</script", "<\\/script")
    html = TEMPLATE.replace("__FACTOR_DATA_JSON__", payload_escaped)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({len(html):,} chars)")


if __name__ == "__main__":
    main()
