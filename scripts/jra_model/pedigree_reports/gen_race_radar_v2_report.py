# -*- coding: utf-8 -*-
"""血統レーダー v2 のレポート(R5 → R6 で複数日・全競馬場に一般化、2026-10-07)。

入力: radar_v2/live/race_radar_v2_{date}.json(radar_v2_live.py、出走表から作った値)を日付ごとに並べ、検証の要約・DLが学んだ要求・
前向き検証の途中集計(radar_v2/prospective_eval.json があれば)を足す。
使い方: python gen_race_radar_v2_report.py --days 20261003 20261004 [--out path.html]
出力(既定): data/jra_pipeline/pedigree_reports/race_radar_v2_report.html(Artifact として公開する本体)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402

BASE = Path(__file__).resolve().parents[3] / "data" / "jra_pipeline" / "pedigree_reports"
LIVE = C.OUT_DIR / "live"
OUT = BASE / "race_radar_v2_report.html"

TEMPLATE = r"""<title>血統レーダーチャート</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f4f6f3; --surface: #ffffff; --fg: #18231d; --muted: #5d6b63; --line: #d6ddd7;
  --accent: #1f6f50; --accent-soft: #e2f1e9; --pos: #2f7d5a; --neg: #b4574a; --zero: #9aa59f;
  --warn: #8f5a00; --warn-soft: #f8edd8; --ref: #7a6a9a; --ref-soft: #eeeaf5;
  --font-body: "IBM Plex Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1713; --surface: #16211b; --fg: #e3ece6; --muted: #95a59b; --line: #2a3a31;
    --accent: #6cc79c; --accent-soft: #173a2a; --pos: #74c79f; --neg: #e3907f; --zero: #5d6b63;
    --warn: #e7b66a; --warn-soft: #3a2c14; --ref: #b6a6db; --ref-soft: #2a2438; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1713; --surface: #16211b; --fg: #e3ece6; --muted: #95a59b; --line: #2a3a31;
  --accent: #6cc79c; --accent-soft: #173a2a; --pos: #74c79f; --neg: #e3907f; --zero: #5d6b63;
  --warn: #e7b66a; --warn-soft: #3a2c14; --ref: #b6a6db; --ref-soft: #2a2438; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); line-height: 1.7; padding-inline: 16px; padding-block: 24px 56px; }
.wrap { max-width: 1080px; margin-inline: auto; display: flex; flex-direction: column; gap: 20px; }
h1 { font-size: clamp(22px, 4vw, 30px); margin: 0; line-height: 1.3; text-wrap: balance; }
h2 { font-size: 18px; margin: 0 0 10px; }
h3 { font-size: 15px; margin: 0 0 6px; }
p { margin: 0 0 8px; }
.kicker { font-family: var(--font-mono); font-size: 12px; letter-spacing: .08em; color: var(--accent); }
.meta { color: var(--muted); font-size: 13px; }
section { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px; min-width: 0; }
.verdict { background: var(--accent-soft); border-color: var(--accent); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.grid > div { min-width: 0; }
.mono, td.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; white-space: nowrap; }
th { font-size: 12px; color: var(--muted); font-weight: 500; }
th .st { display: block; font-size: 10.5px; font-weight: 700; letter-spacing: .03em; }
.st-予測 { color: var(--accent); } .st-記述 { color: var(--warn); } .st-参考 { color: var(--ref); } .st-基準 { color: var(--muted); }
td.faint, th.faint { opacity: .55; }
.sub { display: block; font-size: 11.5px; color: var(--muted); font-weight: 400; }
tr.hrow { cursor: pointer; } tr.hrow:hover { background: var(--accent-soft); } tr.hrow.sel { background: var(--accent-soft); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 5px 12px; background: var(--surface); color: var(--fg); border: 0; border-right: 1px solid var(--line); cursor: pointer; }
.seg button:last-child { border-right: 0; }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--surface); }
.label { font-size: 12px; color: var(--muted); margin-right: 6px; }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tabs button { font: inherit; font-size: 13px; padding: 4px 10px; border: 1px solid var(--line); border-radius: 999px; background: var(--surface); color: var(--fg); cursor: pointer; }
.tabs button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); font-weight: 700; }
.tabs button:disabled { opacity: .45; cursor: default; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; font-size: 13px; min-width: 0; }
.chip b { font-family: var(--font-mono); }
.bar { display: inline-flex; align-items: center; gap: 6px; }
.bar .track { position: relative; width: 74px; height: 9px; background: var(--bg); border-radius: 2px; }
.bar .track::after { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--zero); }
.bar .fill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
.bar .v { font-family: var(--font-mono); font-size: 12px; min-width: 3.4em; text-align: right; color: var(--muted); }
.fitbar .track { width: 110px; }
.na { color: var(--muted); font-size: 12px; }
.fin { font-family: var(--font-mono); font-weight: 700; }
.fin.top3 { color: var(--accent); }
.rhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 20px; margin-bottom: 10px; }
.rhead h2 { margin: 0; }
.res { font-size: 13px; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.res-top { display: flex; flex-wrap: wrap; gap: 2px 12px; }
.res-h b, .res-pay b { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.res-h b { color: var(--accent); }
.res-pay .label { margin-right: 4px; }
.radar-wrap { display: grid; grid-template-columns: minmax(0, 340px) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 720px) { .radar-wrap { grid-template-columns: 1fr; } }
svg text { fill: var(--muted); font-size: 11px; font-family: var(--font-body); }
.note { border-left: 3px solid var(--warn); background: var(--warn-soft); padding: 8px 12px; border-radius: 0 4px 4px 0; font-size: 13px; }
ul { margin: 0 0 8px; padding-left: 1.2em; } li { margin-bottom: 3px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); }
.sw { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }
footer { color: var(--muted); font-size: 12px; }
</style>

<div class="wrap">
<header>
  <div class="kicker">JRA / 血統レーダー v2 / <span id="hdr-date"></span></div>
  <h1>血統レーダー v2 — JRA全場</h1>
  <p class="meta">15年分(2011〜)で測り方を作り直した版。血統の値はレース前年末までのデータ・本人の走を除き、レースの要求・実績はその日より前の走だけで作っています(出走表から作るので、発走前にも同じ値が出ます)。2026年10月7日に旧版(v1、中山9/26)をこの版へ差し替えました。</p>
</header>

<section class="verdict">
  <h2>このレーダーで分かること・分からないこと</h2>
  <div class="grid">
    <div>
      <h3>確かめられたこと(2021〜2026年、<span id="v-n"></span>レース)</h3>
      <ul>
        <li>父の総合力だけのモデルに「適合度」を足すと、着順の予測が良くなる(1レースあたり <b class="mono" id="v-main"></b>)。</li>
        <li>効いているのはほぼ <b>芝ダ適性</b> と <b>新馬適性</b>。新馬戦・未勝利戦で特に効く。</li>
        <li>1位予想の的中率は <span class="mono" id="v-top1"></span>(父の総合力のみ → +適合度)。</li>
        <li id="v-dl-li"><b>DL版の適合度</b>(レース条件から「求められる血統」を学ぶ小さなネット)は、v2の基準よりさらに予測を良くした(<b class="mono" id="v-dl"></b>、6年すべてで正)。</li>
      </ul>
    </div>
    <div>
      <h3>確かめられなかったこと</h3>
      <ul>
        <li><b>「レースが求める脚質・キレに血統が合う馬が来る」ことは未確認</b>。脚質・キレの列は、血統の傾向の記述として見てください。</li>
        <li><b>オッズに対する上積みは、v2の適合度では無し</b>(<span class="mono" id="v-mkt"></span>)。DL版はごく小さな上積み(<span class="mono" id="v-dlmkt"></span>、結果を見た後の探索で、確認ではない)。</li>
        <li>当日のトラックバイアス・馬場の変化は入っていません。</li>
      </ul>
    </div>
  </div>
</section>

<section>
  <div class="controls">
    <div><span class="label">開催日</span><span class="tabs" id="tabs-day"></span></div>
  </div>
  <div class="controls" style="margin-top:8px">
    <div><span class="label">競馬場</span><span class="tabs" id="tabs-venue"></span></div>
  </div>
  <div class="controls" style="margin-top:8px">
    <div><span class="label">レース</span><span class="tabs" id="tabs"></span></div>
  </div>
  <div class="controls" style="margin-top:10px">
    <div><span class="label">馬の値</span><span class="seg" id="seg-mode"><button data-v="ped" aria-pressed="true">血統のみ</button><button data-v="own">血統+実績</button></span></div>
    <div><span class="label">想定ペース</span><span class="seg" id="seg-pace"><button data-v="全体" aria-pressed="true">指定なし</button><button data-v="速い">速い</button><button data-v="平均">平均</button><button data-v="遅い">遅い</button></span></div>
    <div><span class="label">馬場(DL)</span><span class="seg" id="seg-going"><button data-v="auto" aria-pressed="true">自動</button><button data-v="良">良</button><button data-v="稍重">稍重</button><button data-v="重">重</button><button data-v="不良">不良</button></span></div>
    <div><span class="label">並び</span><span class="seg" id="seg-sort"><button data-v="fit" aria-pressed="true">適合度</button><button data-v="dl">DL適合度</button><button data-v="umaban">馬番</button></span></div>
    <div id="fin-ctl"><span class="label">着順</span><span class="seg" id="seg-fin"><button data-v="hide">隠す</button><button data-v="show" aria-pressed="true">表示</button></span></div>
  </div>
</section>

<section id="race"></section>

<section id="pros"></section>

<section id="learned"></section>

<section>
  <h2>列の読み方</h2>
  <div class="grid">
    <div>
      <ul>
        <li>値は「父(芝ダ・距離帯は父・母父・父父の平均)の産駒が、その条件で普段よりどれだけ走るか」を、真の差のばらつき(τ)を1とした単位で表したもの。0が平均、±1で血統間の典型的な差。</li>
        <li><b>適合度</b> = 各軸の値 × レースの要求 × 軸の信頼度 の合計(R4で検証した式)。想定ペースを変えると脚質・キレの要求が変わります。</li>
        <li><b>血統+実績</b>: その日より前の本人の走で血統の値を更新(走数が多いほど本人の値に寄る)。芝ダ・距離帯は「本人の芝ダ・距離帯別の成績差」です。</li>
        <li><b>DL適合度</b>: レース条件(競馬場・芝ダ・距離・内外回り・馬場・クラス・頭数・過去のペース傾向・出走馬の過去の位置取り)から、9本の血統の値それぞれに掛ける重みを学習したもの。血統のみで計算します(切替の影響を受けません)。馬場は発走前には分からないため4通り計算してあり、「自動」は結果のある日は実際の馬場、まだの日は良です。</li>
        <li>「血統表未取得」の馬は、馬柱の父・母父の名前から血統IDを引いています(父父は父から)。</li>
      </ul>
    </div>
    <div>
      <ul>
        <li><span class="st-予測"><b>予測</b></span>: 予測上の価値を確認(芝ダ)。</li>
        <li><span class="st-記述"><b>記述</b></span>: 偏りと信頼性は旧版より大きく改善したが、予測上の価値は未確認(脚質・キレ)。</li>
        <li><span class="st-参考"><b>参考</b></span>: 距離帯(はっきりしない、長距離帯は信頼度低)・新馬(効くが偏りが一部残る)。薄く表示。</li>
        <li>レースの要求は、同じコース・クラス・開催日目の過去レース(足りなければ段階的に広げる)の上位3頭と出走馬平均の差から作っています。想定ペース別の要求は、そのペースだった過去レースだけで作ります(2026-10-07に、実際のペース以外の区分がほぼ0になる表示の不具合を直しました。評価に使った「指定なし」は不変)。</li>
      </ul>
    </div>
  </div>
</section>

<footer id="foot"></footer>
</div>

<script>
const D = __DATA__;
const AX = ["S", "front", "kire", "surface", "dbucket", "debut"];
const state = { day: null, venue: null, race: null, mode: "ped", pace: "全体", going: "auto", sort: "fit", fin: "show", sel: null };
const $ = (s, el = document) => el.querySelector(s);
const fmt = (x, d = 2) => (x === null || x === undefined || !isFinite(x)) ? "—" : (x > 0 ? "+" : "") + x.toFixed(d);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const dayObj = () => D.days.find(d => d.date === state.day);
const raceObj = () => dayObj().races.find(r => r.race_id === state.race);
const GOINGS = ["良", "稍重", "重", "不良"];

function goingOf(race) { return state.going !== "auto" ? state.going : (GOINGS.includes(race.actual_going) ? race.actual_going : "良"); }
function v3of(h, race) { return h.v3 ? h.v3[goingOf(race)] : null; }
function relOf(axis, race) {
  if (axis === "surface") return D.rel.surface[race.surface] || 0;
  if (axis === "dbucket") return D.rel.dbucket[race.dbucket] || 0;
  return D.rel[axis] || 0;
}
function vals(h) { return state.mode === "own" && h.own ? h.own : h.ped; }
function q(race, a) { const v = race.q[a][state.pace]; return v === null ? race.q[a]["全体"] : v; }
function fit(race, h) {
  const v = vals(h), z = x => (x === null || x === undefined) ? 0 : x;
  return relOf("front", race) * z(q(race, "front")) * z(v.front) + relOf("kire", race) * z(q(race, "kire")) * z(v.kire)
       + relOf("surface", race) * z(v.surface) + relOf("dbucket", race) * z(v.dbucket)
       + (race.is_debut ? relOf("debut", race) * z(v.debut) : 0);
}
function bar(x, lim = 2.5, cls = "") {
  if (x === null || x === undefined || !isFinite(x)) return '<span class="na">—</span>';
  const c = Math.max(-lim, Math.min(lim, x)) / lim * 50;
  const left = c >= 0 ? 50 : 50 + c, w = Math.abs(c);
  const col = x >= 0 ? "var(--pos)" : "var(--neg)";
  return `<span class="bar ${cls}"><span class="track"><span class="fill" style="left:${left}%;width:${w}%;background:${col}"></span></span><span class="v">${fmt(x)}</span></span>`;
}
function statusOf(a) { return D.axes[a].status; }
function faint(a, race) {
  if (a === "debut" && !race.is_debut) return true;
  if (a === "dbucket" && race.dbucket && race.dbucket.startsWith("長距離")) return true;
  return statusOf(a) === "参考";
}
function wd(ds) { return "日月火水木金土"[new Date(ds + "T00:00:00").getDay()]; }

function tabBtn(text, pressed, onclick, title, disabled) {
  const b = document.createElement("button");
  b.textContent = text; if (title) b.title = title;
  b.setAttribute("aria-pressed", String(pressed)); b.disabled = !!disabled; b.onclick = onclick;
  return b;
}
function pickVenue() {
  const vs = [...new Set(dayObj().races.map(r => r.venue))];
  if (!vs.includes(state.venue)) state.venue = vs[0];
  return vs;
}
function pickRace() {
  const rs = dayObj().races.filter(r => r.venue === state.venue);
  if (!rs.find(r => r.race_id === state.race && !r.skipped)) state.race = (rs.find(r => !r.skipped) || {}).race_id;
  return rs;
}
function renderTabs() {
  const td = $("#tabs-day"); td.innerHTML = "";
  D.days.forEach(d => td.appendChild(tabBtn(d.date.slice(5).replace("-", "/") + "(" + wd(d.date) + ")", d.date === state.day,
    () => { state.day = d.date; state.sel = null; pickVenue(); pickRace(); render(); }, d.has_results ? "結果あり" : "発走前")));
  const tv = $("#tabs-venue"); tv.innerHTML = "";
  pickVenue().forEach(v => tv.appendChild(tabBtn(v, v === state.venue, () => { state.venue = v; state.sel = null; pickRace(); render(); })));
  const t = $("#tabs"); t.innerHTML = "";
  pickRace().forEach(r => t.appendChild(tabBtn(r.race_number + "R", r.race_id === state.race,
    () => { state.race = r.race_id; state.sel = null; render(); }, r.race_name + (r.skipped ? "(" + r.skipped + ")" : ""), r.skipped)));
  $("#fin-ctl").hidden = !dayObj().has_results;
}

function reqChip(label, v, extra) {
  const word = v === null ? "—" : v > 0.5 ? "強く求める" : v > 0.15 ? "やや求める" : v < -0.5 ? "不利(逆)" : v < -0.15 ? "やや不利" : "ほぼ中立";
  return `<div class="chip">${label}: <b>${fmt(v)}</b> <span class="meta">${word}${extra ? " / " + extra : ""}</span></div>`;
}
const CLS = { 0: "未勝利", 1: "1勝クラス", 2: "2勝クラス", 3: "3勝クラス", 4: "オープン", 5: "G3", 6: "G2", 7: "G1" };

function renderRace() {
  const race = raceObj();
  const el = $("#race");
  if (!race) { el.innerHTML = "<p class='meta'>表示できるレースがありません。</p>"; return; }
  const showFin = state.fin === "show" && dayObj().has_results;
  const hs = race.horses.map(h => ({ ...h, F: fit(race, h), V3: v3of(h, race) }));
  if (state.sort === "fit") hs.sort((a, b) => b.F - a.F);
  else if (state.sort === "dl") hs.sort((a, b) => (b.V3 ?? -99) - (a.V3 ?? -99));
  else hs.sort((a, b) => a.umaban - b.umaban);
  const dmax = Math.max(0.3, ...hs.map(h => Math.abs(h.V3 ?? 0)));
  const fmax = Math.max(0.3, ...hs.map(h => Math.abs(h.F)));
  if (!state.sel && hs.length) state.sel = hs[0].umaban;
  const cols = [["S", "父の総合力"], ["front", "脚質(前)"], ["kire", "キレ"], ["surface", race.surface === "芝" ? "芝適性" : "ダ適性"],
                ["dbucket", "距離帯(" + (race.dbucket || "").replace(/\(.*\)/, "") + ")"], ["debut", "新馬"]];
  const rule = a => `${race.req_rule[a] || "—"}・${race.req_n[a] ?? "—"}件`;
  const g = goingOf(race);
  const cls = race.is_debut ? "新馬" : (CLS[race.class_ord] ?? "クラス不明");
  let html = `<div class="rhead"><h2>${esc(race.venue)} ${race.race_number}R ${esc(race.race_name)} <span class="meta">${race.surface}${race.distance}m・${cls}${race.start_time ? "・" + race.start_time + "発走" : ""}</span></h2>${showFin ? resultHTML(race) : ""}</div>
  <div class="chips" style="margin-bottom:10px">
    ${reqChip("脚質(前に行く)の要求", q(race, "front"), rule("front"))}
    ${reqChip("キレの要求", q(race, "kire"), rule("kire"))}
    <div class="chip">芝ダ: <b>${race.surface}</b> <span class="meta">その芝ダの適性を加点</span></div>
    <div class="chip">距離帯: <b>${race.dbucket || "—"}</b></div>
    <div class="chip">DLの馬場: <b>${g}</b> <span class="meta">${state.going === "auto" ? (GOINGS.includes(race.actual_going) ? "実際の馬場" : "発走前のため良と仮定") : "手動で選択"}</span></div>
  </div>
  ${race.v3_req ? v3ReqHTML(race) : ""}
  <p class="meta">要求の単位: 2018〜2020年のレースの基準のばらつきを1とする。想定ペース「${state.pace === "全体" ? "指定なし" : state.pace}」。適合度は脚質・キレ・芝ダ・距離帯・新馬の合計で、父の総合力(強さ)は含みません。</p>
  <div class="scroll"><table><thead><tr><th>馬番</th><th>馬名<span class="sub">父 / 母父</span></th><th>適合度</th><th>DL適合度<span class="st st-予測">予測</span></th>`;
  cols.forEach(([a, l]) => { html += `<th class="${faint(a, race) ? "faint" : ""}">${l}<span class="st st-${statusOf(a)}">${statusOf(a)}</span></th>`; });
  html += `${showFin ? "<th>着順</th>" : ""}</tr></thead><tbody>`;
  hs.forEach(h => {
    const v = vals(h), key = h.umaban;
    const src = h.ped_src === "名前から" ? " ・血統表未取得(名前から)" : h.ped_src === "不明" ? " ・血統不明" : "";
    html += `<tr class="hrow ${state.sel === key ? "sel" : ""}" data-k="${key}"><td class="num">${h.umaban}</td>
      <td>${esc(h.horse_name)}<span class="sub">${esc(h.sire || "—")} / ${esc(h.bms || "—")}${src}${state.mode === "own" ? " ・前走まで" + (h.own ? h.own.n_prev : 0) + "走" : ""}</span></td>
      <td>${bar(h.F, fmax, "fitbar")}</td><td>${bar(h.V3, dmax, "fitbar")}</td>`;
    cols.forEach(([a]) => { html += `<td class="${faint(a, race) ? "faint" : ""}">${bar(v[a])}</td>`; });
    if (showFin) html += `<td class="fin ${h.finish && h.finish <= 3 ? "top3" : ""}">${h.finish ?? esc(h.finish_raw || "—")}</td>`;
    html += "</tr>";
  });
  html += `</tbody></table></div>`;
  const sh = hs.find(h => h.umaban === state.sel) || hs[0];
  if (sh) html += `<div class="radar-wrap" style="margin-top:14px"><div>${radarSVG(race, sh)}</div><div id="detail">${detail(race, sh)}</div></div>`;
  el.innerHTML = html;
  el.querySelectorAll("tr.hrow").forEach(tr => tr.onclick = () => { state.sel = Number(tr.dataset.k); renderRace(); });
}

const MARU = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱";
const yen = v => "¥" + Number(v).toLocaleString();
function resultHTML(race) {
  if (!race.top5 || !race.top5.length) return `<div class="res"><span class="meta">結果はまだありません</span></div>`;
  let s = `<div class="res"><div class="res-top">`;
  race.top5.forEach(([p, u, n, pop]) => { s += `<span class="res-h"><b>${p}着</b> ${MARU[u - 1] || u} ${esc(n)}${pop ? `<span class="meta">(${pop}人気)</span>` : ""}</span>`; });
  s += `</div>`;
  const P = race.pay || {};
  const fmtPay = k => (P[k] || []).map(([c, v, pop]) => `${MARU[Number(c) - 1] || c} <b>${yen(v)}</b>${pop ? `<span class="meta">(${pop}人気)</span>` : ""}`).join("、");
  if (P["単勝"] || P["複勝"]) s += `<div class="res-pay"><span class="label">単勝</span>${fmtPay("単勝") || "—"}　<span class="label">複勝</span>${fmtPay("複勝") || "—"}</div>`;
  return s + `</div>`;
}

const ZL = {"z_front": "脚質(前)", "z_kire": "キレ", "z_s|芝": "芝", "z_s|ダ": "ダ", "z_b|短距離(~1400m)": "短距離", "z_b|マイル(1401-1800m)": "マイル",
            "z_b|中距離(1801-2200m)": "中距離", "z_b|長距離(2201m~)": "長距離", "z_debut": "新馬"};
function v3ReqHTML(race) {
  const r = race.v3_req[goingOf(race)]; if (!r) return "";
  const mx = Math.max(0.2, ...Object.values(r).map(Math.abs));
  let s = `<div style="margin-bottom:8px"><span class="label">DLが学んだこのレースの要求(各血統の値への重み、馬場: ${goingOf(race)})</span><div class="chips">`;
  Object.keys(ZL).forEach(k => { s += `<div class="chip">${ZL[k]} ${bar(r[k], mx)}</div>`; });
  return s + `</div></div>`;
}
function renderLearned() {
  const L = D.v3_learned; if (!L) { $("#learned").hidden = true; return; }
  const keys = Object.keys(ZL);
  const mx = Math.max(0.2, ...Object.values(L).flatMap(g => keys.map(k => Math.abs(g[k]))));
  let s = `<h2>DLが学んだ要求(区分ごとの平均、2026年のレース)</h2><p class="meta">各血統の値に掛かる重みの平均。結果を見た後の記述(探索)で、確認検証ではありません。</p><div class="scroll"><table><tr><th>区分</th><th>レース数</th>`;
  keys.forEach(k => s += `<th>${ZL[k]}</th>`);
  s += `</tr>`;
  Object.entries(L).sort().forEach(([g, v]) => { s += `<tr><td>${g}</td><td class="num">${v.n_races}</td>`; keys.forEach(k => s += `<td>${bar(v[k], mx)}</td>`); s += `</tr>`; });
  $("#learned").innerHTML = s + `</table></div>`;
}
function renderPros() {
  const P = D.prospective;
  let s = `<h2>前向きの記録(2026年10月7日以降の開催)</h2>`;
  s += `<p>発走前に出走表から作った値をレースごとに保存し(最初の記録が正本)、結果が出てから答え合わせします(事前登録 追補6)。係数は確認評価で決めた値に固定します。</p>
  <ul><li><b>P1</b>: DL版(v3)対 v2の基準+軸ごとの係数(C2)。<b>2027年3月28日までの開催分で1回だけ判定</b>(見込み約1,600レース)。</li>
  <li><b>P2</b>: v2の主仮説(父の総合力+適合度 対 父の総合力のみ)。効果が小さいため <b>2027年12月28日までの開催分で1回だけ判定</b>(見込み約4,300レース。効果が小さいため、この件数でも検出力は6割ほど)。</li>
  <li>判定は各 片側p &lt; 0.0125(2つの比較の分を補正)。途中の数字は記述のみで、途中で判定はしません。</li></ul>`;
  if (!P || !P.coverage || !P.coverage.days) s += `<p class="note">まだ記録はありません。最初の記録は10月10日(土)の開催からです。</p>`;
  else {
    const c = P.coverage;
    s += `<p class="meta">記録 ${c.days}日・${c.races_recorded}レース(発走前 ${c.races_before_start}、結果あり ${c.races_with_results})</p>`;
    ["P1", "P2"].forEach(k => { const x = P[k]; if (x && x.n_races) s += `<div class="chip" style="margin:4px 0">${k} ${esc(x.name)}: <b>${fmt(x.dll_per_race, 4)}</b> [${x.ci95_block ? x.ci95_block.map(v => v.toFixed(4)).join(", ") : "—"}] / ${x.n_races}レース ${x.final ? (x.pass ? "(判定: 合格)" : "(判定: 不合格)") : "(途中、記述のみ)"}</div>`; });
  }
  $("#pros").innerHTML = s;
}
function radarSVG(race, h) {
  const labels = ["父の総合力", "脚質(前)", "キレ", race.surface === "芝" ? "芝適性" : "ダ適性", "距離帯", "新馬"];
  const W = 340, cx = 170, cy = 165, R = 115, lim = 2.5, n = AX.length;
  const pt = (i, v) => { const r = (Math.max(-lim, Math.min(lim, v ?? 0)) + lim) / (2 * lim) * R; const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; };
  let s = `<svg viewBox="0 0 ${W} 330" width="100%" role="img" aria-label="${esc(h.horse_name)}のレーダー">`;
  [-2, -1, 0, 1, 2].forEach(g => {
    const p = AX.map((_, i) => pt(i, g).join(",")).join(" ");
    s += `<polygon points="${p}" fill="none" stroke="var(--line)" stroke-width="${g === 0 ? 1.6 : 0.8}" ${g === 0 ? 'stroke-dasharray="3 3"' : ""}/>`;
  });
  AX.forEach((a, i) => {
    const [x, y] = pt(i, lim); s += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--line)" stroke-width="0.8"/>`;
    const [lx, ly] = pt(i, lim + 0.62);
    s += `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" opacity="${faint(a, race) ? 0.6 : 1}">${labels[i]}</text>`;
  });
  const poly = (v, col, dash, op) => `<polygon points="${AX.map((a, i) => pt(i, v[a]).join(",")).join(" ")}" fill="${col}" fill-opacity="${op}" stroke="${col}" stroke-width="2" ${dash ? 'stroke-dasharray="5 4"' : ""}/>`;
  if (h.own) s += poly(state.mode === "own" ? h.ped : h.own, "var(--ref)", true, 0.0);
  s += poly(vals(h), "var(--accent)", false, 0.18);
  s += `</svg>`;
  s += `<div class="legend"><span><span class="sw" style="background:var(--accent)"></span>${state.mode === "own" ? "血統+実績" : "血統のみ"}</span>${h.own ? `<span><span class="sw" style="background:var(--ref)"></span>${state.mode === "own" ? "血統のみ" : "血統+実績"}(比較)</span>` : ""}<span>点線の輪 = 0(平均)、外側ほど +</span></div>`;
  return s;
}

function detail(race, h) {
  const v = vals(h);
  const parts = [["脚質", relOf("front", race) * (q(race, "front") || 0) * (v.front || 0)], ["キレ", relOf("kire", race) * (q(race, "kire") || 0) * (v.kire || 0)],
                 ["芝ダ", relOf("surface", race) * (v.surface || 0)], ["距離帯", relOf("dbucket", race) * (v.dbucket || 0)],
                 ["新馬", race.is_debut ? relOf("debut", race) * (v.debut || 0) : 0]];
  let s = `<h3>${h.umaban} ${esc(h.horse_name)}</h3><p class="meta">父 ${esc(h.sire || "—")} / 母父 ${esc(h.bms || "—")}</p>`;
  s += `<table><tr><th>適合度の内訳</th><th>寄与</th></tr>`;
  parts.forEach(([l, x]) => { s += `<tr><td>${l}</td><td>${bar(x, 1.0)}</td></tr>`; });
  s += `<tr><td><b>合計</b></td><td class="num"><b>${fmt(fit(race, h), 3)}</b></td></tr></table>`;
  if (state.mode === "own" && h.own && h.own.n_prev === 0) s += `<p class="note">この馬は過去の出走が無いため、血統のみの値と同じです。</p>`;
  return s;
}

function render() { renderTabs(); renderRace(); }
function seg(id, key) {
  $(id).querySelectorAll("button").forEach(b => b.onclick = () => {
    state[key] = b.dataset.v; $(id).querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); renderRace();
  });
}
(function init() {
  $("#hdr-date").textContent = D.days.map(d => d.date.slice(5).replace("-", "/")).join("・");
  const V = D.validation;
  $("#v-n").textContent = V.main.n_races.toLocaleString();
  $("#v-main").textContent = fmt(V.main.dll, 4) + " [" + V.main.ci.map(x => x.toFixed(4)).join(", ") + "]";
  $("#v-mkt").textContent = fmt(V.market.dll, 5);
  if (V.dl) { $("#v-dl").textContent = fmt(V.dl.dll, 4) + " [" + V.dl.ci.map(x => x.toFixed(4)).join(", ") + "]"; $("#v-dlmkt").textContent = fmt(V.dl.market, 4) + " [" + V.dl.market_ci.map(x => x.toFixed(4)).join(", ") + "]"; }
  else { $("#v-dl-li").hidden = true; $("#v-dlmkt").textContent = "—"; }
  $("#v-top1").textContent = (V.top1.S * 100).toFixed(1) + "% → " + (V.top1["S+F"] * 100).toFixed(1) + "%";
  $("#foot").innerHTML = `時点表: ${D.table_window_end}までのデータ(${D.table_year}年用)。作成: ${D.generated_at}。検証: 事前登録 RADAR_V2_PREREG_2026_10_06.md(追補1〜6)、radar_v2_eval.json・radar_v2_eval_addendum.json・radar_v2_dl_eval.json。`;
  seg("#seg-mode", "mode"); seg("#seg-pace", "pace"); seg("#seg-going", "going"); seg("#seg-sort", "sort"); seg("#seg-fin", "fin");
  state.day = D.days[D.days.length - 1].date;
  try { const s = JSON.parse(localStorage.getItem("rv2_pos") || "null"); if (s && D.days.find(d => d.date === s.day)) Object.assign(state, { day: s.day, venue: s.venue, race: s.race }); } catch (e) {}
  pickVenue(); pickRace();
  render(); renderLearned(); renderPros();
  document.addEventListener("click", () => { try { localStorage.setItem("rv2_pos", JSON.stringify({ day: state.day, venue: state.venue, race: state.race })); } catch (e) {} });
})();
</script>
"""


def common() -> dict:
    """検証の要約・軸の区分・DLが学んだ要求(日付によらない部分)。"""
    import build_race_radar_v2 as B
    import radar_v2_eval as E
    R4 = json.loads((C.OUT_DIR / "radar_v2_eval.json").read_text(encoding="utf-8"))
    AD = json.loads((C.OUT_DIR / "radar_v2_eval_addendum.json").read_text(encoding="utf-8"))
    out = {"axes": B.axis_meta(), "table_year": 2026, "table_window_end": "2025年末",
           "rel": {"front": E.rel("front"), "kire": E.rel("kire"), "debut": E.rel("debut"),
                   "surface": {s: E.rel("surface", s) for s in ["芝", "ダ"]}, "dbucket": {b: E.rel("dbucket", b) for b in E.BUCKETS}},
           "validation": {"main": {"dll": R4["main"]["dll_per_race"], "ci": R4["main"]["ci95_block"], "n_races": R4["n_test_races"]},
                          "market": {"dll": R4["secondary"][-1]["dll_per_race"], "ci": R4["secondary"][-1]["ci95_block"]},
                          "top1": {"S": AD["effect_size"]["S"]["top1_accuracy"], "S+F": AD["effect_size"]["S+F"]["top1_accuracy"]}}}
    d = json.loads((C.OUT_DIR / "radar_v2_dl_eval.json").read_text(encoding="utf-8"))
    out["validation"]["dl"] = {"dll": d["primary_5"]["dll_per_race"], "ci": d["primary_5"]["ci95_block"],
                               "market": d["market"]["v3"]["dll_per_race"], "market_ci": d["market"]["v3"]["ci95_block"]}
    v3 = json.loads((C.OUT_DIR / "v3_export_2026.json").read_text(encoding="utf-8"))
    out["v3_learned"] = v3["learned_by_group"]
    pe = C.OUT_DIR / "prospective_eval.json"
    out["prospective"] = json.loads(pe.read_text(encoding="utf-8")) if pe.exists() else None
    return out


def add_results(day: dict, d: str):
    """結果のある日: 各レースの1〜5着(馬番・馬名・人気)と単勝・複勝の払戻を足す(表示のみ、値の計算には使わない)。"""
    import pandas as pd
    res_p = C.RESULTS_DIR / d[:4] / f"{d}.csv"
    pay_p = C.PROJECT_ROOT / "data" / "payouts" / d[:4] / f"{d}.csv"
    if not res_p.exists():
        return
    r = pd.read_csv(res_p, dtype=str)
    r["pos"] = C.parse_finish(r["finish_pos"])
    pay = pd.read_csv(pay_p, dtype=str) if pay_p.exists() else None
    for race in day["races"]:
        rid = race["race_id"]
        g = r[(r["race_id"] == rid) & (r["pos"] <= 5)].sort_values(["pos", "umaban"], key=lambda s: pd.to_numeric(s, errors="coerce"))
        race["top5"] = [[int(p), int(u), n, int(pp) if str(pp).isdigit() else None]
                        for p, u, n, pp in zip(g["pos"], g["umaban"], g["horse_name"], g["popularity"])]
        if pay is not None:
            pr = pay[(pay["race_id"] == rid) & pay["bet_type"].isin(["単勝", "複勝"])]
            race["pay"] = {k: [[c, int(v), int(pp) if str(pp).isdigit() else None] for c, v, pp in zip(x["combination"], x["payout"], x["popularity"])]
                           for k, x in pr.groupby("bet_type")}


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    data = common()
    data["days"] = []
    for d in sorted(a.days):
        day = json.loads((LIVE / f"race_radar_v2_{d}.json").read_text(encoding="utf-8"))
        day.pop("check_vs_r5", None)
        add_results(day, d)
        data["days"].append(day)
    data["generated_at"] = time.strftime("%Y-%m-%d %H:%M")
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    a.out.write_text(html, encoding="utf-8")
    print("wrote", a.out, round(a.out.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
