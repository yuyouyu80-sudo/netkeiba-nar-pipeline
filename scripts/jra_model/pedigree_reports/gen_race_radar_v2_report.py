# -*- coding: utf-8 -*-
"""血統レーダー v2 の R5: race_radar_v2_data.json から HTML レポートを作る(2026-10-06)。
出力: data/jra_pipeline/pedigree_reports/race_radar_v2_report.html(Artifact として公開する本体)
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / "data" / "jra_pipeline" / "pedigree_reports"
DATA = BASE / "race_radar_v2_data.json"
OUT = BASE / "race_radar_v2_report.html"

TEMPLATE = r"""<title>血統レーダー v2</title>
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
  <h1>血統レーダー v2 — 中山 <span id="hdr-date2"></span></h1>
  <p class="meta">15年分(2011〜)で測り方を作り直した版。血統の値はレース前年末までのデータ・本人の走を除いて作成。旧版(v1)のレポートはそのまま残しています。</p>
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
      </ul>
    </div>
    <div>
      <h3>確かめられなかったこと</h3>
      <ul>
        <li><b>「レースが求める脚質・キレに血統が合う馬が来る」ことは未確認</b>。脚質・キレの列は、血統の傾向の記述として見てください。</li>
        <li><b>オッズに対する上積みは無し</b>(<span class="mono" id="v-mkt"></span>)。人気に織り込まれている情報です。</li>
        <li>当日のトラックバイアス・馬場の変化は入っていません。</li>
      </ul>
    </div>
  </div>
</section>

<section>
  <div class="controls">
    <div><span class="label">レース</span><span class="tabs" id="tabs"></span></div>
  </div>
  <div class="controls" style="margin-top:10px">
    <div><span class="label">馬の値</span><span class="seg" id="seg-mode"><button data-v="ped" aria-pressed="true">血統のみ</button><button data-v="own">血統+実績</button></span></div>
    <div><span class="label">想定ペース</span><span class="seg" id="seg-pace"><button data-v="全体" aria-pressed="true">指定なし</button><button data-v="速い">速い</button><button data-v="平均">平均</button><button data-v="遅い">遅い</button></span></div>
    <div><span class="label">並び</span><span class="seg" id="seg-sort"><button data-v="fit" aria-pressed="true">適合度</button><button data-v="umaban">馬番</button></span></div>
    <div><span class="label">着順</span><span class="seg" id="seg-fin"><button data-v="hide" aria-pressed="true">隠す</button><button data-v="show">表示</button></span></div>
  </div>
</section>

<section id="race"></section>

<section>
  <h2>列の読み方</h2>
  <div class="grid">
    <div>
      <ul>
        <li>値は「父(芝ダ・距離帯は父・母父・父父の平均)の産駒が、その条件で普段よりどれだけ走るか」を、真の差のばらつき(τ)を1とした単位で表したもの。0が平均、±1で血統間の典型的な差。</li>
        <li><b>適合度</b> = 各軸の値 × レースの要求 × 軸の信頼度 の合計(R4で検証した式)。想定ペースを変えると脚質・キレの要求が変わります。</li>
        <li><b>血統+実績</b>: その日より前の本人の走で血統の値を更新(走数が多いほど本人の値に寄る)。芝ダ・距離帯は「本人の芝ダ・距離帯別の成績差」です。</li>
      </ul>
    </div>
    <div>
      <ul>
        <li><span class="st-予測"><b>予測</b></span>: 予測上の価値を確認(芝ダ)。</li>
        <li><span class="st-記述"><b>記述</b></span>: 偏りと信頼性は旧版より大きく改善したが、予測上の価値は未確認(脚質・キレ)。</li>
        <li><span class="st-参考"><b>参考</b></span>: 距離帯(はっきりしない、長距離帯は信頼度低)・新馬(効くが偏りが一部残る)。薄く表示。</li>
        <li>レースの要求は、同じコース・クラス・開催日目の過去レース(足りなければ段階的に広げる)の上位3頭と出走馬平均の差から作っています。</li>
      </ul>
    </div>
  </div>
</section>

<footer id="foot"></footer>
</div>

<script>
const D = __DATA__;
const AX = ["S", "front", "kire", "surface", "dbucket", "debut"];
const state = { race: null, mode: "ped", pace: "全体", sort: "fit", fin: "hide", sel: null };
const $ = (s, el = document) => el.querySelector(s);
const fmt = (x, d = 2) => (x === null || x === undefined || !isFinite(x)) ? "—" : (x > 0 ? "+" : "") + x.toFixed(d);

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

function renderTabs() {
  const t = $("#tabs");
  t.innerHTML = "";
  D.races.forEach(r => {
    const b = document.createElement("button");
    b.textContent = r.race_number + "R";
    b.title = r.race_name;
    if (r.skipped) { b.disabled = true; b.title += "(" + r.skipped + ")"; }
    b.setAttribute("aria-pressed", String(state.race === r.race_id));
    b.onclick = () => { state.race = r.race_id; state.sel = null; render(); };
    t.appendChild(b);
  });
}

function reqChip(label, v, extra) {
  const word = v === null ? "—" : v > 0.5 ? "強く求める" : v > 0.15 ? "やや求める" : v < -0.5 ? "不利(逆)" : v < -0.15 ? "やや不利" : "ほぼ中立";
  return `<div class="chip">${label}: <b>${fmt(v)}</b> <span class="meta">${word}${extra ? " / " + extra : ""}</span></div>`;
}

function renderRace() {
  const race = D.races.find(r => r.race_id === state.race);
  const el = $("#race");
  const hs = race.horses.map(h => ({ ...h, F: fit(race, h) }));
  if (state.sort === "fit") hs.sort((a, b) => b.F - a.F); else hs.sort((a, b) => a.umaban - b.umaban);
  const fmax = Math.max(0.3, ...hs.map(h => Math.abs(h.F)));
  if (!state.sel && hs.length) state.sel = hs[0].umaban;
  const cols = [["S", "父の総合力"], ["front", "脚質(前)"], ["kire", "キレ"], ["surface", race.surface === "芝" ? "芝適性" : "ダ適性"],
                ["dbucket", "距離帯(" + (race.dbucket || "").replace(/\(.*\)/, "") + ")"], ["debut", "新馬"]];
  const rule = a => `${race.req_rule[a] || "—"}・${race.req_n[a] ?? "—"}件`;
  let html = `<h2>${race.race_number}R ${race.race_name} <span class="meta">${race.surface}${race.distance}m${race.is_debut ? "・新馬戦" : ""}</span></h2>
  <div class="chips" style="margin-bottom:10px">
    ${reqChip("脚質(前に行く)の要求", q(race, "front"), rule("front"))}
    ${reqChip("キレの要求", q(race, "kire"), rule("kire"))}
    <div class="chip">芝ダ: <b>${race.surface}</b> <span class="meta">その芝ダの適性を加点</span></div>
    <div class="chip">距離帯: <b>${race.dbucket || "—"}</b></div>
  </div>
  <p class="meta">要求の単位: 2018〜2020年のレースの基準のばらつきを1とする。想定ペース「${state.pace === "全体" ? "指定なし" : state.pace}」${state.fin === "show" && race.actual_pace ? "・実際のペース: " + race.actual_pace : ""}。適合度は脚質・キレ・芝ダ・距離帯・新馬の合計で、父の総合力(強さ)は含みません。</p>
  <div class="scroll"><table><thead><tr><th>馬番</th><th>馬名<span class="sub">父 / 母父</span></th><th>適合度</th>`;
  cols.forEach(([a, l]) => { html += `<th class="${faint(a, race) ? "faint" : ""}">${l}<span class="st st-${statusOf(a)}">${statusOf(a)}</span></th>`; });
  html += `${state.fin === "show" ? "<th>着順</th>" : ""}</tr></thead><tbody>`;
  hs.forEach(h => {
    const v = vals(h), key = h.umaban;
    html += `<tr class="hrow ${state.sel === key ? "sel" : ""}" data-k="${key}"><td class="num">${h.umaban}</td>
      <td>${h.horse_name}<span class="sub">${h.sire || "—"} / ${h.bms || "—"}${state.mode === "own" ? (h.own ? " ・前走まで" + h.own.n_prev + "走" : " ・実績なし(血統のみ)") : ""}</span></td>
      <td>${bar(h.F, fmax, "fitbar")}</td>`;
    cols.forEach(([a]) => { html += `<td class="${faint(a, race) ? "faint" : ""}">${bar(v[a])}</td>`; });
    if (state.fin === "show") html += `<td class="fin ${h.finish && h.finish <= 3 ? "top3" : ""}">${h.finish ?? (h.ran ? "—" : "中止/取消")}</td>`;
    html += "</tr>";
  });
  html += `</tbody></table></div>`;
  const sh = hs.find(h => h.umaban === state.sel) || hs[0];
  html += `<div class="radar-wrap" style="margin-top:14px"><div>${radarSVG(race, sh)}</div><div id="detail">${detail(race, sh)}</div></div>`;
  el.innerHTML = html;
  el.querySelectorAll("tr.hrow").forEach(tr => tr.onclick = () => { state.sel = Number(tr.dataset.k); renderRace(); });
}

function radarSVG(race, h) {
  const labels = ["父の総合力", "脚質(前)", "キレ", race.surface === "芝" ? "芝適性" : "ダ適性", "距離帯", "新馬"];
  const W = 340, cx = 170, cy = 165, R = 115, lim = 2.5, n = AX.length;
  const pt = (i, v) => { const r = (Math.max(-lim, Math.min(lim, v ?? 0)) + lim) / (2 * lim) * R; const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; };
  let s = `<svg viewBox="0 0 ${W} 330" width="100%" role="img" aria-label="${h.horse_name}のレーダー">`;
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
  let s = `<h3>${h.umaban} ${h.horse_name}</h3><p class="meta">父 ${h.sire || "—"} / 母父 ${h.bms || "—"}</p>`;
  s += `<table><tr><th>適合度の内訳</th><th>寄与</th></tr>`;
  parts.forEach(([l, x]) => { s += `<tr><td>${l}</td><td>${bar(x, 1.0)}</td></tr>`; });
  s += `<tr><td><b>合計</b></td><td class="num"><b>${fmt(fit(race, h), 3)}</b></td></tr></table>`;
  if (state.mode === "own" && !h.own) s += `<p class="note">この馬は今回の出走記録が無いため(取消・中止など)、血統のみを表示しています。</p>`;
  return s;
}

function render() { renderTabs(); renderRace(); }
function seg(id, key) {
  $(id).querySelectorAll("button").forEach(b => b.onclick = () => {
    state[key] = b.dataset.v; $(id).querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); renderRace();
  });
}
(function init() {
  const d = D.race_date.replace(/-/g, "/").slice(5);
  $("#hdr-date").textContent = D.race_date; $("#hdr-date2").textContent = d;
  const V = D.validation;
  $("#v-n").textContent = V.main.n_races.toLocaleString();
  $("#v-main").textContent = fmt(V.main.dll, 4) + " [" + V.main.ci.map(x => x.toFixed(4)).join(", ") + "]";
  $("#v-mkt").textContent = fmt(V.market.dll, 5);
  $("#v-top1").textContent = (V.top1.S * 100).toFixed(1) + "% → " + (V.top1["S+F"] * 100).toFixed(1) + "%";
  $("#foot").innerHTML = `時点表: ${D.table_window_end}までのデータ(${D.table_year}年用)。検証: 事前登録 RADAR_V2_PREREG_2026_10_06.md(追補1〜3)、radar_v2_eval.json・radar_v2_eval_addendum.json` + (V.dl ? `、DL挑戦モデル: ${fmt(V.dl.dll, 4)} [${V.dl.ci.map(x => x.toFixed(4)).join(", ")}]` : "") + "。";
  seg("#seg-mode", "mode"); seg("#seg-pace", "pace"); seg("#seg-sort", "sort"); seg("#seg-fin", "fin");
  state.race = (D.races.find(r => !r.skipped) || {}).race_id;
  try { const s = localStorage.getItem("rv2_race"); if (s && D.races.find(r => r.race_id === s && !r.skipped)) state.race = s; } catch (e) {}
  render();
  document.addEventListener("click", () => { try { localStorage.setItem("rv2_race", state.race); } catch (e) {} });
})();
</script>
"""


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT, round(OUT.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
