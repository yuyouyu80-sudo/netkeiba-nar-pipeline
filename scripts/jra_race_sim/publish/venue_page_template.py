# -*- coding: utf-8 -*-
"""競馬場ごとの相互作用シミュレーションHTMLページ(全レース分をhash切り替えで閲覧できる版)の
静的アセット(CSS + 共通JS)を返すモジュール。build_venue_artifact.py から使う。
horse_run_pair.html(苗場特別1レース専用版)のCSS/JSを、data-role属性ベースの
マウント関数(mountRace/unmount)へリファクタリングしたもの。
"""

STYLE = """
:root {
  --bg: #F3EEE3; --bg-elev: #FBF8F1; --bg-card: #FFFFFF;
  --ink: #241C13; --ink-muted: #6E6152; --ink-faint: #8B7E6D;
  --rule: #E0D6C2; --rule-strong: #C9BB9E;
  --accent: #B5541A; --accent-ink: #FFFFFF; --accent-soft: #F3DFCB; --accent-soft-ink: #7A3510;
  --track-rail: #A99678; --track-surface: #E7DAC2;
  --shadow: 0 1px 2px rgba(40,28,14,0.08), 0 8px 20px -12px rgba(40,28,14,0.22);
  --serif: "Yu Mincho", "YuMincho", "Hiragino Mincho ProN", "Noto Serif JP", Georgia, serif;
  --sans: "Yu Gothic", "YuGothic", "Hiragino Sans", "Noto Sans JP", "Segoe UI", sans-serif;
  --mono: "Cascadia Mono", "SF Mono", Consolas, "Yu Gothic", monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16130E; --bg-elev: #1C1710; --bg-card: #221C14;
    --ink: #EFE7D8; --ink-muted: #B3A691; --ink-faint: #8F8370;
    --rule: #362D21; --rule-strong: #493C2B;
    --accent: #E4874A; --accent-ink: #201004; --accent-soft: #3A2A1C; --accent-soft-ink: #F2C39B;
    --track-rail: #6B5C46; --track-surface: #2A2318;
    --shadow: 0 1px 2px rgba(0,0,0,0.35), 0 10px 24px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #16130E; --bg-elev: #1C1710; --bg-card: #221C14;
  --ink: #EFE7D8; --ink-muted: #B3A691; --ink-faint: #8F8370;
  --rule: #362D21; --rule-strong: #493C2B;
  --accent: #E4874A; --accent-ink: #201004; --accent-soft: #3A2A1C; --accent-soft-ink: #F2C39B;
  --track-rail: #6B5C46; --track-surface: #2A2318;
  --shadow: 0 1px 2px rgba(0,0,0,0.35), 0 10px 24px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #F3EEE3; --bg-elev: #FBF8F1; --bg-card: #FFFFFF;
  --ink: #241C13; --ink-muted: #6E6152; --ink-faint: #8B7E6D;
  --rule: #E0D6C2; --rule-strong: #C9BB9E;
  --accent: #B5541A; --accent-ink: #FFFFFF; --accent-soft: #F3DFCB; --accent-soft-ink: #7A3510;
  --track-rail: #A99678; --track-surface: #E7DAC2;
  --shadow: 0 1px 2px rgba(40,28,14,0.08), 0 8px 20px -12px rgba(40,28,14,0.22);
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
  line-height: 1.6; font-feature-settings: "palt";
}
main { max-width: 1100px; margin: 0 auto; padding: 28px 20px 56px; }

.masthead h1 { font-family: var(--serif); font-size: 22px; margin: 0; letter-spacing: 0.01em; }
.masthead-sub { color: var(--ink-muted); font-size: 13px; margin: 4px 0 0; }
.venue-nav { display: flex; gap: 10px; margin: 14px 0 22px; flex-wrap: wrap; }
.venue-nav a {
  font: 700 12px/1 var(--sans); color: var(--ink-muted); background: var(--bg-elev);
  border: 1px solid var(--rule); border-radius: 999px; padding: 7px 14px; text-decoration: none;
}
.venue-nav a.is-current { color: var(--accent-ink); background: var(--accent); border-color: var(--accent); }
.venue-nav a:hover:not(.is-current) { border-color: var(--rule-strong); }

.race-index { list-style: none; margin: 0 0 26px; padding: 0; display: grid; gap: 6px; }
.race-row {
  display: grid; grid-template-columns: 44px 1fr auto auto; align-items: center; gap: 10px;
  padding: 9px 12px; border: 1px solid var(--rule); border-radius: 10px; background: var(--bg-card);
  cursor: pointer; box-shadow: var(--shadow);
}
.race-row.is-disabled { cursor: default; opacity: 0.55; }
.race-row.is-active { border-color: var(--accent); box-shadow: 0 0 0 1.5px var(--accent) inset; }
.race-row:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.race-rnum { font: 700 13px/1 var(--mono); color: var(--accent); }
.race-name { font-size: 13px; color: var(--ink); }
.race-name .race-sub { display: block; color: var(--ink-faint); font-size: 11px; margin-top: 2px; }
.race-badge {
  font: 700 10px/1 var(--sans); padding: 4px 8px; border-radius: 999px; white-space: nowrap;
  background: var(--bg-elev); color: var(--ink-muted); border: 1px solid var(--rule);
}
.race-badge.badge-straight { color: var(--accent-soft-ink); background: var(--accent-soft); border-color: transparent; }
.race-badge.badge-excluded { color: var(--ink-faint); }
.race-arrow { color: var(--ink-faint); font-size: 13px; }

.panels { display: grid; grid-template-columns: minmax(340px, 1.85fr) minmax(230px, 0.85fr); gap: 18px; align-items: start; }
@media (max-width: 800px) { .panels { grid-template-columns: 1fr; } }

.panel {
  background: var(--bg-card); border: 1px solid var(--rule); border-radius: 14px; padding: 16px 18px 18px;
  box-shadow: var(--shadow);
}
.panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
.panel h2 { font-family: var(--serif); font-size: 14px; margin: 0; color: var(--ink-muted); font-weight: 700; }

.track-svg { width: 100%; height: auto; display: block; }
.track-rail { fill: var(--track-surface); stroke: var(--track-rail); stroke-width: 2.5; }
.track-line { fill: none; stroke: var(--rule-strong); stroke-width: 1; stroke-dasharray: 3 4; }
.track-marker-label { font: 700 5.4px/1 var(--sans); text-anchor: middle; fill: #fff; pointer-events: none; }
.track-goal-line { stroke: var(--accent); stroke-width: 2.5; }
.track-goal-text { font: 700 8.5px/1 var(--sans); fill: var(--accent); text-anchor: middle; }
.track-start-text { font: 700 8px/1 var(--sans); fill: var(--ink-faint); text-anchor: middle; }
.track-stretch-band { fill: var(--accent); opacity: 0.05; }
.horse-marker { stroke: var(--bg-card); stroke-width: 1; cursor: pointer; }
.horse-marker.is-dim { opacity: 0.35; }
.horse-marker.needs-halo { stroke: var(--ink-faint); stroke-width: 1.3; }
.horse-marker.is-estimated { stroke-dasharray: 1.4 1; }
.horse-marker.is-hi { stroke: var(--ink); stroke-width: 1.6; stroke-dasharray: none; }

.rank-list { list-style: none; margin: 12px 0 0; padding: 0; font-size: 11.5px; max-height: 320px; overflow-y: auto; }
.rank-item {
  display: grid; grid-template-columns: 18px 12px 20px 1fr auto; align-items: center; gap: 7px;
  padding: 3px 4px; border-radius: 5px; cursor: pointer;
}
.rank-item:hover, .rank-item.is-hi { background: var(--bg-elev); }
.rank-item:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.rank-num { color: var(--ink-faint); text-align: right; font-variant-numeric: tabular-nums; }
.rank-swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; border: 1px solid var(--rule-strong); }
.rank-style {
  font-size: 10px; text-align: center; color: var(--ink-muted); border: 1px solid var(--rule);
  border-radius: 4px; line-height: 15px; cursor: help;
}
.rank-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); }
.rank-name .est-mark { color: var(--ink-faint); font-size: 9.5px; margin-left: 3px; }
.rank-dist { font: 700 11px/1 var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-muted); }

.hover-info {
  margin-top: 10px; padding: 9px 12px; background: var(--bg-elev); border: 1px solid var(--rule);
  border-radius: 8px; font-size: 11.5px; color: var(--ink); font-variant-numeric: tabular-nums;
}
.hover-info.is-empty { color: var(--ink-faint); font-style: italic; }
.hover-info b { color: var(--accent); font-weight: 700; }

.chart-svg { width: 100%; height: auto; display: block; }
.axis-line { stroke: var(--rule-strong); stroke-width: 1; }
.grid-line { stroke: var(--rule); stroke-width: 1; }
.axis-label { font: 400 10px/1 var(--sans); fill: var(--ink-faint); }
.axis-title { font: 700 10.5px/1 var(--sans); fill: var(--ink-muted); }
.metric-path { fill: none; stroke-width: 1.6; opacity: 0.8; }
.metric-path.is-hi { stroke-width: 2.8; opacity: 1; }
.metric-path.is-dim { opacity: 0.15; }
.metric-path-halo { fill: none; stroke: var(--ink-faint); stroke-width: 2.9; opacity: 0.4; }
.metric-path-halo.is-hi { stroke-width: 3.9; opacity: 0.55; }
.metric-path-halo.is-dim { opacity: 0.08; }
.metric-dot { stroke: var(--bg-card); stroke-width: 1; }
.metric-dot.needs-halo { stroke: var(--ink-faint); stroke-width: 1.3; }
.metric-dot.is-hi { stroke: var(--ink); stroke-width: 1.4; r: 5; }
.drive-band { fill: var(--accent); opacity: 0.06; }
.drive-band-label { font: 700 9.5px/1 var(--sans); fill: var(--accent); text-anchor: middle; }

.event-log {
  list-style: none; margin: 10px 0 0; padding: 0; font-size: 11px; color: var(--ink-muted);
  max-height: 90px; overflow-y: auto; border-top: 1px solid var(--rule); padding-top: 8px;
}
.event-log li { padding: 2px 0; }
.event-log b { color: var(--ink); }

.controls-panel { margin-bottom: 18px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
.play-btn {
  font: 700 13px/1 var(--sans); color: var(--accent-ink); background: var(--accent); border: none;
  border-radius: 999px; padding: 9px 20px; cursor: pointer; flex: none;
}
.play-btn:hover { filter: brightness(1.06); }
.speed-btns { display: inline-flex; gap: 4px; flex: none; }
.speed-btn {
  font: 700 11.5px/1 var(--mono); color: var(--ink-muted); background: var(--bg-elev);
  border: 1px solid var(--rule); border-radius: 6px; padding: 6px 9px; cursor: pointer;
}
.speed-btn.is-active { color: var(--accent-ink); background: var(--accent); border-color: var(--accent); }
.scrub { flex: 1 1 200px; min-width: 160px; accent-color: var(--accent); }
.time-readout { font: 700 13px/1 var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-muted); flex: none; min-width: 62px; }

.caveat-details { margin: 0 0 22px; border: 1px solid var(--rule); border-radius: 10px; background: var(--bg-elev); }
.caveat-details summary {
  cursor: pointer; padding: 11px 16px; font: 700 12.5px/1 var(--sans); color: var(--ink-muted);
  list-style: none;
}
.caveat-details summary::-webkit-details-marker { display: none; }
.caveat-details summary::before { content: "▸ "; color: var(--accent); }
.caveat-details[open] summary::before { content: "▾ "; }
.caveat-body { padding: 0 16px 14px; font-size: 11.5px; color: var(--ink-muted); line-height: 1.7; }
.caveat-body b { color: var(--ink); }

.race-caveat {
  margin: 14px 0 0; padding: 10px 14px; background: var(--bg-elev); border: 1px solid var(--rule);
  border-radius: 8px; font-size: 11px; color: var(--ink-muted); line-height: 1.6;
}

.lap-table-wrap { overflow-x: auto; }
.lap-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; }
.lap-table th, .lap-table td { padding: 7px 9px; text-align: center; border-bottom: 1px solid var(--rule); white-space: nowrap; }
.lap-table th { font-family: var(--serif); font-weight: 700; color: var(--ink-muted); font-size: 11.5px; }
.lap-table td:first-child, .lap-table th:first-child { text-align: left; color: var(--ink-muted); position: sticky; left: 0; background: var(--bg-card); }
.lap-table td.cumulative { font-weight: 700; color: var(--ink); }
.lap-table td.split { color: var(--ink-muted); }
.lap-table tr.is-actual-row td:first-child { color: var(--ink-faint); font-style: italic; }

.results-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; }
.results-table th, .results-table td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
.results-table th { font-family: var(--serif); font-weight: 700; color: var(--ink-muted); font-size: 11.5px; }
.results-table th:nth-child(1), .results-table td:nth-child(1),
.results-table th:nth-child(2), .results-table td:nth-child(2) { text-align: left; }
.results-table th:nth-child(1), .results-table td:nth-child(1) { position: sticky; left: 0; background: var(--bg-card); }
.results-table td:nth-child(2) { color: var(--ink); }
.results-table td.rt-group-start, .results-table th.rt-group-start { border-left: 1px solid var(--rule-strong); }
.results-table td.rt-pair-start, .results-table th.rt-pair-start { border-left: 1px solid var(--rule); }
.results-table tr:hover td { background: var(--bg-elev); }
.results-table .rt-swatch {
  display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px;
  border: 1px solid var(--rule-strong); vertical-align: middle;
}
.results-table .rt-na { color: var(--ink-faint); }
.results-table .rt-est { color: var(--ink-muted); font-size: 9.5px; }
.results-table .rt-est-low {
  color: var(--accent-soft-ink); background: var(--accent-soft);
  border-radius: 3px; padding: 0 4px; font-weight: 700;
}
.lap-swatch {
  display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px;
  border: 1px solid var(--rule-strong); vertical-align: middle;
}
.player-empty {
  padding: 40px 20px; text-align: center; color: var(--ink-faint); font-size: 13px;
  border: 1px dashed var(--rule-strong); border-radius: 14px;
}

.model-update-panel {
  margin: 0 0 22px; background: var(--bg-card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 18px; box-shadow: var(--shadow); font-size: 12.5px;
}
.model-update-panel h2 {
  font: 700 12.5px/1 var(--sans); margin: 0 0 8px; color: var(--accent); letter-spacing: 0.02em;
}
.model-update-panel p { margin: 0 0 8px; color: var(--ink-muted); line-height: 1.7; }
.model-update-panel p:last-child { margin-bottom: 0; }
.model-update-panel b { color: var(--ink); }

.no-change-note {
  margin: 0 0 22px; background: transparent; border: 1px dashed var(--rule);
  border-radius: 10px; padding: 10px 18px; font-size: 11.5px; color: var(--ink-faint);
  line-height: 1.7;
}
.no-change-note b { color: var(--ink-muted); }
"""

PLAYER_MARKUP = """
<div class="panel controls-panel">
  <button type="button" class="play-btn" data-role="playBtn">&#9654; 再生</button>
  <span class="speed-btns" data-role="speedBtns">
    <button type="button" class="speed-btn" data-mult="0.2">0.2x</button>
    <button type="button" class="speed-btn" data-mult="0.5">0.5x</button>
    <button type="button" class="speed-btn" data-mult="1">1x</button>
    <button type="button" class="speed-btn is-active" data-mult="4">4x</button>
    <button type="button" class="speed-btn" data-mult="8">8x</button>
  </span>
  <input type="range" class="scrub" data-role="scrub" min="0" max="120" step="0.1" value="0">
  <span class="time-readout" data-role="timeReadout">0.0秒</span>
</div>

<div class="panels">
  <section class="panel">
    <div class="panel-head"><h2>コース走行(相互作用あり)</h2></div>
    <svg class="track-svg" viewBox="0 0 380 220" data-role="trackSvg"></svg>
    <p class="hover-info is-empty" data-role="hoverInfo">馬(マーカーまたは一覧)をクリック・タップすると詳細が固定表示されます</p>
    <ol class="rank-list" data-role="rankList"></ol>
    <ul class="event-log" data-role="eventLog"></ul>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>距離 × 瞬間スピード</h2></div>
    <svg class="chart-svg" viewBox="0 0 620 420" data-role="chartSvgV"></svg>
    <div class="panel-head" style="margin-top:16px"><h2>距離 × スタミナ残り</h2></div>
    <svg class="chart-svg" viewBox="0 0 620 420" data-role="chartSvgS"></svg>
  </section>
</div>

<p class="race-caveat" data-role="raceCaveat"></p>

<section class="panel" style="margin-top:18px">
  <div class="panel-head"><h2>結果比較(実測 vs シミュレーション)</h2></div>
  <p class="hover-info" style="margin-top:0" data-role="resultsCaption">実際着順の順に並んでいます。「差」はシミュレーション側の値から実測を引いた秒数(プラス=シミュレーションの方が遅い/長い)。</p>
  <div class="lap-table-wrap">
    <table class="results-table" data-role="resultsTable"></table>
  </div>
</section>

<section class="panel" style="margin-top:18px">
  <div class="panel-head"><h2>先頭馬ラップタイム(200mごと、実測 vs シミュレーション)</h2></div>
  <div class="lap-table-wrap">
    <table class="lap-table" data-role="lapTable"></table>
  </div>
</section>
"""

SHARED_CAVEAT_HTML = """
<b>このシミュレーションについて:</b> 出走馬全頭が<b>互いに干渉しながら</b>走ります。
序盤(スタートから第3コーナー相当地点まで。コースごとに長さが異なります)の速度形状は
苗場特別(新潟6R・実測200mラップ)の実測に基づいて再較正した1レース分の形状を全レース共通で
流用しています(3段階モデル: (a)スタート直後は静止状態(0m/s)から約20mでゲート直後の速度まで
加速、(b)続けて「スタートダッシュ」で200m地点まで一度だけ最高速度に到達、(c)以降は実測ビンを
線形補間)。実装している機構は6つです。(1)<b>内側への進路変更</b>: ホームストレッチに入る前、内側(レール方向)に
縦1.5m以上の隙間があれば距離短縮のためレールへ寄ります(スタミナ消耗は実際に走った距離
基準で計算しており、外を回った馬ほど同じペースでもスタミナの減りが速くなります)。
(2)<b>ドラフティング</b>: 他馬の直後(縦6m以内・同じ進路上)を走ると、スタミナ消耗が
わずかに軽減されます。(3)<b>ホームストレッチでのブロック回避</b>: ここから先は横移動が
距離短縮にならないため、前に馬がいて3m未満に迫った場合のみ左右3m先への移動を試み、
そこが縦方向に1m以上空いていればそちらへ回避、なければ前走馬の速度まで減速します
(この5つの閾値は2026-08-08時点でユーザー指定のテスト値です。以前は苗場特別1レースの
着順のみへのグリッドサーチ較正値を使っていましたが、下記「結果比較」テーブルと同じ
指標で8日分約280レースの実測と突き合わせて検証しています)。(4)<b>先頭馬のペース配分</b>: 先頭馬は直後の追走馬より
スタミナが明らかに劣勢になったらペースを落とします(ヒステリシス付き)。第3コーナー相当地点
(下記(5)のキック開始地点)以降はこの配分は行いません。(5)<b>ラストスパート(2026-08-09〜)</b>:
第3コーナー相当地点から、ゴールでスタミナがおよそ30%残るよう速度を上げます。最高速度は
各馬の「スピード指数(直近5走平均)」を参考に芝/ダート別の式で換算した値を上限にします
(この特徴量だけでは実測終盤ペースの個体差の大部分は説明できないため、あくまで参考程度の
上限です)。実測または推定の上がり3Fタイムは、この区間の速度には直接反映されない参考値に
なりました(以前はこのタイムに一致するよう速度を逆算していましたが、2026-08-09に
ユーザー指示で撤廃しています)。(6)<b>脚質による位置取り・キック補正</b>: 各馬の脚質
(逃げ/先行/差し/追込。直近成績から判定、新馬戦など過去走データの無い馬は補正なしの
中立扱い)に応じて、道中(スタートダッシュ後の200m以降)の巡航速度形状と、ラストスパートの
目標スタミナをわずかに補正しています。279レースの実測(コーナー通過順・上がり3Fの脚質別
傾向)を基準に較正した結果、隊列内の位置は逃げが最も前目・追込が最も後方寄り(先行・差しは
その中間)、終盤の脚(上がり3F)は差しが最も速く追込・先行はほぼ同水準、逃げが最も遅いという
傾向が再現されます(上がり3Fは「追込が一番脚を使う」という一般的なイメージとは異なりますが、
279レースの実測に基づく傾向です)。位置取りと終盤の脚は道中のペース配分を通じて互いに
影響し合うため、2つの補正量は独立にではなく組み合わせて較正しています。脚質内でも個体差が
大きいため、単一レースで教科書通りの隊列になるとは限りません。スタート位置は馬番順に1頭あたり約1.25mの間隔、トラック実幅は
22mを想定した未実測の推定値です(いずれも頭数の異なる全レース共通で流用)。
<br><br>
<b>3競馬場36レースへの拡張にともなう既知の限界:</b>
序盤速度形状・進路変更/回避の閾値は苗場特別(新潟ダート1800m)1レースのみの較正であり、
他競馬場・他距離帯・芝コースへの一般化は未検証です。実測ビン(300〜1100m)を超える距離帯は
フラット外挿で扱います(影響が大きいのは札幌4R・芝2600mで、序盤速度形状の約900m分が
外挿範囲です)。中京競馬場の「スパイラルカーブ」(半径が連続的に変化する特殊形状)は
単一半径の円弧近似で簡略化しています。SVGアニメーションの見た目のプロポーション
(直線とコーナーの視覚的な比率)は全レース共通の汎用オーバル形状を流用しており、
各競馬場の実際の見た目とは異なります(背後の物理シミュレーション自体は競馬場ごとの
実測周長・実測直線長で正しく計算されています)。新潟の芝1000m(アイビスSD等)は
コーナーの無い直線専用コースのため、専用の直線レイアウトで表示しています(この場合の
ラストスパート開始地点は従来通りゴール前600m地点を使います)。
出走馬の一部(実測の持続タイムデータが無い馬、*推定表記)は、実測データを持つ馬の頭数に
応じて3段階で推定します。(a)実測データを持つ馬が6頭以上いる場合は回帰推定。
(b)実測データを持つ馬が1〜5頭の場合、回帰は少数標本で不安定になるため使わず、
そのレース自身の実測馬の中央値をspeed_idxの差分で緩やかに調整するのみの簡易推定です
(結果比較テーブルで「*推定(低確度)」と表示、実測検証では3段階中もっとも精度が低く、
8日間の検証対象レースの4割強が該当します)。(c)2歳新馬・未勝利戦など出走馬全頭に
実測データが無いレースでは、レース内の相対的な総合スピード指数のみに基づく粗い近似
(代表的な巡航速度を基準に加減)を使っています。いずれの段階でも、ここで推定される
上がり3Fタイムは上記(5)の通りラストスパートの参考値にとどまり、実測データを持つ馬との
扱いの差は無くなりました。
ディープリンクで1レースだけ見たい場合も、同じ競馬場の全レース分のデータをまとめて
ダウンロードすることになります。
<br><br>
<b>コース横幅の誇張表示について:</b> 馬群の動きを見やすくするため、コースの見た目の
横幅(太さ)と各馬の横方向の位置は実寸の3倍に誇張して描画しています。ドラフティング圏内
(縦6m以内)・ブロック回避(3m未満)等の判定やイベントログの表示は実寸メートルのまま
計算しており、誇張表示には影響されません。また実測したところ、馬群の横の広がりは
主にスタート時の枠順位置(1頭あたり約1.25m間隔)に由来し、レース中の変化は比較的
小さいため、3倍表示にしてもレース中の横の動き自体が大きく見えるようになるわけではない
点にご留意ください(直線コースのアイビスSD等1レースのみ、数式の性質上この拡大表示が
適用されず実寸のまま表示されます)。
<br><br>
<b>「結果比較」「先頭馬ラップタイム」テーブルについて:</b> simタイム・sim上がり3Fは
各馬の走行ログを線形補間して求めた推定値です。「simコーナー通過相当」は直線入り地点
(ゴール前のホームストレッチ長だけ手前の地点)を通過した時刻の順位で、実測の
「通過順位」(周回コーナーごとにレース実況で判定される値)とは判定タイミングが異なる
代理指標のため、値が近いかどうかの方向感の参考にとどめてください(直線コースは
対象外で「—」表示です)。先頭馬ラップの実測行は、実測ラップタイムの区間定義
(第1区間はdistance%200mの端数、以降200mごと)に合わせてsim側も同じ区切りで
再計算しています。
"""


def model_update_panel_html(esc, mean_diff, ci_lo, ci_hi, n_races):
    """2026-08-15、K1/K2再較正に伴う開示パネル(UI/UXレビュー反映)。
    ページ/会場単位の注記のため、レース固有のrace_caveatとは別関数にしている。
    esc: build_venue_artifact系が使っているエスケープ関数を呼び出し側から渡す。
    mean_diff/ci_lo/ci_hi: 採用候補-現行値のm6平均差(独立holdout、Bonferroni補正後CI)。"""
    return f"""
<div class="model-update-panel">
  <h2>このページのデータについて(2026-08-15更新)</h2>
  <p>
    このページのシミュレーションは2026-08-15時点のモデル(脚質補正 K1=-0.55/K2=0.0)で
    再計算したものです。公開当初(8/1・8/2)は2026-08-11生成の古いシミュレーション結果を
    表示しており、その後8/13にコーナー通過順位判定のバグ修正と脚質補正パラメータの変更、
    8/14にスタミナオフセットの追加が行われましたが未反映でした。
  </p>
  <p>
    2026-08-15、コーナー通過順位の精度指標(m6、値が小さいほど実測の通過順位に近い)の
    近傍を再探索し、K1=-0.55/K2=0.0へ変更しました。これまで一度も使っていない独立検証データ
    ({esc(n_races)}レース)で、m6が平均{esc(f"{mean_diff:+.4f}")}(改善方向)、
    ペアブートストラップ95%CI=[{esc(f"{ci_lo:+.4f}")}, {esc(f"{ci_hi:+.4f}")}]と
    統計的に有意に改善しています(0をまたがない)。着順(m1)・走破タイム(m2)に有意な
    悪化はありません。詳細な検証経緯は<code>horse_baseline.py</code>のRUNNING_STYLE_K1/K2
    コメント(2026-08-15更新分)を参照してください。
  </p>
</div>
"""


def no_change_panel_html(esc):
    """2026-08-18、ラストスパート系5パラメータの動画較正で「変更なし」と判断した際の
    軽量開示エントリ(UI/UXレビュー反映: 通常の更新エントリ=model-update-panelとは
    視覚的に区別する。検証区分バッジ=探索的診断、N=17・確認用5)。"""
    return f"""
<div class="no-change-note">
  <b>検証実施・変更なし(2026-08-18・探索的診断・N=17、確認用5)</b><br>
  ユーザー提供のJRA実況動画フレーム(8/1・8/2の17レース、視覚読み取り+二重読取で再現性
  確認済み)を使い、ラストスパート/スタートダッシュ系5パラメータ(R_MIN・R_MAX・
  KICK_EFFORT_EXPONENT・KICK_START_MIN_M・DASH_PEAK_DIST_M)の再較正を検証しましたが、
  いずれも有意・一貫した改善候補が見つからず、現行値を維持しています(探索用12レース+
  確認用5レースの二段階検証。DASH_PEAK_DIST_Mは探索段階でわずかな改善傾向が見えたものの
  確認用データで再現されませんでした)。このページのシミュレーション自体に変更はありません。
  詳細は{esc("horse_baseline.py")}のKICK_START_MIN_M/DASH_PEAK_DIST_Mコメント
  (2026-08-18追記分)を参照してください。
</div>
"""


# --- JS本体。mountRace(container, meta, data) はレースごとに1回呼ばれ、unmount関数を返す。
# 元の horse_run_pair.html(苗場特別1レース専用IIFE)を、data-role経由のDOM取得・
# 明示的なunmount・楕円/直線コース両対応に一般化した。
SCRIPT_JS = r"""
(function () {
  var WAKU_COLORS = { 1: '#FFFFFF', 2: '#1A1A1A', 3: '#D8332B', 4: '#1660B0', 5: '#EFC22A', 6: '#1F8A45', 7: '#E8791A', 8: '#E0538C' };
  var WAKU_NEEDS_HALO = { 1: true, 2: true };
  var RUNNING_STYLE_LABEL = { '逃': '逃げ', '先': '先行', '差': '差し', '追': '追込' };
  var svgNS = "http://www.w3.org/2000/svg";

  // --- 楕円コース幾何(全レース共通の固定プロポーション。horse_run_all.htmlと同一関数) ---
  function trackGeometry() {
    var viewW = 380, viewH = 220, pad = 46;
    var CIRCUMFERENCE_FIXED = 1472.0, homeStretchFixed = 353.9;
    var straightFrac = Math.min(0.45, Math.max(0.15, homeStretchFixed / CIRCUMFERENCE_FIXED));
    var halfW = (viewW - 2 * pad) / 2;
    var ry = (viewH - 2 * pad) / 2;
    var rx = ry * 0.85;
    var straightHalfLen = Math.min(halfW - rx - 8, Math.max(24, halfW * (straightFrac / 0.3)));
    var cx = viewW / 2, cyBottom = pad + 2 * ry, cyTop = pad;
    var perim = 2 * (2 * straightHalfLen) + 2 * (Math.PI * ry);
    var finishLapT = (2 * straightHalfLen) / perim;
    return { viewW: viewW, viewH: viewH, pad: pad, straightHalfLen: straightHalfLen, ry: ry, rx: rx, cx: cx, cyBottom: cyBottom, cyTop: cyTop, perim: perim, finishLapT: finishLapT };
  }
  function cornerNormal(g, local) {
    var s = Math.sin(Math.PI * local), c = Math.cos(Math.PI * local);
    var nx = g.ry * s, ny = g.rx * c;
    var nmag = Math.sqrt(nx * nx + ny * ny) || 1;
    return { nx: nx / nmag, ny: ny / nmag };
  }
  function pointAtLapT(g, lapT, lane) {
    lapT = ((lapT % 1) + 1) % 1;
    var sh = g.straightHalfLen;
    var strFrac = (2 * sh) / g.perim;
    var turnFrac = (Math.PI * g.ry) / g.perim;
    var t = lapT;
    if (t < strFrac) {
      var local = t / strFrac;
      return { x: (g.cx - sh) + 2 * sh * local, y: g.cyBottom + lane };
    }
    t -= strFrac;
    if (t < turnFrac) {
      var localR = t / turnFrac;
      var cxR = g.cx + sh, cyR = (g.cyTop + g.cyBottom) / 2;
      var n = cornerNormal(g, localR);
      return { x: cxR + g.rx * Math.sin(Math.PI * localR) + lane * n.nx, y: cyR + g.ry * Math.cos(Math.PI * localR) + lane * n.ny };
    }
    t -= turnFrac;
    if (t < strFrac) {
      var localT = t / strFrac;
      return { x: (g.cx + sh) - 2 * sh * localT, y: g.cyTop - lane };
    }
    t -= strFrac;
    var localL = Math.min(1, t / turnFrac);
    var cxL = g.cx - sh, cyL = (g.cyTop + g.cyBottom) / 2;
    var n2 = cornerNormal(g, localL);
    return { x: cxL - g.rx * Math.sin(Math.PI * localL) - lane * n2.nx, y: cyL - g.ry * Math.cos(Math.PI * localL) - lane * n2.ny };
  }

  // --- 直線コース幾何(コーナーの無い専用コース、新潟の芝1000m等) ---
  function straightGeometry() {
    var viewW = 380, viewH = 220, padX = 30, padY = 30;
    return { viewW: viewW, viewH: viewH, padX: padX, padY: padY, trackLen: viewW - 2 * padX, laneBand: viewH - 2 * padY };
  }
  function pointAtDistanceStraight(g, d, distance, laneUnits, laneMaxUnits) {
    var x = g.padX + Math.max(0, Math.min(1, d / distance)) * g.trackLen;
    var frac = laneMaxUnits > 0 ? Math.max(0, Math.min(1, laneUnits / laneMaxUnits)) : 0.5;
    var y = g.padY + frac * g.laneBand;
    return { x: x, y: y };
  }

  var PLAYER_TEMPLATE = document.getElementById("player-template").innerHTML;

  // --- mountRace: containerにこのレースの完全なUIを構築し、unmount関数を返す ---
  window.mountRace = function (container, meta, data) {
    container.innerHTML = PLAYER_TEMPLATE;
    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };

    var isStraight = !!meta.isStraightCourse;
    var CIRCUMFERENCE = meta.circumferenceM;   // このレースの実測周長(楕円コースのみ)
    var HOME_STRETCH_M = meta.homeStretchM;    // このレースの実測ホームストレッチ長
    var METERS_PER_UNIT = data.metersPerUnit;
    // 見た目のコース幅・馬の横方向の広がりだけを誇張する倍率。物理判定(ドラフト圏内
    // 2.5m等、laneMの生値を使う箇所)には一切適用しない。+4/+9/-12等の装飾オフセットは
    // 定数のままスケールしない(viewBox=380x220 pad=46に対し全35レース最悪ケースで
    // 収まることを検算済み)。
    var LANE_VISUAL_SCALE = 3;

    var umabanList = Object.keys(data.horses).map(Number).sort(function (a, b) { return a - b; });
    var wakuSeenCount = {};
    var horses = umabanList.map(function (u, i) {
      var h = data.horses[u];
      var waku = h.waku;
      wakuSeenCount[waku] = (wakuSeenCount[waku] || 0) + 1;
      return {
        umaban: u, name: h.name, totalTime: h.totalTime, isEstimated: h.isEstimated, pts: h.pts,
        waku: waku, color: WAKU_COLORS[waku] || '#999', needsHalo: !!WAKU_NEEDS_HALO[waku], laneIndex: i,
        wakuMate: wakuSeenCount[waku] === 2,
        actualFinishPos: h.actualFinishPos != null ? h.actualFinishPos : null,
        actualTime: h.actualTime != null ? h.actualTime : null,
        actualTimeSec: h.actualTimeSec != null ? h.actualTimeSec : null,
        actualLast3f: h.actualLast3f != null ? h.actualLast3f : null,
        actualPassingOrder: h.actualPassingOrder != null ? h.actualPassingOrder : null,
        runningStyle: h.runningStyle || null,  // 逃/先/差/追、新旧race_json混在時の防御的アクセス
      };
    });
    var maxTotalTime = Math.max.apply(null, horses.map(function (h) { return h.totalTime; }));

    var g, laps, startLapT;
    function lapTAtDistance(d) { return ((startLapT + d / CIRCUMFERENCE) % 1 + 1) % 1; }
    var gs;

    var trackSvg = q("trackSvg");
    if (isStraight) {
      gs = straightGeometry();
      trackSvg.setAttribute("viewBox", "0 0 " + gs.viewW + " " + gs.viewH);
    } else {
      g = trackGeometry();
      trackSvg.setAttribute("viewBox", "0 0 " + g.viewW + " " + g.viewH);
      laps = data.distance / CIRCUMFERENCE;
      startLapT = ((g.finishLapT - laps) % 1 + 1) % 1;
    }

    function railPoints(lane) {
      var n = 260, out = [];
      for (var i = 0; i <= n; i++) { out.push(pointAtLapT(g, i / n, lane)); }
      return out;
    }
    function pointsToPath(points, reverse) {
      var pts = reverse ? points.slice().reverse() : points;
      var d = "";
      pts.forEach(function (p, i) { d += (i === 0 ? "M " : "L ") + p.x.toFixed(2) + " " + p.y.toFixed(2) + " "; });
      return d + "Z";
    }
    function addPath(parent, cls, dAttr) {
      var el = document.createElementNS(svgNS, "path");
      el.setAttribute("class", cls); el.setAttribute("d", dAttr);
      parent.appendChild(el);
      return el;
    }

    var laneMaxUnits = (LANE_VISUAL_SCALE * data.trackWidthM + 4) / METERS_PER_UNIT;
    var stretchStartD = data.distance - HOME_STRETCH_M;

    function posFor(d, laneUnits) {
      if (isStraight) return pointAtDistanceStraight(gs, d, data.distance, laneUnits, laneMaxUnits);
      return pointAtLapT(g, lapTAtDistance(d), laneUnits);
    }

    if (isStraight) {
      // 直線コース: シンプルなレーン+ゴールラインのみ描画
      var railTop = document.createElementNS(svgNS, "line");
      railTop.setAttribute("class", "track-line");
      railTop.setAttribute("x1", gs.padX); railTop.setAttribute("y1", gs.padY - 6);
      railTop.setAttribute("x2", gs.padX + gs.trackLen); railTop.setAttribute("y2", gs.padY - 6);
      trackSvg.appendChild(railTop);
      var railRect = document.createElementNS(svgNS, "rect");
      railRect.setAttribute("class", "track-rail");
      railRect.setAttribute("x", gs.padX); railRect.setAttribute("y", gs.padY - 6);
      railRect.setAttribute("width", gs.trackLen); railRect.setAttribute("height", gs.laneBand + 12);
      trackSvg.insertBefore(railRect, railTop);
      var stretchX = gs.padX + Math.max(0, Math.min(1, stretchStartD / data.distance)) * gs.trackLen;
      var stretchBand = document.createElementNS(svgNS, "rect");
      stretchBand.setAttribute("class", "track-stretch-band");
      stretchBand.setAttribute("x", stretchX); stretchBand.setAttribute("y", gs.padY - 6);
      stretchBand.setAttribute("width", (gs.padX + gs.trackLen) - stretchX); stretchBand.setAttribute("height", gs.laneBand + 12);
      trackSvg.appendChild(stretchBand);
      var goalLineS = document.createElementNS(svgNS, "line");
      goalLineS.setAttribute("class", "track-goal-line");
      goalLineS.setAttribute("x1", gs.padX + gs.trackLen); goalLineS.setAttribute("y1", gs.padY - 10);
      goalLineS.setAttribute("x2", gs.padX + gs.trackLen); goalLineS.setAttribute("y2", gs.padY + gs.laneBand + 10);
      trackSvg.appendChild(goalLineS);
      var goalTextS = document.createElementNS(svgNS, "text");
      goalTextS.setAttribute("class", "track-goal-text");
      goalTextS.setAttribute("x", gs.padX + gs.trackLen); goalTextS.setAttribute("y", gs.padY - 14);
      goalTextS.textContent = "ゴール";
      trackSvg.appendChild(goalTextS);
      var startTextS = document.createElementNS(svgNS, "text");
      startTextS.setAttribute("class", "track-start-text");
      startTextS.setAttribute("x", gs.padX); startTextS.setAttribute("y", gs.padY - 14);
      startTextS.textContent = "スタート";
      trackSvg.appendChild(startTextS);
    } else {
      var innerPts = railPoints(0);
      var outerPts = railPoints(laneMaxUnits);
      addPath(trackSvg, "track-rail", pointsToPath(innerPts, false) + " " + pointsToPath(outerPts, true));
      addPath(trackSvg, "track-line", pointsToPath(railPoints(laneMaxUnits / 2), false));

      var stretchInnerPts = [], stretchOuterPts = [];
      var STRETCH_N = 60;
      for (var si = 0; si <= STRETCH_N; si++) {
        var sd = stretchStartD + (data.distance - stretchStartD) * (si / STRETCH_N);
        var slt = lapTAtDistance(sd);
        stretchInnerPts.push(pointAtLapT(g, slt, 0));
        stretchOuterPts.push(pointAtLapT(g, slt, laneMaxUnits));
      }
      addPath(trackSvg, "track-stretch-band", pointsToPath(stretchInnerPts, false) + " " + pointsToPath(stretchOuterPts, true));

      var goalA = pointAtLapT(g, g.finishLapT, -4);
      var goalB = pointAtLapT(g, g.finishLapT, laneMaxUnits + 4);
      var goalLine = document.createElementNS(svgNS, "line");
      goalLine.setAttribute("class", "track-goal-line");
      goalLine.setAttribute("x1", goalA.x); goalLine.setAttribute("y1", goalA.y);
      goalLine.setAttribute("x2", goalB.x); goalLine.setAttribute("y2", goalB.y);
      trackSvg.appendChild(goalLine);
      var goalLabelPt = pointAtLapT(g, g.finishLapT, laneMaxUnits + 9);
      var goalText = document.createElementNS(svgNS, "text");
      goalText.setAttribute("class", "track-goal-text");
      goalText.setAttribute("x", goalLabelPt.x); goalText.setAttribute("y", goalLabelPt.y + 3);
      goalText.textContent = "ゴール";
      trackSvg.appendChild(goalText);

      var startPt = pointAtLapT(g, startLapT, -12);
      var startText = document.createElementNS(svgNS, "text");
      startText.setAttribute("class", "track-start-text");
      startText.setAttribute("x", startPt.x); startText.setAttribute("y", startPt.y + 3);
      startText.textContent = "スタート";
      trackSvg.appendChild(startText);
    }

    horses.forEach(function (h) {
      h.markerEl = document.createElementNS(svgNS, "circle");
      h.markerEl.setAttribute("class", "horse-marker" + (h.needsHalo ? " needs-halo" : "") + (h.isEstimated ? " is-estimated" : ""));
      h.markerEl.setAttribute("r", 3.6);
      h.markerEl.setAttribute("fill", h.color);
      h.markerEl.setAttribute("data-umaban", h.umaban);
      trackSvg.appendChild(h.markerEl);
      h.labelEl = document.createElementNS(svgNS, "text");
      h.labelEl.setAttribute("class", "track-marker-label");
      h.labelEl.textContent = h.umaban;
      trackSvg.appendChild(h.labelEl);
    });

    // --- 順位リスト ---
    var rankList = q("rankList");
    var hoverInfoEl = q("hoverInfo");
    horses.forEach(function (h) {
      var li = document.createElement("li");
      li.className = "rank-item"; li.setAttribute("data-umaban", h.umaban);
      li.setAttribute("tabindex", "0"); li.setAttribute("role", "button");
      li.setAttribute("aria-pressed", "false");

      var numEl = document.createElement("span");
      numEl.className = "rank-num"; numEl.setAttribute("data-rank", "");
      var swatchEl = document.createElement("span");
      swatchEl.className = "rank-swatch"; swatchEl.style.background = h.color;
      var styleEl = document.createElement("span");
      styleEl.className = "rank-style";
      if (h.runningStyle && RUNNING_STYLE_LABEL[h.runningStyle]) {
        styleEl.textContent = h.runningStyle;
        styleEl.title = "脚質: " + RUNNING_STYLE_LABEL[h.runningStyle];
      } else {
        styleEl.textContent = "—"; styleEl.title = "脚質: データ無し(補正なし)";
      }
      var nameEl = document.createElement("span");
      nameEl.className = "rank-name";
      nameEl.appendChild(document.createTextNode(h.umaban + " " + h.name));
      if (h.isEstimated) {
        var estEl = document.createElement("span");
        estEl.className = "est-mark"; estEl.textContent = "*推定";
        nameEl.appendChild(estEl);
      }
      var distEl = document.createElement("span");
      distEl.className = "rank-dist"; distEl.setAttribute("data-dist", "");

      li.appendChild(numEl); li.appendChild(swatchEl); li.appendChild(styleEl); li.appendChild(nameEl); li.appendChild(distEl);
      rankList.appendChild(li);
      h.rankItemEl = li;
      h.rankNumEl = numEl;
      h.rankDistEl = distEl;
    });

    var eventLogEl = q("eventLog");
    var loggedEvents = {};
    function logEvent(key, text, t) {
      if (loggedEvents[key]) return;
      loggedEvents[key] = true;
      var li = document.createElement("li");
      li.innerHTML = "<b>" + t.toFixed(1) + "秒</b> " + text;
      eventLogEl.insertBefore(li, eventLogEl.firstChild);
    }

    var xMax = data.distance;
    var metricIdx = { v: 3, stamina: 4 };

    function makeChart(role, metricKey, domainMax, unit) {
      var svg = q(role);
      var CW = 620, CH = 420;  // 横幅は据え置き、縦幅のみ2倍(要望対応)
      svg.setAttribute("viewBox", "0 0 " + CW + " " + CH);
      var margin = { left: 42, right: 16, top: 24, bottom: 30 };
      var plotW = CW - margin.left - margin.right;
      var plotH = CH - margin.top - margin.bottom;

      function chartX(d) { return margin.left + (d / xMax) * plotW; }
      function chartY(v) { return margin.top + plotH - (v / domainMax) * plotH; }

      var driveStart = xMax - HOME_STRETCH_M;
      var driveRect = document.createElementNS(svgNS, "rect");
      driveRect.setAttribute("class", "drive-band");
      driveRect.setAttribute("x", chartX(driveStart)); driveRect.setAttribute("y", margin.top);
      driveRect.setAttribute("width", chartX(xMax) - chartX(driveStart)); driveRect.setAttribute("height", plotH);
      svg.appendChild(driveRect);
      var driveLabel = document.createElementNS(svgNS, "text");
      driveLabel.setAttribute("class", "drive-band-label");
      driveLabel.setAttribute("x", (chartX(driveStart) + chartX(xMax)) / 2); driveLabel.setAttribute("y", margin.top - 6);
      driveLabel.textContent = "ホームストレッチ(横移動が距離短縮にならない区間)";
      svg.appendChild(driveLabel);

      var xStep = xMax > 3000 ? 500 : 300;
      for (var xd = 0; xd <= xMax; xd += xStep) {
        var gx = chartX(xd);
        var gl = document.createElementNS(svgNS, "line");
        gl.setAttribute("class", "grid-line");
        gl.setAttribute("x1", gx); gl.setAttribute("y1", margin.top);
        gl.setAttribute("x2", gx); gl.setAttribute("y2", margin.top + plotH);
        svg.appendChild(gl);
        var xl = document.createElementNS(svgNS, "text");
        xl.setAttribute("class", "axis-label"); xl.setAttribute("text-anchor", "middle");
        xl.setAttribute("x", gx); xl.setAttribute("y", margin.top + plotH + 16);
        xl.textContent = xd;
        svg.appendChild(xl);
      }
      var step = domainMax / 4;
      for (var vv = 0; vv <= domainMax + 0.01; vv += step) {
        var gy = chartY(vv);
        var yl = document.createElementNS(svgNS, "text");
        yl.setAttribute("class", "axis-label"); yl.setAttribute("text-anchor", "end");
        yl.setAttribute("x", margin.left - 8); yl.setAttribute("y", gy + 3);
        yl.textContent = Math.round(vv);
        svg.appendChild(yl);
        var gl2 = document.createElementNS(svgNS, "line");
        gl2.setAttribute("class", "grid-line");
        gl2.setAttribute("x1", margin.left); gl2.setAttribute("y1", gy);
        gl2.setAttribute("x2", margin.left + plotW); gl2.setAttribute("y2", gy);
        svg.appendChild(gl2);
      }
      var axisXEl = document.createElementNS(svgNS, "line");
      axisXEl.setAttribute("class", "axis-line");
      axisXEl.setAttribute("x1", margin.left); axisXEl.setAttribute("y1", margin.top + plotH);
      axisXEl.setAttribute("x2", margin.left + plotW); axisXEl.setAttribute("y2", margin.top + plotH);
      svg.appendChild(axisXEl);
      var yTitleEl = document.createElementNS(svgNS, "text");
      yTitleEl.setAttribute("class", "axis-title");
      yTitleEl.setAttribute("x", 10); yTitleEl.setAttribute("y", margin.top - 8);
      yTitleEl.textContent = unit;
      svg.appendChild(yTitleEl);
      var xTitleEl = document.createElementNS(svgNS, "text");
      xTitleEl.setAttribute("class", "axis-title"); xTitleEl.setAttribute("text-anchor", "middle");
      xTitleEl.setAttribute("x", margin.left + plotW / 2); xTitleEl.setAttribute("y", CH - 4);
      xTitleEl.textContent = "距離(m)";
      svg.appendChild(xTitleEl);

      var pathGroup = document.createElementNS(svgNS, "g");
      svg.appendChild(pathGroup);
      var dotGroup = document.createElementNS(svgNS, "g");
      svg.appendChild(dotGroup);

      var idx = metricIdx[metricKey];
      horses.forEach(function (h) {
        var d = "";
        h.pts.forEach(function (p, i) {
          d += (i === 0 ? "M" : "L") + " " + chartX(p[1]).toFixed(2) + " " + chartY(p[idx]).toFixed(2) + " ";
        });

        if (h.needsHalo) {
          var halo = document.createElementNS(svgNS, "path");
          halo.setAttribute("class", "metric-path-halo");
          halo.setAttribute("data-umaban", h.umaban);
          halo.setAttribute("d", d);
          pathGroup.appendChild(halo);
          h.haloEls[metricKey] = halo;
        }

        var el = document.createElementNS(svgNS, "path");
        el.setAttribute("class", "metric-path");
        el.setAttribute("stroke", h.color);
        el.setAttribute("data-umaban", h.umaban);
        if (h.wakuMate) el.setAttribute("stroke-dasharray", "7 3.5");
        el.setAttribute("d", d);
        pathGroup.appendChild(el);
        h.pathEls[metricKey] = el;

        var dot = document.createElementNS(svgNS, "circle");
        dot.setAttribute("class", "metric-dot" + (h.needsHalo ? " needs-halo" : ""));
        dot.setAttribute("r", 3.6);
        dot.setAttribute("fill", h.color);
        dot.setAttribute("data-umaban", h.umaban);
        dotGroup.appendChild(dot);
        h.dotEls[metricKey] = dot;
      });

      return {
        updateDot: function (h, d, value) {
          h.dotEls[metricKey].setAttribute("cx", chartX(d));
          h.dotEls[metricKey].setAttribute("cy", chartY(value));
        },
      };
    }

    horses.forEach(function (h) { h.pathEls = {}; h.dotEls = {}; h.haloEls = {}; });
    var chartV = makeChart("chartSvgV", "v", 20, "m/s");
    var chartS = makeChart("chartSvgS", "stamina", 100, "%");

    function sampleAtTime(h, tq) {
      var pts = h.pts;
      if (tq >= h.totalTime) {
        var frozen = pts[pts.length - 1];
        return { d: frozen[1], laneM: frozen[2], v: frozen[3], stamina: frozen[4], groundDist: frozen[5] };
      }
      tq = Math.max(0, tq);
      var lo = 0, hi = pts.length - 1;
      while (lo < hi) {
        var mid = (lo + hi + 1) >> 1;
        if (pts[mid][0] <= tq) lo = mid; else hi = mid - 1;
      }
      if (lo >= pts.length - 1) {
        var last = pts[pts.length - 1];
        return { d: last[1], laneM: last[2], v: last[3], stamina: last[4], groundDist: last[5] };
      }
      var a = pts[lo], b = pts[lo + 1];
      var span = b[0] - a[0];
      var f = span > 0 ? (tq - a[0]) / span : 0;
      return {
        d: a[1] + (b[1] - a[1]) * f,
        laneM: a[2] + (b[2] - a[2]) * f,
        v: a[3] + (b[3] - a[3]) * f,
        stamina: a[4] + (b[4] - a[4]) * f,
        groundDist: a[5] + (b[5] - a[5]) * f,
      };
    }

    var highlighted = null;
    var pinnedHighlight = null;
    var currentT = 0;
    function setHighlight(umaban) {
      highlighted = umaban;
      horses.forEach(function (h) {
        var hi = highlighted !== null && h.umaban === highlighted;
        var dim = highlighted !== null && !hi;
        h.markerEl.classList.toggle("is-hi", hi);
        h.markerEl.classList.toggle("is-dim", dim);
        ["v", "stamina"].forEach(function (m) {
          h.pathEls[m].classList.toggle("is-hi", hi);
          h.pathEls[m].classList.toggle("is-dim", dim);
          h.dotEls[m].classList.toggle("is-hi", hi);
          if (h.haloEls[m]) {
            h.haloEls[m].classList.toggle("is-hi", hi);
            h.haloEls[m].classList.toggle("is-dim", dim);
          }
        });
        h.rankItemEl.classList.toggle("is-hi", hi);
        h.rankItemEl.setAttribute("aria-pressed", pinnedHighlight === h.umaban ? "true" : "false");
      });
      render(currentT);
    }
    function togglePin(umaban) {
      pinnedHighlight = (pinnedHighlight === umaban) ? null : umaban;
      setHighlight(pinnedHighlight);
    }
    function hoverHighlight(umaban) {
      if (pinnedHighlight === null) setHighlight(umaban);
    }

    var listeners = [];
    function on(el, type, fn) { el.addEventListener(type, fn); listeners.push([el, type, fn]); }

    on(rankList, "mouseleave", function () { hoverHighlight(null); });
    on(rankList, "mouseover", function (ev) {
      var li = ev.target.closest(".rank-item");
      if (li) hoverHighlight(Number(li.getAttribute("data-umaban")));
    });
    on(rankList, "click", function (ev) {
      var li = ev.target.closest(".rank-item");
      if (li) togglePin(Number(li.getAttribute("data-umaban")));
    });
    on(rankList, "keydown", function (ev) {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      var li = ev.target.closest(".rank-item");
      if (!li) return;
      ev.preventDefault();
      togglePin(Number(li.getAttribute("data-umaban")));
    });
    on(trackSvg, "mouseover", function (ev) {
      var m = ev.target.closest(".horse-marker");
      if (m) hoverHighlight(Number(m.getAttribute("data-umaban")));
    });
    on(trackSvg, "mouseleave", function () { hoverHighlight(null); });
    on(trackSvg, "click", function (ev) {
      var m = ev.target.closest(".horse-marker");
      if (m) togglePin(Number(m.getAttribute("data-umaban")));
    });

    var scrub = q("scrub");
    scrub.max = maxTotalTime.toFixed(1);
    var timeReadout = q("timeReadout");

    var lastRankOrder = null;
    var rankHoveringForOrder = false;
    on(rankList, "mouseenter", function () { rankHoveringForOrder = true; });
    on(rankList, "mouseleave", function () { rankHoveringForOrder = false; });

    function render(t) {
      var samples = horses.map(function (h) {
        var s = sampleAtTime(h, t);
        var laneUnits = (LANE_VISUAL_SCALE * s.laneM) / METERS_PER_UNIT;
        var pos = posFor(s.d, laneUnits);
        h.markerEl.setAttribute("cx", pos.x); h.markerEl.setAttribute("cy", pos.y);
        h.labelEl.setAttribute("x", pos.x); h.labelEl.setAttribute("y", pos.y + 1.9);
        chartV.updateDot(h, s.d, s.v);
        chartS.updateDot(h, s.d, s.stamina);
        return { h: h, d: s.d, laneM: s.laneM, v: s.v, stamina: s.stamina, groundDist: s.groundDist };
      });
      samples.sort(function (a, b) {
        if (Math.abs(a.d - b.d) > 1e-6) return b.d - a.d;
        return a.h.totalTime - b.h.totalTime;
      });
      var order = samples.map(function (s) { return s.h.umaban; });
      var orderChanged = !lastRankOrder || order.some(function (u, i) { return u !== lastRankOrder[i]; });
      samples.forEach(function (s, i) {
        s.h.rankNumEl.textContent = (i + 1) + ".";
        s.h.rankDistEl.textContent = Math.round(s.d) + "m(実" + Math.round(s.groundDist) + "m)";
      });
      if (orderChanged && !rankHoveringForOrder) {
        samples.forEach(function (s) { s.h.rankItemEl.parentNode.appendChild(s.h.rankItemEl); });
        lastRankOrder = order;
      }
      timeReadout.textContent = t.toFixed(1) + "秒";
      scrub.value = t;

      for (var si = 0; si < samples.length - 1; si++) {
        var lead = samples[si], trail = samples[si + 1];
        var gapM = lead.d - trail.d;
        if (gapM > 0 && gapM < 6.0 && Math.abs(lead.laneM - trail.laneM) < 2.5 && t > 1) {
          logEvent("draft-" + trail.h.umaban + "-" + lead.h.umaban,
            trail.h.umaban + "番が" + lead.h.umaban + "番のドラフト圏内に入りました", t);
        }
      }
      if (samples.length && samples[0].d >= stretchStartD) {
        logEvent("stretch", "先頭がホームストレッチに入りました(横移動は距離短縮になりません)", t);
      }

      if (highlighted !== null) {
        var hs = samples.filter(function (s) { return s.h.umaban === highlighted; })[0];
        if (hs) {
          hoverInfoEl.classList.remove("is-empty");
          var styleSuffix = (hs.h.runningStyle && RUNNING_STYLE_LABEL[hs.h.runningStyle])
            ? "(" + RUNNING_STYLE_LABEL[hs.h.runningStyle] + ")" : "";
          hoverInfoEl.textContent = hs.h.waku + "枠" + hs.h.umaban + "番 " + hs.h.name + styleSuffix +
            "　距離 " + Math.round(hs.d) + "m(実際の走行距離 " + Math.round(hs.groundDist) + "m) ／ " +
            "レーン " + hs.laneM.toFixed(1) + "m(レール基準) ／ " +
            "速度 " + hs.v.toFixed(1) + "m/s ／ スタミナ " + Math.round(hs.stamina) + "%";
        }
      } else {
        hoverInfoEl.classList.add("is-empty");
        hoverInfoEl.textContent = "馬(マーカーまたは一覧)をクリック・タップすると詳細が固定表示されます";
      }
    }

    var playing = false, speedMult = 4, rafStart = 0, rafBaseT = 0, rafId = null, mounted = true;
    var playBtn = q("playBtn");

    function setPlaying(v) {
      playing = v;
      playBtn.innerHTML = playing ? "&#10074;&#10074; 一時停止" : (currentT >= maxTotalTime ? "&#8635; もう一度再生" : "&#9654; 再生");
      if (playing) { rafStart = performance.now(); rafBaseT = currentT >= maxTotalTime ? 0 : currentT; }
    }
    on(playBtn, "click", function () { setPlaying(!playing); });

    on(q("speedBtns"), "click", function (ev) {
      var btn = ev.target.closest(".speed-btn");
      if (!btn) return;
      container.querySelectorAll(".speed-btn").forEach(function (b) { b.classList.remove("is-active"); });
      btn.classList.add("is-active");
      speedMult = parseFloat(btn.getAttribute("data-mult"));
      rafStart = performance.now(); rafBaseT = currentT;
    });

    on(scrub, "input", function () {
      setPlaying(false);
      currentT = parseFloat(scrub.value);
      render(currentT);
    });

    on(document, "visibilitychange", function () {
      if (!document.hidden && playing) {
        rafStart = performance.now();
        rafBaseT = currentT;
      }
    });

    function frame(now) {
      if (!mounted) return;
      if (playing) {
        var elapsed = rafBaseT + (now - rafStart) / 1000 * speedMult;
        if (elapsed >= maxTotalTime) {
          currentT = maxTotalTime; render(currentT); setPlaying(false);
        } else {
          currentT = elapsed; render(currentT);
        }
      }
      rafId = requestAnimationFrame(frame);
    }

    function formatLapTime(t) {
      if (t >= 60) {
        var m = Math.floor(t / 60), s = t - m * 60;
        return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
      }
      return t.toFixed(1);
    }
    function formatSigned(v, digits) {
      var sign = v >= 0 ? "+" : "";
      return sign + v.toFixed(digits);
    }
    // horse_pair_sim.time_at_distance / compare_sim_vs_actual_multi.time_at_distance_from_ptsと
    // 同じ二分探索補間。ptsは[t, d_rail, lane, v, stamina, ground_d]の配列(index1がd_rail)。
    function timeAtDistanceFromPts(pts, dTarget) {
      if (dTarget <= pts[0][1]) return pts[0][0];
      var lo = 0, hi = pts.length - 1;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (pts[mid][1] < dTarget) lo = mid + 1; else hi = mid;
      }
      if (lo === 0) return pts[0][0];
      var a = pts[lo - 1], b = pts[lo];
      var span = b[1] - a[1];
      var f = span > 0 ? (dTarget - a[1]) / span : 0;
      return a[0] + (b[0] - a[0]) * f;
    }

    function buildLapTable() {
      var rows = data.leaderLapTable;
      if (!rows || !rows.length) return;
      var actualRows = data.actualLeaderLapTable;
      var hasActual = !!(actualRows && actualRows.length === rows.length);
      var table = q("lapTable");

      var thead = document.createElement("thead");
      var headRow = document.createElement("tr");
      headRow.appendChild(document.createElement("th"));
      rows.forEach(function (r) {
        var th = document.createElement("th");
        th.textContent = r.distance + "m";
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      var tbody = document.createElement("tbody");

      function addRow(label, cellsFn, cls) {
        var tr = document.createElement("tr");
        if (cls) tr.className = cls;
        var labelTd = document.createElement("td");
        labelTd.textContent = label;
        tr.appendChild(labelTd);
        rows.forEach(function (r, i) {
          var td = document.createElement("td");
          cellsFn(td, r, i);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
        return tr;
      }

      if (hasActual) {
        addRow("通過タイム(実測)", function (td, r, i) {
          td.className = "cumulative";
          td.textContent = formatLapTime(actualRows[i].cumulative);
        }, "is-actual-row");
      }
      addRow("通過タイム(シミュレーション)", function (td, r) {
        td.className = "cumulative";
        td.textContent = formatLapTime(r.cumulative);
      });
      if (hasActual) {
        addRow("差(sim-実測)", function (td, r, i) {
          td.textContent = formatSigned(r.cumulative - actualRows[i].cumulative, 1);
        });
      }

      if (hasActual) {
        addRow("区間(実測)", function (td, r, i) {
          td.className = "split";
          td.textContent = actualRows[i].split.toFixed(1);
        }, "is-actual-row");
      }
      addRow("区間(シミュレーション)", function (td, r) {
        td.className = "split";
        td.textContent = r.split.toFixed(1);
      });

      addRow("先頭馬(シミュレーション)", function (td, r) {
        var horse = horses.filter(function (h) { return h.umaban === r.umaban; })[0];
        var dot = document.createElement("span");
        dot.className = "lap-swatch";
        dot.style.background = horse ? horse.color : "transparent";
        td.appendChild(dot);
        td.appendChild(document.createTextNode(r.umaban + "番"));
      });

      table.appendChild(tbody);
    }
    buildLapTable();

    function buildResultsTable() {
      var table = q("resultsTable");
      if (!table) return;
      var stretchStartD = isStraight ? null : data.distance - HOME_STRETCH_M;
      var hasStretch = stretchStartD !== null && stretchStartD > 0;

      // horse_baseline.pyの_MIN_REGRESSION_N(=6)と対応した3段階の推定信頼度。
      // have=1〜5(low)は回帰を使わず、そのレース自身の実測馬の中央値をspeed_idxの
      // 差分でわずかに調整するのみの簡易推定で、実測検証では3段階中もっとも精度が低い
      // (着順footrule平均が"全頭推定"のhave=0より悪い、というサブエージェントレビューで
      // 発見された逆説的な結果)。*推定マークを一律にせず、このケースだけ強調する。
      var haveCount = horses.filter(function (h) { return !h.isEstimated; }).length;
      var estTier = haveCount === 0 ? "all" : (haveCount < 6 ? "low" : "reg");

      var captionEl = q("resultsCaption");
      if (captionEl) {
        var extra = captionEl.querySelector(".rt-caption-extra");
        if (extra) extra.remove();
        if (estTier === "low") {
          var span = document.createElement("span");
          span.className = "rt-caption-extra";
          span.textContent = " このレースは実測データのある馬が" + haveCount + "頭のみのため、" +
            "*推定(低確度)馬同士の差は精度が低い参考値です。";
          captionEl.appendChild(span);
        }
      }

      var byTime = horses.slice().sort(function (a, b) { return a.totalTime - b.totalTime; });
      var simTimeRank = {};
      byTime.forEach(function (h, i) { simTimeRank[h.umaban] = i + 1; });

      var simStretchRank = {};
      if (hasStretch) {
        var byStretch = horses.map(function (h) {
          return { umaban: h.umaban, t: timeAtDistanceFromPts(h.pts, stretchStartD) };
        }).sort(function (a, b) { return a.t - b.t; });
        byStretch.forEach(function (r, i) { simStretchRank[r.umaban] = i + 1; });
      }

      function naCell(text, cls) {
        var td = document.createElement("td");
        if (text === null || text === undefined || text === "") {
          td.textContent = "—";
          td.className = "rt-na" + (cls ? " " + cls : "");
        } else {
          td.textContent = text;
          if (cls) td.className = cls;
        }
        return td;
      }

      var thead = document.createElement("thead");
      var headRow = document.createElement("tr");
      ["馬番", "馬名", "脚質", "実際着順", "sim着順", "実際タイム", "simタイム(差)",
       "実際上がり3F", "sim上がり3F(差)", "実際通過順", "simコーナー通過相当"].forEach(function (label, i) {
        var th = document.createElement("th");
        th.textContent = label;
        if (i === 3) th.className = "rt-group-start";
        else if (i === 5 || i === 7 || i === 9) th.className = "rt-pair-start";
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.innerHTML = "";
      table.appendChild(thead);

      var tbody = document.createElement("tbody");
      var rows = horses.slice().sort(function (a, b) {
        var fa = parseInt(a.actualFinishPos, 10), fb = parseInt(b.actualFinishPos, 10);
        fa = isNaN(fa) ? 999 : fa; fb = isNaN(fb) ? 999 : fb;
        return fa - fb || a.umaban - b.umaban;
      });

      rows.forEach(function (h) {
        var tr = document.createElement("tr");

        var umabanTd = document.createElement("td");
        var dot = document.createElement("span");
        dot.className = "rt-swatch";
        dot.style.background = h.color;
        umabanTd.appendChild(dot);
        umabanTd.appendChild(document.createTextNode(String(h.umaban)));
        tr.appendChild(umabanTd);

        var nameTd = document.createElement("td");
        nameTd.textContent = h.name;
        if (h.isEstimated) {
          var est = document.createElement("span");
          est.className = "rt-est" + (estTier === "low" ? " rt-est-low" : "");
          est.textContent = estTier === "low" ? " *推定(低確度)" : " *推定";
          est.title = estTier === "low"
            ? "実測データのある馬がこのレースは" + haveCount + "頭のみのため、推定馬同士の差は精度が低い参考値です。"
            : estTier === "all"
            ? "出走馬全頭に実測データが無く、相対的なスピード指数のみに基づく推定です。"
            : "実測データを持つ" + haveCount + "頭の傾向から回帰推定した値です。";
          nameTd.appendChild(est);
        }
        tr.appendChild(nameTd);

        tr.appendChild(naCell(h.runningStyle && RUNNING_STYLE_LABEL[h.runningStyle] ? RUNNING_STYLE_LABEL[h.runningStyle] : null));

        tr.appendChild(naCell(h.actualFinishPos, "rt-group-start"));
        tr.appendChild(naCell(simTimeRank[h.umaban]));
        tr.appendChild(naCell(h.actualTime, "rt-pair-start"));

        var simTimeText = h.totalTime.toFixed(1) + "s";
        if (h.actualTimeSec != null) simTimeText += "(" + formatSigned(h.totalTime - h.actualTimeSec, 1) + ")";
        tr.appendChild(naCell(simTimeText));

        tr.appendChild(naCell(h.actualLast3f != null ? h.actualLast3f.toFixed(1) : null, "rt-pair-start"));

        var l3fStartT = timeAtDistanceFromPts(h.pts, Math.max(0, data.distance - 600));
        var simL3f = h.totalTime - l3fStartT;
        var simL3fText = simL3f.toFixed(1) + "s";
        if (h.actualLast3f != null) simL3fText += "(" + formatSigned(simL3f - h.actualLast3f, 1) + ")";
        tr.appendChild(naCell(simL3fText));

        tr.appendChild(naCell(h.actualPassingOrder, "rt-pair-start"));
        tr.appendChild(naCell(hasStretch ? simStretchRank[h.umaban] : null));

        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
    }
    buildResultsTable();

    var raceCaveatEl = q("raceCaveat");
    if (raceCaveatEl && meta.raceCaveat) raceCaveatEl.innerHTML = meta.raceCaveat;

    render(0);
    rafId = requestAnimationFrame(frame);

    return function unmount() {
      mounted = false;
      if (rafId !== null) cancelAnimationFrame(rafId);
      listeners.forEach(function (l) { l[0].removeEventListener(l[1], l[2]); });
      container.innerHTML = "";
    };
  };
})();
"""

# --- レース一覧・hashルーティング・ページ全体の初期化を行うJS(RACESオブジェクトは
# build_venue_artifact.py がページごとに埋め込む) ---
INIT_JS = r"""
(function () {
  var currentUnmount = null;
  var playerEl = document.getElementById("player");
  var emptyEl = document.getElementById("player-empty");

  function setActiveRow(raceId) {
    document.querySelectorAll(".race-row").forEach(function (row) {
      row.classList.toggle("is-active", row.getAttribute("data-race-id") === raceId);
    });
  }

  function mount(raceId) {
    var entry = RACES[raceId];
    if (!entry) return;
    if (currentUnmount) { currentUnmount(); currentUnmount = null; }
    playerEl.style.display = "";
    emptyEl.style.display = "none";
    currentUnmount = window.mountRace(playerEl, entry.meta, entry.data);
    setActiveRow(raceId);
    if (location.hash !== "#race-" + raceId) {
      history.replaceState(null, "", "#race-" + raceId);
    }
  }

  document.querySelectorAll(".race-row").forEach(function (row) {
    if (row.classList.contains("is-disabled")) return;
    var raceId = row.getAttribute("data-race-id");
    row.addEventListener("click", function () { mount(raceId); playerEl.scrollIntoView({ behavior: "smooth", block: "start" }); });
    row.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      ev.preventDefault();
      mount(raceId);
      playerEl.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  var hashRaceId = location.hash.indexOf("#race-") === 0 ? location.hash.slice(6) : null;
  var defaultRaceId = Object.keys(RACES).sort(function (a, b) { return RACES[a].meta.raceNumber - RACES[b].meta.raceNumber; })[0];
  if (hashRaceId && RACES[hashRaceId]) {
    mount(hashRaceId);
    setTimeout(function () { playerEl.scrollIntoView({ behavior: "auto", block: "start" }); }, 0);
  } else if (defaultRaceId) {
    mount(defaultRaceId);
  }
})();
"""
