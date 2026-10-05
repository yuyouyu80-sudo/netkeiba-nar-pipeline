# -*- coding: utf-8 -*-
"""予想ページ(馬柱データ予想 / 馬柱データ予想２)の各レースカードに、
荒れ指数・荒れない指数・予想3連複払戻額 の1行を差し込む後処理(2026-10-05追加)。

入力: レポートHTML と out_scores/scores_{date}.json(s6_race_scores.py が作成)
処理: <article class="race-card" id="race-{race_id}"> の </header> 直後に .arere-strip を挿入。
      スコアが無いレース(障害戦など)には何も付けない。既に差し込み済みの行は一度消してから入れ直す(何度実行しても同じ結果)。
      CSSは既存トークン(--accent-soft / --place2-soft 系)だけを使い、ページ先頭の <nav class="jumpnav"> の直前に1回だけ追加。
使い方:
  python inject_arere_strip.py IN.html OUT.html            # out_scores/ にある全日付のスコアを使う
"""
import glob
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPORT_URL = "https://claude.ai/artifact/VxmLdnnPrZH7wyZXbhUGzx"
MARK_A, MARK_B = "<!--arere-strip-->", "<!--/arere-strip-->"
CSS_A, CSS_B = "<!--arere-css-->", "<!--/arere-css-->"
CSS = CSS_A + """<style>
/* 荒れ指数・荒れない指数・予想3連複払戻額(2026-10-05追加)。既存トークンのみ使用。 */
.arere-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: -4px 0 10px; font-size: 11.5px; }
.arere-strip .as-label { font: 700 10px/1 var(--sans); color: var(--ink-faint); letter-spacing: 0.03em; flex: none; }
.arere-strip .as-chip { display: inline-flex; align-items: baseline; gap: 3px; padding: 3px 7px; border-radius: 6px;
  background: var(--score-track); color: var(--ink-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.arere-strip .as-chip b { color: var(--ink); font-weight: 700; }
.arere-strip .as-chip.is-arere { background: var(--accent-soft); color: var(--accent-soft-ink); }
.arere-strip .as-chip.is-arere b { color: inherit; }
.arere-strip .as-chip.is-katai { background: var(--place2-soft); color: var(--place2-soft-ink); }
.arere-strip .as-chip.is-katai b { color: inherit; }
.arere-strip .as-src { font-size: 10px; color: var(--ink-faint); }
.arere-strip a { color: var(--ink-faint); font-size: 10px; }
</style>""" + CSS_B


def yen(v):
    return f"{int(round(v, -1)):,}円"


def strip_html(s):
    a, k = s["arere"] * 100, s["katai"] * 100
    title = ("荒れ指数=3連複が1万円以上になる確率、荒れない指数=1,000円以下になる確率、予想払戻=3連複払戻の目安"
             "(中央値と、実際の払戻の半分が入る25〜75%の幅)。レース条件と単勝オッズの形だけから計算した参考値で、"
             "本番のscore・予想順位・3連複印とは無関係。払戻額の外れ幅は中央値で約2.3倍。")
    return (f'{MARK_A}<div class="arere-strip" title="{title}">'
            f'<span class="as-label">荒れ度</span>'
            f'<span class="as-chip{" is-arere" if a >= 50 else ""}">荒れ指数 <b>{a:.0f}%</b></span>'
            f'<span class="as-chip{" is-katai" if k >= 30 else ""}">荒れない指数 <b>{k:.0f}%</b></span>'
            f'<span class="as-chip">予想3連複 <b>{yen(s["pay50"])}</b>({yen(s["pay25"])}〜{yen(s["pay75"])})</span>'
            f'<span class="as-src">{s["odds_source"]}で計算</span>'
            f'<a href="{REPORT_URL}" target="_blank" rel="noopener">説明</a></div>{MARK_B}')


def main():
    src, dst = sys.argv[1:3]
    html = open(src, encoding="utf-8").read()
    scores = {}
    for p in sorted(glob.glob(str(HERE / "out_scores" / "scores_*.json"))):
        scores.update(json.load(open(p, encoding="utf-8")))
    html = re.sub(r"\n      " + re.escape(MARK_A) + ".*?" + re.escape(MARK_B), "", html, flags=re.S)
    html = re.sub(re.escape(CSS_A) + ".*?" + re.escape(CSS_B) + r"\n", "", html, flags=re.S)
    assert html.count('<nav class="jumpnav">') == 1, "jumpnav が見つからない/複数ある"
    html = html.replace('<nav class="jumpnav">', CSS + '\n<nav class="jumpnav">')
    n_cards = n_done = 0
    out, pos = [], 0
    for m in re.finditer(r'<article class="race-card" id="race-(\d{12})"', html):
        n_cards += 1
        rid = m.group(1)
        if rid not in scores:
            continue
        h = html.find("</header>", m.end())
        nxt = html.find('<article class="race-card"', m.end())
        if h < 0 or (nxt >= 0 and h > nxt):
            continue
        h += len("</header>")
        out.append(html[pos:h])
        out.append("\n      " + strip_html(scores[rid]))
        pos = h
        n_done += 1
    out.append(html[pos:])
    open(dst, "w", encoding="utf-8").write("".join(out))
    print(f"レースカード {n_cards}件中 {n_done}件に差し込み(スコア {len(scores)}レース分)")


if __name__ == "__main__":
    main()
