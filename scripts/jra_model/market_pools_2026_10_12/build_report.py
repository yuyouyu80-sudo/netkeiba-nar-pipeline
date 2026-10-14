# -*- coding: utf-8 -*-
"""市場の外の情報3方向(T1 単勝×複勝、T2 単勝オッズの動き、T3 厩舎コメント本文)のHTMLレポートを作る。

入力: data/jra_pipeline/market_pools_2026_10_12/ の t1/t2/t3_result.json、lambda_2017_2025.json、
      review_notes.json(レビューと修正の記録、任意)、comment_labels.csv(コメント例)
出力: data/jra_pipeline/market_pools_2026_10_12/report.html
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "jra_pipeline" / "market_pools_2026_10_12"
E = html.escape


def load(name, default=None):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def f2(x):
    return f"{x:+.2f}".replace("-", "−")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


CSS = """
/* 誌面: 1カラムの検証記録。判定表→図→方向ごとの詳細→レビュー→再現の順 */
:root{
  --bg:#f6f6f2;--fg:#1c1d1a;--mut:#62645c;--card:#ffffff;--bd:#dcddd4;--soft:#ecede6;
  --ok:#1f7a45;--ng:#a8321f;--acc:#2c5a8c;--mark:#c48a12;
  --f-disp:"Zen Kaku Gothic New","Hiragino Sans","Yu Gothic",sans-serif;
  --f-body:"Noto Sans JP","Hiragino Sans","Yu Gothic",sans-serif;
  --f-num:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#151613;--fg:#e9eae3;--mut:#a2a499;--card:#1e1f1b;--bd:#34362f;--soft:#26281f;
  --ok:#62c48d;--ng:#ef8a78;--acc:#8db4e6;--mark:#e3b453;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#151613;--fg:#e9eae3;--mut:#a2a499;--card:#1e1f1b;--bd:#34362f;--soft:#26281f;
  --ok:#62c48d;--ng:#ef8a78;--acc:#8db4e6;--mark:#e3b453;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px;line-height:1.75}
.wrap{max-width:880px;margin:0 auto;padding-inline:16px;padding-block:28px 64px;display:flex;flex-direction:column;gap:34px}
h1,h2,h3{font-family:var(--f-disp);text-wrap:balance;line-height:1.35;margin:0}
h1{font-size:1.85rem;font-weight:700}
h2{font-size:1.3rem;font-weight:700;padding-bottom:6px;border-bottom:1px solid var(--bd)}
h3{font-size:1.05rem;font-weight:700}
p{margin:0;max-width:68ch}
section{display:flex;flex-direction:column;gap:14px;min-width:0}
.eyebrow{font-size:.78rem;letter-spacing:.08em;color:var(--mut)}
.lead{font-size:1.02rem}
.mut{color:var(--mut);font-size:.88rem}
.num{font-family:var(--f-num);font-variant-numeric:tabular-nums}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.vc{background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:6px;min-width:0}
.vc .k{font-size:.8rem;color:var(--mut)}
.vc .t{font-family:var(--f-disp);font-weight:700}
.pill{display:inline-block;font-size:.75rem;font-weight:700;padding:1px 9px;border-radius:999px;letter-spacing:.04em;width:fit-content}
.pill.ng{color:var(--ng);border:1px solid var(--ng)}
.pill.ok{color:var(--ok);border:1px solid var(--ok)}
.pill.ref{color:var(--mut);border:1px solid var(--bd)}
.tbl{overflow-x:auto;border:1px solid var(--bd);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--bd);text-align:left;vertical-align:top}
th{font-weight:700;background:var(--soft);white-space:nowrap}
td.r,th.r{text-align:right;font-family:var(--f-num);font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.fig{background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:12px;overflow-x:auto}
.fig svg{display:block;max-width:100%;height:auto}
.fig text{fill:var(--fg);font-family:var(--f-body)}
.fig .ax{stroke:var(--bd)}
.fig .zero{stroke:var(--mut);stroke-dasharray:3 3}
.fig .ci{stroke:var(--acc);stroke-width:2.5}
.fig .pt{fill:var(--acc)}
.fig .pl{fill:var(--mark)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:.82rem;color:var(--mut)}
.legend i{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:middle}
ul{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:4px;max-width:70ch}
.quote{border-left:3px solid var(--bd);padding-left:12px;color:var(--fg)}
.rev{display:flex;flex-direction:column;gap:10px}
.rv{background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:12px 14px;display:flex;flex-direction:column;gap:6px;min-width:0}
.rv .who{font-size:.78rem;color:var(--mut);letter-spacing:.04em}
.code{font-family:var(--f-num);font-size:.82rem;word-break:break-all}
a{color:var(--acc)}
a:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
"""


def pill(ok, txt=None):
    if ok is None:
        return '<span class="pill ref">参考</span>'
    return f'<span class="pill {"ok" if ok else "ng"}">{txt or ("合格" if ok else "不合格")}</span>'


def forest(rows):
    """rows: [(label, dll, lo, hi, placebo_max, primary)] を ΔLL の区間図(SVG)にする。"""
    vals = [v for r in rows for v in r[1:5]]
    lo_, hi_ = min(vals + [0]) - 0.6, max(vals + [0]) + 0.6
    W, L, R, H = 760, 250, 30, 34
    h = H * len(rows) + 46
    sx = lambda v: L + (v - lo_) / (hi_ - lo_) * (W - L - R)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="仮説ごとの ΔLL と95%区間、プラセボ最大">']
    import math
    step = 1 if hi_ - lo_ <= 10 else 2
    t = math.ceil(lo_)
    while t <= hi_:
        x = sx(t)
        parts.append(f'<line class="ax" x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{h - 30}"/>')
        parts.append(f'<text x="{x:.1f}" y="{h - 12}" font-size="11" text-anchor="middle">{f2(t) if t else "0"}</text>')
        t += step
    parts.append(f'<line class="zero" x1="{sx(0):.1f}" y1="8" x2="{sx(0):.1f}" y2="{h - 30}"/>')
    for i, (lab, d, lo, hi, pmax, prim) in enumerate(rows):
        y = 20 + i * H
        w = "700" if prim else "400"
        parts.append(f'<text x="8" y="{y + 4}" font-size="12.5" font-weight="{w}">{E(lab)}</text>')
        parts.append(f'<line class="ci" x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}"/>')
        parts.append(f'<circle class="pt" cx="{sx(d):.1f}" cy="{y}" r="5"/>')
        x = sx(pmax)
        parts.append(f'<path class="pl" d="M{x:.1f} {y - 7} l5 7 l-5 7 l-5 -7z"/>')
    parts.append("</svg>")
    return "".join(parts)


def or_env(r, col=None):
    """新特徴の係数の95%区間(ブロック単位とレース単位の広い方)を1SDあたりのオッズ比で返す。"""
    import math
    out = {}
    for c, v in r.get("coef_ci", {}).items():
        lo, hi, sd = v[0], v[1], v[2]
        if len(v) >= 5:
            lo, hi = min(lo, v[3]), max(hi, v[4])
        out[c] = (math.exp(lo * sd), math.exp(hi * sd))
    return out[col] if col else out


def or_ci(r):
    out = [f'{c} {a:.2f}〜{b:.2f}' for c, (a, b) in or_env(r).items()]
    return "<br>".join(out) or "—"


def hyp_row(r, primary):
    return (f'<tr><td>{"<b>" if primary else ""}{E(r["name"])}{"</b>" if primary else ""}</td>'
            f'<td class="r">{r["n_races"]:,}</td><td class="r">{f2(r["dll"])}</td>'
            f'<td class="r">[{f2(r["lo"])}, {f2(r["hi"])}]</td><td class="r">{f2(r["placebo_max"])}</td>'
            f'<td class="r">{r["perm_p"]:.2f}</td><td class="r">{or_ci(r)}</td>'
            f'<td>{pill(r["pass"]) if primary else pill(None)}</td></tr>')


def main():
    lam = load("lambda_2017_2025.json", {})
    t1, t2, t3 = load("t1_result.json"), load("t2_result.json"), load("t3_result.json")
    rev = load("review_notes.json", {"rounds": []})
    lab = pd.read_csv(OUT / "comment_labels.csv")

    prim = [("T1", t1["H1"]), ("T2", t2["H2"]), ("T3", t3["H3"])]
    secs = [t1.get("H1b"), t1.get("H1_sync"), t1.get("H1_popres"), t2.get("H2b"), t2.get("H2c"), t2.get("H2_rel"),
            t3.get("H3b"), t3.get("H3r"), t3.get("H3c")]
    extra = [x for x in rev.get("extra_hyps", [])]
    rows = [(r["name"], r["dll"], r["lo"], r["hi"], r["placebo_max"], True) for _, r in prim]
    rows += [(r["name"], r["dll"], r["lo"], r["hi"], r["placebo_max"], False) for r in secs if r]

    h = []
    h.append('<title>市場の外の情報3方向</title>')
    h.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    h.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@500;700&family=Noto+Sans+JP:wght@400;700&family=IBM+Plex+Mono:wght@400;500&display=swap">')
    h.append(f"<style>{CSS}</style>")
    h.append('<main class="wrap">')

    # ---- 冒頭
    h.append('<header style="display:flex;flex-direction:column;gap:10px">')
    h.append('<div class="eyebrow">JRA 予想・検証記録 / 2026-10-12 / 事前登録 PREREG_market_pools_2026_10_12</div>')
    h.append('<h1>市場の外の情報3方向</h1>')
    h.append('<p class="lead">確定単勝オッズに入っていない情報を3つの方向から探しました。単勝と複勝の売れ方の食い違い、'
             '馬ごとの単勝オッズの動き、厩舎コメントの本文です。3つとも、この標本(2026年の数百レース)では確定オッズへの上乗せを検出できませんでした。'
             f'効果の大きさ(1標準偏差あたりのオッズ比の95%区間)は T1 {or_env(t1["H1"], "D")[0]:.2f}〜{or_env(t1["H1"], "D")[1]:.2f}、'
             f'T2 {or_env(t2["H2"], "drift60")[0]:.2f}〜{or_env(t2["H2"], "drift60")[1]:.2f}、T3 {or_env(t3["H3"], "E")[0]:.2f}〜{or_env(t3["H3"], "E")[1]:.2f} です。'
             '上乗せが小さいと言えるのは T1 だけで、T2・T3 はプラス・マイナス2割程度の効果を排除できません。'
             '1レースあたり +0.007 程度の対数尤度に当たる大きさで、このプロジェクトで他に合格した特徴と同じ水準です。'
             '「上乗せが無い」とも「小さい」とも言えず、T2・T3 は判定できなかったと読むのが正確です。</p>')
    h.append('<p class="mut">発端は Bolton &amp; Chapman (1986)「Searching for Positive Returns at the Track」。'
             'この論文の多項ロジット(条件付きロジット)と、Benter (1994) 型の市場との組み合わせは、このプロジェクトでは'
             ' jra_market_model.py(2026-08-22〜)・finish_v4 などで既に使っています。今回は手法ではなく、使う情報を新しくしました。</p>')
    h.append('</header>')

    h.append('<section><h2>判定</h2><div class="verdict">')
    names = {"T1": "単勝と複勝の食い違い", "T2": "単勝オッズの動き", "T3": "厩舎コメント本文"}
    for k, r in prim:
        h.append(f'<div class="vc"><div class="k">{k}・主仮説</div><div class="t">{names[k]}</div>'
                 f'<div class="num">ΔLL {f2(r["dll"])}  [{f2(r["lo"])}, {f2(r["hi"])}]</div>'
                 f'<div class="mut">{r["n_races"]:,}レース・{r["n_days"]}開催日</div>{pill(r["pass"], "No-Go" if not r["pass"] else "合格")}</div>')
    h.append('</div>')
    h.append('<p class="mut">ΔLL は、確定オッズだけのモデルに新しい特徴を足したときの対数尤度の増え方(開催日を1日ずつ抜いて当てた合計)。'
             '括弧は開催日×競馬場ブロックのブートストラップ4,000回の95%区間(係数を固定した条件付きの区間)。合格条件は「区間の下限が0より上」かつ'
             '「特徴をレース内でシャッフルした20回(プラセボ)の最大より上」。順位p は主仮説のプラセボ200回の中で実際の ΔLL が何番目かを表す参考値です。'
             '1日抜きの交差検証では、効かない特徴を足すと ΔLL は負に寄る(プラセボの平均 −0.5〜−1.2)ため、区間が負でも「害がある」とは読みません。'
             'また ΔLL の区間は係数を固定したときのばらつきだけを表し、効果の大きさの範囲にはなりません。'
             '物差しは2つあります。効果の大きさは表の右端の「オッズ比の95%区間」(開催日×競馬場ブロック単位とレース単位でそれぞれ復元抽出して'
             '係数を当て直した1,000回の、広い方)で見ます。一方、合否の判定(ΔLL の区間の下限が0より上)は採用のための厳しい関門で、'
             '係数の検定に直すとおよそ z≒4 を求めています。偽の合格をほぼ出さない代わりに、中くらいの効果は通りません。'
             '罰則(L2=1)は新しい特徴と評価コード等の基準側の列に掛かり、市場の列には掛けていません。標準化していない列(評価コード・U)は縮みが大きめです。</p></section>')

    # ---- 図
    h.append('<section><h2>仮説ごとの上乗せ</h2>')
    h.append(f'<div class="fig">{forest(rows)}</div>')
    h.append('<div class="legend"><span><i style="background:var(--acc);border-radius:50%"></i>ΔLL と95%区間</span>'
             '<span><i style="background:var(--mark);transform:rotate(45deg)"></i>プラセボ20回の最大</span>'
             '<span>太字は主仮説(合否に使う)、細字は参考</span></div>')
    h.append('<div class="tbl"><table><thead><tr><th>仮説</th><th class="r">レース</th><th class="r">ΔLL</th>'
             '<th class="r">95%区間</th><th class="r">プラセボ最大</th><th class="r">順位p</th><th class="r">オッズ比の95%区間<br>(1SDあたり)</th><th>判定</th></tr></thead><tbody>')
    for _, r in prim:
        h.append(hyp_row(r, True))
    for r in secs:
        if r:
            h.append(hyp_row(r, False))
    for r in extra:
        h.append(hyp_row(r, False))
    h.append('</tbody></table></div>')
    h.append('<p class="mut">オッズ比は、主仮説の勝ち馬モデルでは勝つ確率の、3着内モデルでは3着内に入る確率のオッズ比です。'
             '「1SDあたり」は特徴が1標準偏差ふえたときの値で、1.00 が上乗せなしを表します。</p>')
    pw = load("power.json")
    if pw:
        h.append('<h3>検出力の目安</h3><p>実データの特徴はそのままにして、結果(勝ち馬・3着内)だけを「確定オッズ+β×特徴」から作り直し、'
                 '同じ判定(区間の下限が0より上)を何回通るかを数えました(プラセボの条件は省いた甘い目安)。</p>')
        betas = [r["beta"] for r in pw["H1"]["rows"]]
        h.append('<div class="tbl"><table><thead><tr><th>主仮説</th>' + "".join(
            f'<th class="r">オッズ比 {pw["H1"]["rows"][i]["odds_ratio_per_sd"]:.3g}</th>' for i in range(len(betas)))
            + '</tr></thead><tbody>')
        for k in ("H1", "H2", "H3"):
            if k in pw:
                h.append(f'<tr><td>{k}({pw[k]["n_races"]:,}R、{pw[k]["n_sim"]}回)</td>' + "".join(
                    f'<td class="r">{pct(r["p_lo_gt0"], 0)}</td>' for r in pw[k]["rows"]) + '</tr>')
        h.append('</tbody></table></div><p class="mut">数字は「区間の下限が0より上」になった割合(1SDあたりのオッズ比ごと)。'
                 'オッズ比1.00 が0%なのは、この判定が偽の合格をほとんど出さない代わりに、本物の上乗せもなかなか拾えないことを示します。'
                 '検出力が低いのは標本の小ささより、合格基準の厳しさが主な理由です(T2 でオッズ比1.22 の効果なら、係数の通常の検定で約7割は検出できる)。'
                 '3着内(H1)は各馬を独立に引く近似です。</p>')
    h.append('</section>')

    # ---- T1
    h.append('<section><h2>T1 単勝と複勝の食い違い</h2>')
    h.append('<p>Hausch・Ziemba・Rubinstein (1981) の考え方です。単勝オッズから計算した3着内確率と、複勝オッズが示す3着内確率を比べ、'
             'ずれている馬が実際によく3着内に来るかを見ました。</p>')
    h.append('<ul>'
             f'<li>単勝からの3着内確率 P3w: Benter 型の縮めた Plackett-Luce(2着 ∝ p<sup>λ2</sup>、3着 ∝ p<sup>λ3</sup>)。'
             f'2017〜2025年 {lam.get("n_races", 0):,}レースで λ2 = {lam.get("lambda2", 0):.3f}、λ3 = {lam.get("lambda3", 0):.3f}。</li>'
             '<li>複勝からの確率 q: 「確定」取得(発走10分後に取った実質の最終オッズ)の複勝オッズ下限・上限の幾何中点の逆数を、レース内で合計3に正規化。</li>'
             '<li>特徴 D = logit(q) − logit(P3w)。複勝が2着までの払いになる出走7頭以下のレースは除外。</li>'
             '<li>最終オッズ同士の比較なので、仮に合格しても購入時点では使えません。前向きに記録するなら発走前の時点で取ります。</li></ul>')
    dbp = t1.get("D_by_pop", {})
    if dbp:
        ks = sorted(dbp, key=int)
        h.append('<h3>人気順ごとの D の平均(標準化前)</h3><div class="tbl"><table><thead><tr><th>人気</th>'
                 + "".join(f'<th class="r">{k if int(k) < 12 else "12〜"}</th>' for k in ks) + '</tr></thead><tbody><tr><td>D</td>'
                 + "".join(f'<td class="r">{f2(dbp[k])}</td>' for k in ks) + '</tr></tbody></table></div>'
                 '<p class="mut">幾何中点の変換は人気馬の q を低く出す偏りがあります(1・2番人気で負)。'
                 'この偏りを人気順ごとに引いた版(参考・事後)でも上乗せは出ませんでした。</p>')
    cal = t1["calibration"]["mean_P3w_by_bin"]
    h.append('<h3>3着内確率の較正(P3w の5分位)</h3><div class="tbl"><table><thead><tr><th>分位</th>'
             '<th class="r">単勝から P3w</th><th class="r">複勝から q</th><th class="r">実際の3着内率</th></tr></thead><tbody>')
    for i, (a, b, c) in enumerate(cal, 1):
        h.append(f'<tr><td>{i}</td><td class="r">{pct(a)}</td><td class="r">{pct(b)}</td><td class="r">{pct(c)}</td></tr>')
    h.append('</tbody></table></div>')
    h.append('<h3>複勝の買い方(参考): P3w × 複勝下限オッズ ≥ T の馬を100円ずつ</h3>'
             '<div class="tbl"><table><thead><tr><th>確率</th><th class="r">T</th><th class="r">点数</th>'
             '<th class="r">的中率</th><th class="r">回収率</th><th class="r">95%区間</th></tr></thead><tbody>')
    for b in t1["bets"]:
        ci = f'[{pct(b["lo"])}, {pct(b["hi"])}]' if "lo" in b else "—"
        h.append(f'<tr><td>{E(str(b["prob"]))}</td><td class="r">{b["T"] if b["T"] else "—"}</td><td class="r">{b["n_bets"]:,}</td>'
                 f'<td class="r">{pct(b["hit"])}</td><td class="r">{pct(b["roi"])}</td><td class="r">{ci}</td></tr>')
    h.append(f'</tbody></table></div><p class="mut">払戻が揃う {t1["bet_dates"][0]}〜{t1["bet_dates"][-1]} のみ。'
             '「全馬」は対象レースの全馬の複勝を買った場合(控除の目安)。P3w で選ぶと、P3w が大穴の3着内を高く見積もる'
             '(最下位5分位で予測 3.1%・実際 1.9%)ため大穴ばかりを拾い、回収率はほぼ0になります(57点中50点が単勝100倍超、3着内の見込みは約1頭)。'
             'これは食い違いの検定ではなく、回収率 3.9% から読み取れることはありません。q で選ぶ版は11点だけで、中身はほぼ複勝下限1.1倍前後の人気馬です。'
             '較正表のとおり、最下位分位では P3w だけでなく q も過大です(予測 2.9%・実際 1.9%)。</p>'
             '<p class="mut">3着内の基準 P3w はオッズ帯で外れ方が違います(単勝100倍超で約5割の過大、10〜30倍で約1割の過小)。'
             '基準を直線で入れるだけでは歪みが残り、D がその一部を拾う可能性があります。前向きに見直すなら、オッズ帯ごとに較正した基準に対して食い違いを足します。</p>'
             '<p class="mut">もう一つの注意: 「確定」の複勝オッズは発走10分後の取得で、レース中・レース後の売れ方の影響が入っています。'
             '発走30秒前から「確定」までの複勝下限の変化は、3着内の馬で −0.7%、外れた馬で +2.1% と差があり(約5標準誤差)、結果に依存したデータです。'
             '係数はほぼ0なので判定は変わりませんが、T1 は本来、発走前の時点で見るべきものです。'
             'レビュアー(Fable 5.1)が発走2分前・10分前の複勝で同じ手順を回した検算(103レース・5日)でも、係数は +0.01〜+0.02 で区間は広く、判断材料にはなりません。</p></section>')

    # ---- T2
    h.append('<section><h2>T2 馬ごとの単勝オッズの動き</h2>')
    h.append('<p>発走1時間前と10分前の単勝オッズ(予想パイプラインが記録した推移)から確定までの動き'
             ' drift = log p<sub>確定</sub> − log p<sub>前</sub> を作りました。締切前に売れた馬は、確定オッズの見込み以上に来るか、を見ています。'
             '記録には取得が遅れたものがあり(1時間前の取得のうち約20レースは発走後)、予定時刻から60秒以内に取れたスナップショットだけを使っています。</p>')
    h.append('<h3>1時間前からの動きの5分位</h3><div class="tbl"><table><thead><tr><th>分位</th><th class="r">頭数</th>'
             '<th class="r">平均 drift</th><th class="r">実際の勝率</th><th class="r">確定オッズの勝率</th><th class="r">単勝回収率</th></tr></thead><tbody>')
    for q in t2["roi_quintile"]:
        h.append(f'<tr><td>{E(q["bin"])}</td><td class="r">{q["n"]:,}</td><td class="r">{f2(q["drift_mean"])}</td>'
                 f'<td class="r">{pct(q["win_rate"])}</td><td class="r">{pct(q["mkt_p"])}</td><td class="r">{pct(q["roi"])}</td></tr>')
    h.append('</tbody></table></div>')
    h.append(f'<p class="mut">どの分位でも、実際の勝率は確定オッズの勝率とほぼ同じです。ただし動きは人気と強く結びついていて、'
             f'1番人気の {pct(t2["fav_share_sold"], 0)} が動きの上位3分の1(売れた側)に入ります。'
             'そこで人気帯(1・2・3番人気、4〜6、7〜9、10番人気以下)ごとの中央値を引いた「人気帯の中での動き」でも見ました(参考・事後)。</p>')
    if t2.get("roi_quintile_rel"):
        h.append('<h3>人気帯の中で見た動きの5分位(参考)</h3><div class="tbl"><table><thead><tr><th>分位</th><th class="r">頭数</th>'
                 '<th class="r">平均</th><th class="r">実際の勝率</th><th class="r">確定オッズの勝率</th><th class="r">単勝回収率</th></tr></thead><tbody>')
        for q in t2["roi_quintile_rel"]:
            h.append(f'<tr><td>{E(q["bin"])}</td><td class="r">{q["n"]:,}</td><td class="r">{f2(q["drift_mean"])}</td>'
                     f'<td class="r">{pct(q["win_rate"])}</td><td class="r">{pct(q["mkt_p"])}</td><td class="r">{pct(q["roi"])}</td></tr>')
        h.append('</tbody></table></div><p class="mut">人気帯をそろえても、売れた馬・売れなかった馬の勝率は確定オッズの見込みどおりでした。'
                 '確定オッズも発走10分後の取得なので、締切直前の売れ方は確定オッズの側に入っています。'
                 '係数は負の側に寄っています(売れた馬は確定オッズほど勝たない方向)が、10分前からの動きでは符号が逆で、どちらもノイズの範囲です。'
                 '区間の下限(1標準偏差あたり約0.8)は小さくなく、どちら向きの効果も排除できていません。'
                 '人気帯の中で「売れなかった」分位の回収率が低いのは、勝率が見込みどおり(4.0%対4.1%)なので大穴の配当のぶれにすぎません。</p>')
    h.append('</section>')

    # ---- T3
    h.append('<section><h2>T3 厩舎コメント本文</h2>')
    h.append('<p>このPCに言語モデルのAPIキーが無いため、この会話の Claude(Opus 5.5)が 4,681件のコメントを読み、'
             '馬名・レース・日付を伏せ、無作為な通し番号の順で点数を付けました。点数の基準は読む前に事前登録しています。</p>'
             '<ul><li>E 期待度(−2〜+2): +2 勝ち負け・チャンスを明言 / +1 好調・前進 / 0 中立 / −1 課題・様子見 / −2 使いつつ・厳しい</li>'
             '<li>U 不安(0/1): 状態・気性・距離・馬場などの不安を明記</li></ul>')
    h.append('<h3>期待度 E ごとの成績</h3><div class="tbl"><table><thead><tr><th class="r">E</th><th class="r">頭数</th>'
             '<th class="r">実際の勝率</th><th class="r">確定オッズの勝率</th><th class="r">3着内率</th><th class="r">市場の3着内確率</th>'
             '<th class="r">単勝回収率</th></tr></thead><tbody>')
    for d in t3["by_E"]:
        h.append(f'<tr><td class="r">{f2(d["E"]).replace(".00", "")}</td><td class="r">{d["n"]:,}</td><td class="r">{pct(d["win"])}</td>'
                 f'<td class="r">{pct(d["mkt_p"])}</td><td class="r">{pct(d["top3"])}</td><td class="r">{pct(d["mkt_P3"])}</td>'
                 f'<td class="r">{pct(d["roi_win"])}</td></tr>')
    h.append('</tbody></table></div>')
    cr = t3["rating_x_E"]
    codes = ["01", "02", "03"]
    h.append('<h3>既存の評価コードとの対応(コメントのある馬)</h3><div class="tbl"><table><thead><tr><th>評価コード</th>')
    es = sorted(cr.keys(), key=float)
    for e in es:
        h.append(f'<th class="r">E {f2(float(e)).replace(".00", "")}</th>')
    h.append('</tr></thead><tbody>')
    for c in codes:
        h.append(f'<tr><td>{c}</td>' + "".join(f'<td class="r">{cr[e].get(c, 0):,}</td>' for e in es) + "</tr>")
    h.append('</tbody></table></div>')
    h.append('<h3>点数の例</h3><div class="tbl"><table><thead><tr><th class="r">E</th><th class="r">U</th><th>コメント</th></tr></thead><tbody>')
    for e in (2, 1, 0, -1, -2):
        for _, r in lab[lab["E"] == e].sample(2, random_state=7).iterrows():
            h.append(f'<tr><td class="r">{f2(e).replace(".00", "")}</td><td class="r">{int(r["U"])}</td><td>{E(str(r["stable_comment"]))}</td></tr>')
    h.append('</tbody></table></div>')
    br = t3.get("by_rating", [])
    if br:
        h.append('<h3>評価コードごとの成績(コメント掲載レース)</h3><div class="tbl"><table><thead><tr><th>評価コード</th><th class="r">頭数</th>'
                 '<th class="r">実際の勝率</th><th class="r">確定オッズの勝率</th><th class="r">単勝回収率</th></tr></thead><tbody>')
        for d in br:
            if d["rc"] in ("01", "02", "03"):
                h.append(f'<tr><td>{d["rc"]}</td><td class="r">{d["n"]:,}</td><td class="r">{pct(d["win"])}</td>'
                         f'<td class="r">{pct(d["mkt_p"])}</td><td class="r">{pct(d["roi_win"])}</td></tr>')
        h.append('</tbody></table></div><p class="mut">評価コード03(弱気)の大半は、単勝の中央値が100倍を超える、市場がすでに見放した大穴です。'
                 '勝ちは4頭(市場の見込み約6.5頭、偶然でこれ以下になる確率は約2割)で、市場に足す検定(H3r)でも区別できません。'
                 '低い回収率はこの少数の差で出ているだけなので、使える数字として扱いません。</p>')
    ag = load("rater_agreement.json")
    if ag:
        h.append('<h3>採点の一致度(盲検の別評価者 200件)</h3>'
                 f'<p>結果を見ていない別の評価者(サブエージェント、同じ基準)が無作為の200件を採点しました。期待度 E は完全一致 {pct(ag["E_exact"], 0)}、'
                 f'±1以内 {pct(ag["E_within1"], 0)}、重み付きκ {ag["E_kappa_quadratic"]:.2f} で、十分に再現できます。'
                 f'不安 U は κ {ag["U_kappa"]:.2f}(U=1 の割合が {pct(ag["U_rate_1"], 0)} 対 {pct(ag["U_rate_2"], 0)})と基準の解釈が割れたため、'
                 'U を使う副仮説(H3b)は信頼性が低いものとして扱います。'
                 'なお別評価者も同じ系統のモデルなので、この一致は「Claude 同士で読みが再現する」ことを示すだけで、人の読みとの一致は確かめていません。</p>')
    h.append(f'<p class="mut">コメントはレース単位で全頭に載るか載らないかで、結果のある {t3.get("n_races_all_results", 0):,}レースのうち掲載は'
             f' {t3["H3"]["n_races"]:,}レースでした。評価はこの掲載レースだけで行っています。'
             '期待度は市場の勝率とほぼ同じ順に並び、評価コード(01 強気〜03 弱気)ともほぼ一致します。'
             '本文を読んでも、評価コードと確定オッズ以上の情報はこの標本では検出できませんでした。'
             'H3 の順位p が 0.97 と高いのは、期待度 E が評価コードとほぼ重なっていて、1日抜きのたびに係数が揺れるためと考えられます(害の証拠ではありません)。</p>')
    cm = load("t3_class_mix.json")
    if cm:
        h.append('<h3>コメントが載ったレースのクラス</h3><div class="tbl"><table><thead><tr><th>クラス</th>'
                 '<th class="r">結果のあるレース</th><th class="r">うちコメント掲載</th></tr></thead><tbody>')
        for r in cm:
            h.append(f'<tr><td>{E(r["klass"])}</td><td class="r">{r["n_races"]:,}</td><td class="r">{r["n_comment_races"]:,}</td></tr>')
        h.append('</tbody></table></div><p class="mut">クラスはレース名から判定。コメントは新馬戦と上のクラスに偏り、未勝利戦には1レースも載っていません。'
                 'T3 の結論は未勝利戦と1勝クラスの大半を含まない標本のものです。過去走の無い新馬戦は、コメントの価値が最も高いと考えられるため、'
                 '前向きに別枠で記録します。</p>')
    h.append('</section>')

    # ---- レビュー
    if rev.get("rounds"):
        h.append('<section><h2>レビューと修正</h2>')
        if rev.get("intro"):
            h.append(f'<p>{E(rev["intro"])}</p>')
        for rd in rev["rounds"]:
            h.append(f'<h3>{E(rd["title"])}</h3><div class="rev">')
            for it in rd["items"]:
                h.append(f'<div class="rv"><div class="who">{E(it.get("who", ""))}・重要度 {E(it.get("sev", ""))}</div>'
                         f'<div>{E(it["finding"])}</div><div class="mut">対応: {E(it["action"])}</div></div>')
            h.append('</div>')
        h.append('</section>')

    # ---- 限界・次の手・再現
    h.append('<section><h2>読み方の注意と次の手</h2><ul>')
    for s in rev.get("limits", [
        "データは2026年分のみで標本が小さい(T1 約300R、T2 約630R、T3 約930R)。小さな上乗せを検出する力は弱く、「無い」と断定はできない。",
        "単勝オッズ推移は予想パイプラインが記録した1日24〜36レースに限られる。",
        "事前登録どおり、標本が2倍以上になった時点で同じ式のまま1回だけ再判定してよい。それ以外の見直しはしない。",
    ]):
        h.append(f"<li>{E(s)}</li>")
    h.append('</ul>')
    if rev.get("next"):
        h.append('<h3>前向きに記録するもの(優先順)</h3><ol style="margin:0;padding-left:1.4em;display:flex;flex-direction:column;gap:4px;max-width:70ch">')
        for s in rev["next"]:
            h.append(f"<li>{E(s)}</li>")
        h.append('</ol>')
    h.append('</section>')
    h.append('<section><h2>再現</h2><p class="mut">スクリプト: <span class="code">scripts/jra_model/market_pools_2026_10_12/</span>'
             '(mp_common.py・t1_win_place.py・t2_odds_drift.py・t3_comments.py・t3_class_mix.py・rater_agreement.py・power_check.py・inputs_manifest.py・build_report.py)<br>'
             '入力の目録: <span class="code">inputs_manifest.json</span>(使ったファイルの sha256・対象日・パッケージの版。対象期間の終わりは 2026-10-12 に固定)<br>'
             '結果: <span class="code">data/jra_pipeline/market_pools_2026_10_12/</span>(*_result.json・comment_labels.csv)<br>'
             '事前登録: <span class="code">PREREG_market_pools_2026_10_12.md</span>(追補に結果と修正を記録)</p></section>')
    h.append('</main>')
    (OUT / "report.html").write_text("\n".join(h), encoding="utf-8")
    print("書き出し:", OUT / "report.html")


if __name__ == "__main__":
    main()
