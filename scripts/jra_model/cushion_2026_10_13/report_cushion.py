# -*- coding: utf-8 -*-
"""クッション値・含水率を予想に盛り込む検証のHTMLレポート(2026-10-14)。

入力: data/jra_pipeline/cushion_2026_10_13/ の forecast_result_{dev,confirm}_{none,night,morning}.json、
      horse_result.json(開発)、confirm_C1〜C5.json・confirm_build.json(確認)、step1_result.json
出力: data/jra_pipeline/cushion_2026_10_13/report.html
見た目の部品は市場の外の情報3方向のレポート(market_pools_2026_10_12/build_report.py)と共通。
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data" / "jra_pipeline" / "cushion_2026_10_13"
_spec = importlib.util.spec_from_file_location("mp_report", ROOT / "scripts" / "jra_model" / "market_pools_2026_10_12" / "build_report.py")
BR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BR)
E, f2, pct, pill = BR.E, BR.f2, BR.pct, BR.pill

CSS = BR.CSS.replace("--acc:#2c5a8c", "--acc:#2f6a3e").replace("--acc:#8db4e6", "--acc:#8fcf9c")
TNAME = {"cushion": "芝のクッション値", "moist_turf": "芝の含水率(ゴール前、%)", "moist_dirt": "ダートの含水率(ゴール前、%)"}
WNAME = {"none": "天気なし", "night": "前夜版(前日21時までの天気)", "morning": "当日朝版(当日7時までの天気)"}


def load(name, default=None):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def or_env(r):
    out = {}
    for c, v in r.get("coef_ci", {}).items():
        lo, hi, sd = v[0], v[1], v[2]
        if len(v) >= 5:
            lo, hi = min(lo, v[3]), max(hi, v[4])
        out[c] = (math.exp(lo * sd), math.exp(hi * sd))
    return out


def or_txt(r):
    return "<br>".join(f"{c.split('_')[0]} {a:.2f}〜{b:.2f}" for c, (a, b) in or_env(r).items()) or "—"


def forecast_table(h, period):
    h.append('<div class="tbl"><table><thead><tr><th>予想するもの</th><th>天気</th><th class="r">開催日</th><th class="r">このモデル</th>'
             '<th class="r">直前の測定値そのまま</th><th class="r">競馬場×月の平均</th><th class="r">天気なし(同じ日)</th>'
             '<th class="r">差(モデル−直前値)の95%区間</th><th class="r">±0.5以内</th></tr></thead><tbody>')
    for t in TNAME:
        for w in WNAME:
            r = load(f"forecast_result_{period}_{w}.json")
            if not r or t not in r:
                continue
            x = r[t]
            nw = f'{x["mae_no_weather_same_days"]:.3f}' if "mae_no_weather_same_days" in x else "—"
            h.append(f'<tr><td>{E(TNAME[t])}</td><td>{E(WNAME[w])}</td><td class="r">{x["n"]:,}</td>'
                     f'<td class="r"><b>{x["mae_model"]:.3f}</b></td><td class="r">{x["mae_persistence"]:.3f}</td>'
                     f'<td class="r">{x["mae_venue_month"]:.3f}</td><td class="r">{nw}</td>'
                     f'<td class="r">[{x["mae_diff_ci"][0]:+.3f}, {x["mae_diff_ci"][1]:+.3f}]</td>'
                     f'<td class="r">{pct(x["within05_model"], 0)}</td></tr>')
    h.append('</tbody></table></div>')


def hyp_rows(h, rows):
    h.append('<div class="tbl"><table><thead><tr><th>仮説</th><th class="r">レース</th><th class="r">ΔLL</th><th class="r">95%区間</th>'
             '<th class="r">プラセボ最大</th><th class="r">順位p</th><th class="r">オッズ比の95%区間<br>(1SDあたり)</th><th>判定</th></tr></thead><tbody>')
    for label, r, primary in rows:
        h.append(f'<tr><td>{"<b>" if primary else ""}{E(label)}{"</b>" if primary else ""}</td><td class="r">{r["n_races"]:,}</td>'
                 f'<td class="r">{f2(r["dll"])}</td><td class="r">[{f2(r["lo"])}, {f2(r["hi"])}]</td><td class="r">{f2(r["placebo_max"])}</td>'
                 f'<td class="r">{r["perm_p"]:.2f}</td><td class="r">{or_txt(r)}</td><td>{pill(r["pass"])}</td></tr>')
    h.append('</tbody></table></div>')


def main():
    st1 = load("step1_result.json", {})
    dev3 = load("horse_result.json", {})
    cb = load("confirm_build.json", {})
    C = {k: load(f"confirm_{k}.json") for k in ("C1", "C2", "C3", "C4", "C5")}
    fc_dev, fc_cf = load("forecast_result_dev_none.json"), load("forecast_result_confirm_none.json")
    fm_cf = load("forecast_result_confirm_morning.json")
    c1 = C["C1"]

    h = ['<title>クッション値の予想と検証</title>',
         '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@500;700&family=Noto+Sans+JP:wght@400;700&family=IBM+Plex+Mono:wght@400;500&display=swap">',
         f"<style>{CSS}</style>", '<main class="wrap">']

    # ---- 冒頭
    h.append('<header style="display:flex;flex-direction:column;gap:10px">')
    h.append('<div class="eyebrow">JRA 予想・検証記録 / 2026-10-13〜14 / 事前登録 PREREG_cushion_2026_10_13</div>')
    h.append('<h1>クッション値の予想と検証</h1>')
    cf = fc_cf["cushion"] if fc_cf else None
    lead = ('クッション値・含水率を予想に盛り込むための検証です。クッション値は開催日の朝に測られて当日朝に発表されるので、'
            '当日の予想ではそのまま使えます。発表前(前夜)のために、直前の測定値・前日の天気と馬場・競馬場・季節・コース・気象庁の雨量や気温から'
            '当日朝の値を予想するモデルを作りました。')
    if cf:
        lead += (f'2020〜2024年の確認でも、予想の誤差は直前の測定値をそのまま使うより小さくなりました(平均誤差 {cf["mae_model"]:.3f} 対 {cf["mae_persistence"]:.3f})。')
    if c1:
        lead += (' 馬ごとの「硬い・柔らかい馬場への向き不向き」は、確認で確定オッズへの上乗せを'
                 + ('確かめられました。' if c1["pass"] else '検出できませんでした(不合格)。'
                    + (' 区間が狭く、効果があっても1標準偏差あたりオッズ比 {:.2f}〜{:.2f} 程度の小ささです。'.format(*list(or_env(c1).values())[0]))))
    h.append(f'<p class="lead">{lead}</p>')
    h.append('<p class="mut">開発(2025-01〜2026-08)で定義を決めて凍結し、収集用の会話が新しく取ってきた2020-01〜2024-12で1回だけ確かめています。'
             '合否の条件は「市場の外の情報3方向」と同じです。</p></header>')

    # ---- 判定
    h.append('<section><h2>判定</h2><div class="verdict">')
    if cf:
        h.append(f'<div class="vc"><div class="k">手順2・確認</div><div class="t">クッション値の予想(前夜版)</div>'
                 f'<div class="num">平均誤差 {cf["mae_model"]:.3f}(直前値 {cf["mae_persistence"]:.3f})</div>'
                 f'<div class="mut">{cf["n"]:,}開催日・差の区間 [{cf["mae_diff_ci"][0]:+.3f}, {cf["mae_diff_ci"][1]:+.3f}]</div>{pill(cf["pass"])}</div>')
    if fm_cf:
        x = fm_cf["moist_dirt"]
        h.append(f'<div class="vc"><div class="k">手順2・確認(参考)</div><div class="t">ダート含水率の予想(当日朝版)</div>'
                 f'<div class="num">平均誤差 {x["mae_model"]:.2f}%(直前値 {x["mae_persistence"]:.2f}%)</div>'
                 f'<div class="mut">天気なしでは {x["mae_no_weather_same_days"]:.2f}%。天気がよく効く</div>{pill(x["pass"])}</div>')
    if c1:
        h.append(f'<div class="vc"><div class="k">手順3・確認の主仮説</div><div class="t">馬自身×クッション値(芝)</div>'
                 f'<div class="num">ΔLL {f2(c1["dll"])}  [{f2(c1["lo"])}, {f2(c1["hi"])}]</div>'
                 f'<div class="mut">{c1["n_races"]:,}レース・順位p {c1["perm_p"]:.2f}</div>{pill(c1["pass"], "合格" if c1["pass"] else "No-Go")}</div>')
    h.append('</div><p class="mut">手順2の合格の目安は「開催回を1つずつ抜いて当てた平均誤差が、直前の測定値をそのまま使うより小さく、'
             '開催回ブロックのブートストラップで差の区間が0をまたがない」。手順3は「ΔLL の区間の下限が0より上 かつ プラセボ先頭20回の最大より上」。</p></section>')

    # ---- 発走前の情報
    h.append('<section><h2>クッション値は発走前の情報</h2>')
    h.append('<p>JRA のアーカイブ(開催回が終わってから公開)では、開催日の測定時刻が 8:20、金曜が 9:30 です。値は当日朝に JRA の馬場情報ページで発表されているので、'
             '過去分も「発走前に分かった値」として使えます。「事後データ」だったのはアーカイブの公開時期だけです。'
             '2026-10-13 から、収集用の会話が開催日の朝の値を当日中に取得しています(fetch_jra_baba_live.py)。</p>')
    h.append('<ul><li>当日の予想: 朝の発表値をそのまま使う。</li><li>前夜の予想(朝の発表より前): 下のモデルで当日朝の値を予想する。</li></ul></section>')

    # ---- 手順2
    h.append('<section><h2>手順2 当日朝の値を前もって予想する</h2>')
    h.append('<p>目的は「直前の測定(金曜か前日朝)からの変化」。説明に使う情報は、直前の測定値と測定からの日数・直前の含水率・競馬場・月(季節)・使用コース(A〜D)と変更・'
             '開催の何日目か・前日のその競馬場の天気と馬場状態、天気版ではさらに気象庁の雨量合計・最大1時間雨量・平均気温・日照・平均風速です。線形(リッジ)で、開催回を1つずつ抜いて当てました。</p>')
    h.append('<h3>開発(2025-01〜2026-08)</h3>')
    forecast_table(h, "dev")
    h.append('<h3>確認(2020-01〜2024-12、天気版は2020-09〜)</h3>')
    forecast_table(h, "confirm")
    h.append('<p class="mut">芝のクッション値は、前日の雨・曇り、競馬場(京都が高め、札幌・函館が低め)、直前の値が高すぎ・低すぎると戻る動き、コースの変更でよく説明できます。'
             '前夜までの天気は、レース結果に載る前日の天気でほぼ足りていて、気象庁の観測値を足しても前夜版は縮みませんでした。'
             '夜のうちの雨まで入る当日朝版では縮みます。含水率は天気がはっきり効きます。</p>')
    if fc_cf:
        x = fc_cf["cushion"]
        h.append(f'<p class="mut">直前の値から0.5以上動いた日(確認で {x["big_move_days"]}日)は、モデルでも平均誤差 {x["mae_on_big_moves_model"]:.2f} が残ります。'
                 '前夜の予報(気象庁の降水確率は 2026-10-13 から取得開始)を足すのが次の改良です。</p>')
    h.append('</section>')

    # ---- 手順3
    h.append('<section><h2>手順3 馬ごとの向き不向き</h2>')
    h.append('<p>各走の「オッズに対する着順の上振れ」(人気順位 − 着順を頭数で割ったもの)を、その日の馬場の偏差(競馬場ごとの平均からの差)に回帰した傾きを、'
             '前日までの成績だけで父・母の父・馬自身ごとに求め、今回の馬場の偏差と掛けました。芝はクッション値、ダートは含水率です。'
             '確定オッズに足して、勝ち馬の当たり方が良くなるかを見ています。</p>')
    if st1.get("B"):
        b = st1["B"]["H_front_x_cv"]
        h.append(f'<p class="mut">下調べ: 「前に行く馬 × 硬い馬場」も確定オッズへの上乗せは無し(芝 {b["n_races"]:,}レース、ΔLL {f2(b["dll"])})。'
                 '4角先頭の馬の3着内率は柔らかめ 49%・標準 47%・硬め 47% で、「硬いほど前残り」は2025〜2026年では見られませんでした。</p>')
    if dev3.get("main"):
        h.append('<h3>開発(2025-01〜2026-08、縮め k=50)</h3>')
        names = {"芝_F1": "芝 父×クッション値", "芝_F2": "芝 母の父×クッション値", "芝_F3": "芝 馬自身×クッション値", "芝_all": "芝 3つまとめて",
                 "ダ_F1": "ダート 父×含水率", "ダ_F2": "ダート 母の父×含水率", "ダ_F3": "ダート 馬自身×含水率", "ダ_all": "ダート 3つまとめて"}
        hyp_rows(h, [(names[k], v, False) for k, v in dev3["main"].items() if k in names])
    if any(C.values()):
        h.append('<h3>確認(2020-01〜2024-12、縮め k=200、定義は開発で凍結)</h3>')
        rows = []
        for k, lab in (("C1", "C1 芝 馬自身×クッション値(主仮説)"), ("C2", "C2 芝 母の父×クッション値"), ("C3", "C3 芝 3つまとめて"),
                       ("C4", "C4 ダート 3つまとめて(含水率)"), ("C5", "C5 オッズを使わない着順予想 [u_FINAL] + 芝 3つ(2020年)")):
            if C[k]:
                rows.append((lab, C[k], k == "C1"))
        hyp_rows(h, rows)
        frs = [(lab, r["dll"], r["lo"], r["hi"], r["placebo_max"], prim) for lab, r, prim in rows]
        h.append(f'<div class="fig">{BR.forest(frs)}</div>')
        h.append('<div class="legend"><span><i style="background:var(--acc);border-radius:50%"></i>ΔLL と95%区間</span>'
                 '<span><i style="background:var(--mark);transform:rotate(45deg)"></i>プラセボ20回の最大</span></div>')
        if cb:
            h.append(f'<p class="mut">確認の走数 {cb["n_runs"]:,}。馬場の値が分かる走は芝 {pct(cb["share_x_known_turf"], 0)}・ダート {pct(cb["share_x_known_dirt"], 0)}。'
                     '馬場の偏差はその日より前の開催日だけの平均から計算し、SD は開発期間の値で固定しました。'
                     'C5 は確定オッズの代わりに finish_v4(オッズを使わない着順予想)の点数を基準にしています。</p>')
    h.append('</section>')

    # ---- 盛り込み方
    h.append('<section><h2>盛り込み方</h2><ul>')
    h.append('<li>当日の予想・表示: 開催日朝のクッション値・含水率(fetch_jra_baba_live.py)をそのまま使う。</li>')
    h.append('<li>前夜の予想・表示: 手順2のモデル(前夜版)の予想値を使い、朝の発表で差し替える。ダートの含水率は当日朝版(7時までの雨)が特に効く。</li>')
    if c1 and not c1["pass"]:
        h.append('<li>馬ごとの向き不向きは、確定オッズへの上乗せが確かめられなかったので、本番の印・見送りの判断には入れない。'
                 '表示する場合は「参考」とする。</li>')
    elif c1:
        h.append('<li>馬自身×クッション値は確認で合格。本番に入れる前に、前向きの記録(発走前の値で計算)で確かめる。</li>')
    h.append('<li>確定オッズに上乗せしにくい理由: クッション値は当日朝に公開され、馬場状態の発表や当日の先行レースの結果とともに、投票する人にも見えているため。</li>')
    h.append('</ul></section>')

    # ---- 読み方の注意・再現
    h.append('<section><h2>読み方の注意</h2><ul>'
             '<li>2024年以前のアーカイブには使用コース(A〜D)と測定時刻が無い。コースの特徴は0、測定時刻は金曜 9:30・開催日 8:00 とみなした。</li>'
             '<li>気象庁の観測点は競馬場の最寄り(0.8〜17.6km)。新潟の雨は11.4km離れた松浜。</li>'
             '<li>前夜の天気予報は気象庁が過去分を出さないため、2026-10-13 からの記録しか無い。</li>'
             '<li>開発と確認は時期が違う(確認の方が昔)。馬ごとの特徴は、確認期間の中だけの成績で作っている。</li></ul></section>')
    h.append('<section><h2>再現</h2><p class="mut">スクリプト: <span class="code">scripts/jra_model/cushion_2026_10_13/</span>'
             '(step1_feasibility.py・cushion_forecast.py・horse_features.py・horse_confirm.py・report_cushion.py)<br>'
             '結果: <span class="code">data/jra_pipeline/cushion_2026_10_13/</span>(forecast_result_*.json・horse_result.json・confirm_C*.json)<br>'
             '事前登録: <span class="code">PREREG_cushion_2026_10_13.md</span>(追補1 開発の結果と凍結、追補2 確認の実行前の変更)<br>'
             '使ったデータ: data/jra_baba(クッション値・含水率)、data/jma_amedas(気象庁)、data/race_results、血統の父・母の父(radar_v2 ped_ids)</p></section>')
    h.append('</main>')
    (OUT / "report.html").write_text("\n".join(h), encoding="utf-8")
    print("書き出し:", OUT / "report.html")


if __name__ == "__main__":
    main()
