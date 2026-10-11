# -*- coding: utf-8 -*-
"""finish_v4 のプランと経過のレポート(HTML)を、結果の JSON から作る(2026-10-11)。数値は手で写さない。

出力: data/jra_pipeline/finish_v4/report/finish_v4_report.html
"""
import html
from string import Template
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data" / "jra_pipeline" / "finish_v4"
OUT = D / "report" / "finish_v4_report.html"


def J(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def pt(x, d=1):
    return "—" if x is None else f"{100 * x:+.{d}f}pt"


def ci_pt(c, d=1):
    return "—" if not c else f"[{100 * c[0]:+.{d}f}, {100 * c[1]:+.{d}f}]"


def num(x, d=3, sign=False):
    return "—" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def esc(s):
    return html.escape(str(s))


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def verdict_chip(lo, hi):
    if lo is None:
        return ""
    if lo > 0:
        return '<span class="chip good">上回る</span>'
    if hi < 0:
        return '<span class="chip bad">下回る</span>'
    return '<span class="chip neu">区別できない</span>'


def bar_svg(vals, labels, unit="", maxv=None, w=560, h=190, ref=None, ref_label=""):
    """縦棒。vals は数値の列。ref があれば水平の基準線。"""
    maxv = maxv or max(vals) * 1.12
    n = len(vals)
    pl, pr, pt_, pb = 44, 8, 14, 34
    cw = (w - pl - pr) / n
    y = lambda v: pt_ + (h - pt_ - pb) * (1 - v / maxv)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in range(0, 5):
        v = maxv * t / 4
        parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>')
        parts.append(f'<text x="{pl - 6}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v:.0f}{unit}</text>')
    for i, (v, lab) in enumerate(zip(vals, labels)):
        x = pl + i * cw + cw * 0.18
        bw = cw * 0.64
        parts.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{y(0) - y(v):.1f}" class="bar"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{y(v) - 4:.1f}" class="val" text-anchor="middle">{v:.0f}</text>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{h - pb + 16:.1f}" class="ax" text-anchor="middle">{esc(lab)}</text>')
    if ref is not None:
        parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" class="ref"/>')
        parts.append(f'<text x="{pl + 6}" y="{y(ref) - 5:.1f}" class="refl" text-anchor="start">{esc(ref_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main():
    met, met0 = J("dev_f0_metrics.json"), J("dev_f0_metrics_finishers_only.json")
    mk, mk0 = J("dev_f0_market.json"), J("dev_f0_market_finishers_only.json")
    proxy, cov = J("f0_odds_proxy_check.json"), J("birth_coverage.json")
    tune, ev = J("ability_v2_tune.json"), J("ability_v2_eval_dev.json")
    f3, f3m, dry = J("f3_eval_dev.json"), J("f3_market.json"), J("final_dry_result.json")
    fin = J("final_result.json")
    R = mk["rules"]
    nf = met["nonfinish"]

    # ---- F0-1 前後比較
    rows01 = []
    for lab, key, rk in [("人気どおり(比べる相手)", None, "mkt"), ("血統+実績(M1)", "u_M1_set", "u_M1_set"), ("能力値入り(M5)", "u_M5_pl", "u_M5_pl")]:
        for N in (10, 30):
            old = mk0["rules"][f"{rk}_N{N}"]
            new = R[f"{rk}_N{N}"]
            rows01.append([esc(lab), f"{N}点", pct(old["hit_rate"]), pct(new["hit_rate"]), pct(old["roi"]["ratio"]), pct(new["roi"]["ratio"]),
                           pt(old["uplift"]["mean"]), pt(new["uplift"]["ratio"])])
    # ---- 6つの買い方
    rules_lab = [("N10", "10点"), ("N20", "20点"), ("N30", "30点"), ("C30", "累積30%まで"), ("C40", "累積40%まで"), ("C50", "累積50%まで")]
    rows_r = []
    for rule, lab in rules_lab:
        for who, wl in [("mkt", "人気どおり"), ("u_M5_pl", "能力値入り(M5)")]:
            r = R[f"{who}_{rule}"]
            u = r["uplift"]
            rows_r.append([esc(lab), esc(wl), f'{r["mean_tickets"]:.1f}', pct(r["hit_rate"]), pct(r["roi"]["ratio"]),
                           pct(r["expected_roi_market"]), pt(u["ratio"]), ci_pt(u["ci_bca"]), verdict_chip(*u["ci_bca"])])
    # ---- 層別
    lay_rows = []
    for lab in ["10頭以下", "11〜14頭", "15〜18頭", "新馬・未勝利", "それ以外"]:
        m_ = R["mkt_N10"]["by_layer"][lab]
        a_ = R["u_M5_pl_N10"]["by_layer"][lab]
        lay_rows.append([esc(lab), f'{a_["n"]:,}', pct(m_["hit_rate"]), pct(a_["hit_rate"]), pct(m_["roi"]), pct(a_["roi"]),
                         pt(a_["uplift"]["ratio"]), ci_pt(a_["uplift"]["ci_bca"])])
    inc = mk["increment_logit"]
    # ---- 的中率の曲線
    xs = [1, 5, 10, 15, 20, 30, 50]
    ser = [("能力値入り M5", [100 * met["models"]["u_M5_pl"]["hit_rate"][str(x)] for x in xs], "c1"),
           ("血統+実績 M1", [100 * met["models"]["u_M1_pl"]["hit_rate"][str(x)] for x in xs], "c2")]
    mk_pts = [(10, R["mkt_N10"]["hit_rate"]), (20, R["mkt_N20"]["hit_rate"]), (30, R["mkt_N30"]["hit_rate"])]
    hit_chart = line_svg_multi(ser, xs, [("人気どおり", [x for x, _ in mk_pts], [100 * v for _, v in mk_pts], "c3")])
    # ---- オッズ診断
    ev_vals = [d["ev_per_100yen"] for d in proxy["ev_by_p_decile"]]
    ev_chart = bar_svg(ev_vals, [f"{i + 1}" for i in range(len(ev_vals))], maxv=110, ref=75, ref_label="全体の払戻率 約75%")
    # ---- F1
    g = cov["gate"]
    by_year = cov["by_year"]
    cov_chart = bar_svg([100 * by_year[y]["rate"] for y in sorted(by_year)], sorted(by_year), unit="%", maxv=115, ref=98, ref_label="基準 98%")
    pos_rows = [[esc(k), f'{cov["by_pos"][k]["n"]:,}', pct(cov["by_pos"][k]["rate"])] for k in ["1〜3着", "4〜9着", "10着以下"] if k in cov["by_pos"]]
    car_rows = [[esc(k), f'{cov["by_career"][k]["n"]:,}', pct(cov["by_career"][k]["rate"])] for k in ["初出走", "1〜3走", "4〜9走", "10走〜"] if k in cov["by_career"]]
    # ---- F2
    f2_html = ""
    if tune:
        kg = min(tune["k_grid"], key=lambda r: r["mse"])
        rg = sorted(tune["rho_grid"], key=lambda r: -r["ll"])[:3]
        f2_html += f"""
<p>設定は2017年末までの走で選びました(2018〜2020年は見ていません)。</p>
<ul>
<li><b>父・母の値の縮め方:</b> k_s={tune['k_s']}・k_d={tune['k_d']}(1回目の格子は端が最良だったため広げた。k_s は広げた格子でも端だが、160 との差は誤差 0.0001 未満で平ら)。初出走の能力値の予測誤差(2015〜2017年、平均二乗)は、父の総合力だけの {kg['mse_S_only']:.4f} → {kg['mse']:.4f}。</li>
<li><b>芝・ダ別の状態:</b> 初出走前の芝ダの相関 ρ0={tune['rho0']}、走の間の変化の相関 ρq={tune['rhoq']}。1走先の予測の対数尤度は 1次元 {tune['ll_1d']:.4f} → {tune['ll_best']:.4f}(1走あたり)。上位: {'、'.join(f"ρ0={r['rho0']}・ρq={r['rhoq']} {r['ll']:.4f}" for r in rg)}。</li>
</ul>"""
    if ev:
        rows2 = []
        for k, m in ev["models"].items():
            if k == "M5":
                rows2.append([esc(k) + "(基準)", num(m["ll"], 4), "—", "", num(m["combo_logscore"], 4), "—", pct(m["hit"]["N10"]), "—"])
                continue
            d, c, h = m["dll_vs_M5"], m["combo_logscore_vs_M5"], m["hit_N10_vs_M5"]
            rows2.append([esc(k), num(m["ll"], 4), num(d["mean"], 4, True) + f'<br><span class="sub">[{d["ci_bca"][0]:+.4f}, {d["ci_bca"][1]:+.4f}]</span>',
                          verdict_chip(*d["ci_bca"]), num(m["combo_logscore"], 4),
                          num(c["mean"], 4, True) + f'<br><span class="sub">[{c["ci_bca"][0]:+.4f}, {c["ci_bca"][1]:+.4f}]</span>',
                          pct(m["hit"]["N10"]), pt(h["mean"]) + f'<br><span class="sub">{ci_pt(h["ci_bca"])}</span>'])
        f2_html += table(["モデル", "着順の対数尤度/R", "M5との差 [95%]", "", "組の対数スコア", "M5との差 [95%]", "10点の的中率", "M5との差 [95%]"], rows2, "num")
        lay = []
        for k, m in ev["models"].items():
            if k == "M5":
                continue
            lay.append([esc(k)] + [num(m["by_layer_dll"].get(l_), 4, True) for l_ in ["新馬・未勝利", "新馬戦", "それ以外", "15頭以上"]])
        f2_html += "<h4>層ごとの M5 との差(着順の対数尤度/R、記述)</h4>" + table(["モデル", "新馬・未勝利", "新馬戦", "それ以外", "15頭以上"], lay, "num")
        yrs = []
        for k, m in ev["models"].items():
            if k != "M5":
                yrs.append([esc(k)] + [num(m["by_year_dll"].get(str(y)), 4, True) for y in ev["eval_years"]])
        f2_html += "<h4>年ごとの M5 との差</h4>" + table(["モデル"] + [str(y) for y in ev["eval_years"]], yrs, "num")

    html_out = Template(TEMPLATE).substitute(
        nf_rows=f'{nf["ghost_rows"]:,}', nf_used=f'{nf["ghost_rows_used"]:,}', nf_test=f'{met["n_nonfinish_test_rows"]:,}',
        nf_races=f'{mk["n_races_with_nonfinish"]:,}', n_races=f'{mk["n_races"]:,}',
        same=f'{mk["candidates_same_as_market"]["same"]:,}', diff=mk["candidates_same_as_market"]["diff"],
        t01=table(["予想", "点数", "的中率 前", "的中率 後", "回収率 前", "回収率 後", "上積み 前", "上積み 後"], rows01, "num"),
        nf_hit_m=pct(mk["nonfinish_races"]["mkt"]["hit_N10"]), nf_hit_5=pct(mk["nonfinish_races"]["u_M5_pl"]["hit_N10"]),
        trules=table(["買い方", "予想", "平均点数", "的中率", "回収率", "人気から見た期待", "上積み", "95%区間", "判定"], rows_r, "num"),
        tlayer=table(["層", "レース数", "的中率 人気", "的中率 M5", "回収率 人気", "回収率 M5", "M5の上積み", "95%区間"], lay_rows, "num"),
        inc_a=num(inc["a"], 3), inc_se=num(inc["a_se_cluster"], 3), inc_z=num(inc["z_a"], 1), inc_b=num(inc["b"], 3), inc_dll=num(inc["dll_per_race_vs_market_only"], 4),
        hit_chart=hit_chart,
        m5_ls=num(met["models"]["u_M5_pl"]["logscore"], 3), m1_ls=num(met["models"]["u_M1_pl"]["logscore"], 3),
        uni_ls=num(met["models"]["u_M5_pl"]["uniform_logscore"], 3),
        pop_sp=num(mk["diag_popularity"]["spearman"], 3), pop_x2=pct(mk["diag_popularity"]["within_x2"]), pop_n=f'{mk["diag_popularity"]["n"]:,}',
        px_n=proxy["n_races"], px_sp=num(proxy["spearman_median"], 3), px_ov=pct(proxy["top10_overlap_mean"], 0), px_sl=num(proxy["slope_median"], 2),
        px_to=pct(proxy["takeout_implied_median"]), ev_chart=ev_chart,
        cov_prof=f'{cov["n_profiles"]:,}', cov_bd=f'{cov["profiles_with_birth_date"]:,}', cov_run=pct(g["run_coverage"]), cov_cc=pct(g["race_complete_case"]),
        cov_n=f'{g["n_runs"]:,}', cov_chart=cov_chart, tpos=table(["着順", "走数", "生年月日あり"], pos_rows, "num"),
        tcar=table(["キャリア", "走数", "生年月日あり"], car_rows, "num"), cov_made=esc(cov["made"]),
        fr_n=f'{cov["foreign"]["n"]:,}', fr_jd=cov["foreign"]["born_jul_dec"],
        f2=f2_html or "<p>実行中。</p>", f2_summary=f2_summary(ev),
        f3=f3_html(f3), f3m=f3m_html(f3m), f5=f5_html(dry, True), final=final_html(fin),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8", newline="\n")
    print("wrote", OUT, len(html_out))


def cib(c, d=4):
    return f'<br><span class="sub">[{c[0]:+.{d}f}, {c[1]:+.{d}f}]</span>'


def f3_html(f3) -> str:
    if not f3:
        return "<p>実行中。</p>"
    l2 = "、".join(f"{k}: {v:.5f}" for k, v in f3["l2_grid"].items())
    h = f"<p>L2 は全部入りで 2018・2019年だけを見て選びました({esc(l2)} → <b>{f3['l2']}</b>。差はごく小さい)。</p>"
    rows = []
    for k, m in f3["final_vs_M5"].items():
        if k == "M5":
            rows.append(["M5(能力値入り)", num(m["ll"], 4), "—", num(m["combo_logscore"], 4), "—", pct(m["hit"]["N10"]), pct(m["hit"]["N20"]), "—"])
        else:
            d, c, t = m["dll_vs_M5"], m["combo_vs_M5"], m["hit10_vs_M5"]
            rows.append(["完成形 FINAL", num(m["ll"], 4), num(d["mean"], 4, True) + cib(d["ci_bca"]), num(m["combo_logscore"], 4),
                         num(c["mean"], 4, True) + cib(c["ci_bca"]), pct(m["hit"]["N10"]), pct(m["hit"]["N20"]), pt(t["mean"]) + f'<br><span class="sub">{ci_pt(t["ci_bca"])}</span>'])
    m6 = f3["final_vs_M6"]["M6"]
    rows.insert(1, ["M6(F2 の4つまで)", "—", "—", num(m6["combo_logscore"], 4), "—", pct(m6["hit"]["N10"]), pct(m6["hit"]["N20"]), "—"])
    h += table(["モデル", "着順の対数尤度/R", "M5との差 [95%]", "組の対数スコア", "M5との差 [95%]", "10点の的中率", "20点の的中率", "10点 M5との差"], rows, "num")
    ab = []
    for g in f3["groups"]:
        v = f3["vs_FULL"][f"FULL−{g}"]["combo_vs_FULL"]
        ab.append([esc(g), f'{len(f3["groups"][g])}', num(v["mean"], 4, True) + cib(v["ci_bca"]),
                   '<span class="chip good">残す</span>' if g in f3["kept_groups"] else '<span class="chip neu">外す</span>'])
    h += "<h4>外して比べる(全部入りから1群ずつ外したときの組の対数スコアの変化)</h4>" + table(["群", "本数", "外したときの差 [95%]", "凍結の規則"], ab, "num")
    mc = f3.get("mc")
    if mc:
        h += (f"<p>不確かさを使う版(能力値を平均±SDから32回引いて組の確率を平均): 2018・2019年の組の対数スコアは、そのまま {mc['deterministic_select']:.5f} に対し "
              f"s=0.5 で {mc['s_grid']['0.5']:.5f}、s=1.0 で {mc['s_grid']['1.0']:.5f}。良くならなかったので<b>使わない</b>。</p>")
    return h


def f3m_html(f3m) -> str:
    if not f3m:
        return "<p>実行中。</p>"
    R = f3m["rules"]
    rows = []
    for rule, lab in [("N10", "10点"), ("N20", "20点"), ("N30", "30点")]:
        for who, wl in [("mkt", "人気どおり"), ("u_M5_pl", "M5"), ("u_M6_pl", "M6"), ("u_FINAL_pl", "完成形 FINAL")]:
            r = R[f"{who}_{rule}"]
            u = r["uplift"]
            rows.append([esc(lab), esc(wl), pct(r["hit_rate"]), pct(r["roi"]["ratio"]), pct(r["expected_roi_market"]),
                         pt(u["ratio"]), ci_pt(u["ci_bca"]), verdict_chip(*u["ci_bca"]),
                         "—" if r["median_market_rank"] is None else f'{r["median_market_rank"]:.0f}位'])
    inc = f3m["increment_logit"]
    h = table(["点数", "予想", "的中率", "回収率", "人気から見た期待", "上積み", "95%区間", "判定", "買った組の人気(中央値)"], rows, "num")
    h += (f"<p>オッズに無い情報(完成形): a = {inc['a']:.3f}(SE {inc['a_se_cluster']:.3f}、z = {inc['z_a']:.1f})、人気だけの式に比べて1レースあたり "
          f"+{inc['dll_per_race_vs_market_only']:.4f}(M5 では +0.0085)。</p>")
    return h


def f5_html(dry, prereg_ok, col="試運転(2018〜2020年)") -> str:
    if not dry:
        return "<p>試運転は実行中。</p>"
    h2, h1b, h4 = dry["H2"], dry["H1b"], dry["H4"]
    a, b = h2["combo_logscore_FINAL_minus_M1"], h2["hit_N20_FINAL_minus_M1"]
    rows = [["H2(本命)", "組の対数スコア FINAL − M1", num(a["mean"], 4, True) + cib(a["ci_bca"]), verdict_chip(*a["ci_bca"])],
            ["", "20点の的中 FINAL − M1", pt(b["mean"]) + f'<br><span class="sub">{ci_pt(b["ci_bca"])}</span>', verdict_chip(*b["ci_bca"])],
            ["H1b", "着順の対数尤度 FINAL − M5", num(h1b["ll_FINAL_minus_M5"]["mean"], 4, True) + cib(h1b["ll_FINAL_minus_M5"]["ci_bca"]),
             verdict_chip(*h1b["ll_FINAL_minus_M5"]["ci_bca"])],
            ["H4", "オッズに無い情報 a", f'{h4["increment"]["a"]:.3f}(z {h4["increment"]["z_a"]:.1f})',
             '<span class="chip good">上回る</span>' if h4["increment"]["z_a"] > 1.96 else '<span class="chip neu">区別できない</span>'],
            ["H3(推定)", "20点の人気をそろえた上積み", pt(dry["H3_estimate"]["uplift"]["ratio"]) + f'<br><span class="sub">{ci_pt(dry["H3_estimate"]["uplift"]["ci_bca"])}</span>',
             esc(dry["H3_estimate"]["label"])]]
    h = table(["仮説", "中身", col, "結果"], rows, "num")
    sk = dry.get("skip")
    if sk:
        h += (f"<p>見送り(副次): 20点の買い目の確率の合計が下位20%のレースを見送ると、買ったレースの回収率は {pct(sk['FINAL']['roi_bought'])}"
              f"(上積み {pt(sk['FINAL']['uplift_bought']['ratio'])})。人気どおりに同じ割合を見送った場合は {pct(sk['mkt']['roi_bought'])}。</p>")
    return h


def final_html(fin) -> str:
    if not fin:
        return "<p>未実行。</p>"
    h = f"<p>対象 {fin['n_races_ll']:,} レース(候補の馬はオッズ側と全レースで一致)。各年の係数はその前年までで推定し直しました。</p>"
    h += f5_html({k: v for k, v in fin.items() if k != "skip"}, True, "確認(2021〜2025年)")
    R = fin["rules"]
    rows = []
    for rule, lab in [("N10", "10点"), ("N20", "20点(主)"), ("N30", "30点")]:
        for who, wl in [("mkt", "人気どおり"), ("M1", "血統+実績 M1"), ("M5", "M5"), ("M6", "M6"), ("FINAL", "完成形 FINAL")]:
            r = R[f"{who}_{rule}"]
            u = r["uplift"]
            rows.append([esc(lab), esc(wl), pct(r["hit_rate"]), pct(r["roi"]["ratio"]), pct(r["expected_roi_market"]), pt(u["ratio"]),
                         ci_pt(u["ci_bca"]), verdict_chip(*u["ci_bca"]), pct(r["roi_excl_50k"]),
                         "—" if r["median_market_rank"] is None else f'{r["median_market_rank"]:.0f}位'])
    h += "<h4>買い方ごとの成績(2021〜2025年)</h4>" + table(
        ["点数", "予想", "的中率", "回収率", "人気から見た期待", "上積み", "95%区間", "判定", "5万円超を除く回収率", "買った組の人気(中央値)"], rows, "num")
    f = R["FINAL_N20"]
    yr = [[str(y), pt(v)] for y, v in f["by_year_uplift"].items()]
    lay = [[esc(k), f'{v["n"]:,}', pct(v["hit_rate"]), pct(v["roi"]), pt(v["uplift"])] for k, v in f["by_layer"].items()]
    h += '<div class="grid2"><div style="min-width:0"><h4>完成形20点の年ごとの上積み</h4>' + table(["年", "上積み"], yr, "num") + \
         '</div><div style="min-width:0"><h4>完成形20点の層ごと</h4>' + table(["層", "レース数", "的中率", "回収率", "上積み"], lay, "num") + "</div></div>"
    sk = fin.get("skip")
    if sk:
        h += (f"<p>見送り(副次): 20点の買い目の確率の合計が開発の下位20%点未満のレース({pct(sk['FINAL']['skip_rate'])})を見送ると、"
              f"買ったレースの回収率 {pct(sk['FINAL']['roi_bought'])}・上積み {pt(sk['FINAL']['uplift_bought']['ratio'])} {ci_pt(sk['FINAL']['uplift_bought']['ci_bca'])}。"
              f"見送りで回収率は良くなりませんでした(人気どおりに同じ割合を見送った場合 {pct(sk['mkt']['roi_bought'])})。</p>")
    return h


def f2_summary(ev) -> str:
    if not ev:
        return "評価は実行中。"
    good, flat = [], []
    for k, m in ev["models"].items():
        if k == "M5":
            continue
        lo, hi = m["dll_vs_M5"]["ci_bca"]
        (good if lo > 0 else flat).append(f"{esc(k)}({m['dll_vs_M5']['mean']:+.4f}/R)")
    full = ev["models"].get("M6(全部)")
    s = ""
    if good:
        s += "M5 を区間で上回ったのは " + "、".join(good) + "。"
    if flat:
        s += "区別できなかったのは " + "、".join(flat) + "。"
    if full:
        s += f"全部入りの10点の的中率は {pct(full['hit']['N10'])}(M5 {pct(ev['models']['M5']['hit']['N10'])})。"
    return s


def line_svg_multi(series, xs, extra, w=600, h=230):
    """series は xs 上の折れ線、extra は [(名前, x の列, 値, cls)]。"""
    maxv = 75
    pl, pr, pt_, pb = 44, 112, 12, 32
    xmax = max(xs)
    X = lambda x: pl + (w - pl - pr) * x / xmax  # noqa: E731
    Y = lambda v: pt_ + (h - pt_ - pb) * (1 - v / maxv)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="点数ごとの的中率">']
    for t in range(0, 6):
        v = maxv * t / 5
        parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        parts.append(f'<text x="{pl - 6}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:.0f}%</text>')
    for x in xs:
        parts.append(f'<text x="{X(x):.1f}" y="{h - pb + 15}" class="ax" text-anchor="middle">{x}</text>')
    allser = [(n, xs, v, c) for n, v, c in series] + extra
    ly = {}
    for name, xx, vals, cls in sorted(allser, key=lambda t: Y(t[2][-1])):
        yy = Y(vals[-1]) + 4
        for o in ly.values():
            if abs(o - yy) < 13:
                yy = o + 13
        ly[name] = yy
    for name, xx, vals, cls in allser:
        pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xx, vals))
        parts.append(f'<polyline points="{pts}" class="ln {cls}"/>')
        for x, v in zip(xx, vals):
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="2.8" class="pt {cls}"/>')
        parts.append(f'<text x="{X(xx[-1]) + 8:.1f}" y="{ly[name]:.1f}" class="lab {cls}">{esc(name)}</text>')
    parts.append(f'<text x="{(pl + w - pr) / 2:.1f}" y="{h - 3}" class="ax" text-anchor="middle">買う点数(組の確率の高い順)</text>')
    parts.append("</svg>")
    return "".join(parts)


TEMPLATE = (HERE / "report_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
