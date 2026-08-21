# -*- coding: utf-8 -*-
"""JRA「1頭軸流し」予想の回収率検証レポート(scripts/build_artifact_nar.pyのCSS/タブUIを
踏襲した独立ページ)。既存の「馬柱データ予想」レポート(build_artifact.py、scratchpad常駐)とは
別の新規レポートとして構築する(ユーザー依頼: 2026-08-21)。

対象は軸(スコア1位)+相手(スコア2位以降)の「ながし」買い、馬連・ワイド・3連複・
馬単(軸流し/マルチ)・3連単(軸流し/マルチ)の7区分。軸+相手の合計頭数はaxis5(軸+相手4頭)・
axis4(軸+相手3頭)・axis3(軸+相手2頭)の3サイズ。

2026-08-21の検証結果の要点(scripts/jra_model/jra_axis_search_2026_08_21.pyで実施、
統計学者・競馬予想家の2専門家レビュー済み):
  - 軸流し専用の新規500パターン重み探索は3サイズともREJECTED(選択バイアス診断の
    true_edge/sd比が採否ゲート2.0を大幅未達)。
  - 現行box5/4/3の重みをそのまま軸流しに転用した場合、全211レースでは市場を+18〜23pt
    上回るが、この数値は重み自体のfit母集団(105レース)を含んでおり二重に楽観的。
    fit母集団を除外した公正な評価(106レース)では市場超過が消え、点推定はむしろ
    マイナス(-13.66〜-19.50pt、いずれも95%CIは0をまたぎ統計的有意ではない)に転じる。
  - 結論: 「市場に対する優位性が統計的に実証されたモデル」は今回の検証では見つからなかった。
本レポートはこの結果を隠さず、現行box重み転用モデル(実用上の最有力候補)の回収率検証を
主表として示しつつ、上記の限界を明記した透明な記録として構築する
([[project_nar_search500_v4_signals_2026_08_20_finding]]と同じ方針: 統計的に無効な
「おすすめパターン」を本番同格で見せない)。

**追記(2026-08-21同日、ユーザー追加依頼)**: 「軸(スコア1位)が3着以内(複勝圏内)に来る
確率を最大化する」重み探索(scripts/jra_model/jra_axis_top3_search_2026_08_21.py、目的関数を
ワイド回収率から複勝的中率へ差し替え)も実施。統計学者・競馬予想家の2専門家レビュー済み。
市場(1番人気を軸)の複勝的中率57.82%に対し、500パターン探索の最良でも54.03%止まりで、
選択バイアス診断は再びREJECTED(true_edge/sd比0.336)。3つ目の独立した目的関数でも
「市場を統計的に上回るモデルは見つからない」という同一結論に達した。
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

BOX_SIZES = [5, 4, 3]
BET_ORDER = ["馬連", "ワイド", "3連複", "馬単_軸流し", "馬単_マルチ", "3連単_軸流し", "3連単_マルチ"]
BET_ORDER_MAP = {b: i for i, b in enumerate(BET_ORDER)}
BET_LABEL = {
    "馬連": "馬連", "ワイド": "ワイド", "3連複": "3連複",
    "馬単_軸流し": "馬単(軸流し)", "馬単_マルチ": "馬単(マルチ)",
    "3連単_軸流し": "3連単(軸流し)", "3連単_マルチ": "3連単(マルチ)",
}
LOW_SAMPLE_HITS = 10


def esc(s) -> str:
    import html
    return html.escape(str(s), quote=True)


DECISION_LABEL_JA = {"REJECTED": "不採用", "ADOPT_CANDIDATE": "採用"}


def decision_ja(decision: str) -> str:
    return DECISION_LABEL_JA.get(decision, decision)


def short_scope_label(scope: str) -> str:
    m = re.match(r"^全(\d+)レース$", scope)
    if m:
        return f"全{m.group(1)}R"
    m = re.match(r"^高確信度(\d+)レース/日", scope)
    if m:
        return f"{m.group(1)}R/日"
    return scope


def rate_cls(rate: float) -> str:
    return "is-plus" if rate >= 100 else ("is-mid" if rate >= 80 else "is-minus")


def render_table_rows(sub: pd.DataFrame) -> str:
    sub = sub.copy()
    sub["_order"] = sub["bet_type"].map(BET_ORDER_MAP)
    sub = sub.sort_values("_order")
    rows_html = []
    for _, r in sub.iterrows():
        rate = r["return_rate_pct"]
        hit_rate = r["hit_rate_pct"]
        low = "" if r["hit_races"] >= LOW_SAMPLE_HITS else ' <span class="box-sub">(参考値)</span>'
        rows_html.append(f"""<tr>
          <td class="bt-name">{esc(BET_LABEL.get(r['bet_type'], r['bet_type']))}{low}</td>
          <td>{int(r['hit_races'])}/{int(r['races'])}<span class="box-sub">({hit_rate:.1f}%)</span></td>
          <td class="num">¥{int(r['total_stake']):,}</td>
          <td class="num">¥{int(r['total_return']):,}</td>
          <td class="num rate {rate_cls(rate)}">{rate:.1f}%</td>
        </tr>""")
    return "".join(rows_html)


def build_axis_section(box_n: int) -> tuple[str, str]:
    csv_path = DATA_DIR / f"confidence_sweep_axis{box_n}.csv"
    conf = pd.read_csv(csv_path, dtype=str)
    for c in ["races", "hit_races", "total_stake", "total_return", "hit_rate_pct", "return_rate_pct"]:
        conf[c] = pd.to_numeric(conf[c])
    models = list(dict.fromkeys(conf["model"]))
    current_model = next(m for m in models if "現行box" in m and "探索候補" not in m)
    candidate_model = next((m for m in models if "探索候補" in m), None)
    market_model = next((m for m in models if "市場ベンチマーク" in m), None)

    cur = conf[conf["model"] == current_model]
    scopes = list(dict.fromkeys(cur["scope"]))

    id_prefix = f"axis{box_n}"
    tab_elems, tab_panels, tab_css_rules = [], [], []
    for i, scope in enumerate(scopes):
        tab_id = f"conf-tab-{id_prefix}-{i}"
        panel_id = f"conf-panel-{id_prefix}-{i}"
        checked = " checked" if i == 0 else ""
        tab_elems.append(
            f'<input type="radio" name="conf-tab-{id_prefix}" id="{tab_id}"{checked} class="conf-tab-input">'
            f'<label for="{tab_id}" class="conf-tab-label">{esc(short_scope_label(scope))}</label>')
        sub = cur[cur["scope"] == scope]
        tab_panels.append(f"""<div class="conf-tab-panel" id="{panel_id}">
          <p class="conf-scope-label">{esc(scope)}</p>
          <div class="box-table-wrap">
            <table class="box-table">
              <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
              <tbody>{render_table_rows(sub)}</tbody>
            </table>
          </div>
        </div>""")
        tab_css_rules.append(f"#{tab_id}:checked ~ .conf-tabs-panels #{panel_id} {{ display: block; }}")

    n_races_full = int(cur[cur["scope"].str.startswith("全")]["races"].iloc[0])

    # 参考: 市場ベンチマーク(全レースのみ)
    market_html = ""
    if market_model is not None:
        mkt_full = conf[(conf["model"] == market_model) & (conf["scope"].str.startswith("全"))]
        market_html = f"""
        <details class="method box-method">
          <summary>参考: 市場ベンチマーク(軸=1番人気・相手=2〜{box_n}番人気)との比較、全{n_races_full}レース</summary>
          <div class="box-table-wrap">
            <table class="box-table">
              <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
              <tbody>{render_table_rows(mkt_full)}</tbody>
            </table>
          </div>
        </details>"""

    # 参考: 軸流し専用探索候補(不採用)
    candidate_html = ""
    if candidate_model is not None:
        cand_full = conf[(conf["model"] == candidate_model) & (conf["scope"].str.startswith("全"))]
        candidate_html = f"""
        <details class="method box-method">
          <summary>参考・不採用: {esc(candidate_model)}、全{n_races_full}レース</summary>
          <p class="method-note">
            2026-08-21に軸流し専用として500パターンの重み配分を新規探索しましたが、選択バイアス
            診断(ブロック半分割×200反復)のtrue_edge/sd比が採否ゲート(2.0以上)を全サイズで
            大幅に下回り不採用としました。以下は探索結果の中で最良だった1パターン(in-sample選択、
            楽観バイアスを含む参考値)の内訳です。
          </p>
          <div class="box-table-wrap">
            <table class="box-table">
              <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
              <tbody>{render_table_rows(cand_full)}</tbody>
            </table>
          </div>
        </details>"""

    partner_n = box_n - 1
    lede = (
        f"各レースの予想スコア1位を軸、2〜{box_n}位({partner_n}頭)を相手にした「ながし」買いの、"
        f"実際の払い戻しとの答え合わせです(全{n_races_full}レース、現行box{box_n}モデルの重みを"
        "そのまま軸流しに転用)。"
        "<br><br><b>2026-08-21の検証で判明した限界(必ずお読みください)</b>: "
        "上表の回収率は、重み自体の決定に使われた105レース(6開催日)を含む全レースでの評価であり、"
        "選定に使ったデータで選定結果を評価する楽観を含みます。この重複を除いた106レースだけの"
        "公正な評価では市場超過が消え、点推定はむしろマイナスに転じました(詳細は下記「計算方法・"
        "前提条件」参照)。軸流し専用の新規500パターン重み探索も統計的に不採用でした。つまり"
        "「市場に対する優位性が統計的に実証されたモデル」は今回の検証では見つかっていません。"
        "上表はあくまで実用上の最有力候補(現行box重みの転用)の透明な検証記録として掲載しています。"
    )
    method_note_extra = {
        5: "-13.66pt(95%CI=[-36.58,+8.61]pt)",
        4: "-19.50pt(95%CI=[-51.19,+18.52]pt)",
        3: "-13.63pt(95%CI=[-45.10,+21.93]pt)",
    }[box_n]
    method_note = (
        "1点100円換算。馬連・ワイドは相手の頭数分、3連複はC(相手,2)、馬単(軸流し)は相手の頭数分、"
        "馬単(マルチ)はその2倍、3連単(軸流し)はP(相手,2)、3連単(マルチ)はその6倍の点数です。"
        f"重み自体のfit母集団(105レース)を除外した{n_races_full - 105 if n_races_full > 105 else '106'}"
        f"レースだけで再評価した市場超過は{method_note_extra}で、95%信頼区間は0をまたぎ統計的に"
        "有意ではありません。「高確信度Nレース/日」はオッズ・人気を一切使わず、レース内で"
        f"{box_n}位と{box_n + 1}位のスコア差を(1位−最下位)の幅で正規化した比率が大きい順にレースを"
        "選んだものです。的中数が10未満の券種は「参考値」と注記しています(信頼区間が広く、"
        "回収率の解釈には注意が必要です)。詳細な統計検証プロセスは"
        "data/jra_pipeline/jra_axis_search_2026_08_21_report.txt を参照してください。"
    )

    section_html = f"""
    <section class="box-section" id="sec-axis{box_n}">
      <details class="box-details" open>
        <summary class="box-head">予想{box_n}頭軸流し(軸+相手{partner_n}頭) 回収率検証</summary>
        <p class="box-lede">{lede}</p>
        <div class="conf-tabs">
          {''.join(tab_elems)}
          <div class="conf-tabs-panels">{''.join(tab_panels)}</div>
        </div>
        <details class="method box-method">
          <summary>計算方法・前提条件</summary>
          <p class="method-note">{method_note}</p>
        </details>
        {market_html}
        {candidate_html}
      </details>
    </section>"""
    return "\n".join(tab_css_rules), section_html


def build_top3_section() -> str:
    """2026-08-21追加検証: 軸(スコア1位)の複勝的中率を目的関数にした重み探索の結果セクション。
    jra_axis_top3_search_2026_08_21.pyの出力(result.json)を読み込んで描画する。"""
    result = json.loads((DATA_DIR / "jra_axis_top3_search_2026_08_21_result.json").read_text(encoding="utf-8"))
    n_races = result["n_races"]
    mkt_rate = result["market_axis_hit_rate"]
    cur = result["current_box_weight_axis_hit_rate"]
    eq_rate = result["equal_weight_axis_hit_rate"]
    best = result["best_full_population"]
    oof = result["nested_lobo_oof"]
    opt = result["selection_optimism"]
    gate = result["decision_gate_ratio"]
    decision = result["decision"]
    rc = result["return_check"]
    fs = rc["fukusho_single_axis"]
    fsm = rc["fukusho_single_axis_market_comparison"]

    baseline_rows = "".join(f"""<tr>
          <td class="bt-name">現行box{k}重み転用</td>
          <td class="num">{v:.2f}%</td>
        </tr>""" for k, v in sorted(cur.items(), key=lambda kv: -int(kv[0])))

    nagashi_details = []
    for box_n_str, entry in sorted(rc["nagashi_by_box_n"].items(), key=lambda kv: -int(kv[0])):
        box_n = int(box_n_str)
        rows = "".join(f"""<tr>
          <td class="bt-name">{esc(BET_LABEL.get(row['bet_type'], row['bet_type']))}"""
          f"""{' <span class="box-sub">(参考値)</span>' if row['low_sample'] else ''}</td>
          <td>{row['hit_races']}/{row['races']}<span class="box-sub">({row['hit_rate_pct']:.1f}%)</span></td>
          <td class="num">¥{row['stake']:,}</td>
          <td class="num">¥{row['return']:,}</td>
          <td class="num rate {rate_cls(row['return_rate_pct'])}">{row['return_rate_pct']:.1f}%</td>
        </tr>""" for row in entry["rows"])
        nagashi_details.append(f"""
        <details class="method box-method">
          <summary>参考: axis{box_n}(軸+相手{box_n - 1}頭)ナガシの回収率、この重みを使った場合</summary>
          <div class="box-table-wrap">
            <table class="box-table">
              <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </details>""")

    return f"""
    <section class="box-section" id="sec-top3">
      <details class="box-details">
        <summary class="box-head">追加検証: 軸の複勝的中率を最大化する重み探索(2026-08-21)</summary>
        <p class="box-lede">
          「軸(予想スコア1位)が3着以内(複勝圏内)に来る確率」を直接の目的関数にして、
          {result['n_patterns']}パターンの重み配分を再探索しました({n_races}レース、
          box買い用の重みが「集合としての的中力」を最適化したものであり「単騎で馬券圏内に
          入る確率」を直接最適化したものではない、という専門家レビュー指摘を受けた追加検証です)。
          軸(スコア1位)自体はbox_nに依存しないため、探索はaxis5/4/3共通で1回だけ行っています。
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>候補</th><th>軸の複勝的中率</th></tr></thead>
            <tbody>
              <tr><td class="bt-name">市場(1番人気を軸)</td><td class="num rate is-plus">{mkt_rate:.2f}%</td></tr>
              {baseline_rows}
              <tr><td class="bt-name">25本等重み(参考)</td><td class="num">{eq_rate:.2f}%</td></tr>
              <tr><td class="bt-name">500パターン探索・最良(in-sample、参考値)</td><td class="num">{best['hit_rate']:.2f}%</td></tr>
            </tbody>
          </table>
        </div>
        <p class="box-lede" style="margin-top:14px;">
          <b>結論: 探索は不採用。</b>選択バイアス診断(ブロック半分割×200反復)の
          「選ぶことの真の価値」はtrue_edge_pt={opt['true_edge_pt']:+.2f}pt(sd{opt['true_edge_sd']:.2f})、
          採否ゲート(2.0以上)に対する比は{gate:.3f}で大幅未達でした。
          Nested LOBO OOFはfold毎の選択パターンが{oof['n_unique_patterns']}/{oof['n_folds']}
          (全fold同一パターン)というLOBO退化を起こしており、この数値({oof['hit_rate']:.2f}%)は
          in-sample評価として扱い、統計的検定には使っていません(2026-08-20/21にNAR側で発見した
          落とし穴と同型)。<b>市場(1番人気、複勝的中率{mkt_rate:.2f}%)を統計的に上回る重みは
          見つかりませんでした。</b>これはオッズに織り込まれた情報の強さを裏付ける結果と
          解釈しています。
        </p>
        <details class="method box-method">
          <summary>参考: この探索最良パターンを使った場合の回収率(不採用候補、参考値)</summary>
          <p class="method-note">
            軸単体・複勝1点買い: 的中率{fs['hit_rate_pct']:.2f}%、投資額{fs['stake']:,}円、
            払戻額{fs['return']:,}円、回収率{fs['return_rate_pct']:.1f}%
            (95%CI=[{fs['bootstrap_ci']['lo']:.1f}, {fs['bootstrap_ci']['hi']:.1f}])。
            同条件の市場(1番人気)は回収率{fsm['market_return_rate_pct']:.1f}%
            (95%CI=[{fsm['market_bootstrap_ci']['lo']:.1f}, {fsm['market_bootstrap_ci']['hi']:.1f}])で、
            差は{fsm['diff_model_minus_market_pt']:+.1f}pt
            (95%CI=[{fsm['diff_bootstrap_ci']['lo']:+.1f}, {fsm['diff_bootstrap_ci']['hi']:+.1f}]pt)と
            {'統計的に有意でした' if fsm['significant'] else '0をまたぐため統計的に有意差はありません'}。
            これらの数値はすべて不採用候補(in-sample選択パターン)を使った参考値です。
          </p>
          {''.join(nagashi_details)}
        </details>
      </details>
    </section>"""


def build_front123_section() -> str:
    """2026-08-21追加検証(ユーザー依頼「勝率・回収率を上げる工夫をお願いします」):
    3つの質的に異なるアプローチ(新規シグナル・アンサンブル+時系列検証・ステーク配分)を
    box5/4/3・axis5/4/3の両方に適用した結果セクション。"""
    d1 = json.loads((DATA_DIR / "jra_search500_2026_08_21_v3signals_result.json").read_text(encoding="utf-8"))
    d2 = json.loads((DATA_DIR / "jra_axis_search500_2026_08_21_v3signals_result.json").read_text(encoding="utf-8"))
    d3 = json.loads((DATA_DIR / "jra_ensemble_search_2026_08_21_result.json").read_text(encoding="utf-8"))
    d4 = json.loads((DATA_DIR / "jra_axis_ensemble_search_2026_08_21_result.json").read_text(encoding="utf-8"))
    d5 = json.loads((DATA_DIR / "confidence_sweep_stake_box_summary.json").read_text(encoding="utf-8"))
    cal = json.loads((DATA_DIR / "confidence_calibration.json").read_text(encoding="utf-8"))

    def f1_row(label, r, ci_key):
        ci = r[ci_key]
        return f"""<tr>
          <td class="bt-name">{esc(label)}</td>
          <td class="num">{r['nested_lobo_oof']['excess']:+.2f}pt</td>
          <td class="num">{r['nested_lobo_oof']['n_unique_patterns']}/{r['nested_lobo_oof']['n_folds']}</td>
          <td class="num">{r['selection_optimism']['true_edge_pt']:+.2f}(sd{r['selection_optimism']['true_edge_sd']:.1f})</td>
          <td class="num">[{ci['lo']:+.1f}, {ci['hi']:+.1f}]pt</td>
          <td class="num rate is-minus">{decision_ja(r['decision'])}</td>
        </tr>"""

    f1_rows = "".join([
        f1_row("box5", d1["results_by_box"]["5"], "bootstrap_oof_vs_current"),
        f1_row("box4", d1["results_by_box"]["4"], "bootstrap_oof_vs_current"),
        f1_row("box3", d1["results_by_box"]["3"], "bootstrap_oof_vs_current"),
        f1_row("axis5", d2["results_by_box"]["5"], "bootstrap_oof_vs_current_box_weight"),
        f1_row("axis4", d2["results_by_box"]["4"], "bootstrap_oof_vs_current_box_weight"),
        f1_row("axis3", d2["results_by_box"]["3"], "bootstrap_oof_vs_current_box_weight"),
    ])

    def f2_row(label, r):
        ci = r["bootstrap_ensemble_vs_current"]
        return f"""<tr>
          <td class="bt-name">{esc(label)}</td>
          <td class="num">{r['nested_lobo_oof_single']['n_unique_patterns']}/{r['nested_lobo_oof_single']['n_folds']}</td>
          <td class="num">{r['nested_lobo_oof_ensemble']['n_unique_patterns']}/{r['nested_lobo_oof_ensemble']['n_folds']}</td>
          <td class="num">{r['nested_lobo_oof_ensemble']['excess']:+.2f}pt</td>
          <td class="num">[{ci['lo']:+.1f}, {ci['hi']:+.1f}]pt</td>
          <td class="num rate is-minus">{decision_ja(r['decision'])}</td>
        </tr>"""

    f2_rows = "".join([
        f2_row("box5", d3["results_by_box"]["5"]), f2_row("box4", d3["results_by_box"]["4"]),
        f2_row("box3", d3["results_by_box"]["3"]), f2_row("axis5", d4["results_by_box"]["5"]),
        f2_row("axis4", d4["results_by_box"]["4"]), f2_row("axis3", d4["results_by_box"]["3"]),
    ])

    def f3_row(box_n):
        s = d5["summary_by_box"][str(box_n)]
        ci = s["diff_continuous_vs_baseline"]
        return f"""<tr>
          <td class="bt-name">box{box_n}</td>
          <td class="num">{s['baseline']['excess']:+.2f}pt</td>
          <td class="num">{s['continuous_multiplier']['excess']:+.2f}pt</td>
          <td class="num">[{ci['lo']:+.2f}, {ci['hi']:+.2f}]pt</td>
          <td class="num rate is-minus">有意差なし</td>
        </tr>"""

    f3_rows = "".join(f3_row(n) for n in (5, 4, 3))

    cal_rows = "".join(f"""<tr>
      <td class="bt-name">K={k}</td>
      <td class="num">{v['overall_hit_rate_pct']:.1f}%</td>
      <td class="num">{'○' if v['beats_trivial_baseline'] else '×'}</td>
      <td class="num">{'○' if v['min_blocks_ok'] else '×'}</td>
    </tr>""" for k, v in cal["topk_ladder"]["results_by_k"].items())

    return f"""
    <section class="box-section" id="sec-front123">
      <details class="box-details">
        <summary class="box-head">追加検証: 新規シグナル・アンサンブル・ステーク配分の3方向(2026-08-21)</summary>
        <p class="box-lede">
          box買い・軸流し買いいずれの探索も市場優位性が実証できなかったことを受け、質的に異なる
          3つのアプローチで改善余地を探りました。box5/4/3・axis5/4/3の両方に適用しています。
        </p>

        <h3 class="box-subhead">① 新規シグナル(NARのV4シグナル5本を移植、25→30本プール)</h3>
        <p class="method-note">
          持続タイム(hold_just/hold_wide)・騎手乗り替わり(jockey_change)・クラス降格(class_drop)・
          馬体重トレンド(weight_trend)の5本を追加し、500パターンで再探索しました。
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>対象</th><th>OOF市場差</th><th>LOBOユニーク</th><th>選ぶことの真の価値</th>
              <th>現行比95%CI</th><th>判定</th></tr></thead>
            <tbody>{f1_rows}</tbody>
          </table>
        </div>
        <p class="method-note">
          6本すべて不採用。box4は現行モデルとの差の95%CIが完全にマイナス側
          ([-58.87,-6.83]pt)で、統計的に有意に劣る結果でした。的中率はほぼ同水準
          (単勝59.2%対60.2%)なのに全券種で回収率が下がっており(単勝99.0%→84.7%、
          3連単196.8%→30.0%等)、より人気馬寄りの選択に偏り配当妙味が落ちたためと
          考えられます。新規5シグナルの合計重みシェアは6.9〜16.7%と一定の存在感はありますが、
          市場を上回るには至りませんでした。
        </p>

        <h3 class="box-subhead">② アンサンブル(多様性重視の上位8パターンをブレンド)+時系列検証</h3>
        <p class="method-note">
          ①と同じ500パターン母集団から、コサイン類似度で似すぎたものを除きながら上位8本を
          均等ブレンドした重みを試しました。時系列walk-forward検証もロバスト性チェックとして
          併記しています。
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>対象</th><th>単一ベストのLOBOユニーク</th><th>ブレンドのLOBOユニーク</th>
              <th>ブレンドOOF市場差</th><th>現行比95%CI</th><th>判定</th></tr></thead>
            <tbody>{f2_rows}</tbody>
          </table>
        </div>
        <p class="method-note">
          6本すべて不採用。ただしブレンドはfold間で選ばれる組み合わせの安定性を単一ベスト選択
          (2〜4/36ブロックでしか変わらない)より明確に改善しました(最大24/36)。「選択の不安定さを
          均す」という狙い通りの効果はあったものの、それ自体が回収率の統計的な優位性には
          直結しませんでした。
        </p>

        <h3 class="box-subhead">③ 確信度に応じたステーク配分</h3>
        <p class="method-note">
          レース内のスコア差の日次パーセンタイル順位に応じて、賭け金を0.5〜1.5倍の
          範囲で連続的に増減させた場合の回収率を、均等ステークと比較しました(box5/4/3、現行
          本番モデル)。
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>対象</th><th>均等ステーク市場差</th><th>連続乗数市場差</th>
              <th>差の95%CI</th><th>判定</th></tr></thead>
            <tbody>{f3_rows}</tbody>
          </table>
        </div>
        <p class="method-note">
          3サイズとも連続乗数がわずかにプラス方向(box3で+3.9pt)でしたが、いずれも95%CIが0を
          またぎ統計的有意差はありません。土台となる確信度較正(topk_ladder)自体も、
          {cal['fitted_on']['n_blocks']}ブロック(較正に必要な60ブロックに未到達)のため
          まだ統計的に安定した%表示はできていません(K別の内訳):
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>段階</th><th>実測的中率</th><th>ランダム基準を上回るか</th><th>ブロック数十分か</th></tr></thead>
            <tbody>{cal_rows}</tbody>
          </table>
        </div>
        <p class="method-note">
          このためオッズベースの本格的なKelly基準サイジングは今回のスコープに含めず、
          確信度較正がブロック数60以上に到達し安定するまでの将来課題としています。
        </p>

        <p class="box-lede is-tight">
          <b>総合結論:</b> 新規シグナル・アンサンブル+時系列検証・ステーク配分という3つの
          構造的に異なるアプローチいずれでも、市場に対して統計的に実証された優位性は
          見つかりませんでした。これでbox買い・軸流し買いの重み探索(4回)・アンサンブル(1回)・
          ステーク配分(1回)のすべてが同一の結論に達しています。
        </p>
      </details>
    </section>"""


all_css_rules = []
all_sections = [build_front123_section(), build_top3_section()]
jump_items = ['<a href="#sec-front123">新シグナル・アンサンブル・ステーク</a>',
             '<a href="#sec-top3">軸の複勝的中率探索</a>']
for box_n in BOX_SIZES:
    css_rules, section_html = build_axis_section(box_n)
    all_css_rules.append(css_rules)
    all_sections.append(section_html)
    jump_items.append(f'<a href="#sec-axis{box_n}">軸+相手{box_n - 1}頭</a>')

conf_tabs_css = "\n".join(all_css_rules)

CSS = r"""
:root {
  --bg: #ECEEEA; --bg-elev: #F7F8F5; --bg-card: #FFFFFF; --ink: #1A1C1B; --ink-muted: #5B5F5A;
  --ink-faint: #656960; --rule: #D2D5CD; --rule-strong: #B7BBB1; --accent: #2C6E49; --accent-ink: #FFFFFF;
  --accent-soft: #D9E9DE; --accent-soft-ink: #1E4A31; --score-track: #E2E4DD;
  --shadow: 0 1px 2px rgba(20, 22, 18, 0.06), 0 6px 16px -10px rgba(20, 22, 18, 0.18);
  --serif: "Yu Mincho", "YuMincho", "Hiragino Mincho ProN", "Noto Serif JP", "Georgia", serif;
  --sans: "Yu Gothic", "YuGothic", "Hiragino Sans", "Noto Sans JP", "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141815; --bg-elev: #1A1F1B; --bg-card: #1F2521; --ink: #E6E8E2; --ink-muted: #98A093;
    --ink-faint: #9AA294; --rule: #2C332C; --rule-strong: #3B443A; --accent: #63C08A; --accent-ink: #0E1F15;
    --accent-soft: #1E3A28; --accent-soft-ink: #A9E6BF; --score-track: #29302A;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 8px 20px -12px rgba(0, 0, 0, 0.5);
  }
}
:root[data-theme="dark"] {
  --bg: #141815; --bg-elev: #1A1F1B; --bg-card: #1F2521; --ink: #E6E8E2; --ink-muted: #98A093;
  --ink-faint: #9AA294; --rule: #2C332C; --rule-strong: #3B443A; --accent: #63C08A; --accent-ink: #0E1F15;
  --accent-soft: #1E3A28; --accent-soft-ink: #A9E6BF; --score-track: #29302A;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 20px -12px rgba(0,0,0,0.5);
}
:root[data-theme="light"] {
  --bg: #ECEEEA; --bg-elev: #F7F8F5; --bg-card: #FFFFFF; --ink: #1A1C1B; --ink-muted: #5B5F5A;
  --ink-faint: #656960; --rule: #D2D5CD; --rule-strong: #B7BBB1; --accent: #2C6E49; --accent-ink: #FFFFFF;
  --accent-soft: #D9E9DE; --accent-soft-ink: #1E4A31; --score-track: #E2E4DD;
  --shadow: 0 1px 2px rgba(20,22,18,0.06), 0 6px 16px -10px rgba(20,22,18,0.18);
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
  font-feature-settings: "palt"; line-height: 1.6; font-variant-numeric: tabular-nums;
}
main { max-width: 900px; margin: 0 auto; padding: 0 20px 64px; }
.jumpnav {
  position: sticky; top: 0; z-index: 20; background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule);
}
.jumpnav-inner {
  max-width: 900px; margin: 0 auto; padding: 10px 20px;
  display: flex; align-items: center; gap: 14px; overflow-x: auto; scrollbar-width: thin;
}
.jumpnav-label { font: 700 11px/1 var(--sans); letter-spacing: 0.12em; color: var(--ink-faint); flex: none; }
.jumpnav a {
  flex: none; font-size: 13px; color: var(--ink-muted); text-decoration: none;
  padding: 5px 10px; border-radius: 999px; border: 1px solid var(--rule);
  white-space: nowrap; transition: color .15s, border-color .15s, background .15s;
}
.jumpnav a:hover { color: var(--ink); border-color: var(--rule-strong); background: var(--bg-elev); }
.jumpnav a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.masthead { padding: 40px 0 28px; border-bottom: 3px double var(--rule-strong); margin-bottom: 8px; }
.eyebrow { font: 700 11px/1 var(--sans); letter-spacing: 0.16em; color: var(--accent); margin: 0 0 10px; }
.title {
  font-family: var(--serif); font-weight: 700; font-size: clamp(26px, 4.2vw, 38px);
  margin: 0 0 14px; text-wrap: balance; letter-spacing: 0.01em;
}
.lede { font-size: 15px; color: var(--ink-muted); max-width: 66ch; margin: 0 0 18px; }
.callout {
  background: var(--accent-soft); color: var(--accent-soft-ink); border-radius: 10px;
  padding: 14px 18px; font-size: 13.5px; max-width: 66ch; margin: 0 0 18px;
}
.callout b { font-weight: 700; }
.box-section { margin: 32px 0 40px; padding: 20px 22px; background: var(--bg-elev);
  border: 1px solid var(--rule); border-radius: 12px; scroll-margin-top: 80px; }
.box-head { font-family: var(--serif); font-size: 20px; font-weight: 700; margin: 0 0 8px; cursor: pointer; }
.box-lede { font-size: 13px; color: var(--ink-muted); max-width: 68ch; margin: 0 0 16px; }
.box-table-wrap { overflow-x: auto; }
.box-table { width: 100%; border-collapse: collapse; font-size: 13.5px; min-width: 460px; }
.box-table th { text-align: left; font-weight: 500; font-size: 11px; color: var(--ink-faint);
  letter-spacing: 0.04em; padding: 6px 10px; border-bottom: 1px solid var(--rule-strong); }
.box-table td { padding: 7px 10px; border-bottom: 1px solid var(--rule); }
.box-table .bt-name { font-weight: 700; }
.box-table .num { text-align: right; }
.box-sub { color: var(--ink-faint); font-size: 11px; margin-left: 4px; }
.box-table .rate { font-weight: 700; font-family: var(--serif); }
.box-table .rate.is-plus { color: var(--accent); }
.box-table .rate.is-mid { color: var(--ink); }
.box-table .rate.is-minus { color: var(--ink-faint); }
.conf-tabs { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.conf-tab-input { position: absolute; opacity: 0; width: 1px; height: 1px; overflow: hidden; }
.conf-tab-label { cursor: pointer; font-size: 12.5px; color: var(--ink-muted); user-select: none;
  padding: 5px 12px; border-radius: 999px; border: 1px solid var(--rule);
  transition: color .15s, border-color .15s, background .15s; }
.conf-tab-label:hover { border-color: var(--rule-strong); color: var(--ink); }
.conf-tab-input:checked + .conf-tab-label { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.conf-tab-input:focus-visible + .conf-tab-label { outline: 2px solid var(--accent); outline-offset: 2px; }
.conf-tabs-panels { flex-basis: 100%; margin-top: 14px; }
.conf-tab-panel { display: none; }
.conf-scope-label { font-size: 12px; color: var(--ink-faint); margin: 0 0 8px; }
.method { font-size: 13.5px; color: var(--ink-muted); }
.method summary { cursor: pointer; color: var(--ink); font-weight: 600; padding: 6px 0; }
.method-note { margin: 10px 0 0; max-width: 68ch; }
.box-method { margin-top: 14px; }
.box-method summary { font-size: 12.5px; }
.box-subhead { font-family: var(--serif); font-size: 14px; font-weight: 700; margin: 20px 0 8px; color: var(--ink); }
.box-lede.is-tight { margin-top: 16px; }
.pagefoot { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--rule); }
.pagefoot p { font-size: 11.5px; color: var(--ink-faint); margin: 0; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

html_out = f"""<title>JRA軸馬流し予想 回収率検証</title>
<style>{CSS}
{conf_tabs_css}
</style>
<nav class="jumpnav">
  <div class="jumpnav-inner">
    <span class="jumpnav-label">目次</span>
    {''.join(jump_items)}
  </div>
</nav>

<main>
  <header class="masthead">
    <p class="eyebrow">馬柱データ分析・JRA軸流し買い</p>
    <h1 class="title">JRA 1頭軸流し予想 回収率検証</h1>
    <p class="lede">
      予想スコア1位を軸馬、2位以降を相手馬にした「ながし」買い(馬連・ワイド・3連複・馬単・3連単)の
      回収率検証レポートです。既存の「馬柱データ予想」(BOX買い版)とは別の独立レポートとして、
      同じ馬柱データ・予想モデルを軸流し買いに転用した場合の成績を検証しています。
    </p>
    <p class="callout">
      <b>お読みください:</b> 2026-08-21に軸流し専用の重みモデリング探索(500パターン)を実施しましたが、
      統計的に有意な改善は見つからず不採用でした。現行box重みを軸流しに転用した場合の回収率も、
      重み自体の決定に使ったデータを除いた公正な評価では市場超過が消えます。つまり
      <b>「市場に対して優位性が統計的に実証されたモデル」は現時点ではありません。</b>
      各セクションに詳細な検証結果を透明に記載しています。
    </p>
  </header>

  {''.join(all_sections)}

  <footer class="pagefoot">
    <p>
      あくまで公開データに基づく統計的な検証記録であり、的中や利益を保証するものではありません。
      検証・重み探索の詳細は scripts/jra_model/jra_axis_search_2026_08_21.py および
      data/jra_pipeline/jra_axis_search_2026_08_21_report.txt を参照してください。
    </p>
  </footer>
</main>
"""

OUT_PATH = DATA_DIR / "prediction_report_jra_axis.html"
OUT_PATH.write_text(html_out, encoding="utf-8")
print(f"wrote {OUT_PATH} ({len(html_out)} chars)")
