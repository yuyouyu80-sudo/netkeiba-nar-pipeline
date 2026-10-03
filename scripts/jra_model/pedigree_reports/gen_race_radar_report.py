# -*- coding: utf-8 -*-
"""race_radar_data.json から、中山全レース向け「求められるファクター」レーダーチャートHTML(1ページ、レース切替式)を生成する。

元は中山8R専用の gen_race8_radar_report.py。2026-10-03に全レース対応へ一般化し、各馬のミニレーダー
(レース基準との重ね合わせ)を追加。1ページに全レースのセクションを持ち、上部のレース番号ボタンで切替
(チャートは表示時に初めて描画する)。
"""
import html
import json
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)
DATA_PATH = SCRATCH / "race_radar_data.json"
OUT_PATH = SCRATCH / "race_radar_report.html"

GOING_LABELS = {"良": "良", "稍重": "稍重", "重": "重", "不良": "不良"}
SURFACE_NAME = {"ダ": "ダート", "芝": "芝"}
WEEKDAY_JA = "月火水木金土日"
TIER_BADGE_CLASS = {"高": "tier-high", "中": "tier-mid", "低": "tier-low"}
SHORT = {"distance": "距離", "surface": "馬場", "course": "競馬場", "season": "季節", "rest": "間隔",
         "debut": "新馬", "graded": "重賞", "leader_style": "脚質", "speed": "スピード",
         "stamina": "スタミナ", "kire": "キレ"}
FEW_RACES = 15  # これ未満は「少数」の注意書きを付ける


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def fmt_pt(x):
    return f"{x:+.2f}pt" if x is not None else "?"


def fmt_range(rng):
    return f"{rng[0]:+.1f}〜{rng[1]:+.1f}pt" if rng else "?"


def fmt_rate(v):
    return f"{v:.1f}%" if v is not None else "?"


def balance_pattern_note(pattern, sig):
    if "中間が最適" in pattern:
        tail = "統計的に有意な差ではありませんが" if not sig else "統計的にも有意な差として"
        return f"点推定では中位分位が最も複勝率が高く、{tail}「極端な値より中間寄りが良い」というユーザー仮説と整合する結果です。"
    if "高いほど有利" in pattern:
        return "高位分位ほど複勝率が高い、従来通りの単調な関係が点推定では見られます。"
    if "低いほど有利" in pattern:
        return "低位分位ほど複勝率が高い、逆方向の単調な関係が点推定では見られます。"
    return "分位間の信頼区間が重なっており、明確な傾向(単調・非単調とも)は確認できませんでした。"


def bloodline_table(bl, side_label):
    if not bl["rows"]:
        return f'<p class="bal-note">{esc(side_label)}: 該当系統がN不足のため表示できる区分がありません(不明{bl["n_unmatched"]}頭)。</p>'
    max_rate = max(r["top3_rate"] for r in bl["rows"]) or 1
    rows_html = "".join(f"""
        <tr>
          <td class="bt-label">{esc(r['category'])}</td>
          <td class="bl-bar-cell"><span class="bl-bar" style="width:{r['top3_rate']/max_rate*100:.0f}%"></span>
            <span class="bl-rate">{fmt_rate(r['top3_rate'])}</span></td>
          <td>N={r['n']}</td>
          <td class="bt-n">[{fmt_rate(r['ci_low'])}, {fmt_rate(r['ci_high'])}]</td>
        </tr>""" for r in bl["rows"])
    return f"""
    <div class="bl-block">
      <h3>{esc(side_label)}</h3>
      <table class="bt bl-table">
        <thead><tr><th>系統</th><th>複勝率</th><th>頭数</th><th>95%CI</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="bal-note">分類できた{bl['n_matched']}頭中、系統不明{bl['n_unmatched']}頭は除外。
        N&lt;20の系統は非表示。</p>
    </div>"""


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def render_race(d, rno):
    """1レース分のセクションHTMLと、チャート描画用データ(JSON化してページ末尾のJSへ渡す)を返す。"""
    info = d["race_info"]
    rule = d["match_rule"]
    item_labels = d["item_labels"]
    template = d["template"]
    today = d["today"]
    ssk_labels = d["ssk_labels"]
    ssk_template = d["ssk_template"]
    ssk_balance = d["ssk_balance"]
    tiers = d["importance_tier"]
    dam_records = d["dam_records"]
    n_field_horses = d["n_field_horses"]
    n_unique_pairs = d["n_unique_sire_bms_pairs"]
    role_spec = d["role_spec"]
    n_matched = d["n_matched_races"]
    n_scored = d["n_scored_horses"]
    bloodline_sire, bloodline_bms, bloodline_ss = d["bloodline_sire"], d["bloodline_bms"], d["bloodline_ss"]
    today_bloodlines = d.get("today_bloodlines", {})

    venue, surf_name = info["venue"], SURFACE_NAME[info["surface"]]
    dist = int(info["distance"])
    dt = datetime.strptime(info["race_date"], "%Y-%m-%d")
    date_jp = f"{dt.year}年{dt.month}月{dt.day}日({WEEKDAY_JA[dt.weekday()]})"
    race_label = f"{venue}{rno}R"
    cond_label = f"{info['race_name']}・{surf_name}{dist}m"
    post = f"・発走{info['start_time']}" if info.get("start_time") else ""
    day_no = int(rule["day_seg"])

    all_labels = item_labels + ssk_labels
    all_template = dict(template)
    all_template.update(ssk_template)
    axis_keys = [key for key, _l in all_labels]

    # レーダーチャートは高カバレッジな父方3ライン(父/母父/父父)のみ描画する。母方3ライン
    # (母/母母/父母)は産駒数が少なく該当馬が少ないため、レーダーに重ねると「データなし=0pt」が
    # 「平均的」に見えてしまい誤解を招く。数値内訳テーブル(母方ライン参考表)でのみ開示する。
    chart = {
        "labels": [lbl for _key, lbl in all_labels],
        "sire": [all_template[k]["sire"]["mean_pt"] or 0 for k in axis_keys],
        "bms": [all_template[k]["bms"]["mean_pt"] or 0 for k in axis_keys],
        "ss": [all_template[k].get("ss", {}).get("mean_pt") or 0 for k in axis_keys],
        "ref": [_mean([all_template[k].get(r, {}).get("mean_pt") for r in ("sire", "bms", "ss")]) for k in axis_keys],
    }

    ssk_rows = []
    for key, label in ssk_labels:
        s, b = ssk_template[key]["sire"], ssk_template[key]["bms"]
        g = ssk_template[key].get("ss", {"mean_pt": None, "n_horses": 0})
        tier = tiers.get(key, "中")
        badge = f'<span class="tier-badge {TIER_BADGE_CLASS[tier]}">重要度: {tier}</span>'
        ssk_rows.append(f"""
        <tr>
          <td class="bt-label">{esc(label)} {badge}</td>
          <td class="bt-sire">{fmt_pt(s['mean_pt'])}<span class="bt-n">(N={s['n_horses']})</span></td>
          <td class="bt-bms">{fmt_pt(b['mean_pt'])}<span class="bt-n">(N={b['n_horses']})</span></td>
          <td class="bt-ss">{fmt_pt(g['mean_pt'])}<span class="bt-n">(N={g['n_horses']})</span></td>
        </tr>""")

    # --- 母方3ライン(母/母母/父母)の参考内訳表 ---
    DAM_SIDE_ROLES = [r for r in role_spec if r["key"] in ("dam", "dd", "sd")]
    dam_side_rows = []
    for key, label in all_labels:
        cells = []
        for rs in DAM_SIDE_ROLES:
            cell = all_template.get(key, {}).get(rs["key"], {"mean_pt": None, "n_horses": 0})
            cells.append(f'<td class="bt-damside">{fmt_pt(cell["mean_pt"])}<span class="bt-n">(N={cell["n_horses"]})</span></td>')
        dam_side_rows.append(f'<tr><td class="bt-label">{esc(label)}</td>{"".join(cells)}</tr>')
    dam_side_header = "".join(f"<th>{esc(rs['label'])}平均</th>" for rs in DAM_SIDE_ROLES)

    # --- スピード・スタミナ・キレの「良い塩梅」検証 ---
    balance_cards = []
    for key, label in ssk_labels:
        ba = ssk_balance[key]
        if not ba.get("bins"):
            balance_cards.append(f"""
            <div class="bal-block">
              <h3>{esc(label)}</h3>
              <p class="bal-note">{esc(ba.get('pattern', '判定不能'))}(N={ba.get('n_total', 0)})</p>
            </div>""")
            continue
        rows = "".join(f"""
            <tr class="{'bal-best' if b['label'] == ba['best_bin'] else ''}">
              <td class="bt-label">{esc(b['label'])}</td>
              <td>{fmt_range(b['value_range'])}</td>
              <td>N={b['n']}</td>
              <td>{fmt_rate(b['top3_rate'])}<span class="bt-n">[{fmt_rate(b['ci_low'])}, {fmt_rate(b['ci_high'])}]</span></td>
            </tr>""" for b in ba["bins"])
        sig = not ba.get("ci_overlap_best_vs_worst", True)
        note = balance_pattern_note(ba["pattern"], sig)
        cross = ""
        top3_mean = ba.get("top3_combined_mean")
        if key == "kire" and top3_mean is not None:
            rng = ba["best_bin_value_range"]
            inside = rng[0] <= top3_mean <= rng[1]
            if inside:
                tail = ("この最良分位の範囲内に入っており、2通りの見方(上位馬の平均値/全出走馬の分位別複勝率)は"
                        "矛盾していません。")
            else:
                tail = ("この最良分位の範囲(" + fmt_range(rng) + ")の外にあり、2通りの見方(上位馬の平均値/全出走馬の"
                        "分位別複勝率)は一致していません。どちらも点推定で、傾向を断定できる根拠にはなりません。")
            cross = (
                f'<p class="bal-note">クロスチェック: 実際の上位3着馬(1頭ごとに父・母父・父父の'
                f'算出できたライン平均、N={ba.get("top3_combined_n")})のキレ平均は{fmt_pt(top3_mean)}で、{tail}</p>'
            )
        balance_cards.append(f"""
        <div class="bal-block">
          <h3>{esc(label)} <span class="tier-badge {TIER_BADGE_CLASS[tiers.get(key, '中')]}">重要度: {tiers.get(key, '中')}</span></h3>
          <table class="bt bal-table">
            <thead><tr><th>分位</th><th>値の範囲</th><th>頭数</th><th>複勝率[95%CI]</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
          <p class="bal-verdict">判定: <b>{esc(ba['pattern'])}</b> — {note}</p>
          {cross}
        </div>""")

    dam_rows = []
    if today:
        for h in today["horses"]:
            rec = dam_records.get(h["horse_id"])
            if not rec:
                continue
            if rec.get("raced_in_window"):
                detail = (f"{rec['n_runs']}走{rec['n_wins']}勝・"
                          f"出走距離{rec['min_distance_m']}〜{rec['max_distance_m']}m")
            else:
                detail = "2016〜2026年の対象期間に出走記録なし(現役時代が対象期間外の可能性)"
            dam_rows.append(
                f'<div class="dam-item"><b>{esc(h["umaban"])}番 {esc(h["horse_name"])}</b>'
                f'の母 {esc(rec.get("dam_name")) or "-"}: {detail}</div>'
            )
    dam_box_html = ""
    if dam_rows:
        n_raced = sum(1 for v in dam_records.values() if v.get("raced_in_window"))
        dam_box_html = (
            '<div class="dam-box"><div class="dam-caption">参考情報(軸のスコアには含めません): '
            '母自身の現役成績。産駒への遺伝伝達力(上記の父・母父・父父pt差)とは異なる「母馬個体の能力」'
            f'の情報です。2016〜2026年の対象期間内に出走記録がある馬のみ({n_raced}/{len(dam_records)}頭、'
            '現役時代が対象期間外の馬は「記録なし」表示)。</div>'
            + "".join(dam_rows) + "</div>"
        )

    breakdown_rows = []
    for key, label in item_labels:
        s, b = template[key]["sire"], template[key]["bms"]
        g = template[key].get("ss", {"mean_pt": None, "n_horses": 0})
        breakdown_rows.append(f"""
        <tr>
          <td class="bt-label">{esc(label)}</td>
          <td class="bt-sire">{fmt_pt(s['mean_pt'])}<span class="bt-n">(N={s['n_horses']})</span></td>
          <td class="bt-bms">{fmt_pt(b['mean_pt'])}<span class="bt-n">(N={b['n_horses']})</span></td>
          <td class="bt-ss">{fmt_pt(g['mean_pt'])}<span class="bt-n">(N={g['n_horses']})</span></td>
        </tr>""")
    breakdown_rows.extend(ssk_rows)

    # --- チャート下の解説(データから動的に生成。8R専用だった固定文言は廃止) ---
    line_means = {}
    for key, label in item_labels:
        vals = [template[key].get(r, {}).get("mean_pt") for r in ("sire", "bms", "ss")]
        line_means[key] = (label, vals, _mean(vals))
    ranked_axes = sorted([k for k in line_means if line_means[k][2] is not None],
                         key=lambda k: -abs(line_means[k][2]))
    hl_key = ranked_axes[0] if ranked_axes else "leader_style"
    interp = ""
    if ranked_axes:
        lbl, vals, m = line_means[hl_key]
        present = [v for v in vals if v is not None]
        same_sign = len(present) >= 2 and (all(v > 0 for v in present) or all(v < 0 for v in present))
        others = [abs(line_means[k][2]) for k in ranked_axes[1:]]
        others_txt = (f"残る{len(others)}項目はいずれも3ライン平均の絶対値が{max(others):.1f}pt以内です。"
                      if others else "")
        cons = ("父・母父・父父の3ラインとも同じ向きで一致しています。" if same_sign
                else "ただしラインごとに向きが異なる(またはデータのないラインがある)ため、一貫した傾向とは言えません。")
        interp = (f"既存8項目のうち3ライン平均の絶対値が最も大きいのは「{esc(lbl)}」({m:+.1f}pt、"
                  f"父{fmt_pt(vals[0])}/母父{fmt_pt(vals[1])}/父父{fmt_pt(vals[2])})で、{cons}{others_txt}"
                  "これは過去の上位馬の血統の事後集計であり、血統単独モデルは検証で市場超過を確認できていません。"
                  "なお、キレ(=非先行勝率−先行勝率)は脚質(先行)と表裏一体の指標で、先行有利な血統ほど自動的に"
                  "マイナスになります(独立した別証拠ではありません)。")
    hl_label = line_means[hl_key][0] if hl_key in line_means else "脚質(先行)"

    going_html = "".join(
        f'<span class="going-chip">{esc(GOING_LABELS.get(k, k))}: {v}件</span>'
        for k, v in sorted(d["going_breakdown_of_matched_races"].items(), key=lambda kv: -kv[1])
    )

    race_rows = []
    for r in d["matched_races"]:
        top3_html = "".join(
            f'<div class="ev-horse"><b>{h["finish_pos"]}着</b> {esc(h["horse_name"])}'
            f'<span class="ev-ped">父:{esc(h["sire_name"]) or "-"} / 母父:{esc(h["bms_name"]) or "-"}'
            f' / 父父:{esc(h.get("ss_name")) or "-"} / 母:{esc(h.get("dam_name")) or "-"}'
            f' / 母母:{esc(h.get("dd_name")) or "-"} / 父母:{esc(h.get("sd_name")) or "-"}</span></div>'
            for h in r["top3"]
        )
        race_rows.append(f"""
        <tr>
          <td class="ev-date">{esc(r['race_date'])}</td>
          <td class="ev-name">{esc(r['race_name'])}<span class="ev-going">馬場:{esc(GOING_LABELS.get(r['going'], r['going']))}</span></td>
          <td class="ev-top3">{top3_html}</td>
        </tr>""")

    # 今日の出走馬との比較テーブル
    kire_sweet = ssk_balance.get("kire", {}).get("best_bin_value_range")
    kire_best = ssk_balance.get("kire", {}).get("best_bin")
    kire_sig = not ssk_balance.get("kire", {}).get("ci_overlap_best_vs_worst", True)
    DAMSIDE_DISP = (("dam", "母"), ("dd", "母母"), ("sd", "父母"))
    today_rows = []
    horse_radar = []
    if today:
        horses = sorted(today["horses"], key=lambda h: h["composite_combined"] if h["composite_combined"] is not None else -1e9, reverse=True)
        for rank, h in enumerate(horses, 1):
            sire_items = (h.get("sire") or {}).get("items", {})
            bms_items = (h.get("bms") or {}).get("items", {})
            ss_items = (h.get("ss") or {}).get("items", {})
            sire_ssk = (h.get("sire") or {}).get("ssk", {}) or {}
            bms_ssk = (h.get("bms") or {}).get("ssk", {}) or {}
            ss_ssk = (h.get("ss") or {}).get("ssk", {}) or {}
            damside_scores = [h.get(rk) for rk, _lbl in DAMSIDE_DISP]
            damside_composite = [s["composite_pt"] for s in damside_scores if s and s.get("composite_pt") is not None]
            damside_txt = fmt_pt(sum(damside_composite) / len(damside_composite)) if damside_composite else "-"
            damside_tags = "".join(
                f'<span class="damside-tag">{esc(lbl)}</span>' for rk, lbl in DAMSIDE_DISP if h.get(rk)
            ) or '<span class="dim">該当なし</span>'
            sb = today_bloodlines.get(h["horse_id"], {})
            bloodline_tag = (
                f'<span class="bl-tag">{esc(sb.get("sire_bloodline")) or "-"}/'
                f'{esc(sb.get("bms_bloodline")) or "-"}</span>'
            )
            cells = []
            for key, _label in item_labels:
                si, bi, gi = sire_items.get(key), bms_items.get(key), ss_items.get(key)
                s_txt = fmt_pt(si["excess_pt"]) if si else "-"
                b_txt = fmt_pt(bi["excess_pt"]) if bi else "-"
                g_txt = fmt_pt(gi["excess_pt"]) if gi else "-"
                highlight = " hl" if key == hl_key else ""
                cells.append(f'<td class="tc{highlight}"><span class="tc-s">{s_txt}</span><span class="tc-b">{b_txt}</span><span class="tc-g">{g_txt}</span></td>')
            for key, _label in ssk_labels:
                si, bi, gi = sire_ssk.get(key), bms_ssk.get(key), ss_ssk.get(key)
                s_txt = fmt_pt(si["excess_pt"]) if si else "-"
                b_txt = fmt_pt(bi["excess_pt"]) if bi else "-"
                g_txt = fmt_pt(gi["excess_pt"]) if gi else "-"
                mark = ""
                if key == "kire" and kire_sweet:
                    vals = [v["excess_pt"] for v in (si, bi, gi) if v]
                    if vals and any(kire_sweet[0] <= v <= kire_sweet[1] for v in vals):
                        mark = " sweet"
                cells.append(f'<td class="tc tc-ssk{mark}"><span class="tc-s">{s_txt}</span><span class="tc-b">{b_txt}</span><span class="tc-g">{g_txt}</span></td>')
            composite = h["composite_combined"]
            comp_txt = fmt_pt(composite) if composite is not None else "データ不足"
            today_rows.append(f"""
            <tr>
              <td class="tc-uma">{h['umaban']}</td>
              <td class="tc-name">{esc(h['horse_name'])}
                <span class="tc-ped">父:{esc(h.get('sire_name')) or '-'} / 母父:{esc(h.get('bms_name')) or '-'} / 父父:{esc(h.get('ss_name')) or '-'}</span>
                {bloodline_tag}</td>
              <td class="tc-comp">{comp_txt}</td>
              <td class="tc-damside">{damside_txt}<br>{damside_tags}</td>
              {''.join(cells)}
            </tr>""")

            # 各馬のミニレーダー用: 軸ごとに父・母父・父父の取得できたpt差の単純平均(総合適性と同じ父方3ライン)。
            # 全ラインが無い軸はnull(線を結ばない)にして「データなし=0pt=平均的」に見えるのを避ける。
            vals_h, nlines = [], []
            for key in axis_keys:
                per = []
                for r in ("sire", "bms", "ss"):
                    blk = h.get(r) or {}
                    x = (blk.get("items") or {}).get(key) or (blk.get("ssk") or {}).get(key)
                    per.append(x["excess_pt"] if x else None)
                vals_h.append(_mean(per))
                nlines.append(sum(v is not None for v in per))
            horse_radar.append({
                "umaban": h["umaban"], "name": h["horse_name"], "rank": rank,
                "composite": composite, "vals": vals_h, "nlines": nlines,
                "ped": f"{h.get('sire_name') or '-'} / {h.get('bms_name') or '-'} / {h.get('ss_name') or '-'}",
            })
    chart["horses"] = horse_radar

    today_header = "".join(f'<th{" class=hl" if key == hl_key else ""}>{esc(lbl)}</th>' for key, lbl in item_labels)
    today_header += "".join(f'<th class="tc-ssk">{esc(lbl)}</th>' for _key, lbl in ssk_labels)

    # --- 同等レースの定義(根拠) ---
    if rule["day_condition"]:
        day_item = f'<div class="method-item"><b>開催{day_no}日目</b>race_idの日目桁が今日と一致(開催回・年は問わない)</div>'
        day_bullet = f"開催日数(開催の何日目か=開催{day_no}日目のみ一致、開催回や年は問わない)"
    else:
        day_item = ('<div class="method-item"><b>日数は問わず</b>開催日数まで一致するレースが'
                    f'{rule["n_strict"]}件と少ないため条件を外して再検索</div>')
        day_bullet = (f"開催日数は条件に含めていません(開催{day_no}日目まで一致する同等レースが{rule['n_strict']}件と"
                      f"基準{d.get('min_races_strict', 15)}件に満たなかったため、条件を外して全{rule['n_relaxed']}件を対象にしました)")
    kind_txt = f"{rule['kind']}同士(新馬と未勝利は分けて比較)・" if rule.get("kind") else ""
    class_item = f'<div class="method-item"><b>{esc(info["class_name"])}</b>{esc(kind_txt)}年齢条件は問わず同一クラス帯のみ一致</div>'
    few_warn = ""
    if n_matched < FEW_RACES:
        few_warn = (f'<p class="few-warn">⚠ 同等の過去レースが{n_matched}件と少ないため、レース基準(上位馬の平均)は'
                    '偶然の偏りの影響を受けやすく、傾向は不安定です。参考程度にご覧ください。</p>')

    sec = f"""
<section class="race-sec" id="race-sec-{rno}" data-rno="{rno}" hidden>
  <header class="masthead">
    <div class="eyebrow">Race-Type Pedigree Radar</div>
    <h1>血統レーダーチャート: {esc(race_label)}</h1>
    <div class="subtitle">{esc(date_jp)} {esc(race_label)}・{esc(cond_label)}{esc(post)}</div>
  </header>

  <div class="card">
    <h2>同等レースの定義(根拠)</h2>
    <div class="method-grid">
      <div class="method-item"><b>{n_matched}レース</b>該当した過去の同条件レース(2016〜2026年)</div>
      <div class="method-item"><b>{n_scored}頭</b>血統照合できた上位3着馬(延べ)</div>
      {day_item}
      {class_item}
    </div>
    {few_warn}
    <ul>
      <li>一致条件: 競馬場({esc(venue)})・サーフェス({esc(surf_name)})・距離({dist}m、race_results実測値で厳密一致)・
        クラス({esc(info['class_name'])}、年齢条件の違いは同一帯として許容)・{day_bullet}。</li>
      <li>馬場状態(道悪)は今日はまだ確定しておらず(JRA当日発表のため取得ロジックが存在しない)、
        一致条件には含めていません。過去{n_matched}レースの実際の馬場状態の内訳は参考情報として以下に示します:
        {going_html}</li>
      <li>「上位馬」は複勝圏内(3着以内)、各馬の父・母父・父父それぞれについて、そのレース自身の実際の
        条件(距離{dist}m・{esc(surf_name)}・{esc(venue)}・そのレースが行われた季節・その馬自身の実際の休み明け間隔)に
        対する時点非依存の実測勝率差(pt)を算出し、{n_matched}レース分を平均しています。
        父父(SS)は種牡馬プロファイルに自身の産駒が登録されているケースが多く、新規データ取得なしで
        父・母父と同じ方法で第3のラインとして追加しています。</li>
    </ul>
  </div>

  <div class="card">
    <h2>このレースタイプで求められるファクター</h2>
    <p class="small-note">
      距離・サーフェス・競馬場等の既存8項目に加え、スピード(短距離帯pt差)・スタミナ(長距離帯pt差)・
      キレ(瞬発力、非先行-先行勝率差)の3項目を同じレーダーチャートに統合しています。いずれも「上位3着馬
      (延べ{n_scored}頭)の平均pt差」という同一の算出方法で、父・母父・父父の3ラインを重ねて描いています。
    </p>
    <div class="chart-wrap"><canvas id="radar-{rno}"></canvas></div>
    <div class="legend-row">
      <span class="sw"><span class="dot sire"></span>父ライン平均</span>
      <span class="sw"><span class="dot bms"></span>母父ライン平均</span>
      <span class="sw"><span class="dot ss"></span>父父ライン平均</span>
    </div>
    <table class="bt">
      <thead><tr><th>ファクター</th><th>父平均</th><th>母父平均</th><th>父父平均</th></tr></thead>
      <tbody>{''.join(breakdown_rows)}</tbody>
    </table>
    <p class="small-note" style="margin-top:10px;">{interp}</p>
  </div>

  <div class="card">
    <h2>各馬のレーダーチャート(レース基準との重ね合わせ)</h2>
    <p class="small-note">
      上のチャートと同じ11軸で、<b>灰色の破線=このレースの上位3着馬の3ライン平均(レース基準)</b>、
      <b>金色の実線=その馬の父・母父・父父の3ライン平均</b>を重ねています(総合適性スコア順)。
      基準線に近い形ほど「このレースタイプで上位に来た馬の血統」に近いという意味で、
      面積が大きいほど有利という意味ではありません。軸ごとに父・母父・父父のうち取得できたラインのみを
      平均し、3ラインとも無い軸は線を結びません(0ptとは扱いません)。±15ptを超える値は枠に丸めて描画し、
      マウスオーバー(タップ)で実際の値を表示します。
    </p>
    <div class="hr-grid" id="hrGrid-{rno}"></div>
  </div>

  <div class="card">
    <h2>母方ライン(母・母母・父母、参考)</h2>
    <p class="small-note">
      父方3ライン(父・母父・父父)は種牡馬プロファイルの高いカバレッジに支えられていますが、母馬は生涯の産駒数が
      少なく(全体で中央値2頭、5頭以上は約2割のみという既知の制約)、新設の繁殖牝馬プロファイル(role="dam")で
      N≥30(既存の父・母父と同じ基準)を満たす母馬はごく一部です。今日の出走馬{d.get('n_today_horses', 0)}頭中、
      母データあり{d.get('n_damside_hit', {}).get('dam', 0)}頭・
      母母{d.get('n_damside_hit', {}).get('dd', 0)}頭・
      父母{d.get('n_damside_hit', {}).get('sd', 0)}頭という低カバレッジを前提に、参考表として掲載します
      (父方3ラインと同じ重みで判断材料にすることは推奨しません)。
    </p>
    <table class="bt">
      <thead><tr><th>ファクター</th>{dam_side_header}</tr></thead>
      <tbody>{''.join(dam_side_rows)}</tbody>
    </table>
    <p class="dam-side-note"><b>読み方の注意:</b> 「N=0」は該当馬が種牡馬・母父・父父と違って
      産駒データを持たない(自身が繁殖記録の対象期間外、または産駒がまだ少数)ことを意味し、
      「不利」ではなく「不明」です。総合適性スコア(比較表)にはこの3ラインを含めていません
      (12レース版「中山血統適性台帳」と同じ、父・母父・父父の3ライン平均)。母方3ラインは比較表の
      「母方参考」列に別枠で示します。</p>
  </div>

  <div class="card">
    <h2>系統別成績(どの系統が有利か)</h2>
    <p class="small-note">
      netkeiba公式の「血統ビーム出馬表」が持つ父系統色分け(サンデーサイレンス系・ノーザンダンサー系
      など8分類、種牡馬個体に紐づく固定属性)を、2026年分の馬柱データから名前で再構築し、同等の過去
      {n_matched}レース全出走馬(延べ{n_field_horses}頭)を分類、系統ごとの複勝率を実測しました。
      個体ではなく系統という大きな括りで見るため、種牡馬単位のプロファイルより1系統あたりのNは大きく取れます
      (ただし系統間の実力差ではなく特定の強い種牡馬1頭が系統全体の数字を引き上げている可能性は残ります)。
    </p>
    {bloodline_table(bloodline_sire, "父の系統")}
    {bloodline_table(bloodline_bms, "母父の系統")}
    {bloodline_table(bloodline_ss, "父父の系統")}
  </div>

  <div class="card">
    <h2>スピード・スタミナ・キレの「良い塩梅」検証</h2>
    <p class="small-note">
      「値が極端なほど有利とは限らず、レースごとに適した中間の水準があるのでは」というご指摘を受け、
      上位3着馬だけでなく<b>同等の過去{n_matched}レースに出走した全{n_field_horses}頭</b>
      (上位・凡走問わず)を対象に、父・母父・父父ラインの値を単純平均した1頭あたりの数値で3分位
      (低位/中位/高位)に分け、分位ごとの複勝率(3着以内率)を実測しました。上位3着馬だけを見る
      従来の平均値とは独立した検証です。同じ父・母父・父父の組み合わせを持つ出走馬は同じ数値に
      なるため、実質的な独立標本は延べ頭数より少なく(組み合わせは{n_unique_pairs}通り)、
      95%信頼区間が重なる場合は「有意差なし」と正直に記載しています。
    </p>
    {''.join(balance_cards)}
    {dam_box_html}
  </div>

  <div class="card">
    <h2>今日の{esc(race_label)}出走馬との比較</h2>
    <div class="table-scroll">
    <table class="tcmp">
      <thead><tr><th>馬番</th><th>馬名(父/母父/父父・系統)</th><th>総合適性<br><span class="th-sub">(父・母父・父父の3ライン平均)</span></th><th>母方参考<br><span class="th-sub">(母/母母/父母)</span></th>{today_header}</tr></thead>
      <tbody>{''.join(today_rows)}</tbody>
    </table>
    </div>
    <p class="small-note" style="font-size:12px;margin-top:8px;">
      各セル上段=父・中段=母父・下段=父父のpt差(父父はデータが無い馬は「-」)。
      <span style="background:var(--highlight-bg);padding:1px 4px;border-radius:3px;">網掛け</span>は
      上のレーダーチャートで3ライン平均の絶対値が最も大きかった「{esc(hl_label)}」列です。
      <span class="sweet-swatch"></span>キレ列の緑の印は、「良い塩梅」検証で最も複勝率が高かった
      {esc(kire_best or '分位')}の範囲内にある値({'有意差あり' if kire_sig else '信頼区間が重なり有意差なし、参考'})です。
      スピード・スタミナには印を付けていません。
    </p>
  </div>

  <div class="card">
    <details class="evidence">
      <summary>▸ 根拠: 該当した過去{n_matched}レースと上位3着馬の一覧を表示</summary>
      <table class="ev">
        <thead><tr><th>年月日</th><th>レース名・馬場状態</th><th>上位3着馬(父/母父)</th></tr></thead>
        <tbody>{''.join(race_rows)}</tbody>
      </table>
    </details>
  </div>
</section>
"""
    nav = {"rno": rno, "label": f"{rno}R", "title": f"{rno}R {info['race_name']} {surf_name}{dist}m",
           "n_matched": n_matched, "relaxed": not rule["day_condition"]}
    return sec, chart, nav


CSS = r"""
:root {
  --bg: #f7f4ec; --bg-alt: #efe9d8; --ink: #23262f; --ink-soft: #565a66; --line: #d8d0ba;
  --gold: #a8813c; --gold-soft: #cdae6d; --sire: #8b3a3a; --sire-bg: #f7ecec;
  --bms: #33517d; --bms-bg: #ecf0f7; --ss: #3a7d4f; --dim: #9a9686; --card-bg: #ffffff;
  --shadow: 0 1px 2px rgba(35,38,47,0.06); --caution-bg: #f4e6e0; --caution-fg: #8b3a2c;
  --highlight-bg: #fcf3d9; --nav-bg: #fbf9f2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1b1c22; --bg-alt: #232530; --ink: #e9e6db; --ink-soft: #a7a49a; --line: #3a3c46;
    --gold: #d3b262; --gold-soft: #a98a4c; --sire: #e39a9a; --sire-bg: #33252a;
    --bms: #9db4dd; --bms-bg: #232c3d; --ss: #86c99b; --dim: #6d6f7a; --card-bg: #21232c;
    --shadow: 0 1px 3px rgba(0,0,0,0.4); --caution-bg: #3a2a24; --caution-fg: #e6a389;
    --highlight-bg: #3a331c; --nav-bg: #1f2029;
  }
}
:root[data-theme="dark"] {
  --bg: #1b1c22; --bg-alt: #232530; --ink: #e9e6db; --ink-soft: #a7a49a; --line: #3a3c46;
  --gold: #d3b262; --gold-soft: #a98a4c; --sire: #e39a9a; --sire-bg: #33252a;
  --bms: #9db4dd; --bms-bg: #232c3d; --ss: #86c99b; --dim: #6d6f7a; --card-bg: #21232c;
  --shadow: 0 1px 3px rgba(0,0,0,0.4); --caution-bg: #3a2a24; --caution-fg: #e6a389;
  --highlight-bg: #3a331c; --nav-bg: #1f2029;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: "Zen Kaku Gothic New","Hiragino Sans",sans-serif; line-height: 1.6; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
header.masthead { text-align: center; padding: 24px 12px 18px; border-bottom: 3px double var(--gold-soft); margin-bottom: 20px; }
.eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: 0.14em; color: var(--gold); text-transform: uppercase; }
h1 { font-family: "Shippori Mincho", serif; font-weight: 700; font-size: clamp(22px, 5vw, 32px); margin: 8px 0 6px; text-wrap: balance; }
.subtitle { color: var(--ink-soft); font-size: 14px; }
.caution-strip { background: var(--caution-bg); color: var(--caution-fg); font-size: 12px; font-weight: 600; text-align: center; padding: 8px 12px; border-radius: 6px; margin: 18px 0; }
.card { background: var(--card-bg); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin: 18px 0; box-shadow: var(--shadow); overflow-x: auto; }
.card h2 { font-family: "Shippori Mincho", serif; font-size: 17px; margin: 0 0 10px; color: var(--gold); }
.card ul { margin: 6px 0 0; padding-left: 1.2em; font-size: 13.5px; }
.card li { margin: 4px 0; }
.small-note { font-size: 12.5px; color: var(--ink-soft); }
.few-warn { background: var(--caution-bg); color: var(--caution-fg); font-size: 12.5px; font-weight: 600; border-radius: 6px; padding: 6px 10px; margin: 8px 0; }
.method-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 10px; margin: 12px 0; font-size: 12.5px; }
.method-item { background: var(--bg-alt); border-radius: 6px; padding: 8px 10px; }
.method-item b { display: block; font-family: "IBM Plex Mono", monospace; font-size: 15px; color: var(--gold); }
.going-chip { display: inline-block; background: var(--bg-alt); border-radius: 5px; padding: 3px 8px; font-size: 12px; margin: 2px 4px 2px 0; font-family: "IBM Plex Mono", monospace; }

.race-nav { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 30; background: var(--nav-bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin: 14px 0; display: flex; flex-wrap: wrap; gap: 6px; box-shadow: var(--shadow); }
.race-nav button { font-family: "IBM Plex Mono", monospace; font-size: 13px; color: var(--ink); background: var(--bg-alt); border: 1px solid var(--line); border-radius: 5px; padding: 5px 11px; cursor: pointer; }
.race-nav button:hover { border-color: var(--gold); color: var(--gold); }
.race-nav button.is-active { background: var(--gold); color: var(--bg); border-color: var(--gold); font-weight: 600; }
.race-nav button:disabled { opacity: 0.45; cursor: not-allowed; }
.race-nav button .rn-sub { display: block; font-size: 9.5px; font-weight: 400; font-family: "Zen Kaku Gothic New", sans-serif; }
.nav-note { font-size: 11.5px; color: var(--ink-soft); margin: -6px 0 10px; }

.chart-wrap { max-width: 520px; margin: 0 auto; }
.legend-row { display: flex; justify-content: center; gap: 18px; margin-top: 8px; font-size: 12.5px; flex-wrap: wrap; }
.legend-row .sw { display: inline-flex; align-items: center; gap: 5px; }
.legend-row .dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.hr-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; margin-top: 12px; }
.hr-card { border: 1px solid var(--line); border-radius: 8px; padding: 8px 8px 6px; background: var(--card-bg); min-width: 0; }
.hr-head { display: flex; align-items: baseline; gap: 6px; font-size: 13px; font-weight: 600; flex-wrap: wrap; }
.hr-uma { font-family: "IBM Plex Mono", monospace; background: var(--gold); color: var(--bg); border-radius: 4px; padding: 0 6px; font-size: 12px; }
.hr-rank { margin-left: auto; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--ink-soft); font-weight: 400; }
.hr-ped { font-size: 10.5px; color: var(--ink-soft); margin: 2px 0 4px; }
.hr-canvas { position: relative; width: 100%; aspect-ratio: 1 / 1; }

.dot.sire { background: var(--sire); }
.dot.bms { background: var(--bms); }
.dot.ss { background: var(--ss); }

table.bt { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
table.bt th, table.bt td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: center; }
table.bt th { font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-soft); }
.bt-label { text-align: left !important; font-weight: 600; }
.bt-sire { color: var(--sire); font-family: "IBM Plex Mono", monospace; }
.bt-bms { color: var(--bms); font-family: "IBM Plex Mono", monospace; }
.bt-ss { color: var(--ss); font-family: "IBM Plex Mono", monospace; }
.bt-n { color: var(--dim); font-size: 10.5px; margin-left: 4px; }

.table-scroll { overflow-x: auto; }
table.tcmp { border-collapse: collapse; width: 100%; min-width: 900px; font-size: 11.5px; }
table.tcmp th, table.tcmp td { border: 1px solid var(--line); padding: 5px 6px; text-align: center; }
table.tcmp thead th { background: var(--bg-alt); font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-soft); }
table.tcmp thead th.hl { background: var(--highlight-bg); color: var(--ink); }
.tc-uma { font-family: "IBM Plex Mono", monospace; font-weight: 600; }
.tc-name { text-align: left !important; font-weight: 600; min-width: 110px; }
.tc-ped { display: block; font-size: 10px; font-weight: 400; color: var(--ink-soft); }
.tc-comp { font-family: "IBM Plex Mono", monospace; font-weight: 700; }
.tc.hl { background: var(--highlight-bg); }
.tc-s { display: block; color: var(--sire); font-family: "IBM Plex Mono", monospace; }
.tc-b { display: block; color: var(--bms); font-family: "IBM Plex Mono", monospace; }
.tc-g { display: block; color: var(--ss); font-family: "IBM Plex Mono", monospace; }

details.evidence summary { cursor: pointer; font-size: 13px; color: var(--gold); padding: 6px 2px; }
table.ev { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
table.ev th, table.ev td { border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }
table.ev th { text-align: left; font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-soft); }
.ev-date { font-family: "IBM Plex Mono", monospace; white-space: nowrap; }
.ev-going { display: block; font-size: 10.5px; color: var(--ink-soft); }
.ev-horse { margin: 2px 0; }
.ev-ped { display: block; font-size: 10.5px; color: var(--ink-soft); margin-left: 14px; }

footer { text-align: center; color: var(--ink-soft); font-size: 12px; margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); }
footer a { color: var(--gold); }

.tier-badge { display: inline-block; font-size: 10px; font-weight: 700; border-radius: 4px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }
.tier-badge.tier-high { background: var(--gold); color: var(--bg); }
.tier-badge.tier-mid { background: var(--gold-soft); color: var(--bg); }
.tier-badge.tier-low { background: var(--bg-alt); color: var(--ink-soft); border: 1px solid var(--line); }
.tc-ssk { background: rgba(168,129,60,0.08); }
.tc-ssk.sweet { background: rgba(58,139,90,0.22); box-shadow: inset 0 0 0 1px rgba(58,139,90,0.5); }
.sweet-swatch { display: inline-block; width: 10px; height: 10px; background: rgba(58,139,90,0.5); box-shadow: inset 0 0 0 1px rgba(58,139,90,0.8); border-radius: 2px; vertical-align: middle; margin-right: 3px; }
.dam-box { border: 1px dashed var(--line); border-radius: 8px; padding: 12px 14px; margin-top: 10px; background: var(--bg-alt); }
.dam-box .dam-item { font-size: 12.5px; margin: 4px 0; }
.dam-caption { font-size: 12px; color: var(--ink-soft); margin-bottom: 6px; }

.bal-block { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
.bal-block:first-of-type { margin-top: 10px; padding-top: 0; border-top: none; }
.bal-block h3 { font-family: "Shippori Mincho", serif; font-size: 14.5px; margin: 0 0 8px; }
table.bal-table { margin-top: 4px; }
table.bal-table th { text-align: center; }
table.bal-table td { text-align: center; font-size: 12.5px; }
tr.bal-best { background: rgba(58,139,90,0.14); font-weight: 700; }
.bal-verdict { font-size: 13px; margin: 8px 0 2px; }
.bal-note { font-size: 12px; color: var(--ink-soft); margin: 4px 0; }

.bt-damside { color: var(--dim); font-family: "IBM Plex Mono", monospace; font-style: italic; }
.dam-side-note { font-size: 12px; color: var(--ink-soft); margin: 8px 0 0; }
.dam-side-note b { color: var(--ink); font-style: normal; }

.bl-block { margin-top: 16px; }
.bl-block:first-of-type { margin-top: 0; }
.bl-block h3 { font-family: "Shippori Mincho", serif; font-size: 14.5px; margin: 0 0 8px; }
table.bl-table { margin-top: 4px; }
table.bl-table td { text-align: left; }
td.bl-bar-cell { position: relative; min-width: 160px; }
.bl-bar { display: inline-block; height: 9px; background: var(--gold-soft); border-radius: 3px; vertical-align: middle; margin-right: 6px; }
.bl-rate { font-family: "IBM Plex Mono", monospace; font-size: 12px; }
.bl-tag { display: block; font-size: 10px; color: var(--ink-soft); font-family: "IBM Plex Mono", monospace; margin-top: 2px; }

.tc-damside { font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-soft); }
.damside-tag { display: inline-block; font-size: 9px; background: var(--bg-alt); border-radius: 3px; padding: 0 4px; margin: 1px 1px 0 0; color: var(--ink-soft); }
.th-sub { font-weight: 400; font-size: 9px; }
.dim { color: var(--dim); font-style: italic; }
"""

JS = r"""
const RACES = __RACES__;
const SHORT_LABELS = __SHORT__;
const CLAMP = 15;
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n);
const rendered = {};
const clamp = v => v === null ? null : Math.max(-CLAMP, Math.min(CLAMP, v));

function renderRace(rno) {
  if (rendered[rno]) return;
  rendered[rno] = true;
  const R = RACES[rno];
  new Chart(document.getElementById('radar-' + rno), {
    type: 'radar',
    data: {
      labels: R.labels,
      datasets: [
        { label: '父ライン平均pt', data: R.sire, borderColor: '#8b3a3a', backgroundColor: 'rgba(139,58,58,0.18)', pointBackgroundColor: '#8b3a3a', borderWidth: 2 },
        { label: '母父ライン平均pt', data: R.bms, borderColor: '#33517d', backgroundColor: 'rgba(51,81,125,0.15)', pointBackgroundColor: '#33517d', borderWidth: 2 },
        { label: '父父ライン平均pt', data: R.ss, borderColor: '#3a7d4f', backgroundColor: 'rgba(58,125,79,0.12)', pointBackgroundColor: '#3a7d4f', borderWidth: 2 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { r: {
        min: -12, max: 12,
        ticks: { stepSize: 4, backdropColor: 'transparent', color: cssv('--ink-soft') },
        grid: { color: cssv('--line') }, angleLines: { color: cssv('--line') },
        pointLabels: { color: cssv('--ink'), font: { size: 10.5 } }
      } }
    }
  });
  const ink = cssv('--ink'), line = cssv('--line');
  const grid = document.getElementById('hrGrid-' + rno);
  R.horses.forEach(function (h) {
    const card = document.createElement('div');
    card.className = 'hr-card';
    const comp = h.composite === null ? 'データ不足' : (h.composite >= 0 ? '+' : '') + h.composite.toFixed(1) + 'pt';
    const head = document.createElement('div');
    head.className = 'hr-head';
    const uma = document.createElement('span'); uma.className = 'hr-uma'; uma.textContent = h.umaban;
    const nm = document.createElement('span'); nm.textContent = h.name;
    const rk = document.createElement('span'); rk.className = 'hr-rank'; rk.textContent = '総合 ' + comp + ' / ' + h.rank + '位';
    head.append(uma, nm, rk);
    const ped = document.createElement('div'); ped.className = 'hr-ped'; ped.textContent = '父/母父/父父: ' + h.ped;
    const cv = document.createElement('div'); cv.className = 'hr-canvas';
    const canvas = document.createElement('canvas'); cv.appendChild(canvas);
    card.append(head, ped, cv);
    grid.appendChild(card);
    new Chart(canvas, {
      type: 'radar',
      data: {
        labels: SHORT_LABELS,
        datasets: [
          { label: 'レース基準(上位3着馬の3ライン平均)', data: R.ref.map(clamp), borderColor: '#8a8a8a', borderDash: [4, 3], backgroundColor: 'rgba(138,138,138,0.08)', pointRadius: 0, borderWidth: 1.5 },
          { label: h.name + '(3ライン平均)', data: h.vals.map(clamp), borderColor: '#a8813c', backgroundColor: 'rgba(168,129,60,0.22)', pointBackgroundColor: '#a8813c', pointRadius: 2, borderWidth: 2, spanGaps: true }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: function (c) {
            const real = c.datasetIndex === 0 ? R.ref[c.dataIndex] : h.vals[c.dataIndex];
            const extra = c.datasetIndex === 1 ? ' (' + h.nlines[c.dataIndex] + 'ライン)' : '';
            return c.dataset.label.split('(')[0] + ': ' + (real === null ? 'データなし' : (real >= 0 ? '+' : '') + real.toFixed(1) + 'pt') + extra;
          } } }
        },
        scales: { r: {
          min: -CLAMP, max: CLAMP, ticks: { display: false, stepSize: 5 },
          grid: { color: line }, angleLines: { color: line },
          pointLabels: { color: ink, font: { size: 9.5 } }
        } }
      }
    });
  });
}

function showRace(rno, pushHash) {
  document.querySelectorAll('.race-sec').forEach(function (s) { s.hidden = s.dataset.rno !== String(rno); });
  document.querySelectorAll('.race-nav button[data-rno]').forEach(function (b) {
    b.classList.toggle('is-active', b.dataset.rno === String(rno));
  });
  renderRace(rno);
  if (pushHash) { try { history.replaceState(null, '', '#r' + rno); } catch (e) { location.hash = 'r' + rno; } }
}

document.querySelectorAll('.race-nav button[data-rno]').forEach(function (b) {
  b.addEventListener('click', function () { showRace(b.dataset.rno, true); window.scrollTo({ top: 0 }); });
});
(function () {
  const m = /^#r(\d+)$/.exec(location.hash || '');
  const first = Object.keys(RACES)[0];
  showRace(m && RACES[m[1]] ? m[1] : first, false);
})();
"""


def main():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    generated_date = date.today().isoformat()
    races = payload["races"]
    skipped = payload.get("skipped", {})
    rnos = sorted(races, key=int)
    if not rnos:
        raise SystemExit("掲載できるレースがありません")

    sections, charts, navs = [], {}, {}
    for rno in rnos:
        d = races[rno]
        d["min_races_strict"] = payload.get("min_races_strict", 15)
        sec, chart, nav = render_race(d, int(rno))
        sections.append(sec)
        charts[rno] = chart
        navs[rno] = nav

    dt = datetime.strptime(payload["race_date"], "%Y-%m-%d")
    date_jp = f"{dt.year}年{dt.month}月{dt.day}日({WEEKDAY_JA[dt.weekday()]})"
    venue = races[rnos[0]]["race_info"]["venue"]

    nav_buttons = []
    all_rnos = sorted({int(k) for k in list(races) + list(skipped)})
    for rno in all_rnos:
        key = str(rno)
        if key in races:
            n = navs[key]
            sub = f'<span class="rn-sub">{n["n_matched"]}件{"・日数緩和" if n["relaxed"] else ""}</span>'
            nav_buttons.append(f'<button type="button" data-rno="{rno}" title="{esc(n["title"])}">{rno}R{sub}</button>')
        else:
            nav_buttons.append(
                f'<button type="button" disabled title="{esc(skipped[key]["reason"])}">{rno}R'
                f'<span class="rn-sub">対象外</span></button>')
    skipped_txt = "".join(
        f"{rno}R({esc(v['race_name'])}): {esc(v['reason'])}。" for rno, v in sorted(skipped.items(), key=lambda kv: int(kv[0])))

    n_races = len(rnos)
    desc = (f"{date_jp}・{venue}開催の{n_races}レースについて、同条件の過去レース上位3着馬の血統から求められるファクターを"
            "レーダーチャート化し、各馬のレーダーチャートをレース基準と重ねて表示。系統別成績・スピード/スタミナ/キレの検証も掲載。")

    page = f"""<title>血統レーダーチャート</title>
<meta name="description" content="{esc(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{CSS}</style>
<div class="wrap">
  <div class="caution-strip">
    ⚠ 本レポートは過去の同条件レース上位馬の血統を事後集計した記述統計です。血統単独モデルは
    Nested OOF検証で市場超過を確認できず不採用と判定済みであり、検証済みの予想シグナルではありません。
  </div>

  <div class="card">
    <h2>{esc(date_jp)} {esc(venue)}開催 血統レーダー({n_races}レース)</h2>
    <p class="small-note">
      上のレース番号ボタンで切り替えます。各レースとも、同じ競馬場・サーフェス・距離・クラスの過去レース(2016〜2026年)の
      上位3着馬について、父・母父・父父のそのレース条件でのpt差(該当区分の勝率−その祖先の総合勝率)を平均して
      「このレースタイプで求められるファクター」とし、今日の出走馬それぞれの値と重ねています。
      同じ開催日数(開催の何日目か)まで一致するレースが{payload.get('min_races_strict', 15)}件以上あればその条件も使い、
      少ない場合は日数条件を外して件数を確保します(ボタンに「日数緩和」と表示)。緩和後も{payload.get('min_races_publish', 10)}件に
      満たないレースは掲載しません。クラスは出走表のレース名から判定し、新馬と未勝利は分けて比較します。
    </p>
    <p class="small-note">
      <b>総合適性スコアの定義(2026-10-03に統一):</b> 12レース版「中山血統適性台帳」と同じ、父・母父・父父の3ライン平均です。
      母・母母・父母の母方3ラインは該当馬が半数前後で件数も少ないため、総合に含めず「母方参考」列に分けています
      (旧版の6ライン平均では、8Rで旧3位の馬が14位になるなど順位が大きく動きました)。
    </p>
    {('<p class="small-note"><b>対象外のレース:</b> ' + skipped_txt + '</p>') if skipped else ''}
  </div>

  <nav class="race-nav" aria-label="レース切替">{''.join(nav_buttons)}</nav>
  <div class="nav-note">ボタン下段=同等の過去レース件数。URL末尾に #r8 を付けると8Rを直接開きます。</div>

  {''.join(sections)}

  <footer>
    データ: data/race_results(2016〜2026年)× data/pedigree(5代血統表)。
    関連: <a href="https://claude.ai/artifact/6T1zVng4xQNL9enCKJ6NX7" target="_blank" rel="noopener">中山血統適性台帳</a>
    生成日: {generated_date}
  </footer>
</div>
<script>{JS.replace('__RACES__', json.dumps(charts, ensure_ascii=False)).replace('__SHORT__', json.dumps([SHORT.get(k, k) for k, _l in (races[rnos[0]]['item_labels'] + races[rnos[0]]['ssk_labels'])], ensure_ascii=False))}</script>
"""
    OUT_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(page)} chars, {n_races}レース)")


if __name__ == "__main__":
    main()
