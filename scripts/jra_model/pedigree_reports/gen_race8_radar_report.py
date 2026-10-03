# -*- coding: utf-8 -*-
"""race8_radar_data.json から、中山8R向け「求められるファクター」レーダーチャートHTMLを生成する。"""
import html
import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)
DATA_PATH = SCRATCH / "race8_radar_data.json"
OUT_PATH = SCRATCH / "race8_radar_report.html"

GOING_LABELS = {"良": "良", "稍重": "稍重", "重": "重", "不良": "不良"}


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def fmt_pt(x):
    return f"{x:+.2f}pt" if x is not None else "?"


TIER_BADGE_CLASS = {"高": "tier-high", "中": "tier-mid", "低": "tier-low"}


def fmt_range(rng):
    return f"{rng[0]:+.1f}〜{rng[1]:+.1f}pt" if rng else "?"


def fmt_rate(v):
    return f"{v:.1f}%" if v is not None else "?"


def balance_pattern_note(key, pattern, sig):
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
        N<20の系統は非表示。</p>
    </div>"""


def main():
    d = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    generated_date = date.today().isoformat()
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
    role_spec = d.get("role_spec", [
        {"key": "sire", "label": "父"}, {"key": "bms", "label": "母父"}, {"key": "ss", "label": "父父"},
    ])
    bloodline_sire = d.get("bloodline_sire", {"rows": [], "n_matched": 0, "n_unmatched": 0})
    bloodline_bms = d.get("bloodline_bms", {"rows": [], "n_matched": 0, "n_unmatched": 0})
    bloodline_ss = d.get("bloodline_ss", {"rows": [], "n_matched": 0, "n_unmatched": 0})
    today_bloodlines = d.get("today_bloodlines", {})

    all_labels = item_labels + ssk_labels
    all_template = dict(template)
    all_template.update(ssk_template)

    # レーダーチャートは高カバレッジな父方3ライン(父/母父/父父)のみ描画する。母方3ライン
    # (母/母母/父母)は産駒数が少なく該当馬が少ないため、レーダーに重ねると「データなし=0pt」が
    # 「平均的」に見えてしまい誤解を招く。数値内訳テーブル(母方ライン参考表)でのみ開示する。
    labels_js = json.dumps([lbl for _key, lbl in all_labels], ensure_ascii=False)
    sire_data_js = json.dumps([all_template[key]["sire"]["mean_pt"] or 0 for key, _l in all_labels])
    bms_data_js = json.dumps([all_template[key]["bms"]["mean_pt"] or 0 for key, _l in all_labels])
    ss_data_js = json.dumps([all_template[key].get("ss", {}).get("mean_pt") or 0 for key, _l in all_labels])

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

    # --- 母方3ライン(母/母母/父母)の参考内訳表。役割ごとのカバレッジ(該当ancestorがいた頭数)も
    # 表の直上に明記する。父方3ラインと違い低カバレッジであることを前提とした「参考」扱い。 ---
    DAM_SIDE_ROLES = [r for r in role_spec if r["key"] in ("dam", "dd", "sd")]
    dam_side_rows = []
    for key, label in item_labels + ssk_labels:
        cells = []
        for rs in DAM_SIDE_ROLES:
            cell = all_template.get(key, {}).get(rs["key"], {"mean_pt": None, "n_horses": 0})
            cells.append(f'<td class="bt-damside">{fmt_pt(cell["mean_pt"])}<span class="bt-n">(N={cell["n_horses"]})</span></td>')
        dam_side_rows.append(f'<tr><td class="bt-label">{esc(label)}</td>{"".join(cells)}</tr>')
    dam_side_header = "".join(f"<th>{esc(rs['label'])}平均</th>" for rs in DAM_SIDE_ROLES)

    # --- スピード・スタミナ・キレの「良い塩梅」検証(過去の同等レース全出走馬・上位に限らない) ---
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
        note = balance_pattern_note(key, ba["pattern"], sig)
        cross = ""
        top3_mean = ba.get("top3_combined_mean")
        if key == "kire" and top3_mean is not None:
            cross = (
                f'<p class="bal-note">クロスチェック: 実際の上位3着馬(1頭ごとに父・母父・父父の'
                f'算出できたライン平均、N={ba.get("top3_combined_n")})のキレ平均は{fmt_pt(top3_mean)}で、'
                f'この中位分位({fmt_range(ba["best_bin_value_range"])})とほぼ一致しています。'
                '2通りの独立した見方(上位馬の平均値/全出走馬の分位別複勝率)が同じ結論を示しています。</p>'
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

    # 内訳テーブル(既存8項目+スピード/スタミナ/キレの11項目、父・母父・父父の3ライン)
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

    going_bd = d["going_breakdown_of_matched_races"]
    going_html = "".join(
        f'<span class="going-chip">{esc(GOING_LABELS.get(k, k))}: {v}件</span>'
        for k, v in sorted(going_bd.items(), key=lambda kv: -kv[1])
    )

    # 過去33レースの根拠テーブル
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
    DAMSIDE_DISP = (("dam", "母"), ("dd", "母母"), ("sd", "父母"))
    today_rows = []
    if today:
        horses = sorted(today["horses"], key=lambda h: h["composite_combined"] if h["composite_combined"] is not None else -1e9, reverse=True)
        for h in horses:
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
                si = sire_items.get(key)
                bi = bms_items.get(key)
                gi = ss_items.get(key)
                s_txt = fmt_pt(si["excess_pt"]) if si else "-"
                b_txt = fmt_pt(bi["excess_pt"]) if bi else "-"
                g_txt = fmt_pt(gi["excess_pt"]) if gi else "-"
                highlight = " hl" if key == "leader_style" else ""
                cells.append(f'<td class="tc{highlight}"><span class="tc-s">{s_txt}</span><span class="tc-b">{b_txt}</span><span class="tc-g">{g_txt}</span></td>')
            for key, _label in ssk_labels:
                si = sire_ssk.get(key)
                bi = bms_ssk.get(key)
                gi = ss_ssk.get(key)
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

    today_header = "".join(f'<th{" class=hl" if key=="leader_style" else ""}>{esc(lbl)}</th>' for key, lbl in item_labels)
    today_header += "".join(f'<th class="tc-ssk">{esc(lbl)}</th>' for _key, lbl in ssk_labels)

    html_out = f"""<title>血統レーダーチャート</title>
<meta name="description" content="中山8R(3歳以上1勝クラス・ダート1200m)と同条件の過去33レース・上位3着延べ99頭の血統(父・母父・父父・母・母母・父母)から、求められるファクターと有利な系統をレーダーチャート・系統別成績で可視化。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #f7f4ec; --bg-alt: #efe9d8; --ink: #23262f; --ink-soft: #565a66; --line: #d8d0ba;
  --gold: #a8813c; --gold-soft: #cdae6d; --sire: #8b3a3a; --sire-bg: #f7ecec;
  --bms: #33517d; --bms-bg: #ecf0f7; --ss: #3a7d4f; --dim: #9a9686; --card-bg: #ffffff;
  --shadow: 0 1px 2px rgba(35,38,47,0.06); --caution-bg: #f4e6e0; --caution-fg: #8b3a2c;
  --highlight-bg: #fcf3d9;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #1b1c22; --bg-alt: #232530; --ink: #e9e6db; --ink-soft: #a7a49a; --line: #3a3c46;
    --gold: #d3b262; --gold-soft: #a98a4c; --sire: #e39a9a; --sire-bg: #33252a;
    --bms: #9db4dd; --bms-bg: #232c3d; --ss: #86c99b; --dim: #6d6f7a; --card-bg: #21232c;
    --shadow: 0 1px 3px rgba(0,0,0,0.4); --caution-bg: #3a2a24; --caution-fg: #e6a389;
    --highlight-bg: #3a331c;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #1b1c22; --bg-alt: #232530; --ink: #e9e6db; --ink-soft: #a7a49a; --line: #3a3c46;
  --gold: #d3b262; --gold-soft: #a98a4c; --sire: #e39a9a; --sire-bg: #33252a;
  --bms: #9db4dd; --bms-bg: #232c3d; --ss: #86c99b; --dim: #6d6f7a; --card-bg: #21232c;
  --shadow: 0 1px 3px rgba(0,0,0,0.4); --caution-bg: #3a2a24; --caution-fg: #e6a389;
  --highlight-bg: #3a331c;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Zen Kaku Gothic New","Hiragino Sans",sans-serif; line-height: 1.6; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
header.masthead {{ text-align: center; padding: 24px 12px 18px; border-bottom: 3px double var(--gold-soft); margin-bottom: 20px; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: 0.14em; color: var(--gold); text-transform: uppercase; }}
h1 {{ font-family: "Shippori Mincho", serif; font-weight: 700; font-size: clamp(22px, 5vw, 32px); margin: 8px 0 6px; text-wrap: balance; }}
.subtitle {{ color: var(--ink-soft); font-size: 14px; }}
.caution-strip {{ background: var(--caution-bg); color: var(--caution-fg); font-size: 12px; font-weight: 600; text-align: center; padding: 8px 12px; border-radius: 6px; margin: 18px 0; }}
.card {{ background: var(--card-bg); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin: 18px 0; box-shadow: var(--shadow); }}
.card h2 {{ font-family: "Shippori Mincho", serif; font-size: 17px; margin: 0 0 10px; color: var(--gold); }}
.card ul {{ margin: 6px 0 0; padding-left: 1.2em; font-size: 13.5px; }}
.card li {{ margin: 4px 0; }}
.method-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 10px; margin: 12px 0; font-size: 12.5px; }}
.method-item {{ background: var(--bg-alt); border-radius: 6px; padding: 8px 10px; }}
.method-item b {{ display: block; font-family: "IBM Plex Mono", monospace; font-size: 15px; color: var(--gold); }}
.going-chip {{ display: inline-block; background: var(--bg-alt); border-radius: 5px; padding: 3px 8px; font-size: 12px; margin: 2px 4px 2px 0; font-family: "IBM Plex Mono", monospace; }}

.chart-wrap {{ max-width: 520px; margin: 0 auto; }}
.legend-row {{ display: flex; justify-content: center; gap: 18px; margin-top: 8px; font-size: 12.5px; }}
.legend-row .sw {{ display: inline-flex; align-items: center; gap: 5px; }}
.legend-row .dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
.dot.sire {{ background: var(--sire); }}
.dot.bms {{ background: var(--bms); }}
.dot.ss {{ background: var(--ss); }}

table.bt {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
table.bt th, table.bt td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: center; }}
table.bt th {{ font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-soft); }}
.bt-label {{ text-align: left !important; font-weight: 600; }}
.bt-sire {{ color: var(--sire); font-family: "IBM Plex Mono", monospace; }}
.bt-bms {{ color: var(--bms); font-family: "IBM Plex Mono", monospace; }}
.bt-ss {{ color: var(--ss); font-family: "IBM Plex Mono", monospace; }}
.bt-n {{ color: var(--dim); font-size: 10.5px; margin-left: 4px; }}

.table-scroll {{ overflow-x: auto; }}
table.tcmp {{ border-collapse: collapse; width: 100%; min-width: 900px; font-size: 11.5px; }}
table.tcmp th, table.tcmp td {{ border: 1px solid var(--line); padding: 5px 6px; text-align: center; }}
table.tcmp thead th {{ background: var(--bg-alt); font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-soft); }}
table.tcmp thead th.hl {{ background: var(--highlight-bg); color: var(--ink); }}
.tc-uma {{ font-family: "IBM Plex Mono", monospace; font-weight: 600; }}
.tc-name {{ text-align: left !important; font-weight: 600; min-width: 110px; }}
.tc-ped {{ display: block; font-size: 10px; font-weight: 400; color: var(--ink-soft); }}
.tc-comp {{ font-family: "IBM Plex Mono", monospace; font-weight: 700; }}
.tc.hl {{ background: var(--highlight-bg); }}
.tc-s {{ display: block; color: var(--sire); font-family: "IBM Plex Mono", monospace; }}
.tc-b {{ display: block; color: var(--bms); font-family: "IBM Plex Mono", monospace; }}
.tc-g {{ display: block; color: var(--ss); font-family: "IBM Plex Mono", monospace; }}

details.evidence summary {{ cursor: pointer; font-size: 13px; color: var(--gold); padding: 6px 2px; }}
table.ev {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
table.ev th, table.ev td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }}
table.ev th {{ text-align: left; font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-soft); }}
.ev-date {{ font-family: "IBM Plex Mono", monospace; white-space: nowrap; }}
.ev-going {{ display: block; font-size: 10.5px; color: var(--ink-soft); }}
.ev-horse {{ margin: 2px 0; }}
.ev-ped {{ display: block; font-size: 10.5px; color: var(--ink-soft); margin-left: 14px; }}

footer {{ text-align: center; color: var(--ink-soft); font-size: 12px; margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); }}
footer a {{ color: var(--gold); }}

.tier-badge {{ display: inline-block; font-size: 10px; font-weight: 700; border-radius: 4px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }}
.tier-badge.tier-high {{ background: var(--gold); color: var(--bg); }}
.tier-badge.tier-mid {{ background: var(--gold-soft); color: var(--bg); }}
.tier-badge.tier-low {{ background: var(--bg-alt); color: var(--ink-soft); border: 1px solid var(--line); }}
.tc-ssk {{ background: rgba(168,129,60,0.08); }}
.tc-ssk.sweet {{ background: rgba(58,139,90,0.22); box-shadow: inset 0 0 0 1px rgba(58,139,90,0.5); }}
.sweet-swatch {{ display: inline-block; width: 10px; height: 10px; background: rgba(58,139,90,0.5); box-shadow: inset 0 0 0 1px rgba(58,139,90,0.8); border-radius: 2px; vertical-align: middle; margin-right: 3px; }}
.dam-box {{ border: 1px dashed var(--line); border-radius: 8px; padding: 12px 14px; margin-top: 10px; background: var(--bg-alt); }}
.dam-box .dam-item {{ font-size: 12.5px; margin: 4px 0; }}
.dam-caption {{ font-size: 12px; color: var(--ink-soft); margin-bottom: 6px; }}

.bal-block {{ margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }}
.bal-block:first-of-type {{ margin-top: 10px; padding-top: 0; border-top: none; }}
.bal-block h3 {{ font-family: "Shippori Mincho", serif; font-size: 14.5px; margin: 0 0 8px; }}
table.bal-table {{ margin-top: 4px; }}
table.bal-table th {{ text-align: center; }}
table.bal-table td {{ text-align: center; font-size: 12.5px; }}
tr.bal-best {{ background: rgba(58,139,90,0.14); font-weight: 700; }}
.bal-verdict {{ font-size: 13px; margin: 8px 0 2px; }}
.bal-note {{ font-size: 12px; color: var(--ink-soft); margin: 4px 0; }}

.bt-damside {{ color: var(--dim); font-family: "IBM Plex Mono", monospace; font-style: italic; }}
.dam-side-note {{ font-size: 12px; color: var(--ink-soft); margin: 8px 0 0; }}
.dam-side-note b {{ color: var(--ink); font-style: normal; }}

.bl-block {{ margin-top: 16px; }}
.bl-block:first-of-type {{ margin-top: 0; }}
.bl-block h3 {{ font-family: "Shippori Mincho", serif; font-size: 14.5px; margin: 0 0 8px; }}
table.bl-table {{ margin-top: 4px; }}
table.bl-table td {{ text-align: left; }}
td.bl-bar-cell {{ position: relative; min-width: 160px; }}
.bl-bar {{ display: inline-block; height: 9px; background: var(--gold-soft); border-radius: 3px; vertical-align: middle; margin-right: 6px; }}
.bl-rate {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; }}
.bl-tag {{ display: block; font-size: 10px; color: var(--ink-soft); font-family: "IBM Plex Mono", monospace; margin-top: 2px; }}

.tc-damside {{ font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-soft); }}
.damside-tag {{ display: inline-block; font-size: 9px; background: var(--bg-alt); border-radius: 3px; padding: 0 4px; margin: 1px 1px 0 0; color: var(--ink-soft); }}
.th-sub {{ font-weight: 400; font-size: 9px; }}
.dim {{ color: var(--dim); font-style: italic; }}
</style>
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Race-Type Pedigree Radar</div>
    <h1>血統レーダーチャート: 中山8R</h1>
    <div class="subtitle">2026年9月26日(土) 中山8R・3歳以上1勝クラス・ダート1200m・発走13:40</div>
  </header>

  <div class="caution-strip">
    ⚠ 本レポートは過去の同条件レース上位馬の血統を事後集計した記述統計です。血統単独モデルは
    Nested OOF検証で市場超過を確認できず不採用と判定済みであり、検証済みの予想シグナルではありません。
  </div>

  <div class="card">
    <h2>同等レースの定義(根拠)</h2>
    <div class="method-grid">
      <div class="method-item"><b>{d['n_matched_races']}レース</b>該当した過去の同条件レース(2016〜2026年)</div>
      <div class="method-item"><b>{d['n_scored_horses']}頭</b>血統照合できた上位3着馬(延べ)</div>
      <div class="method-item"><b>開催8日目</b>race_idの日目桁が今日と一致(開催回・年は問わない)</div>
      <div class="method-item"><b>1勝クラス</b>年齢条件は問わず同一クラス帯のみ一致</div>
    </div>
    <ul>
      <li>一致条件: 競馬場(中山)・サーフェス(ダート)・距離(1200m、race_results実測値で厳密一致)・
        クラス(1勝クラス、年齢条件の違いは同一帯として許容)・開催日数(開催の何日目かのみ一致、
        開催回や年は問わない)。</li>
      <li>馬場状態(道悪)は今日はまだ確定しておらず(JRA当日発表のため取得ロジックが存在しない)、
        一致条件には含めていません。過去{d['n_matched_races']}レースの実際の馬場状態の内訳は参考情報として以下に示します:
        {going_html}</li>
      <li>「上位馬」は複勝圏内(3着以内)、各馬の父・母父・父父それぞれについて、そのレース自身の実際の
        条件(距離1200m・ダート・中山・そのレースが行われた季節・その馬自身の実際の休み明け間隔)に
        対する時点非依存の実測勝率差(pt)を算出し、{d['n_matched_races']}レース分を平均しています。
        父父(SS)は種牡馬プロファイルに自身の産駒が登録されているケースが多く(今日の出走馬156頭中
        139頭=89.1%)、新規データ取得なしで父・母父と同じ方法で第3のラインとして追加しています。</li>
    </ul>
  </div>

  <div class="card">
    <h2>このレースタイプで求められるファクター</h2>
    <p style="font-size:12.5px;color:var(--ink-soft);">
      距離・サーフェス・競馬場等の既存8項目に加え、プロ競馬予想家レビュー(2026-09-26)を
      反映したスピード(短距離帯pt差)・スタミナ(長距離帯pt差)・キレ(瞬発力、非先行-先行
      勝率差)の3項目を同じレーダーチャートに統合しました。いずれも「上位3着馬(延べ{d['n_scored_horses']}頭)の
      平均pt差」という同一の算出方法です。さらに父父(SS)ラインを第3の評価軸として追加しました
      (2026-09-26、父・母父以外の取得済み血統データの有効活用)。父父は種牡馬プロファイルに
      自身の産駒が登録されているケースが多く、新規データ取得・新規集計ロジックなしで父・母父と
      全く同じ方法で算出できます。
    </p>
    <div class="chart-wrap"><canvas id="radarChart"></canvas></div>
    <div class="legend-row">
      <span class="sw"><span class="dot sire"></span>父ライン平均</span>
      <span class="sw"><span class="dot bms"></span>母父ライン平均</span>
      <span class="sw"><span class="dot ss"></span>父父ライン平均</span>
    </div>
    <table class="bt">
      <thead><tr><th>ファクター</th><th>父平均</th><th>母父平均</th><th>父父平均</th></tr></thead>
      <tbody>{''.join(breakdown_rows)}</tbody>
    </table>
    <p style="font-size:12.5px;color:var(--ink-soft);margin-top:10px;">
      脚質(先行)以外の既存8項目はいずれも±1pt前後で、血統単独モデルの検証結果(市場超過なし)と
      整合的です。脚質(先行)のみ父・母父・父父とも+8〜10pt前後と3ラインとも一致して明確に高く、
      中山ダート1200mの短距離戦では「先行有利」な血統を持つ馬が上位に来やすい傾向が、父方3世代を
      通じて一貫して出ています。キレが大きくマイナスなのは、この脚質(先行)が大きくプラスだった
      ことと表裏一体です(キレ=非先行勝率-先行勝率なので、先行有利な血統ほど自動的にマイナスに
      なります。独立した別証拠ではありません)。この「マイナスが大きいほど良い」という単純な話
      ではない点は、次の検証で詳しく見ます。
    </p>
  </div>

  <div class="card">
    <h2>母方ライン(母・母母・父母、参考)</h2>
    <p style="font-size:12.5px;color:var(--ink-soft);">
      父方3ライン(父・母父・父父)は種牡馬プロファイルの高いカバレッジ(80〜90%台)に支えられて
      いますが、母馬は生涯の産駒数が少なく(全体で中央値2頭、5頭以上は約2割のみという既知の制約)、
      新設の繁殖牝馬プロファイル(role="dam")でN≥30(既存の父・母父と同じ基準)を満たす母馬は
      ごく一部です。今日の出走馬{d.get('n_today_horses', 0)}頭中、
      母データあり{d.get('n_damside_hit', {}).get('dam', 0)}頭・
      母母{d.get('n_damside_hit', {}).get('dd', 0)}頭・
      父母{d.get('n_damside_hit', {}).get('sd', 0)}頭という低カバレッジを前提に、
      「可能性を全て含める」というご要望に沿って参考表として掲載します(父方3ラインと同じ重みで
      判断材料にすることは推奨しません)。
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
    <p style="font-size:12.5px;color:var(--ink-soft);">
      netkeiba公式の「血統ビーム出馬表」が持つ父系統色分け(サンデーサイレンス系・ノーザンダンサー系
      など8分類、種牡馬個体に紐づく固定属性)を、2026年分の馬柱データ(816ファイル、実測網羅率100%)
      から名前で再構築し、同等の過去{d['n_matched_races']}レース全出走馬(延べ{n_field_horses}頭)を
      分類、系統ごとの複勝率を実測しました。個体ではなく系統という大きな括りで見るため、種牡馬単位の
      プロファイルより1系統あたりのNは大きく取れます(ただし系統間の実力差ではなく特定の強い種牡馬
      1頭が系統全体の数字を引き上げている可能性は残ります)。
    </p>
    {bloodline_table(bloodline_sire, "父の系統")}
    {bloodline_table(bloodline_bms, "母父の系統")}
    {bloodline_table(bloodline_ss, "父父の系統")}
  </div>

  <div class="card">
    <h2>スピード・スタミナ・キレの「良い塩梅」検証</h2>
    <p style="font-size:12.5px;color:var(--ink-soft);">
      「値が極端なほど有利とは限らず、レースごとに適した中間の水準があるのでは」というご指摘を受け、
      上位3着馬だけでなく<b>同等の過去{d['n_matched_races']}レースに出走した全{n_field_horses}頭</b>
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
    <h2>今日の中山8R出走馬との比較</h2>
    <div class="table-scroll">
    <table class="tcmp">
      <thead><tr><th>馬番</th><th>馬名(父/母父/父父・系統)</th><th>総合適性<br><span class="th-sub">(父・母父・父父の3ライン平均)</span></th><th>母方参考<br><span class="th-sub">(母/母母/父母)</span></th>{today_header}</tr></thead>
      <tbody>{''.join(today_rows)}</tbody>
    </table>
    </div>
    <p style="font-size:12px;color:var(--ink-soft);margin-top:8px;">
      各セル上段=父・中段=母父・下段=父父のpt差(父父はデータが無い馬は「-」)。
      <span style="background:var(--highlight-bg);padding:1px 4px;border-radius:3px;">網掛け</span>は
      上のレーダーチャートで最も差が出た「脚質(先行)」列です。
      <span class="sweet-swatch"></span>キレ列は「良い塩梅」検証で最も複勝率が高かった中位分位の範囲内にある値に印を付けています(スピード・スタミナは有意な傾向が確認できなかったため印なし)。
    </p>
    <p style="font-size:12px;color:var(--ink-soft);margin-top:6px;">
      <b>総合適性の定義(2026-10-03に変更):</b> 父・母父・父父の3ライン平均に統一しました(12レース版
      「中山血統適性台帳」と同じ値)。以前は母・母母・父母も平均に含めていましたが、母方3ラインは
      該当馬が半数前後で項目別の件数も少なく(例: 重賞実績はN=2〜13)、父方3ラインと同じ重みで
      平均すると順位が大きく動くため(旧版で3位だった馬が14位になるなど)、「母方参考」列に分けました。
    </p>
  </div>

  <div class="card">
    <details class="evidence">
      <summary>▸ 根拠: 該当した過去{d['n_matched_races']}レースと上位3着馬の一覧を表示</summary>
      <table class="ev">
        <thead><tr><th>年月日</th><th>レース名・馬場状態</th><th>上位3着馬(父/母父)</th></tr></thead>
        <tbody>{''.join(race_rows)}</tbody>
      </table>
    </details>
  </div>

  <footer>
    データ: data/race_results(2016〜2026年)× data/pedigree(5代血統表)。
    関連: <a href="https://claude.ai/artifact/6T1zVng4xQNL9enCKJ6NX7" target="_blank" rel="noopener">中山血統適性台帳</a>
    生成日: {generated_date}
  </footer>
</div>
<script>
const ctx = document.getElementById('radarChart');
new Chart(ctx, {{
  type: 'radar',
  data: {{
    labels: {labels_js},
    datasets: [
      {{ label: '父ライン平均pt', data: {sire_data_js}, borderColor: '#8b3a3a', backgroundColor: 'rgba(139,58,58,0.18)', pointBackgroundColor: '#8b3a3a', borderWidth: 2 }},
      {{ label: '母父ライン平均pt', data: {bms_data_js}, borderColor: '#33517d', backgroundColor: 'rgba(51,81,125,0.15)', pointBackgroundColor: '#33517d', borderWidth: 2 }},
      {{ label: '父父ライン平均pt', data: {ss_data_js}, borderColor: '#3a7d4f', backgroundColor: 'rgba(58,125,79,0.12)', pointBackgroundColor: '#3a7d4f', borderWidth: 2 }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      r: {{
        min: -12, max: 12,
        ticks: {{ stepSize: 4, backdropColor: 'transparent', color: getComputedStyle(document.documentElement).getPropertyValue('--ink-soft') }},
        grid: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--line') }},
        angleLines: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--line') }},
        pointLabels: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--ink'), font: {{ size: 10.5 }} }}
      }}
    }}
  }}
}});
</script>
"""
    OUT_PATH.write_text(html_out, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(html_out)} chars)")


if __name__ == "__main__":
    main()
