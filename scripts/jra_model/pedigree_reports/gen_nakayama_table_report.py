# -*- coding: utf-8 -*-
"""nakayama_today_data.json(14項目詳細)+ race_fit_score_data.json(総合適性スコア)から、
今日の中山12レース分の血統適性台帳HTMLを生成する。"""
import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)

DATA_PATH = SCRATCH / "nakayama_today_data.json"
SCORE_PATH = SCRATCH / "race_fit_score_data.json"
OUT_PATH = SCRATCH / "nakayama_today_pedigree_table.html"

FACTOR_COLS = [
    ("surface", "芝・ダート"),
    ("distance", "距離"),
    ("going", "道悪(馬場状態)"),
    ("turf_type", "洋芝/野芝"),
    ("course", "得意競馬場"),
    ("turn", "右/左回り"),
    ("elevation", "坂適性"),
    ("circumference", "コース規模"),
    ("season", "得意季節"),
    ("rest", "休み明け/連闘"),
]

SCORE_ITEM_LABELS = [
    ("distance", "距離(本日該当)"),
    ("surface", "芝ダート(本日)"),
    ("course", "競馬場(中山)"),
    ("season", "季節(秋)"),
    ("turf_type", "洋芝/野芝"),
    ("rest", "休み明け(実間隔)"),
    ("debut", "新馬戦実績"),
    ("graded", "重賞実績"),
    ("leader_style", "脚質(先行)"),
]

LOW_N = 20


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "?"


def fmt_pt(x):
    return f"{x:+.1f}pt" if x is not None else "?"


# ---------- 既存14項目詳細表(従来ロジック、変更なし) ----------

def factor_cell_role(block, key):
    if block is None:
        return '<span class="dim">対象外</span>'
    f = block["factors"].get(key)
    if f is None:
        return '<span class="dim">-</span>'
    low = f["n"] < LOW_N
    cls = "lown" if low else ""
    return (f'<span class="fval">{esc(f["value"])}</span>'
            f'<span class="fpct {cls}">{fmt_pct(f["win_rate"])}(N={f["n"]})</span>')


def factor_cell(sire, bms, key):
    return (
        f'<div class="cell2"><div class="row-sire">{factor_cell_role(sire, key)}</div>'
        f'<div class="row-bms">{factor_cell_role(bms, key)}</div></div>'
    )


def stat_cell_role(block, kind):
    if block is None:
        return '<span class="dim">対象外</span>'
    if kind == "debut":
        d = block.get("debut")
        if d is None:
            return '<span class="dim">-</span>'
        low = d["n"] < LOW_N
        return f'<span class="fpct {"lown" if low else ""}">{fmt_pct(d["win_rate"])}(N={d["n"]})</span>'
    if kind == "graded":
        d = block.get("graded")
        if d is None:
            return '<span class="dim">-</span>'
        low = d["n"] < LOW_N
        return f'<span class="fpct {"lown" if low else ""}">{fmt_pct(d["win_rate"])}(N={d["n"]})</span>'
    if kind == "running_style":
        d = block.get("running_style")
        if d is None:
            return '<span class="dim">-</span>'
        diff = d["leader_win_rate"] - d["non_leader_win_rate"]
        tendency = "先行有利" if diff > 0 else "差追も可"
        low = min(d["leader_n"], d["non_leader_n"]) < LOW_N
        return (f'<span class="fval">{tendency}</span>'
                f'<span class="fpct {"lown" if low else ""}">先{fmt_pct(d["leader_win_rate"])}'
                f'/非先{fmt_pct(d["non_leader_win_rate"])}</span>')
    if kind == "maturation":
        skew = block.get("maturation_age_skew")
        if skew is None:
            return '<span class="dim">-</span>'
        tendency = "早熟" if skew < -0.1 else ("晩成" if skew > 0.1 else "中間")
        return f'<span class="fval">{tendency}</span><span class="fpct">skew{skew:+.2f}</span>'
    return '<span class="dim">-</span>'


def stat_cell(sire, bms, kind):
    return (
        f'<div class="cell2"><div class="row-sire">{stat_cell_role(sire, kind)}</div>'
        f'<div class="row-bms">{stat_cell_role(bms, kind)}</div></div>'
    )


def render_detail_table(race, n_horses):
    rows = []
    for h in race["horses"]:
        sire, bms = h["sire"], h["bms"]
        tds = [
            f'<td class="sticky c-umaban">{h["umaban"]}</td>',
            f'<td class="sticky c-waku waku-{h["waku"]}">{h["waku"]}</td>',
            f'<td class="sticky c-name">{esc(h["horse_name"])}</td>',
            f'<td class="c-pedname"><div class="cell2"><div class="row-sire">{esc(h["sire_name"]) or "-"}</div>'
            f'<div class="row-bms">{esc(h["bms_name"]) or "-"}</div></div></td>',
        ]
        for key, _label in FACTOR_COLS:
            tds.append(f"<td>{factor_cell(sire, bms, key)}</td>")
        tds.append(f"<td>{stat_cell(sire, bms, 'debut')}</td>")
        tds.append(f"<td>{stat_cell(sire, bms, 'graded')}</td>")
        tds.append(f"<td>{stat_cell(sire, bms, 'running_style')}</td>")
        tds.append(f"<td>{stat_cell(sire, bms, 'maturation')}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")

    header_factor_ths = "".join(f'<th>{esc(label)}</th>' for _key, label in FACTOR_COLS)
    return f"""
    <details class="detail-toggle">
      <summary>14項目詳細表(父・母父の史上ベストカテゴリ、{n_horses}頭分)を表示 ▸</summary>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th class="sticky c-umaban" rowspan="2">馬番</th>
              <th class="sticky c-waku" rowspan="2">枠</th>
              <th class="sticky c-name" rowspan="2">馬名</th>
              <th class="c-pedname" rowspan="2">父 / 母父</th>
              <th colspan="{len(FACTOR_COLS)}" class="group-head">適性(各項目: 上段=父 / 下段=母父)</th>
              <th colspan="4" class="group-head">実績指標(上段=父 / 下段=母父)</th>
            </tr>
            <tr>
              {header_factor_ths}
              <th>新馬戦実績</th>
              <th>重賞実績</th>
              <th>脚質傾向</th>
              <th>早晩傾向</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </details>
    """


# ---------- 新設: 総合適性スコアのサマリー層 ----------

def score_pill(score):
    if score is None:
        return '<span class="score-pill s-na">データ不足</span>'
    if score >= 3:
        cls = "s-strong-pos"
    elif score >= 0.5:
        cls = "s-mild-pos"
    elif score > -0.5:
        cls = "s-neutral"
    elif score > -3:
        cls = "s-mild-neg"
    else:
        cls = "s-strong-neg"
    return f'<span class="score-pill {cls}">{fmt_pt(score)}</span>'


def rank_badge(label):
    if label == "得意":
        return '<span class="rank-badge good">得意</span>'
    if label == "不得意":
        return '<span class="rank-badge bad">不得意</span>'
    return ""


def maturation_compact(sire_score, bms_score, ss_score=None):
    parts = []
    if sire_score and sire_score.get("maturation"):
        parts.append("父" + sire_score["maturation"]["tendency"][0])
    if bms_score and bms_score.get("maturation"):
        parts.append("母" + bms_score["maturation"]["tendency"][0])
    if ss_score and ss_score.get("maturation"):
        parts.append("父父" + ss_score["maturation"]["tendency"][0])
    if not parts:
        return '<span class="dim">-</span>'
    return "".join(f'<span class="matur-badge">{p}</span>' for p in parts)


def breakdown_item(label, sire_item, bms_item, ss_item=None):
    def line(item, role_cls):
        if item is None:
            return f'<span class="{role_cls} bd-na">対象外</span>'
        cls = role_cls + ("" if item["significant"] else " insig")
        return f'<span class="{cls}">{fmt_pt(item["excess_pt"])}(N={item["n"]})</span>'

    return (
        f'<div class="bd-item"><span class="bd-label">{esc(label)}</span>'
        f'{line(sire_item, "bd-sire")}{line(bms_item, "bd-bms")}{line(ss_item, "bd-ss")}</div>'
    )


def render_summary_table(race_score, table_id):
    horses = sorted(
        race_score["horses"],
        key=lambda h: h["composite_combined"] if h["composite_combined"] is not None else -1e9,
        reverse=True,
    )
    rows = []
    for h in horses:
        score_attr = "" if h["composite_combined"] is None else h["composite_combined"]
        sire, bms, ss = h.get("sire"), h.get("bms"), h.get("ss")
        breakdown = "".join(
            breakdown_item(label, (sire or {}).get("items", {}).get(key) if sire else None,
                            (bms or {}).get("items", {}).get(key) if bms else None,
                            (ss or {}).get("items", {}).get(key) if ss else None)
            for key, label in SCORE_ITEM_LABELS
        )
        rest_txt = f'{h["rest_days"]}日' if h.get("rest_days") is not None else "-"
        row = f"""
        <tr class="summary-row" data-umaban="{h['umaban']}" data-score="{score_attr}">
          <td class="s-umaban">{h['umaban']}</td>
          <td class="s-waku waku-{h['waku']}">{h['waku']}</td>
          <td class="s-name">{esc(h['horse_name'])}
            <span class="ped-sub">父:{esc(h.get('sire_name')) or '-'} / 母父:{esc(h.get('bms_name')) or '-'} / 父父:{esc(h.get('ss_name')) or '-'}</span>
          </td>
          <td class="s-score">{score_pill(h['composite_combined'])}{rank_badge(h.get('rank_badge'))}</td>
          <td class="s-rest">{rest_txt}</td>
          <td class="s-matur">{maturation_compact(sire, bms, ss)}</td>
          <td class="s-detail">
            <details class="horse-detail"><summary>▸</summary>
              <div class="breakdown-grid">{breakdown}</div>
            </details>
          </td>
        </tr>
        """
        rows.append(row)

    return f"""
    <div class="summary-toolbar">
      <span class="summary-hint">総合適性スコア順に表示中(▸で内訳、右上ボタンで馬番順に切替可)</span>
      <button class="sort-toggle" data-target="{table_id}">馬番順に切替</button>
    </div>
    <div class="summary-scroll">
    <table class="summary-table" id="{table_id}" data-sort="score">
      <thead>
        <tr>
          <th>馬番</th><th>枠</th><th>馬名(父/母父)</th><th>総合適性</th>
          <th>休み明け</th><th>早晩</th><th>内訳</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    """


def render_race_section(race, race_score):
    n_horses = len(race["horses"])
    skip_note = ""
    if race_score and race_score.get("skipped_reason"):
        skip_note = f'<p class="summary-hint">※{esc(race_score["skipped_reason"])}</p>'
    summary_html = (
        render_summary_table(race_score, f"summary-{race['race_number']}")
        if race_score else '<p class="summary-hint">本日のレース条件データが無いため総合適性スコアは未算出です。</p>'
    )
    detail_html = render_detail_table(race, n_horses)
    return f"""
    <section class="race-block" id="race-{race['race_number']}">
      <h2 class="race-title"><span class="race-no">{race['race_number']}R</span>
        <span class="race-sub">中山 {race['race_number']}R・{n_horses}頭</span></h2>
      {skip_note}
      {summary_html}
      {detail_html}
    </section>
    """


def main():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    races = data["races"]
    total_horses = sum(len(r["horses"]) for r in races)

    score_by_number = {}
    if SCORE_PATH.exists():
        score_data = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
        score_by_number = {r["race_number"]: r for r in score_data["races"]}

    nav_links = "".join(
        f'<a href="#race-{r["race_number"]}">{r["race_number"]}R</a>' for r in races
    )
    race_sections = "".join(
        render_race_section(r, score_by_number.get(r["race_number"])) for r in races
    )

    template = Path(__file__).parent / "nakayama_table_template.html"
    tpl = template.read_text(encoding="utf-8")
    out = (tpl
           .replace("__NAV_LINKS__", nav_links)
           .replace("__RACE_SECTIONS__", race_sections)
           .replace("__TOTAL_RACES__", str(len(races)))
           .replace("__TOTAL_HORSES__", str(total_horses)))
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(out)} chars)")


if __name__ == "__main__":
    main()
