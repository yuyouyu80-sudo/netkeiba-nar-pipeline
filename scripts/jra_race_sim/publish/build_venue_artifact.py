# -*- coding: utf-8 -*-
"""競馬場ごとの相互作用シミュレーションHTML(全レースをhash切り替えで閲覧できる版)を組み立てる。
`race_json_display/{race_id}.json`と`data/jra_race_sim/race_names_{date}.csv`から、
`venue_page_template.py`のCSS/JSを使って自己完結HTMLを1本生成する。

2026-08-18、CLI引数化して1本に統合(旧`build_venue_artifact.py`(8/2専用)・
`build_venue_artifact_20260801.py`(8/1専用)は実質`RACE_DATE`/`RACE_NAMES_CSV`/出力ファイル名/
`VENUE_URLS`だけが差分のコピペ運用だった。開催日が増えるたびにファイルが増殖し「展開予想の
成果を継続的に公開できる状態にする」という目的と矛盾するため、シニアエンジニアレビューで
指摘され統合した)。

使用例:
    python build_venue_artifact.py --venue 新潟 --race-date 20260801 --out venue_20260801_新潟.html
"""
import argparse
import datetime
import json
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import corner_passing_metrics as cpm  # noqa: E402
import sim_geometry as sg  # noqa: E402

import actual_race_data as ard
import venue_page_template as tpl

REPO_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
ENGINE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RACE_NAMES_DIR = REPO_ROOT / "data" / "jra_race_sim"
DEFAULT_RACE_JSON_DIR = ENGINE_DIR / "race_json_display"
DEFAULT_VENUE_URLS_JSON = Path(__file__).resolve().parent / "venue_urls.json"

VENUE_ORDER = ["新潟", "中京", "札幌"]
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]

# 2026-08-15較正のStage B(新holdout、Bonferroni補正後97.5%CIをそのまま95%表記で流用せず、
# 実測CI水準を明記)。step2_stageB_result.json(boundary_extreme(-0.55,0.0)候補・m6)より。
# この値は特定の較正イベント(K1/K2再較正)の固定記録であり、開催日によらず共通。
MODEL_UPDATE_M6_MEAN_DIFF = -0.0333
MODEL_UPDATE_M6_CI_LO = -0.0478
MODEL_UPDATE_M6_CI_HI = -0.0200
MODEL_UPDATE_N_HOLDOUT = 69


class _JsonState:
    """race_json(dict)の1頭分をhp.simulate()のstate相当に見せる薄いアダプタ。
    race_m6()が参照する属性(umaban/log/baseline.is_estimated)だけを持つ。"""

    def __init__(self, umaban, log, is_estimated):
        self.umaban = umaban
        self.log = log
        self.baseline = types.SimpleNamespace(is_estimated=is_estimated)


def _states_from_race_json(rj):
    out = []
    for u_str, h in rj["horses"].items():
        pts = h.get("pts")
        if not pts:
            continue
        out.append(_JsonState(int(u_str), [tuple(p) for p in pts], bool(h.get("isEstimated"))))
    return out


_real_df_cache = {}


def _real_df_for_race(race_id, date):
    if date not in _real_df_cache:
        _real_df_cache[date] = pd.read_csv(
            REPO_ROOT / "data" / "race_results" / "2026" / f"{date}.csv", dtype=str)
    df = _real_df_cache[date]
    return df[df["race_id"] == race_id]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def time_at_distance_from_pts(pts, d_target):
    """horse_pair_sim.time_at_distanceのJSON pts版(pts=[t,d_rail,lane,v,stamina,ground_d]の配列)。"""
    if d_target <= pts[0][1]:
        return pts[0][0]
    lo, hi = 0, len(pts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pts[mid][1] < d_target:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return pts[0][0]
    a, b = pts[lo - 1], pts[lo]
    span = b[1] - a[1]
    f = (d_target - a[1]) / span if span > 0 else 0.0
    return a[0] + (b[0] - a[0]) * f


def aligned_sim_leader_lap_table(data, distance_m):
    """horse_pair_sim.leader_lap_table()は200mの倍数でない距離のレース(距離%200!=0)で
    第1区間が欠落する。実測ラップと同じ「第1区間=distance%200の端数」という区切りで
    作り直し、実測ラップと同じdistance列で並べて比較できるようにする。"""
    first_len = distance_m % 200.0
    if first_len == 0:
        first_len = 200.0
    marks = [first_len]
    while marks[-1] < distance_m - 1e-6:
        marks.append(marks[-1] + 200.0)

    horses_items = [(int(u), h["pts"]) for u, h in data["horses"].items()]
    rows = []
    prev_t = 0.0
    for d_mark in marks:
        t_leader, umaban_leader = min((time_at_distance_from_pts(pts, d_mark), u) for u, pts in horses_items)
        rows.append({
            "distance": round(d_mark), "umaban": umaban_leader,
            "cumulative": round(t_leader, 2), "split": round(t_leader - prev_t, 2),
        })
        prev_t = t_leader
    return rows


def build_race_caveat(meta_extra, n_estimated, n_total, extrap_m, m6_value=None):
    parts = []
    if n_estimated > 0:
        parts.append("出走%d頭中%d頭は実測の持続タイムデータが無く回帰推定(*推定表記)です。" % (n_total, n_estimated))
    if extrap_m > 0:
        parts.append("序盤速度形状は実測ビン(300〜1100m)を%.0fm超えて外挿しています。" % extrap_m)
    if meta_extra.get("is_straight"):
        parts.append("このレースはコーナーの無い直線専用コースです。")
    caveat = ("<b>このレース固有の注記:</b> " + " ".join(parts)) if parts else ""
    if m6_value is not None:
        m6_line = "<b>このレースのコーナー通過順位の一致度(m6、参考値):</b> %.3f(0に近いほど実測と一致)" % m6_value
        caveat = (caveat + "<br>" + m6_line) if caveat else m6_line
    return caveat


def build_venue_html(venue, race_date, race_names_dir, race_json_dir, cross_links):
    """cross_links: {"新潟": url_or_None, "中京": url_or_None, "札幌": url_or_None}"""
    race_names_csv = Path(race_names_dir) / f"race_names_{race_date}.csv"
    df = pd.read_csv(race_names_csv, dtype={"race_id": str})
    vdf = df[df["racecourse"] == venue].sort_values("race_number")

    dt = datetime.datetime.strptime(race_date, "%Y%m%d")
    date_label_full = "%d年%d月%d日(%s)" % (dt.year, dt.month, dt.day, WEEKDAY_JA[dt.weekday()])
    date_label_title = "%d年%d月%d日" % (dt.year, dt.month, dt.day)

    race_entries = []
    index_rows_html = []
    for _, row in vdf.iterrows():
        race_id = row["race_id"]
        rnum = int(row["race_number"])
        race_name = row["race_name"]
        is_jump = pd.isna(row["surface"])
        if is_jump:
            index_rows_html.append("""
<li class="race-row is-disabled" data-race-id="%s">
  <span class="race-rnum">%dR</span>
  <span class="race-name">%s</span>
  <span class="race-badge badge-excluded">対象外(障害)</span>
  <span></span>
</li>""" % (esc(race_id), rnum, esc(race_name)))
            continue

        with open(Path(race_json_dir) / f"{race_id}.json", encoding="utf-8") as f:
            data = json.load(f)

        surface = row["surface"]
        surface_full = "芝" if surface == "芝" else "ダート"
        distance = float(row["distance_m"])
        is_straight = bool(data.get("isStraightCourse"))

        # 実測データ(タイム・着順・上がり3F・通過順位・先頭馬ラップ)をhorses/トップレベルへ
        # マージする。sim側に存在しない馬番は無視、実測が無い馬番(sim側)はキー自体を付与しない
        # (JS側は欠損として「—」表示にフォールバックする)。
        actual = ard.get_actual_for_race(race_id, race_date, distance)
        for u_str, h in data["horses"].items():
            a = actual["horses"].get(int(u_str))
            if a:
                h.update(a)
        data["actualLeaderLapTable"] = actual["leaderLapTable"]
        if not is_straight:
            data["leaderLapTable"] = aligned_sim_leader_lap_table(data, distance)

        n_total = len(data["horses"])
        n_estimated = sum(1 for h in data["horses"].values() if h.get("isEstimated"))
        seg1_len = max(0.0, distance - 600.0)
        extrap_m = max(0.0, seg1_len - 1100.0)

        m6_value = None
        if not is_straight:
            real_race = _real_df_for_race(race_id, race_date)
            if not real_race.empty:
                corner_len_m = sg.physics_geometry(data.get("circumferenceM"), data.get("homeStretchM"))["corner_len_m"]
                m6r = cpm.race_m6(_states_from_race_json(data), real_race, distance,
                                   data.get("homeStretchM"), corner_len_m, is_straight)
                m6_value = m6r["overall_footrule"] if m6r else None

        meta = {
            "raceId": race_id, "raceNumber": rnum, "raceName": race_name,
            "racecourse": venue, "surface": surface_full, "distance": distance,
            "circumferenceM": data.get("circumferenceM"), "homeStretchM": data.get("homeStretchM"),
            "isStraightCourse": is_straight,
            "raceCaveat": build_race_caveat({"is_straight": is_straight}, n_estimated, n_total, extrap_m, m6_value),
        }
        race_entries.append((race_id, meta, data))

        badges = []
        if is_straight:
            badges.append('<span class="race-badge badge-straight">直線コース</span>')
        if extrap_m > 400:
            badges.append('<span class="race-badge">外挿%.0fm</span>' % extrap_m)
        badges_html = "".join(badges)

        index_rows_html.append("""
<li class="race-row" data-race-id="%s" tabindex="0" role="button">
  <span class="race-rnum">%dR</span>
  <span class="race-name">%s<span class="race-sub">%s %.0fm ／ 出走%d頭%s</span></span>
  %s
  <span class="race-arrow">&#9656;</span>
</li>""" % (esc(race_id), rnum, esc(race_name), esc(surface_full), distance, n_total,
            ("・推定%d頭" % n_estimated if n_estimated else ""), badges_html))

    races_js_obj = "{\n" + ",\n".join(
        '  "%s": {"meta": %s, "data": %s}' % (rid, json.dumps(meta, ensure_ascii=False), json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        for rid, meta, data in race_entries
    ) + "\n}"

    nav_links = []
    for v in VENUE_ORDER:
        url = cross_links.get(v)
        if v == venue:
            nav_links.append('<span class="venue-nav is-current-wrap"></span><a class="is-current" href="#">%s(このページ)</a>' % esc(v))
        elif url:
            nav_links.append('<a href="%s">%s</a>' % (esc(url), esc(v)))
        else:
            nav_links.append('<span class="race-badge">%s(準備中)</span>' % esc(v))

    html = """<title>JRA %s全レース相互作用シミュレーション — %s</title>
<style>
%s
</style>

<main>
  <div class="masthead">
    <h1>%s 相互作用シミュレーション — 全レース</h1>
    <p class="masthead-sub">%s %s開催 全%dレース(出走馬全頭・相互作用あり)</p>
  </div>
  <nav class="venue-nav">%s</nav>

  %s

  <details class="caveat-details">
    <summary>このシミュレーションについて(共通の仕組み・既知の限界)</summary>
    <div class="caveat-body">%s</div>
  </details>

  <ol class="race-index">%s
  </ol>

  <div id="player-empty" class="player-empty" style="display:none">レースを選択してください</div>
  <div id="player"></div>
</main>

<template id="player-template">%s</template>

<script>
var RACES = %s;
%s
%s
</script>
""" % (
        esc(venue), esc(date_label_title), tpl.STYLE, esc(venue), esc(date_label_full), esc(venue),
        len(vdf), "".join(nav_links),
        tpl.model_update_panel_html(esc, MODEL_UPDATE_M6_MEAN_DIFF, MODEL_UPDATE_M6_CI_LO,
                                     MODEL_UPDATE_M6_CI_HI, MODEL_UPDATE_N_HOLDOUT)
        + tpl.no_change_panel_html(esc),
        tpl.SHARED_CAVEAT_HTML, "".join(index_rows_html), tpl.PLAYER_MARKUP,
        races_js_obj, tpl.SCRIPT_JS, tpl.INIT_JS,
    )
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", required=True, choices=VENUE_ORDER)
    ap.add_argument("--race-date", required=True, help="YYYYMMDD")
    ap.add_argument("--race-names-dir", default=str(DEFAULT_RACE_NAMES_DIR))
    ap.add_argument("--race-json-dir", default=str(DEFAULT_RACE_JSON_DIR))
    ap.add_argument("--venue-urls-json", default=str(DEFAULT_VENUE_URLS_JSON),
                     help='{"YYYYMMDD": {"新潟": url, "中京": url, "札幌": url}, ...} 形式のJSON。'
                          "対象日付のURLが無ければ「準備中」表示になる")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    venue_urls = {}
    if Path(args.venue_urls_json).exists():
        all_urls = json.loads(Path(args.venue_urls_json).read_text(encoding="utf-8"))
        venue_urls = all_urls.get(args.race_date, {})

    html = build_venue_html(args.venue, args.race_date, args.race_names_dir, args.race_json_dir, venue_urls)
    Path(args.out).write_text(html, encoding="utf-8")
    print("wrote", args.out, "size=%.1fKB" % (len(html.encode("utf-8")) / 1024.0))


if __name__ == "__main__":
    main()
