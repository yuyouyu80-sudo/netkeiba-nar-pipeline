# -*- coding: utf-8 -*-
"""今日(2026-09-26)の中山12レースについて、父・母父の「全区分内訳」(2026-09-26新設の
pedigree_bucket_breakdown.csv)と今日の実際のレース条件を突き合わせ、馬ごとの
「総合適性スコア(pt)」を算出する(承認済み計画: 中山血統適性台帳レース別得意/不得意数値化)。

## 合成スコアに含める9項目(専門家レビュー反映済み、詳細は計画ファイル参照)
distance/surface/course/season/turf_type(該当時)/rest(馬ごとの実間隔)/debut/graded/leader_style
除外: going(予報なし)、turn・elevation・circumference(courseと重複するため合成からは除外、
既存の14項目詳細表には従来通り残す)、maturation(勝率ptと単位が異なるため別バッジ扱い)

## 父父(SS)ラインの追加(2026-09-26、ユーザー指摘「父・母父以外のデータも有効活用」対応)
種牡馬プロファイル(pedigree_commentary_profile.csv)はrole="sire"で「対象期間内に自身の産駒が
走ったことのある馬」を全て登録しており、父父(SS)がこの中に含まれるケースが多い(今日の中山
12レース出走馬156頭で実測89.1%が該当、母父父などのより遠い祖先は17〜36%と大きく低い)。
新規データ取得・新規集計ロジックともに不要で、既存の父・母父と全く同じrole="sire"ルックアップを
SSのIDに対して行うだけで「父父ライン」を第3の評価軸として追加できる。母父父等はカバレッジ不足
(既存の血統系統38.4%・生産者ティア13.7%の却下基準と同水準)のため主要軸には含めない。
"""
import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
from src.netkeiba_pipeline.storage.paths import pedigree_csv_path  # noqa: E402
import jra_pedigree_commentary_profile_2026_09_25 as PROF  # noqa: E402

SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)
BREAKDOWN_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_bucket_breakdown.csv"
PROFILE_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_commentary_profile.csv"
COURSE_MASTER_PATH = PROJECT_ROOT / "data" / "jra_course_master.csv"
RACE_CONDITIONS_PATH = SCRATCH / "race_conditions_check.csv"
NEWSPAPER_DIR = PROJECT_ROOT / "data" / "newspaper"
OUT_PATH = SCRATCH / "race_fit_score_data.json"

TODAY = date(2026, 9, 26)
MIN_SUBGROUP_N = PROF.MIN_SUBGROUP_N  # 内訳セルのN下限(既存プロファイルと同一基準を流用)
TOP_BOTTOM_FRACTION = 0.3  # レース内順位バッジ: 上位/下位この割合のみ表示

# 合成スコアに含む「区分マッチ型」項目(dimension名はBREAKDOWN_DIMENSIONSと対応)
BUCKET_ITEMS = ["distance", "surface", "course", "season", "turf_type", "rest"]
# 合成スコアに含む「条件非依存の統計そのもの」項目(profile CSVの列を直接使う)
STAT_ITEMS = ["debut", "graded", "leader_style"]


def se_pt(win_rate, n):
    """二項比率の標準誤差をpt(パーセントポイント)単位で返す。"""
    if n is None or n <= 0 or win_rate is None:
        return None
    return 100.0 * math.sqrt(max(win_rate * (1 - win_rate), 0) / n)


def load_breakdown_lookup():
    bd = pd.read_csv(BREAKDOWN_PATH, dtype={"horse_id_ancestor": str}, encoding="utf-8")
    lut = {}
    for _, r in bd.iterrows():
        key = (r["ancestor_role"], r["horse_id_ancestor"], r["dimension"])
        lut.setdefault(key, {})[r["bucket_label"]] = (float(r["win_rate"]), int(r["n"]))
    return lut


def load_course_tiers():
    cm = pd.read_csv(COURSE_MASTER_PATH, dtype=str, encoding="utf-8")
    for c in ["turf_elevation_m", "dirt_elevation_m", "turf_circumference_m", "dirt_circumference_m"]:
        cm[c + "_num"] = pd.to_numeric(cm[c], errors="coerce")
    medians = {
        "turf_elev": cm["turf_elevation_m_num"].median(),
        "dirt_elev": cm["dirt_elevation_m_num"].median(),
        "turf_circ": cm["turf_circumference_m_num"].median(),
        "dirt_circ": cm["dirt_circumference_m_num"].median(),
    }
    return cm.set_index("venue"), medians


def today_constant_buckets(racecourse, surface, cm, medians):
    """今日のレースに共通の区分(距離・馬ごとの休み明けを除く)を返す。"""
    row = cm.loc[racecourse]
    turn = row["turn_direction"]
    if surface == "芝":
        elev_v, elev_med = row["turf_elevation_m_num"], medians["turf_elev"]
        circ_v, circ_med = row["turf_circumference_m_num"], medians["turf_circ"]
        turf_type = "洋芝" if racecourse in {"札幌", "函館"} else "野芝"
    elif surface == "ダ":
        elev_v, elev_med = row["dirt_elevation_m_num"], medians["dirt_elev"]
        circ_v, circ_med = row["dirt_circumference_m_num"], medians["dirt_circ"]
        turf_type = None  # ダートは洋芝/野芝の対象外
    else:
        elev_v = circ_v = None
        turf_type = None
    elevation = None if pd.isna(elev_v) else ("急坂型(高低差大)" if elev_v >= elev_med else "平坦型(高低差小)")
    circumference = None if pd.isna(circ_v) else ("広いコース型(周長大)" if circ_v >= circ_med else "小回り型(周長小)")
    season = PROF.SEASON_MAP[TODAY.month]
    return {
        "course": racecourse, "turn": turn, "elevation": elevation,
        "circumference": circumference, "turf_type": turf_type, "season": season,
    }


def role_item_excess(lut, role, ancestor_id, overall_wr, dimension, bucket_label):
    """(role, ancestor_id)の該当区分について excess_pt・n・有意フラグを返す。
    該当区分が無い/N不足/ancestor_idがNoneなら全てNoneを返す(平均から除外する)。"""
    if pd.isna(ancestor_id) or bucket_label is None or overall_wr is None:
        return None
    entry = lut.get((role, ancestor_id, dimension), {})
    hit = entry.get(bucket_label)
    if hit is None:
        return None
    win_rate, n = hit
    excess = (win_rate - overall_wr) * 100.0
    se = se_pt(win_rate, n)
    significant = se is not None and abs(excess) > 1.96 * se
    return {"excess_pt": round(excess, 2), "n": n, "win_rate": round(win_rate, 4),
            "bucket_label": bucket_label, "significant": significant}


def role_stat_excess(profile_row, overall_wr, stat_key):
    col_map = {
        "debut": ("debut_win_rate", "debut_n"),
        "graded": ("graded_win_rate", "graded_n"),
        "leader_style": ("leader_win_rate", "leader_n"),
    }
    wr_col, n_col = col_map[stat_key]
    wr, n = profile_row.get(wr_col), profile_row.get(n_col)
    if pd.isna(wr) or pd.isna(n) or overall_wr is None:
        return None
    excess = (float(wr) - overall_wr) * 100.0
    se = se_pt(float(wr), int(n))
    significant = se is not None and abs(excess) > 1.96 * se
    return {"excess_pt": round(excess, 2), "n": int(n), "win_rate": round(float(wr), 4),
            "bucket_label": None, "significant": significant}


def kire_item(profile_row):
    """「キレ(瞬発力)」の代理指標: 非先行(差し・追込)勝率-先行勝率のpt差。
    プロ競馬予想家レビュー(2026-09-26)で「重賞実績=キレ」は強引と指摘され、既存の
    leader_style(先行時勝率-総合勝率)とは別の独立した軸として、2標本比率の差のSEで
    有意性判定する(2群とも比率が異なるため単純な単標本se_ptは使えない)。"""
    lwr, ln = profile_row.get("leader_win_rate"), profile_row.get("leader_n")
    nwr, nn = profile_row.get("non_leader_win_rate"), profile_row.get("non_leader_n")
    if pd.isna(lwr) or pd.isna(nwr) or pd.isna(ln) or pd.isna(nn):
        return None
    ln, nn = int(ln), int(nn)
    if ln <= 0 or nn <= 0:
        return None
    diff = (float(nwr) - float(lwr)) * 100.0
    se = 100.0 * math.sqrt(max(float(lwr) * (1 - float(lwr)), 0) / ln + max(float(nwr) * (1 - float(nwr)), 0) / nn)
    significant = abs(diff) > 1.96 * se
    return {"excess_pt": round(diff, 2), "n": min(ln, nn), "win_rate": None,
            "bucket_label": None, "significant": significant}


def speed_stamina_kire(lut, profile_by_key, role, ancestor_id):
    """血統由来の「スピード・スタミナ・キレ」3項目(プロ競馬予想家レビュー反映済み、
    2026-09-26新設)。既存のcomposite_pt(9項目平均)には含めない別枠の参考指標。
    スピード=距離適性の短距離帯(~1400m)区分pt差、スタミナ=長距離帯(2201m~)区分pt差、
    キレ=非先行-先行勝率差。いずれもN不足・該当区分なしならNoneを返す(平均から除外)。"""
    if pd.isna(ancestor_id) or (role, ancestor_id) not in profile_by_key:
        return None
    prow = profile_by_key[(role, ancestor_id)]
    overall_wr = prow.get("overall_win_rate")
    return {
        "speed": role_item_excess(lut, role, ancestor_id, overall_wr, "distance", "短距離(~1400m)"),
        "stamina": role_item_excess(lut, role, ancestor_id, overall_wr, "distance", "長距離(2201m~)"),
        "kire": kire_item(prow),
    }


def maturation_badge(profile_row):
    skew = profile_row.get("maturation_age_skew")
    if pd.isna(skew):
        return None
    if skew < -0.1:
        return {"tendency": "早熟", "skew": round(float(skew), 2)}
    if skew > 0.1:
        return {"tendency": "晩成", "skew": round(float(skew), 2)}
    return {"tendency": "中間", "skew": round(float(skew), 2)}


def compute_role_score(lut, profile_by_key, role, ancestor_id, buckets, rest_bucket_label):
    if pd.isna(ancestor_id) or (role, ancestor_id) not in profile_by_key:
        return None
    prow = profile_by_key[(role, ancestor_id)]
    overall_wr = prow.get("overall_win_rate")
    items = {}
    bucket_dim_map = {
        "distance": buckets.get("distance_bucket_label"),
        "surface": buckets.get("surface"),
        "course": buckets.get("course"),
        "season": buckets.get("season"),
        "turf_type": buckets.get("turf_type"),
        "rest": rest_bucket_label,
    }
    for dim in BUCKET_ITEMS:
        label = bucket_dim_map[dim]
        if label is None:
            items[dim] = None
            continue
        items[dim] = role_item_excess(lut, role, ancestor_id, overall_wr, dim, label)
    for stat_key in STAT_ITEMS:
        items[stat_key] = role_stat_excess(prow, overall_wr, stat_key)

    applicable = [v["excess_pt"] for v in items.values() if v is not None]
    composite = round(sum(applicable) / len(applicable), 2) if applicable else None
    return {
        "ancestor_id": ancestor_id, "overall_win_rate": round(overall_wr, 4) if overall_wr else None,
        "items": items, "composite_pt": composite, "n_applicable": len(applicable),
        "maturation": maturation_badge(prow),
        "ssk": speed_stamina_kire(lut, profile_by_key, role, ancestor_id),
    }


def main():
    lut = load_breakdown_lookup()
    profile = pd.read_csv(PROFILE_PATH, dtype={"horse_id_ancestor": str}, encoding="utf-8")
    profile_by_key = {(r["ancestor_role"], r["horse_id_ancestor"]): r for _, r in profile.iterrows()}
    cm, medians = load_course_tiers()
    race_cond = pd.read_csv(RACE_CONDITIONS_PATH, dtype=str, encoding="utf-8")

    races_out = []
    for _, rc in race_cond.iterrows():
        race_id = rc["race_id"]
        surface = rc["surface"] if pd.notna(rc["surface"]) and rc["surface"] else None
        distance_m = pd.to_numeric(rc["distance_m"], errors="coerce")
        racecourse = rc["racecourse"]
        skipped_reason = None
        if surface is None or pd.isna(distance_m):
            skipped_reason = "障害戦等のため距離・馬場条件が取得できず、距離/馬場/洋芝野芝は対象外"
            const_buckets = {"course": racecourse, "turn": None, "elevation": None,
                              "circumference": None, "turf_type": None,
                              "season": PROF.SEASON_MAP[TODAY.month]}
        else:
            const_buckets = today_constant_buckets(racecourse, surface, cm, medians)
        buckets = dict(const_buckets)
        buckets["surface"] = surface
        buckets["distance_bucket_label"] = None if pd.isna(distance_m) else PROF.distance_bucket(distance_m)

        news_path = NEWSPAPER_DIR / f"{race_id}.csv"
        df = pd.read_csv(news_path, dtype=str, encoding="utf-8")
        df["_umaban_num"] = pd.to_numeric(df["umaban"], errors="coerce")
        df = df.sort_values("_umaban_num")

        horses = []
        for _, h in df.iterrows():
            horse_id = h["horse_id"]
            past1_date = h.get("past1_date")
            rest_days = None
            if pd.notna(past1_date) and past1_date:
                try:
                    y, m, d = (int(x) for x in str(past1_date).split(".")[:3])
                    rest_days = (TODAY - date(y, m, d)).days
                except (ValueError, TypeError):
                    rest_days = None
            rest_bucket_label = PROF.rest_bucket(rest_days) if rest_days is not None else None

            ped_path = pedigree_csv_path(horse_id)
            sire_id = bms_id = ss_id = sire_name = bms_name = ss_name = None
            sire_score = bms_score = ss_score = None
            if ped_path.exists():
                ped = pd.read_csv(ped_path, dtype=str, encoding="utf-8").iloc[0]
                sire_id, bms_id = ped.get("ped_S_horse_id"), ped.get("ped_DS_horse_id")
                ss_id = ped.get("ped_SS_horse_id")  # 父父(父の父)。種牡馬プロファイルは
                # role="sire"で登録されているため、父父も同じroleでルックアップすれば再計算不要で使える
                sire_name, bms_name = ped.get("ped_S_name_ja"), ped.get("ped_DS_name_ja")
                ss_name = ped.get("ped_SS_name_ja")
                sire_score = compute_role_score(lut, profile_by_key, "sire", sire_id, buckets, rest_bucket_label)
                bms_score = compute_role_score(lut, profile_by_key, "bms", bms_id, buckets, rest_bucket_label)
                ss_score = compute_role_score(lut, profile_by_key, "sire", ss_id, buckets, rest_bucket_label)

            combined = None
            parts = [s["composite_pt"] for s in (sire_score, bms_score, ss_score) if s and s["composite_pt"] is not None]
            if parts:
                combined = round(sum(parts) / len(parts), 2)

            horses.append({
                "umaban": int(h["umaban"]), "waku": int(h["waku"]), "horse_id": horse_id,
                "horse_name": h["horse_name"],
                "sire_name": sire_name if pd.notna(sire_name) else None,
                "bms_name": bms_name if pd.notna(bms_name) else None,
                "ss_name": ss_name if pd.notna(ss_name) else None,
                "rest_days": rest_days, "rest_bucket": rest_bucket_label,
                "sire": sire_score, "bms": bms_score, "ss": ss_score, "composite_combined": combined,
            })

        # レース内順位バッジ(上位/下位 約30%のみ、combinedが無い馬は除外して順位付け)
        ranked = sorted([h for h in horses if h["composite_combined"] is not None],
                         key=lambda h: h["composite_combined"], reverse=True)
        n_badge = max(1, round(len(ranked) * TOP_BOTTOM_FRACTION)) if ranked else 0
        top_ids = {h["horse_id"] for h in ranked[:n_badge]}
        bottom_ids = {h["horse_id"] for h in ranked[-n_badge:]} if n_badge else set()
        for h in horses:
            if h["horse_id"] in top_ids:
                h["rank_badge"] = "得意"
            elif h["horse_id"] in bottom_ids:
                h["rank_badge"] = "不得意"
            else:
                h["rank_badge"] = None

        races_out.append({
            "race_id": race_id, "race_number": int(rc["race_number"]), "race_name": rc["race_name"],
            "racecourse": racecourse, "surface": surface,
            "distance_m": None if pd.isna(distance_m) else int(distance_m),
            "skipped_reason": skipped_reason, "horses": horses,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"today": str(TODAY), "races": races_out}, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_PATH}: {len(races_out)}races")


if __name__ == "__main__":
    main()
