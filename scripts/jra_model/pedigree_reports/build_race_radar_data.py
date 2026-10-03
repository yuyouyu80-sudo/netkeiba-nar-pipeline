# -*- coding: utf-8 -*-
"""今日の中山全レース(race_fit_score_data.jsonの各レース、障害戦など条件が取れないものは除く)について、
「同等の過去レース」(同競馬場・同クラス・同距離・同サーフェス・同じ開催日数)を特定し、その上位3着馬の
血統(父・母父)から「このレースタイプで求められるファクター」をレーダーチャート用に集計する。
(元は中山8R専用の build_race8_radar_data.py。2026-10-03に全レース対応へ一般化。8Rの出力は旧版と
同一であることを確認済み。)

## 全レース対応で追加した規則(2026-10-03)
- 同等レース数が MIN_RACES_STRICT(15)未満のレースは「開催日数」の条件を外して再検索(緩和)する。
  それでも MIN_RACES_PUBLISH(10)未満なら掲載しない。どちらの条件を使ったかは出力に記録し、レポートに明記する。
- クラスは今日のレースのrace_name(race_results、無ければ出走表のrace_name)をclass_ordinal()で正規化。
  未勝利・新馬(class_ord=0)は新馬/未勝利の別を揃える(新馬同士、未勝利同士)。

## 同等レースの定義(根拠)
- 競馬場: 中山(race_idのvenue桁で判定、"06")
- クラス: race_nameをjra_race_sim/race_potential.pyのclass_ordinal()で正規化し、今日と同じ
  序列(1勝クラス=1)のレースを対象(「3歳以上」「4歳以上」等の年齢条件の違いは同一クラス帯として
  許容する — 2016〜2026年で該当33レースと十分なNが確保できたため、年齢条件までは絞り込まない)。
- 距離・サーフェス: 今日と同じ1200m・ダート(race_results実測値で厳密一致)。
- 開催日数: race_idの日目桁(位置8:10)が今日と同じ"08"(開催回・年は問わない、「開催がその
  何日目か」という進行度だけを合わせる設計)。
- 馬場状態(道悪): 今日はまだ確定していない(JRA公式含め本コードベースに予報取得ロジックが
  存在しない)ため、この軸だけは一致条件に含めず、集計結果には過去33レースの実際の馬場状態の
  内訳を参考情報として付記する(除外ではなく透明性確保)。

## 上位馬の定義
finish_pos<=3(複勝圏内)。血統単独モデルが既にNested OOFで不採用と判定済み
([[project_jra_pedigree_theory_verification_2026_08_29]]、[[血統単独モデル検証結果]])である
ことを踏まえ、本スクリプトの出力は「過去に上位入線した馬の血統がどんな傾向を示したか」という
記述統計・事後集計であり、検証済みの予想シグナルではない。

## 各ファクター(pt差)の算出ロジック
このフォルダのbuild_race_fit_score.pyのrole_item_excess/role_stat_excessをそのまま再利用し、
各上位馬の「その歴史的レース自身の実際の条件」(距離1200m・ダート・中山・そのレースの季節・
その馬自身の実際の休み明け間隔)に対する父/母父の該当区分pt差を計算する。新馬戦実績・重賞実績・
脚質傾向は条件非依存の統計そのもの(今日の同一馬の値と同じロジック)。
"""
import glob
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_race_sim"))
from src.netkeiba_pipeline.storage.paths import pedigree_csv_path  # noqa: E402
import jra_pedigree_commentary_profile_2026_09_25 as PROF  # noqa: E402
from race_potential import class_ordinal  # noqa: E402

SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_race_fit_score import (  # noqa: E402
    load_breakdown_lookup, role_item_excess, role_stat_excess, speed_stamina_kire, compute_role_score,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
PROFILE_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_commentary_profile.csv"
SCORE_TODAY_PATH = SCRATCH / "race_fit_score_data.json"
BLOODLINE_MAP_PATH = SCRATCH / "bloodline_name_map.json"
NEWSPAPER_DIR = PROJECT_ROOT / "data" / "newspaper"
OUT_PATH = SCRATCH / "race_radar_data.json"

MIN_RACES_STRICT = 15   # 開催日数まで一致する同等レースがこれ未満なら開催日数の条件を外す
MIN_RACES_PUBLISH = 10  # 緩和後もこれ未満なら掲載しない
TOP_N = 3

SURFACE_NAME = {"ダ": "ダート", "芝": "芝"}


def item_labels_for(venue, surface, distance):
    return [
        ("distance", f"距離({int(distance)}m帯)"),
        ("surface", f"サーフェス({SURFACE_NAME.get(surface, surface)})"),
        ("course", f"競馬場({venue})"),
        ("season", "季節"),
        ("rest", "休み明け間隔"),
        ("debut", "新馬戦実績"),
        ("graded", "重賞実績"),
        ("leader_style", "脚質(先行)"),
    ]


SSK_LABELS = [("speed", "スピード"), ("stamina", "スタミナ"), ("kire", "キレ(瞬発力)")]

# 「このレースタイプでの参考重要度」判定ルール(プロ競馬予想家レビュー2026-09-26反映)。
# 統計的に検定したものではなく、一般的な血統・展開理論に基づく事前ルール(データを見てから
# 後付けで作ったものではない)。距離帯→サーフェス→クラスの順に段階調整する。
_TIER_ORDER = ["低", "中", "高"]
_BASE_TIER_BY_DISTANCE = {
    "短距離(~1400m)": {"speed": "高", "stamina": "低", "kire": "低"},
    "マイル(1401-1800m)": {"speed": "中", "stamina": "低", "kire": "中"},
    "中距離(1801-2200m)": {"speed": "中", "stamina": "中", "kire": "高"},
    "長距離(2201m~)": {"speed": "低", "stamina": "高", "kire": "中"},
}


def _shift_tier(tier: str, delta: int) -> str:
    i = _TIER_ORDER.index(tier)
    return _TIER_ORDER[max(0, min(len(_TIER_ORDER) - 1, i + delta))]


def importance_tier(distance_bucket_label: str, surface: str, class_ord) -> dict:
    tiers = dict(_BASE_TIER_BY_DISTANCE.get(distance_bucket_label, {"speed": "中", "stamina": "中", "kire": "中"}))
    if surface == "ダ":  # ダートは瞬発戦になりにくく、キレの重要度を1段階下げる
        tiers["kire"] = _shift_tier(tiers["kire"], -1)
    if class_ord == 0:  # 新馬・未勝利は素質(スピード)重視、スタミナ・キレは相対的に低い
        tiers["speed"] = _shift_tier(tiers["speed"], +1)
        tiers["stamina"] = _shift_tier(tiers["stamina"], -1)
        tiers["kire"] = _shift_tier(tiers["kire"], -1)
    elif class_ord is not None and class_ord >= 5:  # 重賞級はスタミナ・キレの重要度が上がる
        tiers["stamina"] = _shift_tier(tiers["stamina"], +1)
        tiers["kire"] = _shift_tier(tiers["kire"], +1)
    return tiers


def load_all_results() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(RESULTS_DIR / "*" / "*.csv"))):
        year_name = Path(p).parent.name
        if not (year_name.isdigit() and int(year_name) >= PROF.MIN_RESULTS_YEAR):
            continue  # 2016年以降に固定(収集側の過去バックフィルで集計期間が動かないように)
        frames.append(pd.read_csv(p, dtype=str, encoding="utf-8"))
    df = pd.concat(frames, ignore_index=True)
    df["finish_pos_num"] = pd.to_numeric(df["finish_pos"], errors="coerce")
    df["distance_num"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["race_date_dt"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["class_ord"] = df["race_name"].map(class_ordinal)
    df["day_seg"] = df["race_id"].str[8:10]
    df = df.sort_values(["horse_id", "race_date_dt"])
    df["prev_race_date"] = df.groupby("horse_id")["race_date_dt"].shift(1)
    df["rest_days"] = (df["race_date_dt"] - df["prev_race_date"]).dt.days
    df["rest_bucket"] = df["rest_days"].map(PROF.rest_bucket)
    df["season"] = df["race_date_dt"].dt.month.map(PROF.SEASON_MAP)
    return df


def score_horse(lut, profile_by_key, role, ancestor_id, overall_wr, distance_bucket_label,
                 surface, course, season, rest_bucket_label):
    if pd.isna(ancestor_id) or (role, ancestor_id) not in profile_by_key:
        return None
    prow = profile_by_key[(role, ancestor_id)]
    items = {
        "distance": role_item_excess(lut, role, ancestor_id, overall_wr, "distance", distance_bucket_label),
        "surface": role_item_excess(lut, role, ancestor_id, overall_wr, "surface", surface),
        "course": role_item_excess(lut, role, ancestor_id, overall_wr, "course", course),
        "season": role_item_excess(lut, role, ancestor_id, overall_wr, "season", season),
        "rest": role_item_excess(lut, role, ancestor_id, overall_wr, "rest", rest_bucket_label),
        "debut": role_stat_excess(prow, overall_wr, "debut"),
        "graded": role_stat_excess(prow, overall_wr, "graded"),
        "leader_style": role_stat_excess(prow, overall_wr, "leader_style"),
    }
    return items


ROLE_SPEC = [
    # (内部キー, ped列コード, 表示名, プロファイルrole)
    ("sire", "S", "父", "sire"),
    ("bms", "DS", "母父", "bms"),
    ("ss", "SS", "父父", "sire"),   # 種牡馬プロファイルに自身の産駒として登録されていれば再利用
    ("dam", "D", "母", "dam"),      # 2026-09-27追加: 新設role="dam"(産駒数が少なく低カバレッジ)
    ("dd", "DD", "母母", "dam"),
    ("sd", "SD", "父母", "dam"),
]


@lru_cache(maxsize=None)
def read_pedigree_row(horse_id):
    ped_path = pedigree_csv_path(horse_id)
    if not ped_path.exists():
        return None
    return pd.read_csv(ped_path, dtype=str, encoding="utf-8").iloc[0]


def build_field_records(df_finishers, lut, profile_by_key, distance_bucket_label, surface, venue):
    """出走馬(上位に限らず全頭)について、父・母父・父父・母・母母・父母それぞれの14項目pt差+
    スピード/スタミナ/キレを算出する。父父(SS)は種牡馬プロファイル(role="sire")に、
    母・母母・父母(D/DD/SD)は繁殖牝馬プロファイル(role="dam"、2026-09-27新設)に自身の産駒が
    登録されているケースがあれば、新規集計不要で同じロールでルックアップするだけで追加の
    評価軸として使える。母系は産駒数が少ないため該当率は低い(既知の制約、詳細はレポート内に
    カバレッジとして明記)。"""
    records = []
    for _, row in df_finishers.iterrows():
        horse_id = row["horse_id"]
        ped = read_pedigree_row(horse_id)
        if ped is None:
            continue
        rest_label = PROF.rest_bucket(row["rest_days"]) if pd.notna(row["rest_days"]) else None

        rec = {
            "race_id": row["race_id"], "race_date": row["race_date"], "race_name": row["race_name"],
            "going": row["going"], "finish_pos": int(row["finish_pos_num"]),
            "horse_name": row["horse_name"],
        }
        any_items = False
        for key, ped_code, _disp, role in ROLE_SPEC:
            anc_id = ped.get(f"ped_{ped_code}_horse_id")
            anc_name = ped.get(f"ped_{ped_code}_name_ja")
            rec[f"{key}_id"] = anc_id if pd.notna(anc_id) else None
            rec[f"{key}_name"] = anc_name if pd.notna(anc_name) else None
            items = ssk = None
            if pd.notna(anc_id) and (role, anc_id) in profile_by_key:
                overall = profile_by_key[(role, anc_id)]["overall_win_rate"]
                items = score_horse(lut, profile_by_key, role, anc_id, overall,
                                     distance_bucket_label, surface, venue,
                                     row["season"], rest_label)
                ssk = speed_stamina_kire(lut, profile_by_key, role, anc_id)
                any_items = True
            rec[f"{key}_items"] = items
            rec[f"{key}_ssk"] = ssk
        if not any_items:
            continue
        records.append(rec)
    return records


SSK_ROLE_KEYS = ["sire", "bms", "ss", "dam", "dd", "sd"]


def combined_ssk_value(rec, key):
    """算出できた祖先ライン(最大6本)のスピード/スタミナ/キレpt差の単純平均(既存の
    composite_combined と同じ「使える方だけ平均」方式)。"""
    vals = []
    for role_key in SSK_ROLE_KEYS:
        ssk = rec.get(f"{role_key}_ssk")
        if ssk and ssk.get(key) and ssk[key].get("excess_pt") is not None:
            vals.append(ssk[key]["excess_pt"])
    return sum(vals) / len(vals) if vals else None


def tercile_balance_analysis(field_records, key, min_n=30):
    """「最大値が最も有利」という前提を置かず、過去の同等レース全出走馬(上位に限らない)を
    3分位に分け、複勝圏内(3着以内)率を分位ごとに実測する。中位分位が最も複勝率が高ければ
    「中間が最適(非単調)」というユーザー仮説を裏付ける証拠、両端のどちらかが最も高ければ
    従来通り単調な関係と判断する。信頼区間が重なる場合は「明確な差はない」と正直に記載する。"""
    pairs = [(combined_ssk_value(r, key), r["finish_pos"] <= TOP_N) for r in field_records]
    pairs = [(v, t3) for v, t3 in pairs if v is not None]
    n_total = len(pairs)
    if n_total < min_n:
        return {"n_total": n_total, "bins": [], "pattern": "サンプル不足(N<30)のため判定不能"}

    vals = np.array([v for v, _ in pairs], dtype=float)
    top3_flags = np.array([1.0 if t else 0.0 for _, t in pairs], dtype=float)
    try:
        bin_idx = pd.qcut(pd.Series(vals), 3, labels=False, duplicates="drop")
    except ValueError:
        return {"n_total": n_total, "bins": [], "pattern": "値の重複が多く3分位に分割できず判定不能"}
    bin_idx = bin_idx.to_numpy()
    n_bins = int(np.nanmax(bin_idx)) + 1
    if n_bins < 3:
        return {"n_total": n_total, "bins": [], "pattern": f"値の重複により{n_bins}分位にしか分割できず判定不能"}

    bin_names = ["低位1/3", "中位1/3", "高位1/3"]
    bins_out = []
    for b in range(3):
        mask = bin_idx == b
        n = int(mask.sum())
        rate = float(top3_flags[mask].mean()) * 100.0 if n > 0 else None
        se = 100.0 * math.sqrt(max(rate / 100 * (1 - rate / 100), 0) / n) if (n > 0 and rate is not None) else None
        bins_out.append({
            "label": bin_names[b], "n": n,
            "value_range": [round(float(vals[mask].min()), 2), round(float(vals[mask].max()), 2)],
            "value_mean": round(float(vals[mask].mean()), 2),
            "top3_rate": round(rate, 1) if rate is not None else None,
            "ci_low": round(max(0.0, rate - 1.96 * se), 1) if se is not None else None,
            "ci_high": round(min(100.0, rate + 1.96 * se), 1) if se is not None else None,
        })

    best = max(bins_out, key=lambda x: x["top3_rate"])
    worst = min(bins_out, key=lambda x: x["top3_rate"])
    ci_overlap = True
    if best["ci_low"] is not None and worst["ci_high"] is not None:
        ci_overlap = not (best["ci_low"] > worst["ci_high"])

    rates = [b["top3_rate"] for b in bins_out]
    if best["label"] == "中位1/3":
        pattern = "中間が最適(非単調)"
    elif best["label"] == "高位1/3" and rates[0] <= rates[1] <= rates[2]:
        pattern = "高いほど有利(単調)"
    elif best["label"] == "低位1/3" and rates[0] >= rates[1] >= rates[2]:
        pattern = "低いほど有利(単調)"
    else:
        pattern = "明確な傾向なし"

    return {
        "n_total": n_total, "bins": bins_out, "best_bin": best["label"],
        "best_bin_value_mean": best["value_mean"], "best_bin_value_range": best["value_range"],
        "pattern": pattern, "ci_overlap_best_vs_worst": ci_overlap,
    }


def bloodline_category_analysis(field_records, name_field, bloodline_map, min_n=20):
    """netkeiba「血統ビーム出馬表」の父系統色分け(サンデーサイレンス系/ノーザンダンサー系等
    8分類、種牡馬個体に紐づく固定属性)で、過去の同等レース全出走馬を分類し、系統ごとの
    複勝率(3着以内率)を実測する。系統マップは2026年分newspaper CSV(816ファイル、実測網羅率
    100%)から名前で構築(2026-09-27)。分類できない馬は「不明」として除外集計する。"""
    groups = defaultdict(list)
    unmatched = 0
    for r in field_records:
        name = r.get(name_field)
        if not name:
            continue
        cat = bloodline_map.get(name)
        if cat is None:
            unmatched += 1
            continue
        groups[cat].append(r["finish_pos"] <= TOP_N)

    rows = []
    for cat, flags in groups.items():
        n = len(flags)
        if n < min_n:
            continue
        rate = 100.0 * sum(flags) / n
        se = 100.0 * math.sqrt(max(rate / 100 * (1 - rate / 100), 0) / n)
        rows.append({
            "category": cat, "n": n, "top3_rate": round(rate, 1),
            "ci_low": round(max(0.0, rate - 1.96 * se), 1),
            "ci_high": round(min(100.0, rate + 1.96 * se), 1),
        })
    rows.sort(key=lambda x: -x["top3_rate"])
    n_matched = sum(len(v) for v in groups.values())
    return {"rows": rows, "n_matched": n_matched, "n_unmatched": unmatched,
            "n_categories_below_min_n": len(groups) - len(rows)}


def today_bloodline_labels(race_id):
    """今日の出走馬自身の系統色分けは、netkeiba出馬表(newspaper CSV)にbias_sire_bloodline/
    bias_dam_sire_bloodlineとしてそのまま収録済み(実測網羅率100%)なので新規取得・推定不要。"""
    news_path = NEWSPAPER_DIR / f"{race_id}.csv"
    if not news_path.exists():
        return {}
    df = pd.read_csv(news_path, dtype=str, encoding="utf-8")
    out = {}
    for _, r in df.iterrows():
        out[r["horse_id"]] = {
            "sire_bloodline": r.get("bias_sire_bloodline") if pd.notna(r.get("bias_sire_bloodline")) else None,
            "bms_bloodline": r.get("bias_dam_sire_bloodline") if pd.notna(r.get("bias_dam_sire_bloodline")) else None,
        }
    return out


def race_kind(race_name, class_ord):
    """未勝利・新馬(class_ord=0)は新馬/未勝利の別を揃える。それ以外はNone(クラス序列のみで比較)。"""
    if class_ord == 0 or class_ord is None:
        if "新馬" in str(race_name):
            return "新馬"
        if "未勝利" in str(race_name):
            return "未勝利"
    return None


def find_matched_races(results, cfg):
    """今日のレースと同じ競馬場・サーフェス・距離・クラスの過去レース。開催日数まで一致するものが
    MIN_RACES_STRICT未満なら日数条件を外す(緩和)。戻り値: (matched_df, rule_dict)"""
    uniq = results.drop_duplicates("race_id")
    base = uniq[
        (uniq["racecourse"] == cfg["venue"]) & (uniq["surface"] == cfg["surface"])
        & (uniq["distance_num"] == cfg["distance"]) & (uniq["class_ord"] == cfg["class_ord"])
        & (uniq["race_id"] != cfg["race_id"])
    ]
    if cfg["kind"] is not None:
        base = base[base["race_name"].map(lambda n: race_kind(n, cfg["class_ord"]) == cfg["kind"])]
    strict = base[base["day_seg"] == cfg["day_seg"]]
    rule = {"n_strict": int(len(strict)), "n_relaxed": int(len(base)), "day_seg": cfg["day_seg"],
            "kind": cfg["kind"]}
    if len(strict) >= MIN_RACES_STRICT:
        rule["day_condition"] = True
        return strict, rule
    rule["day_condition"] = False
    return base, rule


def build_one(cfg, results, lut, profile_by_key, bloodline_map, today_race):
    """1レース分の出力辞書(旧race8_radar_data.jsonと同じ構造+match_rule/race_info)を返す。
    同等レースが少なすぎて掲載できない場合は理由文字列を返す。"""
    venue, surface, distance = cfg["venue"], cfg["surface"], cfg["distance"]
    race_id = cfg["race_id"]
    item_labels = item_labels_for(venue, surface, distance)

    matched, rule = find_matched_races(results, cfg)
    match_race_ids = set(matched["race_id"])
    print(f"[{race_id[-2:]}R] 同等過去レース: {len(match_race_ids)}件 (開催日数条件={'あり' if rule['day_condition'] else 'なし(緩和)'}, "
          f"厳密{rule['n_strict']}件/緩和{rule['n_relaxed']}件)")
    if len(match_race_ids) < MIN_RACES_PUBLISH:
        return f"同等の過去レースが{len(match_race_ids)}件しかなく(掲載基準{MIN_RACES_PUBLISH}件)集計できません"

    field_all = results[
        results["race_id"].isin(match_race_ids) & results["finish_pos_num"].notna()
    ].copy()
    distance_bucket_label = PROF.distance_bucket(distance)

    field_records = build_field_records(field_all, lut, profile_by_key, distance_bucket_label, surface, venue)
    horse_records = [r for r in field_records if r["finish_pos"] <= TOP_N]
    print(f"    血統照合済み(全出走) 延べ{len(field_records)}頭 / うち上位{TOP_N}着 延べ{len(horse_records)}頭")

    role_id_keys = [f"{k}_id" for k, _c, _d, _r in ROLE_SPEC]
    n_unique_pairs = len({
        tuple(r.get(rk) for rk in role_id_keys) for r in field_records
        if any(r.get(rk) is not None for rk in role_id_keys)
    })

    ssk_balance = {key: tercile_balance_analysis(field_records, key) for key, _label in SSK_LABELS}
    for key, _label in SSK_LABELS:
        combined_top3 = [combined_ssk_value(r, key) for r in horse_records]
        combined_top3 = [v for v in combined_top3 if v is not None]
        ssk_balance[key]["top3_combined_mean"] = (
            round(sum(combined_top3) / len(combined_top3), 2) if combined_top3 else None
        )
        ssk_balance[key]["top3_combined_n"] = len(combined_top3)

    # --- 系統別成績(父系統・母父系統・父父系統、netkeiba血統ビーム色分けの再利用) ---
    bloodline_sire = bloodline_category_analysis(field_records, "sire_name", bloodline_map)
    bloodline_bms = bloodline_category_analysis(field_records, "bms_name", bloodline_map)
    bloodline_ss = bloodline_category_analysis(field_records, "ss_name", bloodline_map)
    today_bloodlines = today_bloodline_labels(race_id)

    # --- テンプレート(過去上位馬の平均プロファイル)算出。父父(ss)は父と同じ種牡馬プロファイル、
    # 母・母母・父母(dam/dd/sd)は新設の繁殖牝馬プロファイルを再利用する第4〜6のラインとして扱う ---
    template = {}
    for key, _label in item_labels:
        for role, _code, _disp, _prole in ROLE_SPEC:
            items_field = f"{role}_items"
            vals = [
                r[items_field][key]["excess_pt"] for r in horse_records
                if r[items_field] is not None and r[items_field].get(key) is not None
            ]
            template.setdefault(key, {})[role] = {
                "mean_pt": round(sum(vals) / len(vals), 2) if vals else None,
                "n_horses": len(vals),
            }

    ssk_template = {}
    for key, _label in SSK_LABELS:
        for role, _code, _disp, _prole in ROLE_SPEC:
            ssk_field = f"{role}_ssk"
            vals = [
                r[ssk_field][key]["excess_pt"] for r in horse_records
                if r[ssk_field] is not None and r[ssk_field].get(key) is not None
            ]
            ssk_template.setdefault(key, {})[role] = {
                "mean_pt": round(sum(vals) / len(vals), 2) if vals else None,
                "n_horses": len(vals),
            }
    tiers = importance_tier(distance_bucket_label, surface, cfg["class_ord"])

    going_counts = matched["going"].value_counts().to_dict()

    race_list = []
    for rid in sorted(match_race_ids):
        r = matched[matched["race_id"] == rid].iloc[0]
        finishers = [h for h in horse_records if h["race_id"] == rid]
        finishers.sort(key=lambda h: h["finish_pos"])
        race_list.append({
            "race_id": rid, "race_date": r["race_date"], "race_name": r["race_name"],
            "going": r["going"],
            "top3": [{"finish_pos": h["finish_pos"], "horse_name": h["horse_name"],
                      "sire_name": h["sire_name"], "bms_name": h["bms_name"],
                      "ss_name": h["ss_name"], "dam_name": h["dam_name"],
                      "dd_name": h["dd_name"], "sd_name": h["sd_name"]} for h in finishers],
        })

    today_score = today_race

    # --- 補足参考情報: 今日の出走馬の「母自身」の現役時代の記録(産駒への遺伝伝達力とは別物、
    # プロ競馬予想家レビュー指摘: 個体の能力と遺伝伝達力を混同しないよう、軸には含めず別掲する) ---
    dam_records = {}
    ped_cache = {}
    news_path = NEWSPAPER_DIR / f"{race_id}.csv"
    if today_score and news_path.exists():
        news = pd.read_csv(news_path, dtype=str, encoding="utf-8")
        dam_ids = {}
        for _, nrow in news.iterrows():
            hped = read_pedigree_row(nrow["horse_id"])
            if hped is None:
                continue
            ped_cache[nrow["horse_id"]] = hped
            dam_id = hped.get("ped_D_horse_id")
            if pd.notna(dam_id):
                dam_ids[nrow["horse_id"]] = (dam_id, hped.get("ped_D_name_ja"))
        all_dam_ids = {v[0] for v in dam_ids.values()}
        dam_runs = results[results["horse_id"].isin(all_dam_ids)]
        for horse_id, (dam_id, dam_name) in dam_ids.items():
            runs = dam_runs[dam_runs["horse_id"] == dam_id]
            if len(runs) == 0:
                dam_records[horse_id] = {"dam_name": dam_name, "raced_in_window": False}
                continue
            wins = int((pd.to_numeric(runs["finish_pos"], errors="coerce") == 1).sum())
            dam_records[horse_id] = {
                "dam_name": dam_name, "raced_in_window": True,
                "n_runs": int(len(runs)), "n_wins": wins,
                "max_distance_m": int(pd.to_numeric(runs["distance_m"], errors="coerce").max()),
                "min_distance_m": int(pd.to_numeric(runs["distance_m"], errors="coerce").min()),
            }

    # --- 今日の出走馬自身の母・母母・父母(D/DD/SD)ラインのpt差(2026-09-27追加)。
    # 総合適性スコア(composite_combined)は12レース版台帳と同じ定義(父/母父/父父の3ライン平均)のまま変更しない。
    # 母方3ラインは低カバレッジのため、総合適性には含めず「母方参考」として別枠で扱う(2026-10-03、指標統一)。
    n_damside_hit = {"dam": 0, "dd": 0, "sd": 0}
    if today_score:
        month = int(cfg["race_date"][5:7])
        today_buckets = {
            "distance_bucket_label": distance_bucket_label, "surface": surface,
            "course": venue, "season": PROF.SEASON_MAP[month], "turf_type": None,
        }
        today_horses_by_id = {h["horse_id"]: h for h in today_score["horses"]}
        for horse_id, hped in ped_cache.items():
            th = today_horses_by_id.get(horse_id)
            if th is None:
                continue
            rest_label = th.get("rest_bucket")
            for role_key, ped_code in (("dam", "D"), ("dd", "DD"), ("sd", "SD")):
                anc_id = hped.get(f"ped_{ped_code}_horse_id")
                score = compute_role_score(lut, profile_by_key, "dam", anc_id, today_buckets, rest_label)
                th[role_key] = score
                if score is not None:
                    n_damside_hit[role_key] += 1
                anc_name = hped.get(f"ped_{ped_code}_name_ja")
                th[f"{role_key}_name"] = anc_name if pd.notna(anc_name) else None
    print(f"    今日の出走馬{len(ped_cache)}頭中: 母{n_damside_hit['dam']} / 母母{n_damside_hit['dd']} / 父母{n_damside_hit['sd']}頭 データあり")

    return {
        "target_race_id": race_id, "item_labels": item_labels, "ssk_labels": SSK_LABELS,
        "role_spec": [{"key": k, "label": disp} for k, _c, disp, _r in ROLE_SPEC],
        "n_matched_races": len(match_race_ids), "n_scored_horses": len(horse_records),
        "going_breakdown_of_matched_races": going_counts,
        "template": template, "ssk_template": ssk_template, "importance_tier": tiers,
        "ssk_balance": ssk_balance, "n_field_horses": len(field_records),
        "n_unique_sire_bms_pairs": n_unique_pairs,
        "bloodline_sire": bloodline_sire, "bloodline_bms": bloodline_bms, "bloodline_ss": bloodline_ss,
        "today_bloodlines": today_bloodlines,
        "dam_records": dam_records, "n_today_horses": len(ped_cache), "n_damside_hit": n_damside_hit,
        "matched_races": race_list, "today": today_score,
        "match_rule": rule,
        "race_info": {k: cfg[k] for k in ("race_id", "race_number", "race_name", "class_name", "venue", "surface",
                                          "distance", "class_ord", "kind", "race_date", "start_time")},
    }


def main():
    only = {int(a) for a in sys.argv[1:] if a.isdigit()}  # 例: python build_race_radar_data.py 8 で8Rだけ
    print("race_results全期間ロード中...")
    results = load_all_results()
    lut = load_breakdown_lookup()
    profile = pd.read_csv(PROFILE_PATH, dtype={"horse_id_ancestor": str}, encoding="utf-8")
    profile_by_key = {(r["ancestor_role"], r["horse_id_ancestor"]): r for _, r in profile.iterrows()}
    bloodline_map = json.loads(BLOODLINE_MAP_PATH.read_text(encoding="utf-8")) if BLOODLINE_MAP_PATH.exists() else {}
    today_all = json.loads(SCORE_TODAY_PATH.read_text(encoding="utf-8"))
    race_date = today_all["today"]

    by_rid = results.drop_duplicates("race_id").set_index("race_id")
    races_out, skipped = {}, {}
    for tr in today_all["races"]:
        rno = int(tr["race_number"])
        if only and rno not in only:
            continue
        rid = tr["race_id"]
        if tr.get("skipped_reason") or not tr.get("distance_m") or tr.get("surface") not in SURFACE_NAME:
            skipped[str(rno)] = {"race_name": tr["race_name"],
                                 "reason": tr.get("skipped_reason") or "距離・サーフェスが取得できません"}
            continue
        # 今日のレースのクラスはrace_results(確定後)のrace_name、無ければ出走表のrace_nameから判定
        if rid in by_rid.index:
            full_name = by_rid.loc[rid, "race_name"]
            start_time = by_rid.loc[rid, "start_time"] if "start_time" in by_rid.columns else None
        else:
            full_name, start_time = tr["race_name"], None
        class_ord = class_ordinal(full_name)
        cfg = {
            "race_id": rid, "race_number": rno, "race_name": tr["race_name"], "class_name": full_name,
            "venue": tr["racecourse"], "surface": tr["surface"], "distance": float(tr["distance_m"]),
            "class_ord": class_ord, "kind": race_kind(full_name, class_ord), "day_seg": rid[8:10],
            "race_date": race_date, "start_time": start_time if isinstance(start_time, str) else None,
        }
        out = build_one(cfg, results, lut, profile_by_key, bloodline_map, tr)
        if isinstance(out, str):
            skipped[str(rno)] = {"race_name": tr["race_name"], "reason": out}
            print(f"[{rid[-2:]}R] 掲載見送り: {out}")
        else:
            races_out[str(rno)] = out

    payload = {"race_date": race_date, "races": races_out, "skipped": skipped,
               "min_races_strict": MIN_RACES_STRICT, "min_races_publish": MIN_RACES_PUBLISH}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(races_out)}レース掲載, {len(skipped)}レース見送り)")


if __name__ == "__main__":
    main()
