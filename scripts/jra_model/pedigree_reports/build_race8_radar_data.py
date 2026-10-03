# -*- coding: utf-8 -*-
"""今日の中山8R(202606040808、3歳以上1勝クラス・ダート1200m)と「同等の過去レース」
(同競馬場・同クラス・同距離・同サーフェス・同じ開催日数=開催8日目)を特定し、その上位3着馬の
血統(父・母父)から「このレースタイプで求められるファクター」をレーダーチャート用に集計する。

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
OUT_PATH = SCRATCH / "race8_radar_data.json"

TARGET_RACE_ID = "202606040808"
TARGET_VENUE, TARGET_SURFACE, TARGET_DISTANCE, TARGET_DAY_SEG = "中山", "ダ", 1200.0, "08"
TARGET_CLASS_ORD = 1  # 1勝クラス
TOP_N = 3

ITEM_LABELS = [
    ("distance", "距離(1200m帯)"),
    ("surface", "サーフェス(ダート)"),
    ("course", "競馬場(中山)"),
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


def build_field_records(df_finishers, lut, profile_by_key, distance_bucket_label):
    """出走馬(上位に限らず全頭)について、父・母父・父父・母・母母・父母それぞれの14項目pt差+
    スピード/スタミナ/キレを算出する。父父(SS)は種牡馬プロファイル(role="sire")に、
    母・母母・父母(D/DD/SD)は繁殖牝馬プロファイル(role="dam"、2026-09-27新設)に自身の産駒が
    登録されているケースがあれば、新規集計不要で同じロールでルックアップするだけで追加の
    評価軸として使える。母系は産駒数が少ないため該当率は低い(既知の制約、詳細はレポート内に
    カバレッジとして明記)。"""
    records = []
    for _, row in df_finishers.iterrows():
        horse_id = row["horse_id"]
        ped_path = pedigree_csv_path(horse_id)
        if not ped_path.exists():
            continue
        ped = pd.read_csv(ped_path, dtype=str, encoding="utf-8").iloc[0]
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
                                     distance_bucket_label, TARGET_SURFACE, TARGET_VENUE,
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


def main():
    print("race_results全期間ロード中...")
    results = load_all_results()

    matched = results.drop_duplicates("race_id")
    matched = matched[
        (matched["racecourse"] == TARGET_VENUE) & (matched["surface"] == TARGET_SURFACE)
        & (matched["distance_num"] == TARGET_DISTANCE) & (matched["class_ord"] == TARGET_CLASS_ORD)
        & (matched["day_seg"] == TARGET_DAY_SEG) & (matched["race_id"] != TARGET_RACE_ID)
    ]
    match_race_ids = set(matched["race_id"])
    print(f"同等過去レース: {len(match_race_ids)}件")

    field_all = results[
        results["race_id"].isin(match_race_ids) & results["finish_pos_num"].notna()
    ].copy()
    print(f"同等過去レース全出走 延べ{len(field_all)}頭")

    lut = load_breakdown_lookup()
    profile = pd.read_csv(PROFILE_PATH, dtype={"horse_id_ancestor": str}, encoding="utf-8")
    profile_by_key = {(r["ancestor_role"], r["horse_id_ancestor"]): r for _, r in profile.iterrows()}
    distance_bucket_label = PROF.distance_bucket(TARGET_DISTANCE)

    field_records = build_field_records(field_all, lut, profile_by_key, distance_bucket_label)
    print(f"血統照合済み(全出走) 延べ{len(field_records)}頭")
    horse_records = [r for r in field_records if r["finish_pos"] <= TOP_N]
    print(f"うち上位{TOP_N}着 延べ{len(horse_records)}頭")

    role_id_keys = [f"{k}_id" for k, _c, _d, _r in ROLE_SPEC]
    n_unique_pairs = len({
        tuple(r.get(rk) for rk in role_id_keys) for r in field_records
        if any(r.get(rk) is not None for rk in role_id_keys)
    })
    print(f"6ライン(父/母父/父父/母/母母/父母)IDの組み合わせの種類: {n_unique_pairs}通り(疑似反復の目安)")

    ssk_balance = {key: tercile_balance_analysis(field_records, key) for key, _label in SSK_LABELS}
    for key, _label in SSK_LABELS:
        combined_top3 = [combined_ssk_value(r, key) for r in horse_records]
        combined_top3 = [v for v in combined_top3 if v is not None]
        ssk_balance[key]["top3_combined_mean"] = (
            round(sum(combined_top3) / len(combined_top3), 2) if combined_top3 else None
        )
        ssk_balance[key]["top3_combined_n"] = len(combined_top3)
        print(f"{key}: {ssk_balance[key].get('pattern')}")

    # --- 系統別成績(父系統・母父系統・父父系統、netkeiba血統ビーム色分けの再利用) ---
    bloodline_map = json.loads(BLOODLINE_MAP_PATH.read_text(encoding="utf-8")) if BLOODLINE_MAP_PATH.exists() else {}
    bloodline_sire = bloodline_category_analysis(field_records, "sire_name", bloodline_map)
    bloodline_bms = bloodline_category_analysis(field_records, "bms_name", bloodline_map)
    bloodline_ss = bloodline_category_analysis(field_records, "ss_name", bloodline_map)
    print(f"系統別(父)集計: {len(bloodline_sire['rows'])}系統、不明{bloodline_sire['n_unmatched']}頭")
    print(f"系統別(母父)集計: {len(bloodline_bms['rows'])}系統、不明{bloodline_bms['n_unmatched']}頭")
    print(f"系統別(父父)集計: {len(bloodline_ss['rows'])}系統、不明{bloodline_ss['n_unmatched']}頭")
    today_bloodlines = today_bloodline_labels(TARGET_RACE_ID)

    # --- テンプレート(過去上位馬の平均プロファイル)算出。父父(ss)は父と同じ種牡馬プロファイル、
    # 母・母母・父母(dam/dd/sd)は新設の繁殖牝馬プロファイルを再利用する第4〜6のラインとして扱う ---
    template = {}
    for key, _label in ITEM_LABELS:
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

    # --- スピード・スタミナ・キレのテンプレート平均 ---
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
    tiers = importance_tier(distance_bucket_label, TARGET_SURFACE, TARGET_CLASS_ORD)

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

    today_score = None
    if SCORE_TODAY_PATH.exists():
        today_all = json.loads(SCORE_TODAY_PATH.read_text(encoding="utf-8"))
        today_race = next((r for r in today_all["races"] if r["race_id"] == TARGET_RACE_ID), None)
        if today_race:
            today_score = today_race

    # --- 補足参考情報: 今日の出走馬の「母自身」の現役時代の記録(産駒への遺伝伝達力とは別物、
    # プロ競馬予想家レビュー指摘: 個体の能力と遺伝伝達力を混同しないよう、軸には含めず別掲する) ---
    dam_records = {}
    ped_cache = {}
    if today_score:
        news_path = PROJECT_ROOT / "data" / "newspaper" / f"{TARGET_RACE_ID}.csv"
        news = pd.read_csv(news_path, dtype=str, encoding="utf-8")
        dam_ids = {}
        for _, nrow in news.iterrows():
            hp = pedigree_csv_path(nrow["horse_id"])
            if not hp.exists():
                continue
            hped = pd.read_csv(hp, dtype=str, encoding="utf-8").iloc[0]
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
    print(f"母自身の現役記録が確認できた馬: {sum(1 for v in dam_records.values() if v.get('raced_in_window'))}/{len(dam_records)}")

    # --- 今日の出走馬自身の母・母母・父母(D/DD/SD)ラインのpt差(2026-09-27追加)。
    # 父父(ss)と同じ再利用ルックアップだが、対象roleは今回新設の"dam"(産駒数が少なくカバレッジ低)。
    # 総合適性スコアもこの3ラインを含めてこのレポート内でのみ再計算する
    # (12レース版台帳の総合適性スコアとは値が異なる点に注意、そちらは父/母父/父父の3ラインのまま)。
    n_damside_hit = {"dam": 0, "dd": 0, "sd": 0}
    if today_score:
        today_buckets = {
            "distance_bucket_label": distance_bucket_label, "surface": TARGET_SURFACE,
            "course": TARGET_VENUE, "season": PROF.SEASON_MAP[9], "turf_type": None,
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
            parts = [
                s["composite_pt"] for s in
                (th.get("sire"), th.get("bms"), th.get("ss"), th.get("dam"), th.get("dd"), th.get("sd"))
                if s and s.get("composite_pt") is not None
            ]
            th["composite_combined"] = round(sum(parts) / len(parts), 2) if parts else None
    print(f"今日の出走馬{len(ped_cache)}頭中: 母データあり{n_damside_hit['dam']}頭 / "
          f"母母{n_damside_hit['dd']}頭 / 父母{n_damside_hit['sd']}頭")

    out = {
        "target_race_id": TARGET_RACE_ID, "item_labels": ITEM_LABELS, "ssk_labels": SSK_LABELS,
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
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
