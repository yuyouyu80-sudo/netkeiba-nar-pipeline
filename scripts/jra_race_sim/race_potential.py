# -*- coding: utf-8 -*-
"""naeba_potential.py(2026-08-02 新潟6R苗場特別専用・ハードコード)を一般化した版。
race_id・race_distance・surface(芝/ダート)・racecourse(競馬場名)・newspaper CSVパス・
出力先パスを引数に取る build_potential_csv(...) で、任意のレースについて同じ
スピード/スタミナ等ポテンシャル指数テーブルを生成する。

naeba_potential.py からの変更点は以下の2箇所のみ(それ以外のロジックは完全に同一):
  1. 距離帯別成績(data_distance_slot)の分類フィルタが "ダート" 決め打ちだったのを
     surface引数("芝"/"ダート")で切り替える。
  2. コース別成績(data_course_slot)の判定が "新潟" 決め打ちだったのを racecourse引数で
     切り替える(元のnaeba_potential.pyには無かったが、他競馬場のレースを正しく扱うには
     必須の一般化)。
"""
import re
import numpy as np
import pandas as pd

CLASS_ORDINAL = {
    "新馬": 0, "未勝利": 0,
    "1勝": 1, "2勝": 2, "3勝": 3,
    "OP": 4, "オープン": 4, "L": 4,
    "G3": 5, "GIII": 5, "G2": 6, "GII": 6, "G1": 7, "GI": 7,
}


def class_ordinal(text):
    if pd.isna(text):
        return np.nan
    text = str(text).strip()
    for key, val in CLASS_ORDINAL.items():
        if key in text:
            return val
    return np.nan


def to_pct(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("%", "").strip()
    if s in ("", "-", "--", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in ("", "-", "--", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def time_to_sec(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in ("", "-", "--", "nan"):
        return np.nan
    m = re.match(r"^(\d+):(\d+\.?\d*)$", s)
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def parse_weight(v):
    """'504(+2)' -> (504.0, 2.0)"""
    if pd.isna(v):
        return np.nan, np.nan
    m = re.match(r"^(\d+)\(([+-]?\d+)\)$", str(v).strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"^(\d+)$", str(v).strip())
    if m2:
        return float(m2.group(1)), np.nan
    return np.nan, np.nan


def extract_distance(label):
    if pd.isna(label):
        return np.nan
    m = re.search(r"(\d{3,4})m", str(label))
    return float(m.group(1)) if m else np.nan


def minmax_norm(series):
    s = series.astype(float)
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([np.nan] * len(s), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def build_potential_csv(race_id, race_distance, surface, racecourse, newspaper_csv_path, out_csv_path):
    """1レース分のポテンシャル指数CSVを生成して書き出し、DataFrameも返す。

    race_id: str
    race_distance: int/float (メートル)
    surface: "芝" または "ダート"
    racecourse: 競馬場名(例: "新潟"・"中京"・"札幌")。data_course_slotのラベル一致に使う。
    newspaper_csv_path: data/newspaper/{race_id}.csv のパス
    out_csv_path: 出力先CSVパス
    """
    df = pd.read_csv(newspaper_csv_path, dtype=str)
    df = df.sort_values("umaban", key=lambda c: c.astype(int)).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        horse_weight, horse_weight_diff = parse_weight(r.get("bias_horse_weight"))

        agari_list = [to_num(r.get(f"past{i}_agari_3f")) for i in range(1, 6)]
        agari_list = [v for v in agari_list if not pd.isna(v)]
        agari_avg = float(np.mean(agari_list)) if agari_list else np.nan
        agari_best = float(np.min(agari_list)) if agari_list else np.nan
        agari_std = float(np.std(agari_list, ddof=0)) if len(agari_list) >= 2 else np.nan

        speed_1ago = to_num(r.get("speed_index_1race_ago"))
        speed_3ago = to_num(r.get("speed_index_3races_ago"))
        speed_trend = speed_1ago - speed_3ago if not (pd.isna(speed_1ago) or pd.isna(speed_3ago)) else np.nan

        holdtime_just_sec = time_to_sec(r.get("holdtime_just_time"))
        holdtime_just_l3f = to_num(r.get("holdtime_just_l3f"))
        holdtime_long_sec = time_to_sec(r.get("holdtime_long_time"))
        holdtime_long_exp = 0 if pd.isna(r.get("holdtime_long_time")) else 1

        # 距離帯別成績(data_distance_slot1-3、slot4は「全成績」のため除外)。
        # surface引数に一致する表記のスロットのみを対象に、今回距離との差で分類する。
        same_dist = {"runs": 0, "place3": []}
        longer_dist = {"runs": 0, "place3": []}
        shorter_dist = {"runs": 0, "place3": []}
        for slot in range(1, 4):
            label = r.get(f"data_distance_slot{slot}_label")
            if pd.isna(label) or surface not in str(label):
                continue
            d = extract_distance(label)
            if pd.isna(d):
                continue
            runs = to_num(r.get(f"data_distance_slot{slot}_runs")) or 0
            place3 = to_pct(r.get(f"data_distance_slot{slot}_place3_rate"))
            target = same_dist if abs(d - race_distance) <= 100 else (
                longer_dist if d > race_distance + 100 else shorter_dist
            )
            target["runs"] += runs
            # runs=0(そのカテゴリでの出走実績なし)は「0%」ではなく欠測として扱う。
            # netkeibaのcourse_analysisはruns=0でもwin_rate/place3_rateに"0%"を返すため、
            # ここで弾かないと「未経験」が「経験して複勝率0%」と誤って数値化されてしまう。
            if runs > 0 and not pd.isna(place3):
                target["place3"].append((place3, runs))

        def weighted_place3(bucket):
            if not bucket["place3"]:
                return np.nan
            tot_runs = sum(w for _, w in bucket["place3"])
            if tot_runs <= 0:
                return float(np.mean([p for p, _ in bucket["place3"]]))
            return sum(p * w for p, w in bucket["place3"]) / tot_runs

        same_dist_place3 = weighted_place3(same_dist)
        longer_dist_place3 = weighted_place3(longer_dist)
        shorter_dist_place3 = weighted_place3(shorter_dist)
        longer_dist_runs = longer_dist["runs"]

        # コース実績(data_course_slot、方向別・全成績は除外)。racecourse引数の競馬場名で判定。
        course_place3 = np.nan
        for slot in range(1, 4):
            label = r.get(f"data_course_slot{slot}_label")
            if pd.isna(label):
                continue
            if racecourse in str(label):
                runs = to_num(r.get(f"data_course_slot{slot}_runs")) or 0
                if runs > 0:
                    course_place3 = to_pct(r.get(f"data_course_slot{slot}_place3_rate"))
                break

        # 馬場状態別ベスト(data_condition_slot1-4、常に 良/稍重/重/不良 の順)
        cond_labels = ["良", "稍重", "重", "不良"]
        best_cond_label, best_cond_rate = "", np.nan
        for slot, lab in zip(range(1, 5), cond_labels):
            runs = to_num(r.get(f"data_condition_slot{slot}_runs")) or 0
            if runs <= 0:
                continue
            rate = to_pct(r.get(f"data_condition_slot{slot}_place3_rate"))
            if pd.isna(rate):
                continue
            if pd.isna(best_cond_rate) or rate > best_cond_rate:
                best_cond_rate = rate
                best_cond_label = lab

        # クラス変化傾向(前走時点 = 前走クラス - 前々走クラス。正なら格上げ、負なら格下げ)
        cls1 = class_ordinal(r.get("past1_race_class"))
        cls2 = class_ordinal(r.get("past2_race_class"))
        class_trend = cls1 - cls2 if not (pd.isna(cls1) or pd.isna(cls2)) else np.nan

        finish_list = [to_num(r.get(f"past{i}_finish")) for i in range(1, 6)]
        finish_list = [v for v in finish_list if not pd.isna(v)]
        avg_finish = float(np.mean(finish_list)) if finish_list else np.nan

        same_lo, same_hi = int(race_distance - 100), int(race_distance + 100)
        rows.append({
            "馬番": int(r["umaban"]),
            "枠番": int(r["waku"]) if not pd.isna(r.get("waku")) else np.nan,
            "馬名": r.get("horse_name"),
            "性齢": r.get("bias_sex_age"),
            "斤量(kg)": to_num(r.get("bias_weight_carried")),
            "馬体重(kg)": horse_weight,
            "馬体重増減(前走比,kg)": horse_weight_diff,
            "脚質": r.get("ca_running_style_category_label"),
            "スピード指数_最高": to_num(r.get("speed_max_index")),
            "スピード指数_直近5走平均": to_num(r.get("speed_avg_index_5races")),
            "スピード指数_同距離最高": to_num(r.get("speed_max_distance_index")),
            "スピード指数_同コース最高": to_num(r.get("speed_max_course_index")),
            "スピード指数_前走": speed_1ago,
            "スピード指数_トレンド(前走-3走前)": speed_trend,
            "上がり3F_平均_過去5走(秒)": round(agari_avg, 2) if not pd.isna(agari_avg) else np.nan,
            "上がり3F_最速_過去5走(秒)": agari_best,
            "上がり3F_安定度_標準偏差(秒)": round(agari_std, 3) if not pd.isna(agari_std) else np.nan,
            "持続タイム_今回距離帯(秒)": holdtime_just_sec,
            "持続タイム_今回距離帯_上がり3F(秒)": holdtime_just_l3f,
            "持続タイム_長距離帯_経験有無(0/1)": holdtime_long_exp,
            "持続タイム_長距離帯(秒)": holdtime_long_sec,
            f"同距離帯({same_lo}-{same_hi}m)複勝率(%)": round(same_dist_place3, 1) if not pd.isna(same_dist_place3) else np.nan,
            f"距離延長時({same_hi}m超)複勝率(%)": round(longer_dist_place3, 1) if not pd.isna(longer_dist_place3) else np.nan,
            "距離延長時_走破数": longer_dist_runs if longer_dist_runs else np.nan,
            f"距離短縮時({same_lo}m未満)複勝率(%)": round(shorter_dist_place3, 1) if not pd.isna(shorter_dist_place3) else np.nan,
            f"{racecourse}コース複勝率(%)": course_place3,
            "馬場状態別最良条件": best_cond_label,
            "馬場状態別最良複勝率(%)": best_cond_rate,
            "枠番勝率(%)": to_pct(r.get("ca_waku_win_rate")),
            "脚質勝率(%)": to_pct(r.get("ca_running_style_win_rate")),
            "騎手勝率(%)": to_pct(r.get("ca_jockey_win_rate")),
            "調教師勝率(%)": to_pct(r.get("ca_trainer_win_rate")),
            "父": r.get("bias_sire"),
            "父系統": r.get("bias_sire_bloodline"),
            "母父": r.get("bias_dam_sire"),
            "母父系統": r.get("bias_dam_sire_bloodline"),
            "直近5走平均着順": round(avg_finish, 2) if not pd.isna(avg_finish) else np.nan,
            "クラス変化傾向(前走-前々走,+は格上げ)": class_trend,
        })

    out = pd.DataFrame(rows)

    # --- 総合指数(0-100、このレース出走頭数内の相対値。min-max正規化の単純平均) ---
    speed_components = pd.DataFrame({
        "a": minmax_norm(out["スピード指数_最高"]),
        "b": minmax_norm(out["スピード指数_直近5走平均"]),
        "c": minmax_norm(-out["上がり3F_平均_過去5走(秒)"]),
        "d": minmax_norm(-out["上がり3F_最速_過去5走(秒)"]),
    })
    out["総合スピード指数(0-100,場内相対)"] = speed_components.mean(axis=1, skipna=True).round(1)

    same_col = f"同距離帯({same_lo}-{same_hi}m)複勝率(%)"
    longer_col = f"距離延長時({same_hi}m超)複勝率(%)"
    stamina_components = pd.DataFrame({
        "a": minmax_norm(out[same_col]),
        "b": minmax_norm(out[longer_col]),
        "c": out["持続タイム_長距離帯_経験有無(0/1)"].astype(float) * 100.0,
        "d": minmax_norm(-out["持続タイム_今回距離帯(秒)"]),
    })
    out["総合スタミナ指数(0-100,場内相対)"] = stamina_components.mean(axis=1, skipna=True).round(1)

    out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    # 苗場特別で再現確認(naeba_potential.pyの出力と一致するはず)
    out = build_potential_csv(
        race_id="202604020406", race_distance=1800, surface="ダート", racecourse="新潟",
        newspaper_csv_path=r"C:\Users\yuyou\Desktop\新しい作業場所\data\newspaper\202604020406.csv",
        out_csv_path=r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir\race_potential_check.csv",
    )
    print("wrote", out.shape)
    print(out[["馬番", "馬名", "総合スピード指数(0-100,場内相対)", "総合スタミナ指数(0-100,場内相対)"]].to_string(index=False))
