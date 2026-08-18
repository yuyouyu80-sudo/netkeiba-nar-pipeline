# -*- coding: utf-8 -*-
"""動画位置データ(data/video_positions/{race_id}.csv)の検証ゲート。
2026-08-18のラストスパート較正タスクで確立した設計(旧`step3_validation_gate2.py`)を
繰り返し使えるよう一般化したもの。

設計上の教訓(重要、詳細はMETHODOLOGY.md「検証ゲート設計」参照):
単純に「動画の最終チェックポイント」を公式着順(finish_pos)と突き合わせる設計は、動画UIの
バー消失・カメラ切替・左右反転を「読み取りエラー」と誤判定してしまう(2026-08-18、燕特別で
発覚)。そのため全チェックポイントの時系列footruleトレンドを見る方式にしている。

このスクリプトは「読み取り手法が健全か」の自動スモークテストと、「要確認区間」の自動フラグ
出しまでを行う。フラグが立った区間がgenuine(実際の動き)かartifact(UI起因)かの最終判断は
人間(または専任サブエージェントによる追加フレーム investigation)が行い、
`data/video_positions/exclusions.csv`に記録する(このスクリプトは判断を自動確定しない)。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
VIDEO_POSITIONS_DIR = REPO_ROOT / "data" / "video_positions"
RESULTS_DIR = REPO_ROOT / "data" / "race_results"

# footruleが「ほぼ完全な反転」に近づいたと判断する閾値(video_position_metric.race_m7と
# 同じfootrule_norm定義: 0=完全一致, 1=完全反転相当)
REVIEW_THRESHOLD = 0.75
# 直近区間からの単調な悪化が続く点数(この点数以上、footruleが連続して悪化し続けたら
# 系統的な反転を疑うフラグを立てる。ステップ3で浜松特別の反転がこのパターンだった)
MONOTONIC_STREAK_FOR_FLAG = 4


def _rerank_within_common(values: dict, common):
    """commonの部分集合「内」でのみ順位を振り直す(元のスケールのまま比較すると、videoが
    読み取れた馬だけの部分集合と公式finish_pos(全頭スケール)とのスケール食い違いで
    footruleが1.0を超えてしまうため。corner_passing_metrics.pyの_stratified_footrule()と
    同じ考え方)。タイは平均順位。"""
    items = sorted(((k, values[k]) for k in common), key=lambda kv: kv[1])
    n = len(items)
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


def footrule(rank_a: dict, rank_b: dict):
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None, 0
    ra = _rerank_within_common(rank_a, common)
    rb = _rerank_within_common(rank_b, common)
    raw = sum(abs(ra[k] - rb[k]) for k in common)
    max_possible = (n * n) // 2
    return (raw / max_possible if max_possible > 0 else None), n


def load_finish_rank(race_id: str, date: str) -> dict:
    rdf = pd.read_csv(RESULTS_DIR / date[:4] / f"{date}.csv", encoding="utf-8")
    rdf = rdf[rdf["race_id"] == int(race_id)].copy()
    rdf["finish_pos_num"] = pd.to_numeric(rdf["finish_pos"], errors="coerce")
    return {
        int(r["umaban"]): int(r["finish_pos_num"])
        for _, r in rdf.iterrows() if not pd.isna(r["finish_pos_num"])
    }


def race_trajectory(race_id: str, date: str):
    """1レース分の全チェックポイント時系列footruleトレンドを計算する。
    戻り値: {"trajectory": [...], "best_footrule":.., "best_t":.., "flagged_windows": [...]}"""
    vdf = pd.read_csv(VIDEO_POSITIONS_DIR / f"{race_id}.csv", encoding="utf-8")
    vdf["umaban"] = vdf["umaban"].astype(int)
    vdf["t_sec"] = vdf["t_sec"].astype(float)
    finish_rank = load_finish_rank(race_id, date)

    trajectory = []
    for t_sec, grp in vdf.groupby("t_sec"):
        vr = {int(r["umaban"]): int(r["rank_official"]) for _, r in grp.iterrows()}
        fr, n = footrule(vr, finish_rank)
        trajectory.append({"t_sec": float(t_sec), "footrule": fr, "n": n})
    trajectory.sort(key=lambda x: x["t_sec"])

    valid = [p for p in trajectory if p["footrule"] is not None and p["n"] >= 3]
    best = min(valid, key=lambda p: p["footrule"]) if valid else None

    # 要確認区間の自動フラグ: footrule>=REVIEW_THRESHOLDの状態がMONOTONIC_STREAK_FOR_FLAG点
    # 以上連続する区間を機械的に拾う。
    # 2026-08-18の設計変更履歴(教訓): 当初「単調悪化トレンドの形状」で検出しようとしたが、
    # (a) 厳密な単調増加判定は途中1点の横ばいだけで区間が分断され検出漏れする、
    # (b) 横ばいを許容すると逆に、実際のレース展開として自然にfootruleが緩やかに上下する
    #     健全なレースまで広範囲に誤検知してしまう(17レース中15レースが該当する事態になった)。
    # 「トレンド形状」はgenuine(実展開)とartifact(UI起因)の判別に使えるほど頑健ではないと
    # 判断し、単純に「閾値以上が一定点数以上続くか」だけを機械的な一次フィルタとし、
    # genuine/artifactの最終判定は必ず人間(または専任サブエージェントでの追加フレーム調査)に
    # 委ねる設計にした(METHODOLOGY.md「検証ゲート設計」参照)。
    flagged = []
    streak_start = None
    streak_len = 0
    for i, p in enumerate(valid):
        if p["footrule"] >= REVIEW_THRESHOLD:
            if streak_start is None:
                streak_start = p["t_sec"]
            streak_len += 1
        else:
            if streak_len >= MONOTONIC_STREAK_FOR_FLAG:
                flagged.append({"t_start": streak_start, "t_end": valid[i - 1]["t_sec"],
                                 "reason": f"footrule>={REVIEW_THRESHOLD}が{streak_len}点連続(要目視確認)"})
            streak_start, streak_len = None, 0
    if streak_len >= MONOTONIC_STREAK_FOR_FLAG:
        flagged.append({"t_start": streak_start, "t_end": valid[-1]["t_sec"],
                         "reason": f"footrule>={REVIEW_THRESHOLD}が{streak_len}点連続(要目視確認、"
                                   "系列末尾まで継続=開区間の可能性)"})

    return {
        "race_id": race_id, "trajectory": trajectory,
        "best_footrule": best["footrule"] if best else None,
        "best_t": best["t_sec"] if best else None,
        "flagged_windows": flagged,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(VIDEO_POSITIONS_DIR / "_manifest.csv"),
                     help="race_id,date列を持つmanifest CSV(省略時はdata/video_positions/_manifest.csv)")
    ap.add_argument("--race-ids", default=None,
                     help="カンマ区切りでrace_idを直接指定(--manifestの代わりに使う場合)")
    ap.add_argument("--out", default=str(VIDEO_POSITIONS_DIR / "_validation_gate_result.json"))
    args = ap.parse_args()

    if args.race_ids:
        race_ids = args.race_ids.split(",")
        manifest = pd.read_csv(args.manifest, encoding="utf-8", dtype=str)
        date_by_rid = dict(zip(manifest["race_id"], manifest["date"]))
    else:
        manifest = pd.read_csv(args.manifest, encoding="utf-8", dtype=str)
        race_ids = manifest["race_id"].tolist()
        date_by_rid = dict(zip(manifest["race_id"], manifest["date"]))

    report = {}
    needs_review = []
    print(f"{'race_id':14s} {'best_fr':>8s} {'@t':>6s}  flagged_windows")
    for rid in race_ids:
        date = date_by_rid.get(rid)
        if not date:
            print(f"{rid:14s}  スキップ(manifestにdateが無い)")
            continue
        result = race_trajectory(rid, date)
        report[rid] = result
        bf = result["best_footrule"]
        bf_s = f"{bf:.3f}" if bf is not None else "n/a"
        bt = result["best_t"]
        print(f"{rid:14s} {bf_s:>8s} {bt if bt is not None else '':>6}  {result['flagged_windows']}")
        if result["flagged_windows"] or bf is None or bf > 0.25:
            needs_review.append(rid)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print(f"要確認(needs_review推奨): {needs_review if needs_review else 'なし'}")


if __name__ == "__main__":
    main()
