# -*- coding: utf-8 -*-
"""単勝・複勝・ワイド・馬連・馬単・3連複・3連単の回収率110%台帳を、現行
data/jra_pipeline/factor_database.json(2026-09-15時点、レーダー細分化9グループ・
順位フィルタの1〜2/1〜4/1〜6位追加を含む81グループ・290atom)を候補プールに
ビームサーチで(再)構築する。候補プールはjra_factor_registry.FACTOR_GROUPSから
毎回動的に読み込むため、本スクリプト自体は登録内容の変化に無改造で追随する。

背景(重要): 既存6台帳(単勝/複勝/馬連/ワイド/3連複/3連単、2026-09-07生成)を作った元の
ビームサーチスクリプト(`search_jra_bettype_return.py`、旧台帳の`<script>`ロジックから
ファイル名が確認できるのみ)はリポジトリに存在しない(2026-09-14の台帳再検証タスクで
既に判明済み)。本スクリプトは同じ設計思想を踏襲して探索アルゴリズム自体を新規実装した
もの(元アルゴリズムの再現ではない)。馬単は既存6台帳に無かった新規券種。

候補プール: jra_factor_registry.FACTOR_GROUPS 全81グループ・290atom(2026-09-14の
レーダー細分化9グループ・2026-09-15の順位フィルタ拡張(score_rank系3グループ+
レーダー系16グループへ1〜2/1〜4/1〜6位を追加)を含む)。1パターン=最大6グループから
1atomずつ選んだ組み合わせ(グループ間AND、グループ内は元々1値のみなのでOR相当の
意味は無い)。

母集団: factor_database.jsonの全631レースのうち、直近2開催日をホールドアウトとして
除外(旧台帳のJSロジック`AND_HOLDOUT`と同じ設計を踏襲、`build_ledger_artifact_2026_09_15.py`
のAND再計算パネルも同じ母集団を使う)。

ビームサーチ: depth 1で236atom全てを単独評価、coverage>=40%のものを候補化。depth 2以降は
前段の生存パターン×未使用グループのatomを1つ追加、coverage>=40%を満たすものだけ次段へ。
各段の生存パターンはreturn_rate_pct降順でBEAM_WIDTH件に絞ってから次段を展開する。
どの段であってもcoverage>=40%かつreturn_rate_pct>110%を満たしたパターンは全件
「台帳候補」として記録する(次段へ進めるかどうかとは独立)。券種ごとに独立実行
(ビームの伸長方向=その券種のreturn_rate_pct降順のため)。

決済(高速化): 実際の払戻テーブルは1レースあたり数件(単勝1件・複勝上位数件・
ワイド最大3件・馬連/馬単/3連複/3連単は1件)しか無い疎な構造であることを利用し、
「そのレースの払戻対象馬が全員、パターンの該当馬集合に含まれているか」をnumpyの
ブールインデックスだけで判定する(itertools.combinationsによる総当たりを行わない)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
import jra_dataset_wide as JDW  # noqa: E402
import jra_factor_registry as FR  # noqa: E402

OUT_PATH = DATA_DIR / "jra_ledger_search_2026_09_15_result.json"

BEAM_WIDTH = 300
MAX_DEPTH = 6
COVERAGE_MIN_PCT = 40.0
RETURN_THRESHOLD_PCT = 110.0
N_HOLDOUT_DATES = 2

# 券種: (組合せサイズr, 順序ありか, 最低必要頭数=r)
BET_SPECS = {
    "単勝": (1, False),
    "複勝": (1, False),
    "馬連": (2, False),
    "ワイド": (2, False),
    "馬単": (2, True),
    "3連複": (3, False),
    "3連単": (3, True),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("factor_database.json読み込み中...")
    fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
    all_races = fdb["races"]
    all_dates = sorted({r["kaisai_date"] for r in all_races})
    holdout_dates = set(all_dates[-N_HOLDOUT_DATES:])
    races = [r for r in all_races if r["kaisai_date"] not in holdout_dates]
    log(f"全{len(all_races)}レース中、直近{N_HOLDOUT_DATES}開催日"
        f"({sorted(holdout_dates)})をホールドアウトとして除外 -> 探索母集団{len(races)}レース")

    log("payouts読み込み中(jra_dataset_wide)...")
    wide = JDW.load(rebuild=False)
    actual = wide["actual"]
    offered = wide["offered_bet_types"]

    # --------------------------------------------------------------- 馬を1本の配列へ平坦化
    flat_horses = []
    race_offsets = [0]
    race_ids = []
    race_dates = []
    for r in races:
        horses = [h for h in r["horses"] if h.get("umaban") is not None]
        if not horses:
            continue
        flat_horses.extend(horses)
        race_offsets.append(len(flat_horses))
        race_ids.append(r["race_id"])
        race_dates.append(r["kaisai_date"])
    race_offsets = np.array(race_offsets, dtype=np.int64)
    n_races = len(race_ids)
    n_horses = len(flat_horses)
    log(f"平坦化: {n_races}レース・{n_horses}頭")

    umaban_flat = np.array([int(h["umaban"]) for h in flat_horses], dtype=np.int64)

    # --------------------------------------------------------------- atomマスク事前計算
    atoms = []  # [(gid, opt_id), ...]
    atom_masks = {}
    for gid, group in FR.FACTOR_GROUPS.items():
        for opt in group["options"]:
            key = (gid, opt["id"])
            atoms.append(key)
            atom_masks[key] = np.array(
                [FR.evaluate_option(h, group, opt) for h in flat_horses], dtype=bool)
    log(f"atom数: {len(atoms)}  (グループ数: {len(FR.FACTOR_GROUPS)})")

    # --------------------------------------------------------------- 券種ごとのcombo配列
    # combo_members[bt]: shape (n_combo, r) 平坦indexの配列。combo_payout[bt]: (n_combo,)。
    # combo_race_idx[bt]: どのレース(0-based)由来か(hit_races集計用)。
    bet_combo = {}
    for bt, (r_size, ordered) in BET_SPECS.items():
        members_list, payout_list, race_idx_list = [], [], []
        for ri, race_id in enumerate(race_ids):
            m = actual.get(race_id, {}).get(bt, {})
            if not m:
                continue
            base = race_offsets[ri]
            umaban_to_flat = {int(u): base + j for j, u in
                               enumerate(umaban_flat[race_offsets[ri]:race_offsets[ri + 1]])}
            for combo, payout in m.items():
                combo_seq = (combo,) if isinstance(combo, int) else tuple(combo)
                idxs = [umaban_to_flat.get(int(u)) for u in combo_seq]
                if any(i is None for i in idxs):
                    continue  # 払戻対象馬がfactor_database側に存在しない(取消等) -> 一致しようがない
                if len(idxs) != r_size:
                    continue
                members_list.append(idxs)
                payout_list.append(int(payout))
                race_idx_list.append(ri)
        if members_list:
            combo_members = np.array(members_list, dtype=np.int64)
            combo_payout = np.array(payout_list, dtype=np.int64)
            combo_race_idx = np.array(race_idx_list, dtype=np.int64)
        else:
            combo_members = np.zeros((0, r_size), dtype=np.int64)
            combo_payout = np.zeros((0,), dtype=np.int64)
            combo_race_idx = np.zeros((0,), dtype=np.int64)
        bet_combo[bt] = (combo_members, combo_payout, combo_race_idx)
        log(f"  {bt}: 払戻combo {len(members_list)}件")

    # offered_bet_typesマスク(レースごとにその券種が発売されていたか)
    offered_mask = {}
    for bt in BET_SPECS:
        offered_mask[bt] = np.array(
            [bt in offered.get(rid, []) for rid in race_ids], dtype=bool)

    seg_starts = race_offsets[:-1]

    def eval_mask(mask: np.ndarray, bt: str, min_horses: int):
        counts = np.add.reduceat(mask.astype(np.int64), seg_starts)
        qualifies = (counts >= min_horses) & offered_mask[bt]
        n_q = int(qualifies.sum())
        if n_q == 0:
            return None
        r_size, ordered = BET_SPECS[bt]
        k = counts
        if r_size == 1:
            stake_per_race = k
        elif r_size == 2:
            stake_per_race = (k * (k - 1)) if ordered else (k * (k - 1) // 2)
        else:
            stake_per_race = (k * (k - 1) * (k - 2)) if ordered else (k * (k - 1) * (k - 2) // 6)
        stake_per_race = np.where(qualifies, stake_per_race, 0)
        total_stake = int(stake_per_race.sum()) * 100
        combo_members, combo_payout, combo_race_idx = bet_combo[bt]
        total_return = 0
        if combo_members.shape[0] > 0:
            ok = mask[combo_members[:, 0]]
            for c in range(1, r_size):
                ok &= mask[combo_members[:, c]]
            total_return = int(np.where(ok, combo_payout, 0).sum())
        coverage_pct = n_q / n_races * 100.0
        return_rate_pct = (total_return / total_stake * 100.0) if total_stake else 0.0
        return {
            "n_races": n_q, "stake": total_stake, "return": total_return,
            "coverage_pct": coverage_pct, "return_rate_pct": return_rate_pct,
        }

    def finalize_pattern(pattern, mask, bt):
        r_size, ordered = BET_SPECS[bt]
        min_horses = r_size
        stats = eval_mask(mask, bt, min_horses)
        counts = np.add.reduceat(mask.astype(np.int64), seg_starts)
        qualifies = (counts >= min_horses) & offered_mask[bt]
        combo_members, combo_payout, combo_race_idx = bet_combo[bt]
        hit_races = 0
        if combo_members.shape[0] > 0:
            ok = mask[combo_members[:, 0]]
            for c in range(1, r_size):
                ok &= mask[combo_members[:, c]]
            hit_mask = ok & (combo_payout > 0)
            if hit_mask.any():
                hit_flags = np.zeros(n_races, dtype=bool)
                np.logical_or.at(hit_flags, combo_race_idx[hit_mask], True)
                hit_races = int((hit_flags & qualifies).sum())
        n_points = stats["stake"] // 100
        selected = {gid: oid for gid, oid in pattern}
        conditions = [{"group": FR.FACTOR_GROUPS[gid]["label"],
                       "value": next(o["label"] for o in FR.FACTOR_GROUPS[gid]["options"] if o["id"] == oid)}
                      for gid, oid in pattern]
        return {
            "selected": selected, "conditions": conditions,
            "n_races": stats["n_races"], "n_points": int(n_points),
            "hit_races": hit_races,
            "hit_rate_pct": round(hit_races / stats["n_races"] * 100, 4) if stats["n_races"] else 0.0,
            "return_rate_pct": round(stats["return_rate_pct"], 4),
            "stake": stats["stake"], "return": stats["return"],
            "coverage_pct": round(stats["coverage_pct"], 4),
        }

    results = {}
    for bt, (r_size, ordered) in BET_SPECS.items():
        t0 = time.time()
        min_horses = r_size
        log(f"=== {bt} 探索開始 ===")
        frontier = []
        found = {}  # frozenset(pattern) -> pattern tuple (dedup)
        n_evals = 0
        for atom in atoms:
            n_evals += 1
            stats = eval_mask(atom_masks[atom], bt, min_horses)
            if stats is None or stats["coverage_pct"] < COVERAGE_MIN_PCT:
                continue
            pattern = (atom,)
            frontier.append((pattern, atom_masks[atom], stats["return_rate_pct"]))
            if stats["return_rate_pct"] > RETURN_THRESHOLD_PCT:
                found.setdefault(frozenset(pattern), pattern)
        frontier.sort(key=lambda x: -x[2])
        frontier = frontier[:BEAM_WIDTH]
        log(f"  depth1: 候補{len(frontier)}件(coverage>=40%) / 累計found={len(found)} / eval={n_evals}")

        for depth in range(2, MAX_DEPTH + 1):
            next_frontier = []
            for pattern, mask, _ in frontier:
                used_groups = {gid for gid, oid in pattern}
                for gid, oid in atoms:
                    if gid in used_groups:
                        continue
                    new_mask = mask & atom_masks[(gid, oid)]
                    n_evals += 1
                    stats = eval_mask(new_mask, bt, min_horses)
                    if stats is None or stats["coverage_pct"] < COVERAGE_MIN_PCT:
                        continue
                    new_pattern = pattern + ((gid, oid),)
                    next_frontier.append((new_pattern, new_mask, stats["return_rate_pct"]))
                    if stats["return_rate_pct"] > RETURN_THRESHOLD_PCT:
                        found.setdefault(frozenset(new_pattern), new_pattern)
            if not next_frontier:
                log(f"  depth{depth}: 展開先なし、打ち切り")
                break
            next_frontier.sort(key=lambda x: -x[2])
            frontier = next_frontier[:BEAM_WIDTH]
            log(f"  depth{depth}: 候補{len(next_frontier)}件 -> beam{len(frontier)}件に絞込 "
                f"/ 累計found={len(found)} / 累計eval={n_evals}")

        rows = []
        for pattern in found.values():
            mask = atom_masks[pattern[0]].copy()
            for a in pattern[1:]:
                mask &= atom_masks[a]
            rows.append(finalize_pattern(pattern, mask, bt))
        rows.sort(key=lambda x: -x["return_rate_pct"])
        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        results[bt] = {
            "bet_type": bt, "min_race_frac": COVERAGE_MIN_PCT / 100, "return_threshold": RETURN_THRESHOLD_PCT,
            "total_races": n_races, "min_races": int(np.ceil(n_races * COVERAGE_MIN_PCT / 100)),
            "n_patterns": len(rows), "rows": rows,
        }
        log(f"=== {bt} 完了: {len(rows)}パターン ({time.time()-t0:.1f}秒, eval={n_evals}回) ===")

    meta_common = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "beam_width": BEAM_WIDTH, "max_depth": MAX_DEPTH,
        "coverage_min_pct": COVERAGE_MIN_PCT, "return_threshold_pct": RETURN_THRESHOLD_PCT,
        "n_atoms": len(atoms), "n_groups": len(FR.FACTOR_GROUPS),
        "full_population": {
            "normal": fdb["population"]["normal"]["n_races"],
            "shinba": fdb["population"]["shinba"]["n_races"],
            "mishoubi": fdb["population"]["mishoubi"]["n_races"],
            "total": fdb["population"]["total"]["n_races"],
            "date_range": fdb["population"]["date_range"],
            "n_dates": fdb["population"]["n_dates"],
        },
        "holdout_dates": sorted(holdout_dates),
        "search_population": n_races,
    }
    out = {"meta": meta_common, "bet_types": results}
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"書き出し完了: {OUT_PATH} ({OUT_PATH.stat().st_size/1024/1024:.2f}MB)")


if __name__ == "__main__":
    main()
