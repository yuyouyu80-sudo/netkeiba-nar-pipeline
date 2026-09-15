# -*- coding: utf-8 -*-
"""複勝の消し材料台帳の試算: jra_ledger_search_low_2026_09_15.py(回収率25%未満)では
該当5件まで激減したため、基準を「回収率」から「的中率5%未満」に変えた場合に該当件数が
どう変わるかを一度だけ試算する(本番のArtifact化はまだ行わず、件数と上位パターンの
確認が目的)。

候補プール・母集団・coverage>=40%の設計はjra_ledger_search_low_2026_09_15.pyと共通。
違いはビームの展開方向と「台帳候補」判定基準を return_rate_pct から hit_rate_pct に
差し替えた点のみ(昇順に残す=的中率が低いものを掘り下げる)。
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

OUT_PATH = DATA_DIR / "jra_ledger_search_fukusho_hitrate_trial_2026_09_15_result.json"

BEAM_WIDTH = 300
MAX_DEPTH = 6
COVERAGE_MIN_PCT = 40.0
HIT_RATE_THRESHOLD_PCT = 5.0
N_HOLDOUT_DATES = 2
BT = "複勝"
R_SIZE, ORDERED = 1, False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("factor_database.json読み込み中...")
    fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
    all_races = fdb["races"]
    all_dates = sorted({r["kaisai_date"] for r in all_races})
    holdout_dates = set(all_dates[-N_HOLDOUT_DATES:])
    races = [r for r in all_races if r["kaisai_date"] not in holdout_dates]
    log(f"探索母集団{len(races)}レース")

    log("payouts読み込み中(jra_dataset_wide)...")
    wide = JDW.load(rebuild=False)
    actual = wide["actual"]
    offered = wide["offered_bet_types"]

    flat_horses = []
    race_offsets = [0]
    race_ids = []
    for r in races:
        horses = [h for h in r["horses"] if h.get("umaban") is not None]
        if not horses:
            continue
        flat_horses.extend(horses)
        race_offsets.append(len(flat_horses))
        race_ids.append(r["race_id"])
    race_offsets = np.array(race_offsets, dtype=np.int64)
    n_races = len(race_ids)
    log(f"平坦化: {n_races}レース・{len(flat_horses)}頭")

    umaban_flat = np.array([int(h["umaban"]) for h in flat_horses], dtype=np.int64)

    atoms = []
    atom_masks = {}
    for gid, group in FR.FACTOR_GROUPS.items():
        for opt in group["options"]:
            key = (gid, opt["id"])
            atoms.append(key)
            atom_masks[key] = np.array(
                [FR.evaluate_option(h, group, opt) for h in flat_horses], dtype=bool)
    log(f"atom数: {len(atoms)}  (グループ数: {len(FR.FACTOR_GROUPS)})")

    members_list, payout_list, race_idx_list = [], [], []
    for ri, race_id in enumerate(race_ids):
        m = actual.get(race_id, {}).get(BT, {})
        if not m:
            continue
        base = race_offsets[ri]
        umaban_to_flat = {int(u): base + j for j, u in
                           enumerate(umaban_flat[race_offsets[ri]:race_offsets[ri + 1]])}
        for combo, payout in m.items():
            combo_seq = (combo,) if isinstance(combo, int) else tuple(combo)
            idxs = [umaban_to_flat.get(int(u)) for u in combo_seq]
            if any(i is None for i in idxs) or len(idxs) != R_SIZE:
                continue
            members_list.append(idxs)
            payout_list.append(int(payout))
            race_idx_list.append(ri)
    combo_members = np.array(members_list, dtype=np.int64) if members_list else np.zeros((0, R_SIZE), dtype=np.int64)
    combo_payout = np.array(payout_list, dtype=np.int64) if payout_list else np.zeros((0,), dtype=np.int64)
    combo_race_idx = np.array(race_idx_list, dtype=np.int64) if race_idx_list else np.zeros((0,), dtype=np.int64)
    log(f"  {BT}: 払戻combo {len(members_list)}件")

    offered_mask = np.array([BT in offered.get(rid, []) for rid in race_ids], dtype=bool)
    seg_starts = race_offsets[:-1]

    def eval_mask(mask: np.ndarray):
        counts = np.add.reduceat(mask.astype(np.int64), seg_starts)
        qualifies = (counts >= R_SIZE) & offered_mask
        n_q = int(qualifies.sum())
        if n_q == 0:
            return None
        stake_per_race = np.where(qualifies, counts, 0)
        total_stake = int(stake_per_race.sum()) * 100
        total_return = 0
        hit_races = 0
        if combo_members.shape[0] > 0:
            ok = mask[combo_members[:, 0]]
            total_return = int(np.where(ok, combo_payout, 0).sum())
            hit_mask = ok & (combo_payout > 0)
            if hit_mask.any():
                hit_flags = np.zeros(n_races, dtype=bool)
                np.logical_or.at(hit_flags, combo_race_idx[hit_mask], True)
                hit_races = int((hit_flags & qualifies).sum())
        coverage_pct = n_q / n_races * 100.0
        return_rate_pct = (total_return / total_stake * 100.0) if total_stake else 0.0
        hit_rate_pct = (hit_races / n_q * 100.0) if n_q else 0.0
        return {
            "n_races": n_q, "stake": total_stake, "return": total_return,
            "hit_races": hit_races, "coverage_pct": coverage_pct,
            "return_rate_pct": return_rate_pct, "hit_rate_pct": hit_rate_pct,
        }

    def finalize_pattern(pattern, mask):
        stats = eval_mask(mask)
        n_points = stats["stake"] // 100
        selected = {gid: oid for gid, oid in pattern}
        conditions = [{"group": FR.FACTOR_GROUPS[gid]["label"],
                       "value": next(o["label"] for o in FR.FACTOR_GROUPS[gid]["options"] if o["id"] == oid)}
                      for gid, oid in pattern]
        return {
            "selected": selected, "conditions": conditions,
            "n_races": stats["n_races"], "n_points": int(n_points),
            "hit_races": stats["hit_races"],
            "hit_rate_pct": round(stats["hit_rate_pct"], 4),
            "return_rate_pct": round(stats["return_rate_pct"], 4),
            "stake": stats["stake"], "return": stats["return"],
            "coverage_pct": round(stats["coverage_pct"], 4),
        }

    t0 = time.time()
    log(f"=== {BT} 探索開始(消し材料試算、的中率{HIT_RATE_THRESHOLD_PCT:.0f}%未満基準) ===")
    frontier = []
    found = {}
    n_evals = 0
    for atom in atoms:
        n_evals += 1
        stats = eval_mask(atom_masks[atom])
        if stats is None or stats["coverage_pct"] < COVERAGE_MIN_PCT:
            continue
        pattern = (atom,)
        frontier.append((pattern, atom_masks[atom], stats["hit_rate_pct"]))
        if stats["hit_rate_pct"] < HIT_RATE_THRESHOLD_PCT:
            found.setdefault(frozenset(pattern), pattern)
    frontier.sort(key=lambda x: x[2])
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
                stats = eval_mask(new_mask)
                if stats is None or stats["coverage_pct"] < COVERAGE_MIN_PCT:
                    continue
                new_pattern = pattern + ((gid, oid),)
                next_frontier.append((new_pattern, new_mask, stats["hit_rate_pct"]))
                if stats["hit_rate_pct"] < HIT_RATE_THRESHOLD_PCT:
                    found.setdefault(frozenset(new_pattern), new_pattern)
        if not next_frontier:
            log(f"  depth{depth}: 展開先なし、打ち切り")
            break
        next_frontier.sort(key=lambda x: x[2])
        frontier = next_frontier[:BEAM_WIDTH]
        log(f"  depth{depth}: 候補{len(next_frontier)}件 -> beam{len(frontier)}件に絞込 "
            f"/ 累計found={len(found)} / 累計eval={n_evals}")

    rows = []
    for pattern in found.values():
        mask = atom_masks[pattern[0]].copy()
        for a in pattern[1:]:
            mask &= atom_masks[a]
        rows.append(finalize_pattern(pattern, mask))
    rows.sort(key=lambda x: (x["hit_rate_pct"], x["return_rate_pct"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    log(f"=== {BT} 完了: {len(rows)}パターン ({time.time()-t0:.1f}秒, eval={n_evals}回) ===")

    out = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "beam_width": BEAM_WIDTH, "max_depth": MAX_DEPTH,
            "coverage_min_pct": COVERAGE_MIN_PCT, "hit_rate_threshold_pct": HIT_RATE_THRESHOLD_PCT,
            "n_atoms": len(atoms), "n_groups": len(FR.FACTOR_GROUPS),
            "search_population": n_races, "n_patterns": len(rows),
        },
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"書き出し完了: {OUT_PATH}")

    # 上位20件をコンソールにもプレビュー表示
    print("\n--- 上位20件プレビュー(的中率昇順) ---")
    for row in rows[:20]:
        conds = " / ".join(f"{c['group']}:{c['value']}" for c in row["conditions"])
        print(f"#{row['rank']} n_races={row['n_races']} hit_rate={row['hit_rate_pct']}% "
              f"return_rate={row['return_rate_pct']}% conds=[{conds}]")


if __name__ == "__main__":
    main()
