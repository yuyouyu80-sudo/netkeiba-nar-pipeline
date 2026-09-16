# -*- coding: utf-8 -*-
"""単勝・複勝・ワイドの「回収率が低い条件の組み合わせ(消し材料)」台帳を、現行
data/jra_pipeline/factor_database.json(2026-09-15時点、81グループ・290atom)を
候補プールにビームサーチで(再)構築する。エンジン自体はjra_ledger_search_2026_09_15.py
(回収率110%超=高回収率側の探索)とほぼ同一で、探索方向だけを反転させたもの。

背景: 既存3台帳(単勝回収率10%未満・複勝回収率25%未満・ワイド回収率10%未満、
2026-09-07生成)は、現行FACTOR_GROUPSには存在しない専用の「低い方」169atom語彙
(例: ninki_low_ge13)を使っていたが、その元のビームサーチスクリプトは
リポジトリに存在しない(2026-09-14の台帳再検証タスクで既に判明済み)。
今回は「低い方」専用語彙を復元するのではなく、単勝〜3連単110%台帳(2026-09-15、
jra_ledger_search_2026_09_15.py)と全く同じ290atom候補プール・同じビームサーチ
エンジンを流用し、探索方向(回収率の大小)だけを反転させる設計にした
(候補プールを統一した方が「JRAファクター検証データベース」全体との一貫性が高く、
消し材料としての解釈も「この条件を満たす馬は回収率が低い(=消し材料になりうる)」
という意味でそのまま成立するため)。

候補プール: jra_factor_registry.FACTOR_GROUPS 全81グループ・290atom(単勝110%台帳等と共通)。
母集団: factor_database.jsonの全631レースのうち、直近2開催日をホールドアウトとして除外した
561レース(単勝110%台帳等と共通母集団)。

ビームサーチ(方向反転): depth1で290atom全てを単独評価しcoverage>=40%のものを候補化。
depth2以降は前段の生存パターン×未使用グループのatomを1つ追加、coverage>=40%を満たす
もののみ次段へ。各段の生存パターンはreturn_rate_pct**昇順**でBEAM_WIDTH件に絞ってから
次段を展開する(高回収率側は降順で残すのに対し、ここでは最も回収率が低いものを残して
掘り下げる)。どの段であってもcoverage>=40%かつreturn_rate_pctが券種別しきい値
未満(単勝10%未満・複勝25%未満・ワイド10%未満)を満たしたパターンは全件「台帳候補」として
記録する。
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

# 2026-09-16: 「本番予想スコア順位(BOX4/BOX3モデル基準)」は本番モデル自身の出力を条件に
# 使うだけの循環参照(消し材料としての新規情報を持たない)のため、消し材料台帳の候補プールから
# 除外する。110%台帳・JRAファクター検証データベースでは引き続き使うため、
# jra_factor_registry.py本体は変更せずこのプロセス内でのみ除外する。
for _gid in ("score_rank_box4", "score_rank_box3"):
    FR.FACTOR_GROUPS.pop(_gid, None)

OUT_PATH = DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json"

BEAM_WIDTH = 300
MAX_DEPTH = 6
COVERAGE_MIN_PCT = 40.0
N_HOLDOUT_DATES = 2

# 券種: (組合せサイズr, 順序ありか, 回収率しきい値(これ未満を「台帳候補」とする))
BET_SPECS = {
    "単勝": (1, False, 10.0),
    "複勝": (1, False, 25.0),
    "ワイド": (2, False, 10.0),
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
    bet_combo = {}
    for bt, (r_size, ordered, _thr) in BET_SPECS.items():
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
                    continue
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
        r_size, ordered, _thr = BET_SPECS[bt]
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
        r_size, ordered, _thr = BET_SPECS[bt]
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
    for bt, (r_size, ordered, threshold) in BET_SPECS.items():
        t0 = time.time()
        min_horses = r_size
        log(f"=== {bt} 探索開始(消し材料、回収率{threshold:.0f}%未満) ===")
        frontier = []
        found = {}
        n_evals = 0
        for atom in atoms:
            n_evals += 1
            stats = eval_mask(atom_masks[atom], bt, min_horses)
            if stats is None or stats["coverage_pct"] < COVERAGE_MIN_PCT:
                continue
            pattern = (atom,)
            frontier.append((pattern, atom_masks[atom], stats["return_rate_pct"]))
            if stats["return_rate_pct"] < threshold:
                found.setdefault(frozenset(pattern), pattern)
        frontier.sort(key=lambda x: x[2])  # 昇順(回収率が低い順)に残す
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
                    if stats["return_rate_pct"] < threshold:
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
            rows.append(finalize_pattern(pattern, mask, bt))
        rows.sort(key=lambda x: x["return_rate_pct"])  # 昇順(最も回収率が低い=最強の消し材料が1位)
        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        results[bt] = {
            "bet_type": bt, "min_race_frac": COVERAGE_MIN_PCT / 100, "return_threshold": threshold,
            "total_races": n_races, "min_races": int(np.ceil(n_races * COVERAGE_MIN_PCT / 100)),
            "n_patterns": len(rows), "rows": rows,
        }
        log(f"=== {bt} 完了: {len(rows)}パターン ({time.time()-t0:.1f}秒, eval={n_evals}回) ===")

    meta_common = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "beam_width": BEAM_WIDTH, "max_depth": MAX_DEPTH,
        "coverage_min_pct": COVERAGE_MIN_PCT, "direction": "low",
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
