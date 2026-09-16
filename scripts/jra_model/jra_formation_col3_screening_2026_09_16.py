# -*- coding: utf-8 -*-
"""Stage 0.5(3列目改良プラン、2026-09-16): 3列目を絞る余地があるかどうかを、ビームサーチや
Nested LOBO OOFに着手する前に安価な定量スクリーニングで判定する(目視チェックはしない)。

1. 現行3列目(チェックボックス無しのデフォルト状態)の経済性ベースラインを算出する:
   購入対象レース数・的中レース数・「3列目が無ければ不成立だった的中」の数・平均3列目頭数・
   総投資・総払戻・ROI。
   「3列目が無ければ不成立だった的中」の定義: 的中した3頭のうち、1列目にも2列目にも
   属さない(=3列目だけに存在する)馬が1頭以上含まれる的中。
2. 3列目候補母集団(score5≥30 かつ rank4>4、BOX4圏外)に対して、290atom単変量でlift
   (該当群のfp<=3率−母集団全体のfp<=3率)を算出する。
3. 事前に固定した3条件でGo/No-Go判定する(結果を見てから基準を決めない):
   (a) 実測liftが、290atom分の一括max統計量(ラベル置換2000回、多重比較=look-elsewhere
       効果を補正)による片側5%点(消し材料方向=lift最小値の分布)を下回る
   (b) 18開催日ブロック中、符号(liftの正負)が12ブロック以上で一致
   (c) 該当頭数が母集団の15%以上
   いずれか1本でも(a)(b)(c)を全て満たすatomがあればGo、無ければNo-Go。

出力: data/jra_pipeline/jra_search_formation_col3_2026_09_16_result.json
"""
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
LIB_DIR = PROJECT_ROOT / "scripts" / "jra_model"
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))

import jra_factor_registry as FR  # noqa: E402

BASE_PATH = DATA_DIR / "jra_formation_base_2026_09_16.json"
OUT_PATH = DATA_DIR / "jra_search_formation_col3_2026_09_16_result.json"

SCORE3_MIN = 30.0
COVERAGE_MIN = 0.15
SIGN_BLOCKS_MIN = 12
N_PERM = 2000
RNG_SEED = 2026

print("loading Stage 0-C base dataset...")
base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
races = base["races"]
n_races = len(races)
dates = base["meta"]["dates"]
n_dates = len(dates)
print(f"races={n_races} dates={n_dates}")

# ------------------------------------------------------------ 1) 経済性ベースライン(デフォルト
# 状態: 1列目=2列目=BOX4上位4頭、3列目=score5>=30プール、マーク一切無し)
print("computing baseline formation economics (no marks)...")
n_bet = n_hit = n_col3_necessary_hit = 0
stake_sum = ret_sum = 0
col3_sizes = []
for r in races:
    horses = r["horses"]
    box4 = {h["umaban"] for h in horses if h.get("rank4") is not None and h["rank4"] <= 4}
    col1 = col2 = box4
    col3 = {h["umaban"] for h in horses if h.get("score5") is not None and h["score5"] >= SCORE3_MIN}
    if not (col1 and col2 and col3):
        continue
    combos = set()
    for x in col1:
        for y in col2:
            if y == x:
                continue
            for z in col3:
                if z == x or z == y:
                    continue
                combos.add(tuple(sorted((x, y, z))))
    if not combos:
        continue
    n_bet += 1
    col3_sizes.append(len(col3))
    stake = len(combos) * 100
    ret = 0
    for combo in combos:
        key = "-".join(str(u) for u in combo)
        ret += r["pay_sanrenpuku"].get(key, 0)
    stake_sum += stake
    ret_sum += ret
    if ret > 0:
        n_hit += 1
        top3 = sorted((h for h in horses if h.get("finish_pos") is not None), key=lambda h: h["finish_pos"])[:3]
        top3_u = {h["umaban"] for h in top3}
        col12 = col1 | col2
        if any(u not in col12 for u in top3_u):
            n_col3_necessary_hit += 1

baseline_economics = {
    "n_races": n_races, "n_bet": n_bet, "n_hit": n_hit,
    "hit_rate_pct": round(100.0 * n_hit / n_bet, 2) if n_bet else None,
    "n_col3_necessary_hit": n_col3_necessary_hit,
    "col3_necessary_hit_share_of_hits_pct": round(100.0 * n_col3_necessary_hit / n_hit, 2) if n_hit else None,
    "avg_col3_size": round(float(np.mean(col3_sizes)), 2) if col3_sizes else None,
    "stake_sum": stake_sum, "return_sum": ret_sum,
    "roi_pct": round(100.0 * ret_sum / stake_sum, 2) if stake_sum else None,
}
print(json.dumps(baseline_economics, ensure_ascii=False, indent=2))

# ------------------------------------------------------------ 2) 3列目候補母集団(score5>=30 かつ
# rank4>4)を平坦化し、fp<=3ラベル・ブロック(開催日)・atomsを収集
print("building col3 candidate population (score5>=30 & rank4>4)...")
pop_atoms, pop_label, pop_block = [], [], []
for r in races:
    for h in r["horses"]:
        sc5 = h.get("score5")
        rk4 = h.get("rank4")
        if sc5 is None or sc5 < SCORE3_MIN:
            continue
        if rk4 is not None and rk4 <= 4:
            continue  # BOX4圏内は3列目候補としての新規性が無いため除外(=プラン定義通り)
        fp = h.get("finish_pos")
        pop_atoms.append(h["atoms"])
        pop_label.append(bool(fp is not None and fp <= 3))
        pop_block.append(r["kaisai_date"])

n_pop = len(pop_atoms)
label = np.array(pop_label, dtype=bool)
p0 = label.mean()
print(f"col3 population: {n_pop} horses, baseline fp<=3 rate = {p0*100:.2f}%")

print("building atom_mask matrix (290atom x population)...")
atom_keys = []
atom_mask_rows = []
for gid, group in FR.FACTOR_GROUPS.items():
    for opt in group["options"]:
        atom_keys.append((gid, opt["id"], group.get("label", gid), opt.get("label", opt["id"])))
        atom_mask_rows.append(np.array([FR.evaluate_option(h, group, opt) for h in pop_atoms], dtype=bool))
atom_mask = np.array(atom_mask_rows)  # shape (n_atoms, n_pop), bool
n_atoms = atom_mask.shape[0]
print(f"atoms: {n_atoms}")

atom_counts = atom_mask.sum(axis=1).astype(np.float64)  # (n_atoms,)
label_f = label.astype(np.float64)
real_hit_counts = atom_mask @ label_f  # (n_atoms,)
with np.errstate(invalid="ignore", divide="ignore"):
    real_rate = np.where(atom_counts > 0, real_hit_counts / atom_counts, np.nan)
real_lift = real_rate - p0  # (n_atoms,), 消し材料方向(絞りたい)は負

# ------------------------------------------------------------ 3-a) 置換検定(maxT法、290atom
# 分の一括多重比較補正)
print(f"running {N_PERM} label permutations (maxT null, look-elsewhere補正)...")
rng = np.random.default_rng(RNG_SEED)
perm_min_lifts = np.empty(N_PERM, dtype=np.float64)
valid_atom_mask_for_perm = atom_counts > 0
for i in range(N_PERM):
    perm_label = rng.permutation(label_f)
    hit_counts = atom_mask @ perm_label
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(atom_counts > 0, hit_counts / atom_counts, np.nan)
    lifts = rates - p0
    perm_min_lifts[i] = np.nanmin(lifts[valid_atom_mask_for_perm])
threshold_5pct = float(np.percentile(perm_min_lifts, 5))
print(f"maxT null 5th percentile (消し材料方向の有意閾値): {threshold_5pct*100:.3f}pt")

gate_a = real_lift <= threshold_5pct
gate_c = (atom_counts / n_pop) >= COVERAGE_MIN

# ------------------------------------------------------------ 3-b) ブロック(開催日)符号一致
print("checking block-level sign consistency (18 kaisai dates)...")
block_ids = sorted(set(pop_block))
pop_block_arr = np.array(pop_block)
overall_sign = np.sign(real_lift)  # +1/-1/0 per atom
sign_match_counts = np.zeros(n_atoms, dtype=int)
for d in block_ids:
    bmask = pop_block_arr == d
    if bmask.sum() == 0:
        continue
    b_label = label_f[bmask]
    b_p0 = b_label.mean()
    b_atom_mask = atom_mask[:, bmask]
    b_counts = b_atom_mask.sum(axis=1).astype(np.float64)
    b_hits = b_atom_mask @ b_label
    with np.errstate(invalid="ignore", divide="ignore"):
        b_rate = np.where(b_counts > 0, b_hits / b_counts, np.nan)
    b_lift = b_rate - b_p0
    b_sign = np.sign(b_lift)
    match = (b_counts > 0) & (b_sign == overall_sign) & (overall_sign != 0)
    sign_match_counts += match.astype(int)
gate_b = sign_match_counts >= SIGN_BLOCKS_MIN

# ------------------------------------------------------------ 総合判定
all_gates = gate_a & gate_b & gate_c
passing_idx = np.where(all_gates)[0]
print(f"gate_a(置換検定)通過: {int(gate_a.sum())}本  gate_b(符号一致>=12/18)通過: {int(gate_b.sum())}本  "
      f"gate_c(coverage>=15%)通過: {int(gate_c.sum())}本  全条件通過: {len(passing_idx)}本")

decision = "GO" if len(passing_idx) > 0 else "NOT_ATTEMPTED_SCREENING_GATE_FAILED"

top_candidates = []
order = np.argsort(real_lift)  # 最も負(消し材料として強い)順
for idx in order[:15]:
    gid, oid, glabel, olabel = atom_keys[idx]
    top_candidates.append({
        "group_id": gid, "option_id": oid, "group_label": glabel, "option_label": olabel,
        "n_marked": int(atom_counts[idx]), "coverage_pct": round(float(atom_counts[idx] / n_pop * 100), 2),
        "lift_pt": round(float(real_lift[idx] * 100), 3),
        "gate_a_pass": bool(gate_a[idx]), "gate_b_sign_match_blocks": int(sign_match_counts[idx]),
        "gate_b_pass": bool(gate_b[idx]), "gate_c_pass": bool(gate_c[idx]),
        "all_gates_pass": bool(all_gates[idx]),
    })

result = {
    "generated_at": None,
    "decision": decision,
    "population_definition": "score5>=30 かつ rank4>4(BOX4圏外)、314レース(障害戦除外済み)",
    "n_population": n_pop,
    "baseline_fp3_rate_pct": round(float(p0 * 100), 2),
    "baseline_economics": baseline_economics,
    "screening_params": {
        "coverage_min_pct": COVERAGE_MIN * 100, "sign_blocks_min": SIGN_BLOCKS_MIN, "n_blocks": len(block_ids),
        "n_permutations": N_PERM, "rng_seed": RNG_SEED,
        "maxT_null_5th_percentile_lift_pt": round(threshold_5pct * 100, 3),
    },
    "gate_pass_counts": {
        "gate_a_maxT": int(gate_a.sum()), "gate_b_sign_consistency": int(gate_b.sum()),
        "gate_c_coverage": int(gate_c.sum()), "all_three": int(len(passing_idx)),
    },
    "top15_candidates_by_lift": top_candidates,
}
import pandas as pd
result["generated_at"] = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()

OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"wrote {OUT_PATH}")
print(f"\n=== 判定: {decision} ===")
