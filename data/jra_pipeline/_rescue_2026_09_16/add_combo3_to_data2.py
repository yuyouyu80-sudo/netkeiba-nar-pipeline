# -*- coding: utf-8 -*-
"""add_combo_to_data2.pyの拡張版。単一の最良組み合わせ(#1425+#1845)だけでなく、
box5_combo_diagnosis.jsonのcombo_diagnosis(上位3件: combo_top1_add/combo_top2_mult/
combo_top3_mult)を全て計算し、data2_races.jsonへscore_combo{1,2,3}/rank_combo{1,2,3}
として追加する(ユーザー指示: 「上からE:1、E:2〜と順番に並べてください」= 3件とも表示)。
旧来の単一score_combo/rank_comboフィールドは削除して置き換える。"""
import sys, json
import numpy as np

sys.path.insert(0, r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
import jra_dataset
import jra_signals as JS

SP = r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
SEED, N_PATTERNS = 2911, 2000

data = jra_dataset.load(rebuild=False)
races = data["races"]
NAMES = JS.ALL_SIGNALS_V5 + JS.CANDIDATE_SIGNALS_V4
dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
mats_all = JS.signal_matrices(races, priors_fresh, NAMES, JS.CLASS_ORDINAL)


def wvec(d):
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


rng = np.random.default_rng(SEED)
W_POOL = np.column_stack([
    wvec(dict(zip(NAMES, rng.dirichlet([1.0] * len(NAMES))))) for _ in range(N_PATTERNS)
])

diag = json.loads(open(SP + r"\box5_combo_diagnosis.json", encoding="utf-8").read())
combos = diag["combo_diagnosis"]  # 3件、既にmodel_pct降順(top1/2/3)
print("combos:", [(c["pattern_i"], c["op"], c["pattern_j"], c["label"]) for c in combos])

# combo_idx(1-indexed) -> race_id/umaban -> {score, rank}
combo_by_idx = {}
for idx, c in enumerate(combos, start=1):
    i, j, op = c["pattern_i"], c["pattern_j"], c["op"]
    w_i, w_j = W_POOL[:, i], W_POOL[:, j]
    per_horse = {}
    for race, m in zip(races, mats_all):
        num_i, den_i = m["S"] @ w_i, m["A"] @ w_i
        num_j, den_j = m["S"] @ w_j, m["A"] @ w_j
        with np.errstate(divide="ignore", invalid="ignore"):
            si = np.where(den_i > 0, num_i / den_i, 0.0)
            sj = np.where(den_j > 0, num_j / den_j, 0.0)
        if op == "mult":
            a, b = np.clip(si, 0, None), np.clip(sj, 0, None)
            combo = a * b
        else:
            combo = si + sj
        order = np.argsort(-combo, kind="stable")
        rank = np.empty(len(combo), dtype=int)
        rank[order] = np.arange(1, len(combo) + 1)
        df = race["df"]
        for pos, (_, row) in enumerate(df.iterrows()):
            try:
                umaban = int(row["umaban"])
            except (ValueError, TypeError):
                continue
            per_horse[(race["race_id"], umaban)] = {
                "score": float(combo[pos]), "rank": int(rank[pos]),
            }
    combo_by_idx[idx] = per_horse
    print(f"  computed combo #{idx}: pattern#{i} {op} pattern#{j} ({len(per_horse)} horse entries)")

races_out = json.loads(open(SP + r"\data2_races.json", encoding="utf-8").read())
n_matched = [0, 0, 0]
for r in races_out:
    for h in r["horses"]:
        # 旧フィールドは削除して置き換える
        h.pop("score_combo", None)
        h.pop("rank_combo", None)
        key = (r["race_id"], h["umaban"])
        for idx in (1, 2, 3):
            info = combo_by_idx[idx].get(key)
            if info:
                h[f"score_combo{idx}"] = info["score"]
                h[f"rank_combo{idx}"] = info["rank"]
                n_matched[idx - 1] += 1
            else:
                h[f"score_combo{idx}"] = None
                h[f"rank_combo{idx}"] = None

open(SP + r"\data2_races.json", "w", encoding="utf-8").write(
    json.dumps(races_out, ensure_ascii=False, default=str))
print(f"matched horses per combo: {n_matched}, wrote data2_races.json")

# meta.jsonのbox5_combo_experimentを3件のリストへ差し替え
meta = json.loads(open(SP + r"\data2_meta.json", encoding="utf-8").read())
meta["box5_combo_experiment"] = {
    "note": "2026-09-11、BOX5の2000パターン(45信号Dirichletランダム重み)の中から単体成績上位80件の"
           "全ペア(3160通り)×2演算(×/+)=6320通りを評価。上位3件(E1〜E3、E1が最良)を全て表示。"
           "単体探索の候補と異なりheld-out(36ブロック)でも一貫してプラス、最大配当ブロック除外後も"
           "プラスを維持する点で相対的に頑健だが、95%信頼区間はいずれも0を跨ぎ統計的有意性は無い。"
           "6320通りという大規模な多重比較の結果であり、偶然の産物である可能性は排除できない参考値。",
    "combos": [
        {"idx": idx, "pattern_i": c["pattern_i"], "pattern_j": c["pattern_j"], "op": c["op"], "diagnosis": c}
        for idx, c in enumerate(combos, start=1)
    ],
}
open(SP + r"\data2_meta.json", "w", encoding="utf-8").write(
    json.dumps(meta, ensure_ascii=False, indent=2, default=str))
print("wrote data2_meta.json (box5_combo_experiment -> 3-combo list)")
