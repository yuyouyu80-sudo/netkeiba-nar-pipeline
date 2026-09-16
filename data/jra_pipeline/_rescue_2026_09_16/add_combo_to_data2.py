# -*- coding: utf-8 -*-
"""data2_races.jsonへ、BOX5の組み合わせ実験(pattern#1425 add #1845、2026-09-11)の
score/rankを追加する。健全性診断(held-out・最大配当ブロック除外)を通過した唯一の
「単体探索より頑健」な候補だが、統計的有意性は無い(95%CIが0を跨ぐ)ため、
既存のBOX5/4/3列を差し替えず追加の4列目として扱う。"""
import sys, json
import numpy as np

sys.path.insert(0, r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
import jra_dataset
import jra_signals as JS

SP = r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
SEED, N_PATTERNS = 2911, 2000
PAT_I, PAT_J, OP = 1425, 1845, "add"

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
w_i, w_j = W_POOL[:, PAT_I], W_POOL[:, PAT_J]

combo_by_race_umaban = {}
for race, m in zip(races, mats_all):
    num_i, den_i = m["S"] @ w_i, m["A"] @ w_i
    num_j, den_j = m["S"] @ w_j, m["A"] @ w_j
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.where(den_i > 0, num_i / den_i, 0.0)
        sj = np.where(den_j > 0, num_j / den_j, 0.0)
    combo = si + sj  # add合成(健全性診断で掛け算とほぼ同着、足し算をわずかに採用)
    order = np.argsort(-combo, kind="stable")
    rank = np.empty(len(combo), dtype=int)
    rank[order] = np.arange(1, len(combo) + 1)
    df = race["df"]
    for pos, (_, row) in enumerate(df.iterrows()):
        try:
            umaban = int(row["umaban"])
        except (ValueError, TypeError):
            continue
        combo_by_race_umaban[(race["race_id"], umaban)] = {
            "score_combo": float(combo[pos]), "rank_combo": int(rank[pos]),
        }

races_out = json.loads(open(SP + r"\data2_races.json", encoding="utf-8").read())
n_matched = 0
for r in races_out:
    for h in r["horses"]:
        key = (r["race_id"], h["umaban"])
        info = combo_by_race_umaban.get(key)
        if info:
            h["score_combo"] = info["score_combo"]
            h["rank_combo"] = info["rank_combo"]
            n_matched += 1
        else:
            h["score_combo"] = None
            h["rank_combo"] = None

open(SP + r"\data2_races.json", "w", encoding="utf-8").write(
    json.dumps(races_out, ensure_ascii=False, default=str))
print(f"matched {n_matched} horses, wrote data2_races.json")

# meta.jsonにも組み合わせの診断結果を追記
meta = json.loads(open(SP + r"\data2_meta.json", encoding="utf-8").read())
diag = json.loads(open(SP + r"\box5_combo_diagnosis.json", encoding="utf-8").read())
meta["box5_combo_experiment"] = {
    "pattern_i": PAT_I, "pattern_j": PAT_J, "op": OP,
    "note": "2026-09-11、BOX5の2000パターン(45信号Dirichletランダム重み)の中から単体成績上位80件の"
           "全ペア(3160通り)×2演算(×/+)=6320通りを評価。最良は#1425+#1845(足し算)。単体探索の"
           "候補と異なりheld-out(36ブロック)でも一貫してプラス、最大配当ブロック除外後もプラスを維持する"
           "点で相対的に頑健だが、95%信頼区間は0を跨ぎ統計的有意性は無い。6320通りという大規模な"
           "多重比較の結果であり、偶然の産物である可能性は排除できない参考値。",
    "diagnosis": diag["combo_diagnosis"][0],
}
open(SP + r"\data2_meta.json", "w", encoding="utf-8").write(
    json.dumps(meta, ensure_ascii=False, indent=2, default=str))
print("wrote data2_meta.json (+box5_combo_experiment)")
