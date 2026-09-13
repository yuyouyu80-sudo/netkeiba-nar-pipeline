import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import jra_dataset_wide as JDW
import jra_model_scoring as MS
import jra_signals as JS
import jra_eval as JE

data = JDW.load(rebuild=False)
races = data["races"]

by_type = {}
for r in races:
    by_type.setdefault(r["race_type"], []).append(r["df"])
priors = MS.compute_priors_for_population(by_type)
weights = {rt: MS.load_weights(rt) for rt in ("normal","shinba","mishoubi")}

scores = {}
for r in races:
    s = MS.score_race(r["df"], r["race_name"], r["race_type"], priors[r["race_type"]], weights[r["race_type"]])
    scores[r["race_id"]] = s

print("scored races:", len(scores))

# --- cross-check normal (BOX5/winner_v3.json) against jra_eval.score_picks path ---
normal_races = [r for r in races if r["race_type"] == "normal"]
mats = JS.signal_matrices(normal_races, priors["normal"], JS.LEGACY_SIGNALS, JS.CLASS_ORDINAL)
w_vec = np.array([weights["normal"][n] for n in JS.LEGACY_SIGNALS])
max_abs_diff = 0.0
for r, m in zip(normal_races, mats):
    num = m["S"] @ w_vec
    den = m["A"] @ w_vec
    ref_score = np.where(den > 0, num/den, np.nan)
    mine = scores[r["race_id"]].to_numpy()
    d = np.nanmax(np.abs(np.nan_to_num(ref_score,nan=-999) - np.nan_to_num(mine,nan=-999)))
    max_abs_diff = max(max_abs_diff, d)
print("normal max abs diff vs signal_matrices path:", max_abs_diff)
