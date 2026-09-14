# -*- coding: utf-8 -*-
"""Opus5レビュー指摘の「真に未見の214レースでの現行モデル成績」を独立に検算する。
現行モデル(winner_v3/box4/box3.json)のfitted_on(search_dates+holdout_dates=6日・105レース)
を319レース母集団から除外した214レースだけで、市場比較を行う。"""
import json
import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
DATA_DIR = LIB_DIR.parent.parent / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}

out = []
data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
out.append(f"母集団: {len(races)}レース  日付: {data['dates']}")

FITTED_DATES = {"20260711", "20260712", "20260718", "20260719", "20260725", "20260726"}
unseen_races = [r for r in races if r["kaisai_date"] not in FITTED_DATES]
out.append(f"fitted_on日付(6日): {sorted(FITTED_DATES)}")
out.append(f"真に未見のレース数: {len(unseen_races)} (319-{len(races)-len(unseen_races)}=期待214)")

dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)

for BOX_N in (5, 4, 3):
    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    NAMES = list(winner["weights"].keys())
    W = np.array([winner["weights"][n] for n in NAMES])

    # 全319レースでの評価(本文既報値との整合確認)
    ev_all = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_all = ev_all.evaluate(JE.market_picks(races, BOX_N))
    mats_all = JS.signal_matrices(races, priors_fresh, NAMES, JS.CLASS_ORDINAL)
    cur_all = ev_all.evaluate(JE.score_picks(mats_all, W, BOX_N))
    out.append(f"\nbox_n={BOX_N} (pattern{winner['pattern_id']})")
    out.append(f"  [全319R] 市場={mkt_all['model']:.2f}%  現行={cur_all['model']:.2f}%  "
               f"市場差={cur_all['excess']:+.2f}pt   (既報値と一致するはず)")

    # 真に未見の214レースだけでの評価
    ev_u = JE.Evaluator(unseen_races, actual, box_n=BOX_N)
    mkt_u = ev_u.evaluate(JE.market_picks(unseen_races, BOX_N))
    mats_u = JS.signal_matrices(unseen_races, priors_fresh, NAMES, JS.CLASS_ORDINAL)
    cur_u = ev_u.evaluate(JE.score_picks(mats_u, W, BOX_N))
    boot = ev_u.block_bootstrap(JE.score_picks(mats_u, W, BOX_N), n=2000, seed=31)
    out.append(f"  [真に未見214R] 市場={mkt_u['model']:.2f}%  現行={cur_u['model']:.2f}%  "
               f"市場差={cur_u['excess']:+.2f}pt  95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]")

Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\verify_truly_unseen_out.txt"
    ).write_text("\n".join(out), encoding="utf-8")
print("done")
