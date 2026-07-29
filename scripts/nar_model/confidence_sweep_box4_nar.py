# -*- coding: utf-8 -*-
"""NAR 予想4頭BOX の回収率検証(全レース + 高確信度N レース/日)。

nar_dataset / nar_signals / nar_backtest / predict_box4_nar の統合パイプラインで動く。
旧版は scripts/predict_pattern29.py の動的import + 独自の払戻パースだったため、
探索スクリプトと priors が食い違って数値が最大8.6pt ずれていた。

確信度は2026-07-29のnar_confidence_calibrate.pyでの検証結果を反映し、「1位と2位の
スコア差をスコア全幅で割った値(gap_top2)」を使う。以前はBOXの賭け目位置(4位-5位差)を
使っていたが、LOBO較正で実際の複勝的中率とのSpearman相関を比較したところ、
4位-5位差はほぼ無相関(+0.024)だったのに対しgap_top2は+0.218の正相関があり、
自明な基準(常に平均を予測)よりOOF Brier scoreが改善した唯一の候補だった
(詳細はconfidence_calibration_nar.json参照)。この統計量を固定したまま
1日あたりの採用レース数 N だけを 5〜12 で振ることで、N を増やすほど的中レース数が
単調非減少になる(=スイープとして筋が通る)ことを担保している。

枠連は馬柱CSVの waku 列が NAR では 0% 充填のため賭け目が作れず、検証対象外。
旧版は「回収率0.0%」と出力していたが、これは「当たらない」ではなく「買っていない」なので
行自体を出さないようにした。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
DATA_DIR = LIB_DIR.parent.parent / "data" / "nar_pipeline"
sys.path.insert(0, str(LIB_DIR))
import nar_backtest as NB  # noqa: E402
import nar_dataset  # noqa: E402
import nar_signals as NS  # noqa: E402
import predict_box4_nar as P  # noqa: E402

BOX_N = P.BOX_N
N_RANGE = [5, 6, 7, 8, 9, 10, 11, 12]
OUT = DATA_DIR / "confidence_sweep_box4_nar.csv"

data = nar_dataset.load()
races, actual = data["races"], data["actual"]
settler = NB.BoxSettler(races, actual, box_n=BOX_N)

picks, conf_rows = [], []
for i, r in enumerate(races):
    scored = P.score_race(r["df"], P._class_ordinal(r["race_name"]))
    s = scored["_score"].to_numpy(dtype=float)
    filled = np.where(np.isnan(s), -1e18, s)
    order = np.argsort(-filled, kind="stable")
    picks.append(order[:BOX_N])

    ss = filled[order]
    spread = ss[0] - ss[-1]
    gap_pct = (ss[0] - ss[1]) / spread if (len(ss) > 1 and spread > 0) else 0.0
    conf_rows.append({"race_id": r["race_id"], "kaisai_date": r["kaisai_date"],
                      "idx": i, "gap_pct": gap_pct})

conf = pd.DataFrame(conf_rows)
stake, ret = settler.returns_for(picks)


def summarize(idx: np.ndarray, model: str, scope: str) -> pd.DataFrame:
    rows = []
    for j, bt in enumerate(NB.BET_TYPES):
        s = int(stake[idx, j].sum())
        v = int(ret[idx, j].sum())
        hits = int((ret[idx, j] > 0).sum())
        rows.append({
            "model": model, "scope": scope, "bet_type": bt,
            "races": len(idx), "hit_races": hits,
            "hit_rate_pct": round(hits / len(idx) * 100, 1) if len(idx) else 0.0,
            "total_stake": s, "total_return": v,
            "return_rate_pct": round(v / s * 100, 1) if s else 0.0,
        })
    return pd.DataFrame(rows)


MODEL = f"通常戦モデル(BOX4, {P.MODEL_LABEL})"
allidx = np.arange(len(races))
out = [summarize(allidx, MODEL, f"全{len(races)}レース")]

conf_sorted = conf.sort_values("gap_pct", ascending=False, kind="stable")
for n in N_RANGE:
    sel = conf_sorted.groupby("kaisai_date", group_keys=False).head(n)
    idx = np.sort(sel["idx"].to_numpy())
    out.append(summarize(idx, MODEL, f"高確信度{n}レース/日(計{len(idx)}レース)"))

result = pd.concat(out, ignore_index=True)
result.to_csv(OUT, index=False, encoding="utf-8-sig")

# --- 健全性チェック: Nを増やすと的中レース数は単調非減少でなければならない
piv = result.pivot_table(index="scope", columns="bet_type", values="hit_races", aggfunc="first")
order_scopes = [s for s in result["scope"].drop_duplicates() if s.startswith("高確信度")]
bad = []
for bt in NB.BET_TYPES:
    seq = [piv.loc[s, bt] for s in order_scopes]
    if any(b < a for a, b in zip(seq, seq[1:])):
        bad.append((bt, seq))
print(f"races={len(races)}  model={MODEL}")
print(f"monotonicity check: {'OK' if not bad else 'VIOLATION ' + str(bad)}")
print(result.to_string(index=False))
print(f"\nwrote {OUT.name}")
