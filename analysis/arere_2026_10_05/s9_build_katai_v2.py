# -*- coding: utf-8 -*-
"""荒れない指数 改良版(v2)のモデルを作って保存する(2026-10-05、ユーザー決定で本番切替)。

s7_improve.py の検証で選ばれた設定:
  - 単勝オッズからの3連複推定を割引Harville(λ2=0.9, λ3=0.8)に補正
  - 「払戻1,000円以下になりそうな組の確率」(l_le1000)と組確率のエントロピー(t_ent)を特徴に追加
  - 直近重視: 重み 0.5^((2022−年)/8)
学習期間は従来版と同じ 2011〜2022年(検証結果と同じ条件)。荒れ指数は従来版のまま。
出力: out_katai/s1_models_v2.pkl  {"models": {(区分, "条件+オッズ"): pipeline}, "lam": (0.9, 0.8), "num_cols": [...], "halflife": 8}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_stats_model as S  # noqa: E402
import s7_improve as I  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
LAM = (0.9, 0.8)
HALF = 8
REF_YEAR = 2022


def main():
    races = pd.read_csv(S.RACES, dtype={"race_id": str})
    races = races[races["sanrenpuku"].notna() & races["fav_odds"].notna()].copy()
    base = S.add_bins(S.prep_numeric(races)).reset_index(drop=True)
    feats = pd.read_pickle(I.OUT / f"feat_{LAM[0]}_{LAM[1]}.pkl")
    X = I.make_X(base, feats)
    y = (X["sanrenpuku"] <= 1000).astype(int).to_numpy()
    num_cols = list(I.NUM_NEW)
    saved = S.NUM_ODDS
    S.NUM_ODDS = num_cols  # build_model は呼び出し時のモジュール変数を参照する
    models = {}
    for cat in ("一般", "新馬", "未勝利"):
        tr = ((X["category"] == cat) & (X["year"] <= REF_YEAR) & X["m_ge10000"].notna()).to_numpy()
        m, _ = S.build_model(cat, True, True)
        w = 0.5 ** ((REF_YEAR - X.loc[tr, "year"].to_numpy()) / HALF)
        m.fit(X[tr], y[tr], lr__sample_weight=w)
        models[(cat, "条件+オッズ")] = m
        print(cat, "学習", int(tr.sum()), "レース", flush=True)
    S.NUM_ODDS = saved
    with open(HERE / "out_katai" / "s1_models_v2.pkl", "wb") as f:
        pickle.dump({"models": models, "lam": LAM, "num_cols": num_cols, "halflife": HALF}, f)
    print("saved out_katai/s1_models_v2.pkl")


if __name__ == "__main__":
    main()
