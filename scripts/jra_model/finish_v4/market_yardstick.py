# -*- coding: utf-8 -*-
"""finish_v4 F0-3: 評価だけに使う市場の物差し(2026-10-10、プラン第3版)。

**ここで作る値は、予想・印・見送りには一切使わない**(ユーザーの決定 2026-10-10: 市場の予想は評価の物差しとしてだけ可)。
新しい印を作るモジュールはこのファイルを import しない(テストで確認する)。

1. 市場の組確率: 確定単勝オッズ o から u = γ·log(1/o) とし、2・3着段階の縮め係数つき PL(pl_ext)で全3頭組の確率を作る。
   γ・λ2・λ3 は評価する年の前年までのレースの実際の着順の尤度で推定する(払戻は使わない)。候補は確定オッズのある出走馬全員
   (競走中止の馬も含む。取消・除外はオッズが無いので除く)。
2. 市場から見た期待回収率 g: 全レース×全組の実現リターン(当たり組は払戻、外れは0、100円あたり)を、
   市場の組確率の対数のビン × 頭数の層 で平均したもの(前年までのレース)。当たり組だけで較正しない(選択の偏りと log の戻しの偏りを避ける)。
3. 人気をそろえた上積み: 買った組について「実際の払戻 − g による期待払戻」を、1レースの賭け金で割ったもの。
   陰性対照: 市場の組確率の上位 N 組を買うと、上積みは 0±SE になるはず。
4. 市場に対する増分情報: 組の条件付きロジット log P(組) ∝ a·log p_model + b·log p_mkt の a(a>0 ならモデルに市場に無い情報がある)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import pl_ext as P  # noqa: E402
import radar_v2_common as C  # noqa: E402

N_BINS = 30
FIELD_BANDS = [(0, 10), (11, 14), (15, 18)]


def field_band(n: int) -> int:
    return 0 if n <= 10 else (1 if n <= 14 else 2)


def load_odds(years) -> pd.DataFrame:
    frames = []
    for y in years:
        for p in sorted((C.RESULTS_DIR / str(y)).glob("*.csv")):
            frames.append(pd.read_csv(p, dtype=str, usecols=["race_id", "race_name", "surface", "race_date", "umaban", "horse_id",
                                                            "finish_pos", "odds_final"]))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    d = d[~C.is_jump(d["race_name"]) & d["surface"].isin(["芝", "ダ"])]
    d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
    d = d[d["odds"] > 0].copy()                       # 取消・除外(オッズ無し)は候補から除く
    d["pos"] = C.parse_finish(d["finish_pos"])
    d["umaban_num"] = pd.to_numeric(d["umaban"], errors="coerce")
    d["lo"] = np.log(1.0 / d["odds"])
    d["year"] = d["race_date"].str[:4].astype(int)
    d["block"] = d["race_date"].str.replace("-", "") + "_" + d["race_id"].str[4:6]
    return d


def fit_market(train: pd.DataFrame):
    b = P.RaceBatch(train, ["lo"])
    beta, lam = P.fit(b, kind="pl", fit_lam=True)
    return float(beta[0]), lam


def market_combos(g: pd.DataFrame, gamma: float, lam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1レース: (組の馬番 [K,3], 市場の組確率 [K], 馬番の並び)。"""
    um = g["umaban_num"].to_numpy(int)
    combos, p = P.triple_probs(gamma * g["lo"].to_numpy(float), "pl", lam)
    return um[combos], p, um


def combo_keys(trip: np.ndarray) -> np.ndarray:
    """馬番の3頭組 [K,3] → 並びによらない整数キー(小さい順に a·10000 + b·100 + c)。"""
    t = np.sort(np.asarray(trip, int), axis=1)
    return t[:, 0] * 10000 + t[:, 1] * 100 + t[:, 2]


def win_keys(pay_list) -> dict:
    """[(frozenset(馬番), 払戻)] → {キー: 払戻}。"""
    out = {}
    for s, v in pay_list:
        if len(s) == 3:
            a, b, c = sorted(s)
            out[a * 10000 + b * 100 + c] = v
    return out


class ReturnTable:
    """g: 市場の組確率の対数のビン × 頭数の層 → 100円あたりの平均の実現リターン。"""

    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.s = np.zeros((len(FIELD_BANDS), len(edges) + 1))
        self.n = np.zeros_like(self.s)

    def bin(self, lp):
        return np.searchsorted(self.edges, lp)

    def add(self, lp: np.ndarray, ret: np.ndarray, field: int):
        fb = field_band(field)
        b = self.bin(lp)
        np.add.at(self.s[fb], b, ret)
        np.add.at(self.n[fb], b, 1)

    def expect(self, lp: np.ndarray, field: int) -> np.ndarray:
        fb = field_band(field)
        b = self.bin(lp)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.s[fb] / self.n[fb]
        allm = self.s.sum(0) / np.maximum(self.n.sum(0), 1)
        return np.where(self.n[fb][b] >= 200, m[b], allm[b])


def build_return_table(odds: pd.DataFrame, pay: dict, gamma: float, lam) -> ReturnTable:
    # ビンの境界: 組確率の対数の分位点(学習レースから)
    sample = []
    races = list(odds.groupby("race_id"))
    for rid, g in races[::20]:
        _, p, _ = market_combos(g, gamma, lam)
        sample.append(np.log(p))
    edges = np.quantile(np.concatenate(sample), np.linspace(0, 1, N_BINS + 1)[1:-1])
    T = ReturnTable(edges)
    for rid, g in races:
        if rid not in pay or len(g) < 3:
            continue
        c, p, um = market_combos(g, gamma, lam)
        w = win_keys(pay[rid])
        k = combo_keys(c)
        ret = np.array([w.get(int(x), 0) for x in k], float) if len(w) > 1 else np.where(k == next(iter(w), -1), next(iter(w.values()), 0), 0).astype(float)
        T.add(np.log(p), ret, len(um))
    return T
