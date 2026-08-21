# -*- coding: utf-8 -*-
"""JRA 1頭軸流しモデルの評価基盤。scripts/jra_model/jra_eval.pyの軸流し版。

jra_eval.pyとの設計共通点(そのまま踏襲):
  * fold は「開催日 × 競馬場」ブロック。
  * fold平均ではなく pooled(払戻と賭金を積んでから割る)。
  * ブートストラップはブロック単位・比推定量。

`score_picks()`/`market_picks()`/`blocks_of()`は軸・BOXどちらでも「レースごとに上位box_n頭の
行インデックスを、1位から順に並べた配列」を返すという契約が同じ(argsortの結果をそのまま
使っているため既に軸=[0]番目の順序になっている)。よってjra_eval.pyから変更なしで再利用する。

軸流し独自の設計判断:
  * 対象券種は馬連・ワイド・3連複・馬単(軸流し/マルチ)・3連単(軸流し/マルチ)の7区分
    (jra_axis_backtest.BET_TYPES_AXIS)。単勝・複勝・枠連は軸+相手という構造を持たないため対象外。
  * 目的関数(重み探索のfit対象)は「ワイドのコスト加重回収率」のみを使う。3連単・馬単は
    1点あたり配当の分散が極端に大きく(1レースの的中だけでin-sample順位が入れ替わりうる)、
    box買いの目的関数が複勝+ワイドという低分散券種を選んだのと同じ理由でワイド単独を採用する。
    最終レポートでは7区分すべての的中率・回収率をfull_table()で個別表示する。
"""
import numpy as np
import pandas as pd

import jra_axis_backtest as AB
from jra_eval import blocks_of, market_picks, score_picks  # noqa: F401  (軸流しでも変更なし再利用)

OBJ_BETS_AXIS = ["ワイド"]
UNIT = AB.UNIT


def cost_weighted_rate(stake: np.ndarray, ret: np.ndarray, bets=OBJ_BETS_AXIS,
                       idx: np.ndarray = None) -> float:
    """指定券種のコスト加重回収率(%)。Σ払戻 / Σ賭金。"""
    cols = [AB.BET_TYPES_AXIS.index(b) for b in bets]
    s = stake[:, cols] if idx is None else stake[np.ix_(idx, cols)]
    r = ret[:, cols] if idx is None else ret[np.ix_(idx, cols)]
    tot = s.sum()
    return float(r.sum() / tot * 100) if tot else 0.0


class Evaluator:
    """1つのレース集合・1つのbox_n(軸+相手の合計頭数)に対する評価器。"""

    def __init__(self, races: list, actual: dict, box_n: int = 5):
        self.races = races
        self.box_n = box_n
        self.settler = AB.AxisSettler(races, actual, box_n=box_n)
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_stake, self.mkt_ret = self.settler.returns_for(market_picks(races, box_n))

    def evaluate(self, picks: list, idx: np.ndarray = None, multipliers: np.ndarray = None) -> dict:
        """picks に対する目的関数値と、市場ベンチマーク(軸=1番人気、相手=2〜box_n番人気)との差。
        multipliers: レースごとのステーク乗数(2026-08-21、Front3用。Noneなら従来と同一)。"""
        st, rt = self.settler.returns_for(picks)
        mkt_st, mkt_rt = self.mkt_stake, self.mkt_ret
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st, rt = st * m, rt * m
            mkt_st, mkt_rt = mkt_st * m, mkt_rt * m
        model = cost_weighted_rate(st, rt, idx=idx)
        market = cost_weighted_rate(mkt_st, mkt_rt, idx=idx)
        return {"model": model, "market": market, "excess": model - market,
                "stake": st, "return": rt}

    def full_table(self, picks: list, idx: np.ndarray = None,
                   multipliers: np.ndarray = None) -> pd.DataFrame:
        """全券種(7区分)の的中率・回収率(レポート用)。idx/multipliersの意味はjra_eval.py
        と同じ(2026-08-21、Front3用に追加)。"""
        st, rt = self.settler.returns_for(picks)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st, rt = st * m, rt * m
        if idx is not None:
            st, rt = st[idx], rt[idx]
        n_races = st.shape[0]
        rows = []
        for i, bt in enumerate(AB.BET_TYPES_AXIS):
            s, r = float(st[:, i].sum()), float(rt[:, i].sum())
            hits = int((rt[:, i] > 0).sum())
            rows.append({"bet_type": bt, "races": n_races, "hit_races": hits,
                         "hit_rate_pct": round(hits / n_races * 100, 1) if n_races else 0.0,
                         "stake": s, "return": r,
                         "return_rate_pct": round(r / s * 100, 1) if s else 0.0})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- CV
    def lobo_oof(self, fit_fn, mats_all: list) -> dict:
        """Leave-one-block-out の out-of-fold 評価。戻り値に fold ごとの選択パターンindex
        (chosen_pattern_idx)を含める — LOBO退化チェック(全foldで同一パターンしか
        選ばれていないか)に使う。fit_fn(train_idx) は (w, pattern_idx) のタプルを返すこと。
        """
        picks = [None] * len(self.races)
        chosen_pattern_idx = {}
        for b in self.block_ids:
            test_idx = np.where(self.blocks == b)[0]
            train_idx = np.where(self.blocks != b)[0]
            w, pat_idx = fit_fn(train_idx)
            chosen_pattern_idx[b] = pat_idx
            for i in test_idx:
                m = mats_all[i]
                num, den = m["S"] @ w, m["A"] @ w
                score = np.where(den > 0, num / den, -1e18)
                picks[i] = np.argsort(-score, kind="stable")[:self.box_n]
        result = {"picks": picks, "chosen_pattern_idx": chosen_pattern_idx, **self.evaluate(picks)}
        result["n_unique_patterns"] = len(set(chosen_pattern_idx.values()))
        result["n_folds"] = len(chosen_pattern_idx)
        return result

    # --------------------------------------------------------------- CV(時系列)
    def chronological_oof(self, fit_fn, mats_all: list, min_train_blocks: int = 3) -> dict:
        """開催日昇順のexpanding-window walk-forward評価(2026-08-21新設、jra_eval.pyと同一設計)。
        **採否の主判定には使わない**(lobo_oof+選択バイアス診断が主判定)。時間方向のリークが
        無いかを確認する追加のロバスト性チェック専用。"""
        dates = sorted({b.split("_", 1)[0] for b in self.block_ids})
        picks = [None] * len(self.races)
        chosen_pattern_idx = {}
        tested_race_idx = []
        for i, d in enumerate(dates):
            train_dates = set(dates[:i])
            if len(train_dates) < min_train_blocks:
                continue
            test_blocks = [b for b in self.block_ids if b.split("_", 1)[0] == d]
            train_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in train_dates]
            test_idx = np.where(np.isin(self.blocks, test_blocks))[0]
            train_idx = np.where(np.isin(self.blocks, train_blocks))[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            w, pat_idx = fit_fn(train_idx)
            chosen_pattern_idx[d] = pat_idx
            for j in test_idx:
                m = mats_all[j]
                num, den = m["S"] @ w, m["A"] @ w
                score = np.where(den > 0, num / den, -1e18)
                picks[j] = np.argsort(-score, kind="stable")[:self.box_n]
                tested_race_idx.append(j)
        tested_race_idx = np.array(sorted(tested_race_idx), dtype=int)
        safe_picks = [p if p is not None else np.arange(self.box_n) for p in picks]
        result = {"picks": picks, "tested_race_idx": tested_race_idx,
                 "chosen_pattern_idx": chosen_pattern_idx,
                 **self.evaluate(safe_picks, idx=tested_race_idx)}
        result["n_unique_patterns"] = len(set(chosen_pattern_idx.values()))
        result["n_folds"] = len(chosen_pattern_idx)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, bets=OBJ_BETS_AXIS, n: int = 2000,
                        seed: int = 11, block_subset=None,
                        multipliers: np.ndarray = None) -> dict:
        """ブロック単位の比推定量ブートストラップ。block_subsetを指定すると、その
        ブロック集合だけを対象にリサンプルする(例: 重みのfit母集団と重複しないブロックだけで
        「真に未見のデータ」上の信頼区間を出す用途)。multipliersはレースごとのステーク乗数
        (2026-08-21、Front3用)。"""
        st, rt = self.settler.returns_for(picks)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st, rt = st * m, rt * m
        cols = [AB.BET_TYPES_AXIS.index(b) for b in bets]
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(block_subset) if block_subset is not None else list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            s = st[np.ix_(idx, cols)].sum()
            r = rt[np.ix_(idx, cols)].sum()
            out[k] = r / s * 100 if s else 0.0
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}

    def block_bootstrap_diff(self, picks_a: list, picks_b: list, bets=OBJ_BETS_AXIS,
                             n: int = 2000, seed: int = 11, block_subset=None,
                             multipliers: np.ndarray = None) -> dict:
        """2つのpicks(例: 探索モデル vs 現行box重み転用)の差(model_a - model_b)を
        ブロック単位でペアでブートストラップする。同一レースを同一リサンプルに使うことで
        天候・その日の配当水準といった共通ノイズを相殺する。block_subset/multipliersの意味は
        block_bootstrapと同じ。"""
        st_a, rt_a = self.settler.returns_for(picks_a)
        st_b, rt_b = self.settler.returns_for(picks_b)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st_a, rt_a = st_a * m, rt_a * m
            st_b, rt_b = st_b * m, rt_b * m
        cols = [AB.BET_TYPES_AXIS.index(b) for b in bets]
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(block_subset) if block_subset is not None else list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            sa, ra = st_a[np.ix_(idx, cols)].sum(), rt_a[np.ix_(idx, cols)].sum()
            sb, rb = st_b[np.ix_(idx, cols)].sum(), rt_b[np.ix_(idx, cols)].sum()
            rate_a = ra / sa * 100 if sa else 0.0
            rate_b = rb / sb * 100 if sb else 0.0
            out[k] = rate_a - rate_b
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}


def selection_optimism(ev: Evaluator, mats: list, W: np.ndarray, n_rep: int = 200,
                       seed: int = 99) -> dict:
    """「重みを選ぶ」という行為から得られる真の利得を測る(ブロック半分割×n_rep)。"""
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    all_picks = [score_picks(mats, W[:, j], ev.box_n) for j in range(W.shape[1])]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    rng = np.random.default_rng(seed)
    sel, unseen, unseen_mean = [], [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(ids))
        a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
        b = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
        va = np.array([cost_weighted_rate(all_st[j], all_rt[j], idx=a) for j in range(W.shape[1])])
        vb = np.array([cost_weighted_rate(all_st[j], all_rt[j], idx=b) for j in range(W.shape[1])])
        best = int(np.argmax(va))
        sel.append(va[best])
        unseen.append(vb[best])
        unseen_mean.append(vb.mean())
    sel, unseen, unseen_mean = map(np.array, (sel, unseen, unseen_mean))
    return {
        "selected_side": float(sel.mean()),
        "unseen_side": float(unseen.mean()),
        "unseen_all_mean": float(unseen_mean.mean()),
        "optimism_pt": float(sel.mean() - unseen.mean()),
        "true_edge_pt": float((unseen - unseen_mean).mean()),
        "true_edge_sd": float((unseen - unseen_mean).std()),
        "win_rate": float((unseen > unseen_mean).mean()),
    }
