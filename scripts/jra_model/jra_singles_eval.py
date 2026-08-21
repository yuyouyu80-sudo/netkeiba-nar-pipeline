# -*- coding: utf-8 -*-
"""単勝/複勝EVベッティングの評価基盤(2026-08-22新設、Step1)。

jra_axis_eval.pyと同型のEvaluator(fold=開催日×競馬場ブロック、pooled比推定量、ブロック単位
ブートストラップ)を踏襲するが、スコアが線形重み付き平均(box/axis)でなく条件付きロジット確率
(jra_market_model)に変わるため、lobo_oof/chronological_oofは「訓練foldでfit_conditional_logit→
EV閾値でpicks生成」という契約に置き換える(jra_eval.pyのlobo_oofをそのまま流用できない)。
box_nという概念も無く(EV条件を満たす馬だけを可変数賭ける)、picks はレースごとの可変長選択
(空配列=そのレースは見送り)。
"""
import numpy as np
import pandas as pd

import jra_market_model as MM
import jra_singles_backtest as SB
from jra_eval import blocks_of  # 既存のブロック定義をそのまま再利用

OBJ_BET_SINGLES = "単勝"  # 主指標(既存box/axisのOBJ_BETSに相当)


def market_favorite_picks(races: list) -> list:
    """市場ベンチマーク: 単勝1番人気1点(bias_ninki==1)。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        idx = np.where(ninki == 1)[0]
        picks.append(idx[:1])
    return picks


def cost_weighted_rate(stake: np.ndarray, ret: np.ndarray, bet: str = OBJ_BET_SINGLES,
                       idx: np.ndarray = None) -> float:
    col = SB.BET_TYPES_SINGLES.index(bet)
    s = stake[:, col] if idx is None else stake[idx, col]
    r = ret[:, col] if idx is None else ret[idx, col]
    tot = s.sum()
    return float(r.sum() / tot * 100) if tot else 0.0


class Evaluator:
    """単勝/複勝EVベッティングの評価器。1つのレース集合に対して1つ作る(box_nのような
    サイズ違いのバリエーションはない)。"""

    def __init__(self, races: list, actual: dict):
        self.races = races
        self.actual = actual
        self.settler = SB.SinglesSettler(races, actual)
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_stake, self.mkt_ret = self.settler.returns_for(market_favorite_picks(races))

    def evaluate(self, picks: list, idx: np.ndarray = None, bet: str = OBJ_BET_SINGLES) -> dict:
        st, rt = self.settler.returns_for(picks)
        model = cost_weighted_rate(st, rt, bet=bet, idx=idx)
        market = cost_weighted_rate(self.mkt_stake, self.mkt_ret, bet=bet, idx=idx)
        subset = picks if idx is None else [picks[i] for i in idx]
        n_bet_races = int(sum(1 for p in subset if p is not None and len(p) > 0))
        return {"model": model, "market": market, "excess": model - market,
                "stake": st, "return": rt, "n_bet_races": n_bet_races}

    def full_table(self, picks: list, idx: np.ndarray = None) -> pd.DataFrame:
        st, rt = self.settler.returns_for(picks)
        if idx is not None:
            st, rt = st[idx], rt[idx]
        n_races = st.shape[0]
        rows = []
        for i, bt in enumerate(SB.BET_TYPES_SINGLES):
            s, r = float(st[:, i].sum()), float(rt[:, i].sum())
            hits = int((rt[:, i] > 0).sum())
            rows.append({"bet_type": bt, "races": n_races, "hit_races": hits,
                         "hit_rate_pct": round(hits / n_races * 100, 1) if n_races else 0.0,
                         "stake": s, "return": r,
                         "return_rate_pct": round(r / s * 100, 1) if s else 0.0})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- CV
    def lobo_oof(self, fit_fn, feats: list, ev_threshold=MM.DEFAULT_EV_THRESHOLD,
                odds_cap=MM.DEFAULT_ODDS_CAP) -> dict:
        """Leave-one-block-outのout-of-fold評価。fit_fn(train_idx)は条件付きロジットの
        パラメータ配列beta=[beta0,beta1,beta2]を返す契約(500パターン探索のような離散候補
        プールが無いため、jra_eval.lobo_oofのpattern_idx契約ではなくbetaそのものを返す)。"""
        picks = [None] * len(self.races)
        chosen_params = {}
        for b in self.block_ids:
            test_idx = np.where(self.blocks == b)[0]
            train_idx = np.where(self.blocks != b)[0]
            beta = fit_fn(train_idx)
            chosen_params[b] = tuple(np.round(beta, 4))
            test_feats = [feats[i] for i in test_idx]
            test_picks = MM.ev_picks(beta, test_feats, ev_threshold=ev_threshold, odds_cap=odds_cap)
            for local_i, global_i in enumerate(test_idx):
                picks[global_i] = test_picks[local_i]
        result = {"picks": picks, "chosen_params": chosen_params, **self.evaluate(picks)}
        result["n_unique_patterns"] = len(set(chosen_params.values()))
        result["n_folds"] = len(chosen_params)
        return result

    def lobo_oof_nll(self, fit_fn, feats: list) -> dict:
        """パラメータ推定手続き自体をLOBO OOF化し、fold毎のheld-outレースのNLL
        (対数尤度の符号反転、レース単位)を記録する。block_bootstrap_diff_nllの入力に使う
        (ゲート1: モデルのNLLが市場のみのNLLを有意に下回るか)。"""
        nll_per_race = np.full(len(self.races), np.nan)
        chosen_params = {}
        for b in self.block_ids:
            test_idx = np.where(self.blocks == b)[0]
            train_idx = np.where(self.blocks != b)[0]
            beta = fit_fn(train_idx)
            chosen_params[b] = tuple(np.round(beta, 4))
            for i in test_idx:
                nll_per_race[i] = MM.race_nll(beta, feats, idx=[i])
        return {"nll_per_race": nll_per_race, "chosen_params": chosen_params,
                "mean_nll": float(np.nanmean(nll_per_race))}

    # --------------------------------------------------------------- CV(時系列)
    def chronological_oof(self, fit_fn, feats: list, min_train_blocks: int = 3,
                          ev_threshold=MM.DEFAULT_EV_THRESHOLD, odds_cap=MM.DEFAULT_ODDS_CAP) -> dict:
        """開催日昇順のexpanding-window walk-forward評価(jra_axis_eval.pyと同一設計)。
        夏競馬限定データという性質上、本モデルではこれをlobo_oofと並ぶ主判定の一部として扱う
        (ゲート4: 符号がlobo_oofと一致するか)。"""
        dates = sorted({b.split("_", 1)[0] for b in self.block_ids})
        picks = [None] * len(self.races)
        chosen_params = {}
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
            beta = fit_fn(train_idx)
            chosen_params[d] = tuple(np.round(beta, 4))
            test_feats = [feats[i] for i in test_idx]
            test_picks = MM.ev_picks(beta, test_feats, ev_threshold=ev_threshold, odds_cap=odds_cap)
            for local_i, global_i in enumerate(test_idx):
                picks[global_i] = test_picks[local_i]
                tested_race_idx.append(global_i)
        tested_race_idx = np.array(sorted(tested_race_idx), dtype=int)
        safe_picks = [p if p is not None else np.array([], dtype=int) for p in picks]
        result = {"picks": picks, "tested_race_idx": tested_race_idx, "chosen_params": chosen_params,
                 **self.evaluate(safe_picks, idx=tested_race_idx)}
        result["n_unique_patterns"] = len(set(chosen_params.values()))
        result["n_folds"] = len(chosen_params)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, bet: str = OBJ_BET_SINGLES, n: int = 2000,
                        seed: int = 11, block_subset=None) -> dict:
        st, rt = self.settler.returns_for(picks)
        col = SB.BET_TYPES_SINGLES.index(bet)
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(block_subset) if block_subset is not None else list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            s, r = st[idx, col].sum(), rt[idx, col].sum()
            out[k] = r / s * 100 if s else 0.0
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}

    def block_bootstrap_diff(self, picks_a: list, picks_b: list, bet: str = OBJ_BET_SINGLES,
                             n: int = 2000, seed: int = 11, block_subset=None) -> dict:
        st_a, rt_a = self.settler.returns_for(picks_a)
        st_b, rt_b = self.settler.returns_for(picks_b)
        col = SB.BET_TYPES_SINGLES.index(bet)
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(block_subset) if block_subset is not None else list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            sa, ra = st_a[idx, col].sum(), rt_a[idx, col].sum()
            sb, rb = st_b[idx, col].sum(), rt_b[idx, col].sum()
            rate_a = ra / sa * 100 if sa else 0.0
            rate_b = rb / sb * 100 if sb else 0.0
            out[k] = rate_a - rate_b
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}

    def block_bootstrap_diff_nll(self, nll_model: np.ndarray, nll_market_only: np.ndarray,
                                 n: int = 2000, seed: int = 11) -> dict:
        """(nll_market_only - nll_model)の平均をブロック単位でブートストラップする
        (正なら「モデルの方が市場のみより対数尤度が良い」)。NaNレース(結果データ欠損等)は
        nanmeanで自然に除外する。"""
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            out[k] = np.nanmean(nll_market_only[idx] - nll_model[idx])
        return {"mean": float(np.nanmean(out)), "lo": float(np.nanpercentile(out, 2.5)),
                "hi": float(np.nanpercentile(out, 97.5))}


def selection_optimism_thresholds(ev: Evaluator, feats: list, beta: np.ndarray, threshold_grid: list,
                                  bet: str = OBJ_BET_SINGLES, n_rep: int = 200, seed: int = 99) -> dict:
    """jra_eval.selection_optimismの「候補プールから選ぶことの真の価値」診断を、重み候補では
    なくEV閾値・オッズ上限のグリッド(threshold_grid: [(ev_threshold, odds_cap), ...])に
    読み替えたもの。betaは固定(Nested LOBO OOF等で既に決めたパラメータ)、変えるのは
    賭けるかどうかの閾値だけ。"""
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    col = SB.BET_TYPES_SINGLES.index(bet)
    all_picks = [MM.ev_picks(beta, feats, ev_threshold=t, odds_cap=c) for (t, c) in threshold_grid]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)

    def rate(st, rt, idx):
        s, r = st[idx, col].sum(), rt[idx, col].sum()
        return r / s * 100 if s else 0.0

    rng = np.random.default_rng(seed)
    sel, unseen, unseen_mean = [], [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(ids))
        a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
        b = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
        va = np.array([rate(all_st[j], all_rt[j], a) for j in range(len(threshold_grid))])
        vb = np.array([rate(all_st[j], all_rt[j], b) for j in range(len(threshold_grid))])
        best = int(np.argmax(va))
        sel.append(va[best])
        unseen.append(vb[best])
        unseen_mean.append(vb.mean())
    sel, unseen, unseen_mean = map(np.array, (sel, unseen, unseen_mean))
    return {
        "selected_side": float(sel.mean()), "unseen_side": float(unseen.mean()),
        "unseen_all_mean": float(unseen_mean.mean()), "optimism_pt": float(sel.mean() - unseen.mean()),
        "true_edge_pt": float((unseen - unseen_mean).mean()),
        "true_edge_sd": float((unseen - unseen_mean).std()),
        "win_rate": float((unseen > unseen_mean).mean()),
    }


def odds_matched_permutation_test(ev: Evaluator, races: list, actual_picks: list,
                                  bet: str = OBJ_BET_SINGLES, n_perm: int = 2000, seed: int = 77) -> dict:
    """actual_picksと同じ「1レースあたりの選択頭数」を保ったまま、各レース内でランダムに
    同数の馬を選ぶ操作をn_perm回繰り返し、実際の回収率が乱数分布の何パーセンタイルに位置するか
    を返す(オッズ分布はレース内randomなので厳密なオッズ帯マッチではないが、Opusの
    「オッズ分布マッチのランダム選抜」診断の簡易再現として同じ精神を踏襲する)。"""
    rng = np.random.default_rng(seed)
    n_horses = [len(r["df"]) for r in races]
    k_per_race = [len(p) if p is not None else 0 for p in actual_picks]
    col = SB.BET_TYPES_SINGLES.index(bet)
    real_st, real_rt = ev.settler.returns_for(actual_picks)
    real_stake, real_ret = real_st[:, col].sum(), real_rt[:, col].sum()
    real_rate = real_ret / real_stake * 100 if real_stake else 0.0

    sim_rates = np.empty(n_perm)
    for p in range(n_perm):
        rand_picks = []
        for i, k in enumerate(k_per_race):
            if k == 0:
                rand_picks.append(np.array([], dtype=int))
            else:
                rand_picks.append(rng.choice(n_horses[i], size=min(k, n_horses[i]), replace=False))
        st, rt = ev.settler.returns_for(rand_picks)
        s, r = st[:, col].sum(), rt[:, col].sum()
        sim_rates[p] = r / s * 100 if s else 0.0

    return {"real_rate": real_rate, "n_bet_races": int(sum(1 for k in k_per_race if k > 0)),
            "sim_mean": float(sim_rates.mean()), "sim_median": float(np.median(sim_rates)),
            "sim_p95": float(np.percentile(sim_rates, 95)),
            "p_value_ge_real": float((sim_rates >= real_rate).mean()), "n_perm": n_perm}
