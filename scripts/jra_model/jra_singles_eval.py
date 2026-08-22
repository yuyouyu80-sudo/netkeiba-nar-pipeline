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


def market_favorite_picks(races: list, ninki_col: str = "bias_ninki") -> list:
    """市場ベンチマーク: 単勝1番人気1点(ninki_col==1)。ninki_col: 人気列名(既定"bias_ninki"=
    馬柱由来。archiveベース評価では"popularity"を渡す、2026-08-23追加)。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"][ninki_col], errors="coerce").to_numpy(dtype=float)
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

    def __init__(self, races: list, actual: dict, ninki_col: str = "bias_ninki"):
        self.races = races
        self.actual = actual
        self.settler = SB.SinglesSettler(races, actual)
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_stake, self.mkt_ret = self.settler.returns_for(
            market_favorite_picks(races, ninki_col=ninki_col))

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

    # ----------------------------------------------------- CV(2026-08-23追加、選択肢B Phase4)
    def grouped_kfold_oof(self, fit_fn, feats: list, k: int = 10, seed: int = 7,
                          pq_threshold=None, odds_cap: float = float("inf")) -> dict:
        """開催日をk個のグループにランダム分割するgrouped K-fold(補助的なCV)。
        2026-08-22のOpus調査: 拡張母集団(1000+レース、100+ブロック)では連続パラメータ2個の
        MLEに対してLOBO(leave-one-block-out)は過剰(最適化バイアスがO(p/n)で無視できる水準)
        なので、より軽量なgrouped K-foldで代替する。fit_fn(train_idx)->beta=[b0,b1,b2]契約は
        lobo_oofと同じ(pq_picksを使う点のみ既存lobo_oofと異なる)。"""
        import jra_market_model as _MM
        pq_threshold = _MM.DEFAULT_PQ_THRESHOLD if pq_threshold is None else pq_threshold
        dates = sorted({b.split("_", 1)[0] for b in self.block_ids})
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(dates)
        groups = np.array_split(shuffled, k)
        picks = [None] * len(self.races)
        chosen_params = {}
        for gi, test_dates in enumerate(groups):
            test_dates = set(test_dates.tolist())
            test_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in test_dates]
            train_blocks = [b for b in self.block_ids if b.split("_", 1)[0] not in test_dates]
            test_idx = np.where(np.isin(self.blocks, test_blocks))[0]
            train_idx = np.where(np.isin(self.blocks, train_blocks))[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            beta = fit_fn(train_idx)
            chosen_params[gi] = tuple(np.round(beta, 4))
            test_feats = [feats[i] for i in test_idx]
            test_picks = _MM.pq_picks(beta, test_feats, pq_threshold=pq_threshold, odds_cap=odds_cap)
            for local_i, global_i in enumerate(test_idx):
                picks[global_i] = test_picks[local_i]
        safe_picks = [p if p is not None else np.array([], dtype=int) for p in picks]
        result = {"picks": safe_picks, "chosen_params": chosen_params, **self.evaluate(safe_picks)}
        result["n_unique_patterns"] = len(set(chosen_params.values()))
        result["n_folds"] = len(chosen_params)
        return result

    def walk_forward_oof(self, fit_fn, feats: list, burn_in_months: int = 6,
                         pq_threshold=None, odds_cap: float = float("inf")) -> dict:
        """月次リフィットのexpanding-window walk-forward(2026-08-23新設、主判定)。
        開催日をYYYY-MM単位でグルーピングし、burn_in_months ヶ月分のデータが貯まってから
        月ごとに「それより前の全月で学習→当該月をテスト」を繰り返す。実運用(毎月モデルを
        更新して翌月に賭ける)と1対1に対応する検証方法。"""
        import jra_market_model as _MM
        pq_threshold = _MM.DEFAULT_PQ_THRESHOLD if pq_threshold is None else pq_threshold
        dates = sorted({b.split("_", 1)[0] for b in self.block_ids})
        month_of = {d: d[:6] for d in dates}
        months = sorted(set(month_of.values()))
        picks = [None] * len(self.races)
        chosen_params = {}
        monthly_stats = []
        tested_race_idx = []
        for i, m in enumerate(months):
            train_months = set(months[:i])
            if len(train_months) < burn_in_months:
                continue
            test_dates = {d for d in dates if month_of[d] == m}
            train_dates = {d for d in dates if month_of[d] in train_months}
            test_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in test_dates]
            train_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in train_dates]
            test_idx = np.where(np.isin(self.blocks, test_blocks))[0]
            train_idx = np.where(np.isin(self.blocks, train_blocks))[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            beta = fit_fn(train_idx)
            chosen_params[m] = tuple(np.round(beta, 4))
            test_feats = [feats[i] for i in test_idx]
            test_picks = _MM.pq_picks(beta, test_feats, pq_threshold=pq_threshold, odds_cap=odds_cap)
            for local_i, global_i in enumerate(test_idx):
                picks[global_i] = test_picks[local_i]
                tested_race_idx.append(global_i)
            monthly_stats.append({"month": m, "beta": beta.tolist(),
                                  "n_train_races": int(len(train_idx)), "n_test_races": int(len(test_idx))})
        tested_race_idx = np.array(sorted(tested_race_idx), dtype=int)
        safe_picks = [p if p is not None else np.array([], dtype=int) for p in picks]
        result = {"picks": safe_picks, "tested_race_idx": tested_race_idx, "chosen_params": chosen_params,
                 "monthly_stats": monthly_stats, **self.evaluate(safe_picks, idx=tested_race_idx)}
        result["n_unique_patterns"] = len(set(chosen_params.values()))
        result["n_folds"] = len(chosen_params)
        return result

    def walk_forward_oof_nll(self, fit_fn, feats: list, burn_in_months: int = 6) -> dict:
        """walk_forward_oofと同じ月次リフィット手順で、held-outレースのNLLを記録する版
        (ゲート1: block_bootstrap_diff_nllの入力用)。"""
        import jra_market_model as _MM
        dates = sorted({b.split("_", 1)[0] for b in self.block_ids})
        month_of = {d: d[:6] for d in dates}
        months = sorted(set(month_of.values()))
        nll_per_race = np.full(len(self.races), np.nan)
        chosen_params = {}
        for i, m in enumerate(months):
            train_months = set(months[:i])
            if len(train_months) < burn_in_months:
                continue
            test_dates = {d for d in dates if month_of[d] == m}
            train_dates = {d for d in dates if month_of[d] in train_months}
            test_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in test_dates]
            train_blocks = [b for b in self.block_ids if b.split("_", 1)[0] in train_dates]
            test_idx = np.where(np.isin(self.blocks, test_blocks))[0]
            train_idx = np.where(np.isin(self.blocks, train_blocks))[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            beta = fit_fn(train_idx)
            chosen_params[m] = tuple(np.round(beta, 4))
            for i2 in test_idx:
                nll_per_race[i2] = _MM.race_nll(beta, feats, idx=[i2])
        return {"nll_per_race": nll_per_race, "chosen_params": chosen_params,
                "mean_nll": float(np.nanmean(nll_per_race))}

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
                                  bet: str = OBJ_BET_SINGLES, n_rep: int = 200, seed: int = 99,
                                  fit_fn=None, refit_beta_per_split: bool = False,
                                  pick_fn=None) -> dict:
    """jra_eval.selection_optimismの「候補プールから選ぶことの真の価値」診断を、重み候補では
    なくEV閾値・オッズ上限のグリッド(threshold_grid: [(threshold, odds_cap), ...])に
    読み替えたもの。

    2026-08-23修正: 旧仕様(refit_beta_per_split省略時、既定False)はbetaを固定して渡す
    ため、そのbetaが全データ(split両側)を使ってfitされたものだと、split A/Bの両方に
    リークする(jra_market_model_search_2026_08_22.pyがbeta_full_insampleを渡していたのが
    該当)。refit_beta_per_split=True かつ fit_fn を渡すと、split毎にA側のインデックスだけで
    fit_fn(idx_a)を呼んで真にA側だけで学習したbetaを使う(B側は一切参照しない)。
    後方互換のため、fit_fn省略時は旧仕様(固定beta)にフォールバックする。"""
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    col = SB.BET_TYPES_SINGLES.index(bet)
    # 既定はMM.ev_picks(旧仕様と同一の呼び出し規約: ev_threshold/odds_cap)。
    # p/q閾値ベースで評価したい場合はpick_fn=MM.pq_picksを明示的に渡す(2026-08-23、選択肢B)。
    pick_fn = pick_fn if pick_fn is not None else MM.ev_picks

    def rate(st, rt, idx):
        s, r = st[idx, col].sum(), rt[idx, col].sum()
        return r / s * 100 if s else 0.0

    def st_rt_for_beta(beta_):
        out = []
        for (t, c) in threshold_grid:
            p = pick_fn(beta_, feats, pq_threshold=t, odds_cap=c) if pick_fn is MM.pq_picks \
                else pick_fn(beta_, feats, ev_threshold=t, odds_cap=c)
            s, r = ev.settler.returns_for(p)
            out.append((s, r))
        return out

    rng = np.random.default_rng(seed)
    sel, unseen, unseen_mean = [], [], []
    fixed_st_rt = None if (refit_beta_per_split and fit_fn is not None) else st_rt_for_beta(beta)
    for _ in range(n_rep):
        perm = rng.permutation(len(ids))
        a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
        b = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
        if refit_beta_per_split and fit_fn is not None:
            beta_a = fit_fn(a)
            st_rt = st_rt_for_beta(beta_a)
        else:
            st_rt = fixed_st_rt
        va = np.array([rate(s, r, a) for (s, r) in st_rt])
        vb = np.array([rate(s, r, b) for (s, r) in st_rt])
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
        "refit_beta_per_split": bool(refit_beta_per_split and fit_fn is not None),
    }


def odds_matched_permutation_test(ev: Evaluator, races: list, actual_picks: list,
                                  bet: str = OBJ_BET_SINGLES, n_perm: int = 2000, seed: int = 77,
                                  odds_col: str = None, tol_log: float = 0.15) -> dict:
    """【2026-08-23修正】旧仕様は各レース内で完全一様ランダムに馬を選んでいた
    (docstringも自認)。一様ランダムは万馬券も同確率で引くため帰無分布が過度に歪み検定力が
    落ちる。修正版は「実際に選んだ馬と同じレース内で、log(オッズ)の距離がtol_log以内の
    別の馬」からランダムに選ぶ(候補が無ければlog距離最近傍)。帰無仮説を「同じレース・
    同じ値段の別の馬より良かったか」に正しく揃える。odds_col省略時は各レースのdfから
    "odds_final"→"bias_win_odds"の順で自動検出する。"""
    rng = np.random.default_rng(seed)
    col = SB.BET_TYPES_SINGLES.index(bet)

    race_odds = []
    for r in races:
        oc = odds_col
        if oc is None:
            oc = "odds_final" if "odds_final" in r["df"].columns else "bias_win_odds"
        o = pd.to_numeric(r["df"][oc], errors="coerce").to_numpy(dtype=float)
        race_odds.append(o)

    k_per_race = [len(p) if p is not None else 0 for p in actual_picks]
    real_st, real_rt = ev.settler.returns_for(actual_picks)
    real_stake, real_ret = real_st[:, col].sum(), real_rt[:, col].sum()
    real_rate = real_ret / real_stake * 100 if real_stake else 0.0

    sim_rates = np.empty(n_perm)
    for p in range(n_perm):
        rand_picks = []
        for i, k in enumerate(k_per_race):
            if k == 0:
                rand_picks.append(np.array([], dtype=int))
                continue
            odds_i = race_odds[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_odds = np.log(np.where(odds_i > 0, odds_i, np.nan))
            chosen = set()
            for orig_idx in actual_picks[i]:
                target = log_odds[orig_idx] if orig_idx < len(log_odds) else np.nan
                if np.isnan(target):
                    chosen.add(int(orig_idx))
                    continue
                dist = np.abs(log_odds - target)
                dist[list(chosen)] = np.inf
                candidates = np.where(dist <= tol_log)[0]
                if len(candidates) == 0:
                    pick = int(np.nanargmin(dist))
                else:
                    pick = int(rng.choice(candidates))
                chosen.add(pick)
            rand_picks.append(np.array(sorted(chosen), dtype=int))
        st, rt = ev.settler.returns_for(rand_picks)
        s, r = st[:, col].sum(), rt[:, col].sum()
        sim_rates[p] = r / s * 100 if s else 0.0

    return {"real_rate": real_rate, "n_bet_races": int(sum(1 for k in k_per_race if k > 0)),
            "sim_mean": float(sim_rates.mean()), "sim_median": float(np.median(sim_rates)),
            "sim_p95": float(np.percentile(sim_rates, 95)),
            "p_value_ge_real": float((sim_rates >= real_rate).mean()), "n_perm": n_perm,
            "tol_log": tol_log}
