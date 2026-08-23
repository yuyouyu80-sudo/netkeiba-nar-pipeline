# -*- coding: utf-8 -*-
"""馬連特化モデリングの評価基盤(2026-08-22新設、2026-08-23 ex-ante cap化)。

`jra_singles_eval.py`(単勝/複勝、Evaluator)と対になる、馬連(1着-2着の組)専用の評価器。
賭け目選択は`jra_market_model.umaren_pq_picks`(Harville式ペア確率のp/q比、top1固定)、
決済は`jra_umaren_backtest.UmarenSettler`を使う。ブロック定義(`blocks_of`=開催日×競馬場)は
`jra_eval.py`のものをそのまま再利用する(単勝/複勝版と揃える)。

2026-08-23修正(Opus 5サブエージェントのレビューで発見): ユーザー指定の「払い戻し2万円以上は
無視する」を、旧版は**実現後(realized)**の払戻で事後的に無効化する`apply_payout_cap`として
実装していたが、これには2つの不具合があった。
  (a) 実運用不能: 賭ける時点では払戻は分からない。
  (b) 順列検定(Gate5)の帰無分布だけを非対称に切り詰める: 実測側は「たまたま大穴が当たらな
      かった」ため無傷、シミュレーション側は2,000回の毎回、上振れだけを機械的に削られる。
      結果、cap有りだとp=0.000(PASS)・cap無しだとp=0.455(FAIL)という正反対の結論になり、
      「モデルはランダムより有意に賢い」という主張は成立しないことが判明した。
本モジュールは払戻capを**事前(ex-ante)**基準に修正する: `jra_market_model.umaren_pq_picks`
の`max_payout`引数(市場インプライド確率から導出した想定払戻`estimated_umaren_payout`が
上限を超える候補はそもそも賭け目に選ばない)を使い、選抜の時点で一貫してcapを適用する。
以後、決済・ブートストラップ・順列検定は「既にcap済みのpicks」をそのまま扱うだけでよく、
事後的なcap関数は不要になった。
"""
import numpy as np
import pandas as pd

import jra_market_model as MM
import jra_umaren_backtest as UB
from jra_eval import blocks_of

OBJ_BET_UMAREN = "馬連"
DEFAULT_MAX_PAYOUT = 20000  # ユーザー指定: 想定払戻がこれ以上のペアはそもそも賭けない(ex-ante)


def cost_weighted_rate(stake: np.ndarray, ret: np.ndarray, idx: np.ndarray = None) -> float:
    s = stake[:, 0] if idx is None else stake[idx, 0]
    r = ret[:, 0] if idx is None else ret[idx, 0]
    tot = s.sum()
    return float(r.sum() / tot * 100) if tot else 0.0


def market_favorite_pair_picks(races: list, ninki_col: str = "popularity") -> list:
    """市場ベンチマーク: 単勝人気1-2位の馬連1点(参考)。2026-08-23: Opus 5レビューでこの
    ベンチマークがレポートに出力されていなかったことが指摘された。モデルとの差(excess)を
    必ず報告に含める。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"][ninki_col], errors="coerce").to_numpy(dtype=float)
        i1 = np.where(ninki == 1)[0]
        i2 = np.where(ninki == 2)[0]
        if len(i1) and len(i2):
            picks.append((int(i1[0]), int(i2[0])))
        else:
            picks.append(None)
    return picks


class UmarenEvaluator:
    """馬連1点買いの評価器(単勝/複勝版Evaluatorの馬連版)。picksは呼び出し側で既に
    ex-ante払戻capを適用済みという前提(MM.umaren_pq_picksのmax_payout引数を使う)。"""

    def __init__(self, races: list, actual: dict, ninki_col: str = "popularity"):
        self.races = races
        self.actual = actual
        self.settler = UB.UmarenSettler(races, actual)
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_stake, self.mkt_ret = self.settler.returns_for(
            market_favorite_pair_picks(races, ninki_col=ninki_col))

    def evaluate(self, picks: list, idx: np.ndarray = None) -> dict:
        st, rt = self.settler.returns_for(picks)
        model = cost_weighted_rate(st, rt, idx=idx)
        market = cost_weighted_rate(self.mkt_stake, self.mkt_ret, idx=idx)
        subset = picks if idx is None else [picks[i] for i in idx]
        n_bet_races = int(sum(1 for p in subset if p is not None))
        return {"model": model, "market": market, "excess": model - market,
                "stake": st, "return": rt, "n_bet_races": n_bet_races}

    # --------------------------------------------------------------- CV(時系列・月次)
    def walk_forward_oof(self, fit_fn, feats: list, burn_in_months: int = 6,
                         pq_threshold=None, max_payout: float = None) -> dict:
        """月次リフィットのexpanding-window walk-forward。fit_fn(train_idx)は単勝勝率モデルの
        パラメータ配列beta=[beta0,beta1,beta2]を返す契約(jra_singles_eval.walk_forward_oofと
        同一、勝率モデル自体は共通)。ペアの選抜だけがumaren_pq_picksに置き換わる。
        max_payout: MM.umaren_pq_picksへそのまま渡すex-anteの想定払戻上限(Noneならcap無し)。"""
        pq_threshold = MM.DEFAULT_PQ_THRESHOLD if pq_threshold is None else pq_threshold
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
            test_picks = MM.umaren_pq_picks(beta, test_feats, pq_threshold=pq_threshold,
                                            max_payout=max_payout)
            for local_i, global_i in enumerate(test_idx):
                picks[global_i] = test_picks[local_i]
                tested_race_idx.append(global_i)
            monthly_stats.append({"month": m, "beta": beta.tolist(),
                                  "n_train_races": int(len(train_idx)), "n_test_races": int(len(test_idx))})
        tested_race_idx = np.array(sorted(tested_race_idx), dtype=int)
        result = {"picks": picks, "tested_race_idx": tested_race_idx, "chosen_params": chosen_params,
                 "monthly_stats": monthly_stats, **self.evaluate(picks, idx=tested_race_idx)}
        result["n_unique_patterns"] = len(set(chosen_params.values()))
        result["n_folds"] = len(chosen_params)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, n: int = 2000, seed: int = 11, block_subset=None) -> dict:
        """ブロック単位の比推定量ブートストラップ。picksは呼び出し側で既にcap適用済みの
        前提(2026-08-23、事後capを廃止したためここでは何もフィルタしない)。"""
        st, rt = self.settler.returns_for(picks)
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(block_subset) if block_subset is not None else list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            s, r = st[idx, 0].sum(), rt[idx, 0].sum()
            out[k] = r / s * 100 if s else 0.0
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}


def odds_matched_permutation_test(ev: UmarenEvaluator, races: list, actual_picks: list,
                                  n_perm: int = 2000, seed: int = 77, odds_col: str = None,
                                  tol_log: float = 0.15, max_payout: float = None) -> dict:
    """実際に選んだペアの各馬を、同一レース内でlog(オッズ)の距離がtol_log以内の別の馬に
    独立に置き換える(候補が無ければlog距離最近傍)ことでランダムなペアを作る。単勝/複勝版
    (jra_singles_eval.odds_matched_permutation_test)の1頭版ロジックを、ペアの各スロットに
    独立適用したもの。

    2026-08-23修正(Opus 5サブエージェントのレビュー):
      (1) 事後capの非対称バグを修正。実測picksは呼び出し側で既にex-ante cap済みという前提
          (MM.umaren_pq_picksのmax_payout)。シミュレーション側にも**同じex-ante基準**
          (max_payout指定時、estimated_umaren_payoutで想定払戻を計算し超過ならそのレースは
          見送り)を1回ごとに独立適用することで、実測・シミュレーションの両アームを対称に
          そろえる(片方だけ上振れを削る、という旧版の偏りを解消)。
      (2) 単勝版との非対称も修正: 旧版はスロット1で必ず元の馬(i1)を候補から除外していたが
          単勝版は除外していなかった(=元の馬を再選択できた)。ここもexclude集合を単勝版と
          揃え、1本目のスロットは自分自身を再選択できるようにした(2本目は既に選んだ馬との
          重複だけを避ける)。"""
    rng = np.random.default_rng(seed)

    race_odds, race_q = [], []
    for r in races:
        oc = odds_col
        if oc is None:
            oc = "odds_final" if "odds_final" in r["df"].columns else "bias_win_odds"
        o = pd.to_numeric(r["df"][oc], errors="coerce").to_numpy(dtype=float)
        race_odds.append(o)
        race_q.append(MM.normalized_implied_prob(o))

    real_st, real_rt = ev.settler.returns_for(actual_picks)
    real_stake, real_ret = real_st[:, 0].sum(), real_rt[:, 0].sum()
    real_rate = real_ret / real_stake * 100 if real_stake else 0.0
    n_bet_races = int(sum(1 for p in actual_picks if p is not None))

    def _replace_one(log_odds, target, exclude):
        if np.isnan(target):
            return None
        dist = np.abs(log_odds - target)
        if exclude:
            dist[list(exclude)] = np.inf
        candidates = np.where(dist <= tol_log)[0]
        if len(candidates) == 0:
            return int(np.nanargmin(dist))
        return int(rng.choice(candidates))

    sim_rates = np.empty(n_perm)
    for p in range(n_perm):
        rand_picks = []
        for i, pair in enumerate(actual_picks):
            if pair is None:
                rand_picks.append(None)
                continue
            odds_i = race_odds[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_odds = np.log(np.where(odds_i > 0, odds_i, np.nan))
            i1, i2 = pair
            # (2)の修正: 単勝版と揃え、1本目は元の馬(i1)自身も候補になりうる(exclude無し)。
            new1 = _replace_one(log_odds, log_odds[i1] if i1 < len(log_odds) else np.nan, set())
            if new1 is None:
                new1 = i1
            new2 = _replace_one(log_odds, log_odds[i2] if i2 < len(log_odds) else np.nan, {new1})
            if new2 is None:
                new2 = i2 if i2 != new1 else i1
            cand_pair = (new1, new2)
            # (1)の修正: ex-ante capをシミュレーション側にも同じ基準で適用する。
            if max_payout is not None:
                est = MM.estimated_umaren_payout(cand_pair, race_q[i])
                if est > max_payout:
                    rand_picks.append(None)
                    continue
            rand_picks.append(cand_pair)
        st, rt = ev.settler.returns_for(rand_picks)
        s, r = st[:, 0].sum(), rt[:, 0].sum()
        sim_rates[p] = r / s * 100 if s else 0.0

    return {"real_rate": real_rate, "n_bet_races": n_bet_races,
            "sim_mean": float(sim_rates.mean()), "sim_median": float(np.median(sim_rates)),
            "sim_p95": float(np.percentile(sim_rates, 95)),
            "p_value_ge_real": float((sim_rates >= real_rate).mean()), "n_perm": n_perm,
            "tol_log": tol_log, "max_payout": max_payout}
