# -*- coding: utf-8 -*-
"""JRA 通常戦モデルの評価基盤。scripts/nar_model/nar_eval.pyのJRA移植版。

設計判断(NAR版から踏襲):
  * fold は「開催日」ではなく「開催日 × 競馬場」ブロック。ブロック単位にすることで
    ブロック内の相関(馬場・トラックバイアス・その日の騎手の調子)を跨がない評価にする。
  * fold平均ではなく pooled(払戻と賭金を積んでから割る)。単純平均は選択バイアスの
    分散を水増しする(NAR側で実測済み)。
  * 目的関数は「複勝+ワイドのコスト加重回収率」から「同じレースで上位N人気BOXを
    買ったときの同指標」を引いた超過値(市場超過)。同一レース上の対比較にすることで、
    天候・その日の配当水準といった共通ノイズが相殺される。
  * ブートストラップはブロック単位・比推定量(Σreturn/Σstake を同時にリサンプル)。

JRAとNARのアーキテクチャ差分: NARはBOX5/4/3それぞれが独立モデル(predict_box{3,4,5}_nar.py、
各々winner_box{3,4,5}_nar.json)であり、JRAも実際には同じ構造(predict.py=pattern83/BOX5、
predict_box4.py=pattern19/BOX4、predict_box3.py=pattern95/BOX3、winner_v3.json/winner_box4.json/
winner_box3.jsonがそれぞれ独立)であることが判明した(2026-08-11、実装時に確認。当初は
「1本のランキングをtop5→top4→top3と切り詰める」設計だと誤認していたが、predict_box4.py/
predict_box3.pyが既にwinner_box4.json/winner_box3.jsonという独立した重みファイルを持って
いることをコード確認して訂正した)。よってbox_n=5/4/3はNARのnar_search300_2026_08_01.pyと
同じく、それぞれ独立にEvaluatorを作り独立に最良パターンを探索する。
"""
import numpy as np
import pandas as pd

import jra_backtest as JB

OBJ_BETS = ["複勝", "ワイド"]
UNIT = JB.UNIT


def blocks_of(races: list) -> np.ndarray:
    """開催日×競馬場のブロックIDを返す。"""
    return np.array([f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races])


def market_picks(races: list, box_n: int = 5) -> list:
    """上位N人気BOX(市場ベンチマーク)。人気が欠損する馬は最後尾に置く。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), 1e18, ninki)
        picks.append(np.argsort(key, kind="stable")[:box_n])
    return picks


def score_picks(mats: list, w: np.ndarray, box_n: int = 5) -> list:
    """重みベクトル w(names順)でレースごとの上位N頭の行インデックスを返す。"""
    picks = []
    for m in mats:
        num = m["S"] @ w
        den = m["A"] @ w
        score = np.where(den > 0, num / den, -1e18)
        picks.append(np.argsort(-score, kind="stable")[:box_n])
    return picks


def cost_weighted_rate(stake: np.ndarray, ret: np.ndarray, bets=OBJ_BETS,
                       idx: np.ndarray = None) -> float:
    """指定券種のコスト加重回収率(%)。Σ払戻 / Σ賭金。"""
    cols = [JB.BET_TYPES.index(b) for b in bets]
    s = stake[:, cols] if idx is None else stake[np.ix_(idx, cols)]
    r = ret[:, cols] if idx is None else ret[np.ix_(idx, cols)]
    tot = s.sum()
    return float(r.sum() / tot * 100) if tot else 0.0


class Evaluator:
    """1つのレース集合・1つのbox_nに対する評価器。決済テーブルと市場ベンチマークを一度だけ作る。"""

    def __init__(self, races: list, actual: dict, box_n: int = 5):
        self.races = races
        self.box_n = box_n
        self.settler = JB.BoxSettler(races, actual, box_n=box_n)
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_stake, self.mkt_ret = self.settler.returns_for(market_picks(races, box_n))

    def evaluate(self, picks: list, idx: np.ndarray = None) -> dict:
        """picks に対する目的関数値と、市場ベンチマークとの差を返す。"""
        st, rt = self.settler.returns_for(picks)
        model = cost_weighted_rate(st, rt, idx=idx)
        market = cost_weighted_rate(self.mkt_stake, self.mkt_ret, idx=idx)
        return {"model": model, "market": market, "excess": model - market,
                "stake": st, "return": rt}

    def full_table(self, picks: list) -> pd.DataFrame:
        """全券種の的中率・回収率(レポート用)。"""
        st, rt = self.settler.returns_for(picks)
        rows = []
        for i, bt in enumerate(JB.BET_TYPES):
            s, r = int(st[:, i].sum()), int(rt[:, i].sum())
            hits = int((rt[:, i] > 0).sum())
            rows.append({"bet_type": bt, "races": len(picks), "hit_races": hits,
                         "hit_rate_pct": round(hits / len(picks) * 100, 1) if len(picks) else 0.0,
                         "stake": s, "return": r,
                         "return_rate_pct": round(r / s * 100, 1) if s else 0.0})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- CV
    def lobo_oof(self, fit_fn, mats_all: list) -> dict:
        """Leave-one-block-out の out-of-fold 評価。

        fit_fn(train_idx) -> 重みベクトル w を受け取り、各ブロックをテストにして
        そのブロックのpicksを集める。最後に全体をpooledで集計する。
        """
        picks = [None] * len(self.races)
        for b in self.block_ids:
            test_idx = np.where(self.blocks == b)[0]
            train_idx = np.where(self.blocks != b)[0]
            w = fit_fn(train_idx)
            for i in test_idx:
                m = mats_all[i]
                num, den = m["S"] @ w, m["A"] @ w
                score = np.where(den > 0, num / den, -1e18)
                picks[i] = np.argsort(-score, kind="stable")[:self.box_n]
        return {"picks": picks, **self.evaluate(picks)}

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, bets=OBJ_BETS, n: int = 2000,
                        seed: int = 11) -> dict:
        """ブロック単位の比推定量ブートストラップ。レース単位だとブロック内相関を
        無視して信頼区間を過小評価する。"""
        st, rt = self.settler.returns_for(picks)
        cols = [JB.BET_TYPES.index(b) for b in bets]
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            s = st[np.ix_(idx, cols)].sum()
            r = rt[np.ix_(idx, cols)].sum()
            out[k] = r / s * 100 if s else 0.0
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5))}


def selection_optimism(ev: Evaluator, mats: list, W: np.ndarray, n_rep: int = 200,
                       seed: int = 99) -> dict:
    """「重みを選ぶ」という行為から得られる真の利得を測る。

    ブロックを半分に割り、片側で最良の重みを選び、もう片側でその重みを評価する。
    未使用側の全パターン平均も同時に出すことで、
      (未使用側での選抜値) - (未使用側での全パターン平均)
    = 選抜の真の価値、が読める。
    """
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
