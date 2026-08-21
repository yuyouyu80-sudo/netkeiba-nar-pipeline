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
import itertools
import math

import numpy as np
import pandas as pd

import jra_backtest as JB

OBJ_BETS = ["複勝", "ワイド"]
UNIT = JB.UNIT

# JRA公式の券種別控除率(%)。BET_TYPES(単勝/複勝/枠連/馬連/ワイド/馬単/3連複/3連単)の順。
# 2026-08-22追加(Step0: 評価基盤の是正)。breakeven_pct()の計算に使う。
TAKEOUT_RATES = {
    "単勝": 20.0, "複勝": 20.0, "枠連": 22.5, "馬連": 22.5,
    "ワイド": 22.5, "馬単": 25.0, "3連複": 25.0, "3連単": 27.5,
}


def blocks_of(races: list) -> np.ndarray:
    """開催日×競馬場のブロックIDを返す。"""
    return np.array([f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races])


def breakeven_pct(box_n: int, bets=OBJ_BETS) -> float:
    """box_nでbets(既定=複勝+ワイド)を買ったときの理論ブレークイーブン回収率(%)。
    2026-08-22追加(Step0)。1レースあたりの点数(複勝=box_n、ワイド=C(box_n,2)、
    枠連/馬連/3連複も同型のC(box_n,2)、馬単/3連単は順列)でTAKEOUT_RATESを加重平均する。
    「市場差+Npt」が理論ブレークイーブンをまだ下回っているか超えているかを判定する基準線。"""
    n_points = {
        "単勝": box_n, "複勝": box_n,
        "枠連": math.comb(box_n, 2), "馬連": math.comb(box_n, 2), "ワイド": math.comb(box_n, 2),
        "馬単": box_n * (box_n - 1),
        "3連複": math.comb(box_n, 3) if box_n >= 3 else 0,
        "3連単": box_n * (box_n - 1) * (box_n - 2) if box_n >= 3 else 0,
    }
    total_pts = sum(n_points[b] for b in bets)
    if total_pts == 0:
        return 0.0
    weighted = sum(n_points[b] * (100.0 - TAKEOUT_RATES[b]) for b in bets)
    return weighted / total_pts


def held_out_block_subset(fitted_on: dict, races: list) -> list:
    """fitted_on(winner_*.jsonの'fitted_on'辞書、search_dates/holdout_datesを含む)に
    含まれるkaisai_dateのブロックを除いた、真に未見のブロックID一覧を返す。
    2026-08-22追加(Step0)。Evaluator.block_bootstrap(..., block_subset=...)にそのまま渡せる。"""
    fit_dates = set(fitted_on.get("search_dates", [])) | set(fitted_on.get("holdout_dates", []))
    blocks = blocks_of(races)
    return sorted({b for b in blocks if b.split("_", 1)[0] not in fit_dates})


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

    def evaluate(self, picks: list, idx: np.ndarray = None, multipliers: np.ndarray = None) -> dict:
        """picks に対する目的関数値と、市場ベンチマークとの差を返す。
        multipliers: レースごとのステーク乗数(長さ=len(picks))。指定時はmodel・market双方の
        stake/returnに同じ乗数を適用してから比推定量(Σreturn/Σstake)を取る(2026-08-21、
        Front3ステーク配分最適化用に追加。Noneなら従来と数値的に完全に同一)。"""
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
        """全券種の的中率・回収率(レポート用)。idxを指定するとそのレース添字だけに絞る
        (0/1選抜の的中率が正しく出る。multipliersで0埋めすると的中していても賭金0で
        hit_races判定から漏れるため、選抜にはidxを使うこと)。multipliersは連続ステーク乗数
        (2026-08-21、Front3用。0にならない前提ならhit_races判定は変わらない)。"""
        st, rt = self.settler.returns_for(picks)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st, rt = st * m, rt * m
        if idx is not None:
            st, rt = st[idx], rt[idx]
        n_races = st.shape[0]
        rows = []
        for i, bt in enumerate(JB.BET_TYPES):
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
        選ばれていないか)に使う。fit_fn(train_idx) は (w, pattern_idx) のタプルを返すこと
        (2026-08-21、jra_axis_eval.pyと契約を統一)。
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
        """開催日昇順のexpanding-window walk-forward評価(2026-08-21新設)。
        blocks_of()が返す"{kaisai_date}_{racecourse}"の日付部分でグループ化し、
        train=それより前の全開催日・test=対象開催日、という分割を日付昇順に繰り返す。
        lobo_oofと同じ戻り値契約(chosen_pattern_idx/n_unique_patterns/n_folds)を持つが、
        ブロックのholdoutがランダムではなく時間方向に一方向という点だけが異なる。
        **採否の主判定には使わない**(ランダムblock-holdoutのlobo_oof+選択バイアス診断が
        主判定)。時間方向のリークが無いかを確認する追加のロバスト性チェック専用。
        開催日数がmin_train_blocks未満しかない先頭の日はテスト対象から除外する
        (ウォームアップ、学習データが薄すぎるfoldを評価に含めない)。
        """
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
        # 未テスト(ウォームアップ除外)のレースはpicksがNoneのままなのでevaluateに渡せない。
        # tested_race_idxだけの部分集合として評価する(idx引数で絞り込む。picks自体は
        # Noneを含んだ全長のリストのままだが、evaluate()はreturns_for()経由でNone要素には
        # アクセスしない — idxで絞ったcost_weighted_rateの列だけを見るため)。
        safe_picks = [p if p is not None else np.arange(self.box_n) for p in picks]
        result = {"picks": picks, "tested_race_idx": tested_race_idx,
                 "chosen_pattern_idx": chosen_pattern_idx,
                 **self.evaluate(safe_picks, idx=tested_race_idx)}
        result["n_unique_patterns"] = len(set(chosen_pattern_idx.values()))
        result["n_folds"] = len(chosen_pattern_idx)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, bets=OBJ_BETS, n: int = 2000,
                        seed: int = 11, block_subset=None,
                        multipliers: np.ndarray = None) -> dict:
        """ブロック単位の比推定量ブートストラップ。レース単位だとブロック内相関を
        無視して信頼区間を過小評価する。block_subsetを指定すると、そのブロック集合だけを
        対象にリサンプルする(例: 重みのfit母集団と重複しないブロックだけで「真に未見の
        データ」上の信頼区間を出す用途)。multipliersはレースごとのステーク乗数(Front3用)。"""
        st, rt = self.settler.returns_for(picks)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st, rt = st * m, rt * m
        cols = [JB.BET_TYPES.index(b) for b in bets]
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

    def block_bootstrap_diff(self, picks_a: list, picks_b: list, bets=OBJ_BETS,
                             n: int = 2000, seed: int = 11, block_subset=None,
                             multipliers: np.ndarray = None) -> dict:
        """2つのpicks(例: 探索モデル vs 現行重み)の差(model_a - model_b)をブロック単位で
        ペアでブートストラップする。同一レースを同一リサンプルに使うことで天候・その日の
        配当水準といった共通ノイズを相殺する。block_subset/multipliersの意味はblock_bootstrap
        と同じ(2026-08-21、jra_axis_eval.pyから移植・multipliers対応を追加)。"""
        st_a, rt_a = self.settler.returns_for(picks_a)
        st_b, rt_b = self.settler.returns_for(picks_b)
        if multipliers is not None:
            m = np.asarray(multipliers, dtype=float)[:, None]
            st_a, rt_a = st_a * m, rt_a * m
            st_b, rt_b = st_b * m, rt_b * m
        cols = [JB.BET_TYPES.index(b) for b in bets]
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
