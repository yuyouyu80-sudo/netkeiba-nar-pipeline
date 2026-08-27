# -*- coding: utf-8 -*-
"""NAR BOX4 の評価基盤。レビュー2件の指摘をすべて反映した評価だけをここに置く。

設計判断(いずれもレビュー指摘に対応):
  * fold は「開催日」ではなく「開催日 × 競馬場」ブロック。7/25は高知単独開催のため、
    日付foldだと「高知だけでテスト」する回が生まれる。ブロックなら7個前後に増え、
    かつブロック内の相関(馬場・トラックバイアス・その日の騎手の調子)を跨がない。
  * fold平均ではなく pooled(払戻と賭金を積んでから割る)。単純平均は実測で+12.9pt
    水増しし、選択バイアスの分散を1.8倍にする。
  * 目的関数は「複勝+ワイドのコスト加重回収率」から「同じレースで上位4人気BOXを
    買ったときの同指標」を引いた超過値。絶対回収率を最大化すると、市場(オッズ)を
    再発見するか71レースの運を拾うかのどちらかにしかならない。同一レース上の対比較に
    することで、天候・その日の配当水準といった共通ノイズが相殺される。
  * ブートストラップはブロック単位・比推定量(Σreturn/Σstake を同時にリサンプル)。
"""
import numpy as np
import pandas as pd

import nar_backtest as NB

# 目的関数に使う券種と、4頭BOXでの購入点数(=1レースあたりのコスト倍率)。
OBJ_BETS = ["複勝", "ワイド"]
BET_POINTS = NB.POINTS_BOX4
UNIT = NB.UNIT


def blocks_of(races: list) -> np.ndarray:
    """開催日×競馬場のブロックIDを返す。"""
    return np.array([f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races])


def market_picks(races: list, box_n: int = 4) -> list:
    """上位N人気BOX(市場ベンチマーク)。人気が欠損する馬は最後尾に置く。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), 1e18, ninki)
        picks.append(np.argsort(key, kind="stable")[:box_n])
    return picks


def score_picks(mats: list, w: np.ndarray, box_n: int = 4) -> list:
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
    cols = [NB.BET_TYPES.index(b) for b in bets]
    s = stake[:, cols] if idx is None else stake[np.ix_(idx, cols)]
    r = ret[:, cols] if idx is None else ret[np.ix_(idx, cols)]
    tot = s.sum()
    return float(r.sum() / tot * 100) if tot else 0.0


class Evaluator:
    """1つのレース集合に対する評価器。決済テーブルと市場ベンチマークを一度だけ作る。"""

    def __init__(self, races: list, actual: dict, box_n: int = 4):
        self.races = races
        self.box_n = box_n
        self.settler = NB.BoxSettler(races, actual, box_n=box_n)
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
        for i, bt in enumerate(NB.BET_TYPES):
            s, r = int(st[:, i].sum()), int(rt[:, i].sum())
            hits = int((rt[:, i] > 0).sum())
            rows.append({"bet_type": bt, "races": len(picks), "hit_races": hits,
                         "hit_rate_pct": round(hits / len(picks) * 100, 1),
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

    # ----------------------------------------------------- CV (2026-08-27追加)
    # 以下5メソッドは全て既存メソッドへの後方互換な追加であり、lobo_oof/block_bootstrap
    # 等の既存シグネチャ・挙動は無改造。全ファクター統合・高確信度選抜計画(Phase0)。
    def lobo_oof_tracked(self, fit_fn, mats_all: list) -> dict:
        """lobo_oofと同じLOBO OOFだが、fit_fn(train_idx) -> (w, chosen) を受け取り
        fold別argmaxのユニーク数(fold_argmax_unique)を追跡する。ブロック数が多いと
        学習集合がほぼ変わらずargmaxが実質固定される構造的欠陥(2026-08-20発見、
        nar_search500_2026_08_20.pyのbox4/box3で全foldargmax固定を確認済み)を検知する
        ための診断値。既存lobo_oof(fit_fn(train_idx) -> w のみ)は無改造のまま残す
        (nar_search500_2026_08_20.py等の既存呼び出しの再現性を保つため)。"""
        picks = [None] * len(self.races)
        choices = []
        for b in self.block_ids:
            test_idx = np.where(self.blocks == b)[0]
            train_idx = np.where(self.blocks != b)[0]
            w, chosen = fit_fn(train_idx)
            choices.append(chosen)
            for i in test_idx:
                m = mats_all[i]
                num, den = m["S"] @ w, m["A"] @ w
                score = np.where(den > 0, num / den, -1e18)
                picks[i] = np.argsort(-score, kind="stable")[:self.box_n]
        result = {"picks": picks, **self.evaluate(picks)}
        result["fold_argmax_choices"] = choices
        result["fold_argmax_unique"] = len(set(choices))
        return result

    def group_kfold_oof(self, fit_fn, mats_all: list, n_folds: int = 8,
                        seed: int = 13) -> dict:
        """ブロックをn_foldsグループに乱数分割するK分割out-of-fold評価
        (nar_bottom_eval.Evaluator.group_kfold_oofの移植)。leave-one-block-outは
        ブロック数が多い(NARは94〜138)と1ブロック抜いても学習集合がほぼ変わらず
        argmaxが実質固定される構造的欠陥がある。held-out比率を意図的に大きくして
        (既定n_folds=8なら1/8=12.5%)、argmaxが実際に変動しうるようにする。
        fit_fnの契約はlobo_oof_trackedと同じ: fit_fn(train_idx) -> (w, chosen)。"""
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        order = rng.permutation(len(ids))
        folds = [order[i::n_folds] for i in range(n_folds)]
        picks = [None] * len(self.races)
        choices = []
        for fold in folds:
            test_blocks = {ids[i] for i in fold}
            test_idx = np.array([i for i, b in enumerate(self.blocks) if b in test_blocks])
            train_idx = np.array([i for i, b in enumerate(self.blocks) if b not in test_blocks])
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            w, chosen = fit_fn(train_idx)
            choices.append(chosen)
            for i in test_idx:
                m = mats_all[i]
                num, den = m["S"] @ w, m["A"] @ w
                score = np.where(den > 0, num / den, -1e18)
                picks[i] = np.argsort(-score, kind="stable")[:self.box_n]
        result = {"picks": picks, **self.evaluate(picks)}
        result["fold_argmax_choices"] = choices
        result["fold_argmax_unique"] = len(set(choices))
        result["n_folds"] = len(folds)
        return result

    def group_kfold_oof_generic(self, predict_fn, n_folds: int = 8, seed: int = 13) -> dict:
        """group_kfold_oofの汎用版。predict_fn(train_idx, test_idx) -> (test_picks, chosen)
        を受け取る(test_picksはtest_idx順に対応するpicksのリスト、chosenはハッシュ可能な
        値である必要がある — 例: 選ばれたDirichletパターン番号や正則化強度C)。
        mats_all@w という重みベクトル前提のgroup_kfold_oofと異なり、正則化ロジスティック
        回帰(nar_logistic.py)のような任意のfit/predict手順を、Dirichletパターン探索と
        全く同じOOFプロトコル(同じブロック分割・同じ評価指標)に載せて直接比較できる
        (Phase2の本命)。"""
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        order = rng.permutation(len(ids))
        folds = [order[i::n_folds] for i in range(n_folds)]
        picks = [None] * len(self.races)
        choices = []
        for fold in folds:
            test_blocks = {ids[i] for i in fold}
            test_idx = np.array([i for i, b in enumerate(self.blocks) if b in test_blocks])
            train_idx = np.array([i for i, b in enumerate(self.blocks) if b not in test_blocks])
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            test_picks, chosen = predict_fn(train_idx, test_idx)
            choices.append(chosen)
            for i, p in zip(test_idx, test_picks):
                picks[i] = p
        result = {"picks": picks, **self.evaluate(picks)}
        result["fold_argmax_choices"] = choices
        result["fold_argmax_unique"] = len(set(choices))
        result["n_folds"] = len(folds)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, bets=OBJ_BETS, n: int = 2000,
                        seed: int = 11) -> dict:
        """ブロック単位の比推定量ブートストラップ。レース単位だとブロック内相関を
        無視して信頼区間を過小評価する。"""
        st, rt = self.settler.returns_for(picks)
        cols = [NB.BET_TYPES.index(b) for b in bets]
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

    def paired_block_bootstrap(self, picks_a: list, picks_b: list, bets=OBJ_BETS,
                               n: int = 2000, seed: int = 41) -> dict:
        """同一レース上でpicks_aとpicks_bのコスト加重回収率を対にした差分をブロック
        ブートストラップする(nar_bottom_eval.paired_block_bootstrapの移植、payout決済版)。
        天候・その日の配当水準といった共通ノイズを相殺するため、block_bootstrap同士を
        別々に見るより検出力が高い(本命の比較手法、Phase1/2の候補vs現行基準の判定に使う)。"""
        st_a, rt_a = self.settler.returns_for(picks_a)
        st_b, rt_b = self.settler.returns_for(picks_b)
        cols = [NB.BET_TYPES.index(b) for b in bets]
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        diffs = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            sa = st_a[np.ix_(idx, cols)].sum()
            ra = rt_a[np.ix_(idx, cols)].sum()
            sb = st_b[np.ix_(idx, cols)].sum()
            rb = rt_b[np.ix_(idx, cols)].sum()
            va = ra / sa * 100 if sa else 0.0
            vb = rb / sb * 100 if sb else 0.0
            diffs[k] = va - vb
        return {"mean": float(diffs.mean()), "lo": float(np.percentile(diffs, 2.5)),
                "hi": float(np.percentile(diffs, 97.5))}

    def paired_block_bootstrap_subset(self, picks: list, idx: np.ndarray, bets=OBJ_BETS,
                                      n: int = 2000, seed: int = 11) -> dict:
        """paired_block_bootstrapのidx限定版(Phase3: 確信度閾値τで絞った部分集合を、
        同じ部分集合上の市場ベンチマーク(self.mkt_stake/self.mkt_ret)とペア差分比較する)。

        2026-08-28に発見した重要な区別: block_bootstrap_subset(idxに絞った絶対回収率の
        CI)は「市場との比較」ではないため、下限が0を超えるのは回収率が正の値である限り
        ほぼ自明で、Phase3の採否ゲート(市場を統計的に上回るか)には使えない
        (nar_threshold_sweep_2026_08_28.pyの初回実行で9閾値中8つが誤ってPASSした
        バグの原因)。市場超過ptの部分集合CIが必要な場合は必ずこちらを使うこと。"""
        st, rt = self.settler.returns_for(picks)
        cols = [NB.BET_TYPES.index(b) for b in bets]
        idx_set = set(int(i) for i in idx)
        by_block = {}
        for b in self.block_ids:
            rows = [i for i in np.where(self.blocks == b)[0] if i in idx_set]
            if rows:
                by_block[b] = np.array(rows)
        ids = list(by_block.keys())
        if not ids:
            return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_blocks": 0}
        rng = np.random.default_rng(seed)
        diffs = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            rows = np.concatenate([by_block[ids[c]] for c in chosen])
            sm = st[np.ix_(rows, cols)].sum()
            rm = rt[np.ix_(rows, cols)].sum()
            smk = self.mkt_stake[np.ix_(rows, cols)].sum()
            rmk = self.mkt_ret[np.ix_(rows, cols)].sum()
            vm = rm / sm * 100 if sm else 0.0
            vk = rmk / smk * 100 if smk else 0.0
            diffs[k] = vm - vk
        return {"mean": float(diffs.mean()), "lo": float(np.percentile(diffs, 2.5)),
                "hi": float(np.percentile(diffs, 97.5)), "n_blocks": len(ids)}

    def block_bootstrap_subset(self, picks: list, idx: np.ndarray, bets=OBJ_BETS,
                               n: int = 2000, seed: int = 11) -> dict:
        """block_bootstrapのidx限定版(Phase3: 確信度閾値τで絞った部分集合の評価用)。
        idxに含まれるレースが属するブロックだけをリサンプリング対象にし、各ブロック内でも
        idxに含まれる行だけを集計する(1ブロックの中で一部のレースだけがτを超える場合にも
        対応)。該当ブロックが無ければn_blocks=0とし平均・CIとも0.0を返す(呼び出し側で
        n_blocks不足時は結果を採用しない設計、Phase3の最低ブロック数ゲートに使う)。"""
        st, rt = self.settler.returns_for(picks)
        cols = [NB.BET_TYPES.index(b) for b in bets]
        idx_set = set(int(i) for i in idx)
        by_block = {}
        for b in self.block_ids:
            rows = [i for i in np.where(self.blocks == b)[0] if i in idx_set]
            if rows:
                by_block[b] = np.array(rows)
        ids = list(by_block.keys())
        if not ids:
            return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_blocks": 0}
        rng = np.random.default_rng(seed)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            rows = np.concatenate([by_block[ids[c]] for c in chosen])
            s = st[np.ix_(rows, cols)].sum()
            r = rt[np.ix_(rows, cols)].sum()
            out[k] = r / s * 100 if s else 0.0
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
