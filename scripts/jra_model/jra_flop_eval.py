# -*- coding: utf-8 -*-
"""凡走予測(flop)の評価専用ヘルパー(2026-09-11新設)。

計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`

既存の`jra_eval.Evaluator`/`jra_backtest.BoxSettler`は無改造で再利用する。新規に必要なのは
「連続スコア調整によるpicks構築(adjusted_picks)」の1関数のみ(`jra_flop_signals.py`の
`flop_precision_table`と合わせて、統計的検証に必要な最小限の道具立て)。

評価設計(シニアエンジニアレビューを反映済み): 当初「BOX5から最も危険な1頭を機械的に除外して
BOX4にする」という二値除外設計だったが、以下2点の理由でレビューにより「連続スコア調整」方式
(adjusted_score = score - alpha*flop_score、常にbox_n頭ちょうどをargsortで選ぶ)へ変更した。
  1. 除外後の頭数がbox_nと一致しない分岐(全候補NaN時)で`BoxSettler`の決済テーブル引きが
     KeyErrorでクラッシュする実装上の欠陥があった(box_n長のタプルキーしか存在しないため)。
  2. 「独立最適化済みのBOX4モデル」を比較対象にすると、flopシグナル自体の価値ではなく
     「BOX5ランキング vs BOX4ランキングの優劣」を測ってしまう交絡があった。
連続スコア調整方式では、alpha=0のとき`jra_eval.score_picks()`と完全に同一のpicksを返すため
(単体テストで保証)、比較対象は常に「同一ランキング上のalpha=0(現行) vs alpha>0(flop加味)」
という単一変数アブレーションになり、`jra_signal_gate_v4_2026_08_28.py`のG1と同型の
比較構造になる。
"""
import numpy as np


def adjusted_picks(mats_score: list, w_score: np.ndarray, mats_flop: list, w_flop: np.ndarray,
                   alpha: float, box_n: int) -> list:
    """score(_scoreと同じ計算式) から alpha*flop_score を引いた調整済みスコアで
    上位box_n頭を選ぶ。常にちょうどbox_n頭を返すため、Evaluator/BoxSettlerは無改造で使える。

    alpha=0のとき jra_eval.score_picks(mats_score, w_score, box_n) と完全に同一の結果を
    返すことをテストで保証する(tests/test_jra_flop_signals.py参照)。

    mats_score/mats_flopはレース順序が一致している前提(同じ母集団races配列から
    jra_signals.signal_matrices() / jra_flop_signals.flop_signal_matrices() で
    それぞれ作ったものを渡す)。
    """
    picks = []
    for ms, mf in zip(mats_score, mats_flop):
        num, den = ms["S"] @ w_score, ms["A"] @ w_score
        score = np.where(den > 0, num / den, -1e18)

        fnum, fden = mf["S"] @ w_flop, mf["A"] @ w_flop
        # 情報なし(fden<=0)は危険側に倒さず中立(0、調整量ゼロ)にする。
        # jra_eval.score_picks()の-1e18センチネル(スコア計算で情報皆無の馬を最下位に
        # 追いやる用途)とは意味が異なる点に注意。
        flop = np.divide(fnum, fden, out=np.zeros_like(fnum, dtype=float), where=fden > 0)

        adj = score - alpha * flop
        picks.append(np.argsort(-adj, kind="stable")[:box_n])
    return picks


# ============================================================================
# FLOP_SIGNALS_V2網羅探索用(2026-09-11、計画: valiant-cuddling-aho.md)
#
# `jra_eval.selection_optimism`/box3テンプレートの`fit_fn`は、いずれも「単一の比
# score=(S@w)/(A@w)」を前提にした実装(`jra_eval.score_picks`ベース)であり、
# `adjusted_score = score_fixed - alpha*flop_w`(2つの独立した比の差)には数学的に
# そのまま適用できない(シニアエンジニアレビュー指摘C1/C2)。以下の2関数は「流用」ではなく、
# 同じ統計的手続き(split-half選択バイアス診断、group_kfold_oof_genericの
# predict_fn(train_idx, test_idx)契約)をadjusted_picksベースで作り直した新規実装。
# ============================================================================

def flop_selection_optimism(ev, mats_score: list, w_score: np.ndarray, mats_flop: list,
                            W_flop: np.ndarray, alpha: float, n_rep: int = 200, seed: int = 99,
                            bets=None, max_payout: float = None) -> dict:
    """`jra_eval.selection_optimism`のadjusted_picks版。mats_score/w_scoreは固定
    (baseline側の本番重み)、W_flopは列ごとに1パターンのDirichlet重みプール、alphaは
    固定スカラー。ブロックを半分に割り、片側で最良のflop重みパターンを選び、もう片側で
    その重みを評価する(split-halfロジック自体は`jra_eval.selection_optimism`と同一、
    pick生成部分だけ`adjusted_picks`に差し替えている)。
    戻り値のキーは`jra_eval.selection_optimism`と同一(true_edge_pt/true_edge_sd/win_rate等)。
    """
    import jra_eval as JE  # 遅延import(循環import回避、jra_eval側はflop_evalに依存しない)

    bets = JE.OBJ_BETS if bets is None else bets
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    n_patterns = W_flop.shape[1]
    all_picks = [adjusted_picks(mats_score, w_score, mats_flop, W_flop[:, j], alpha, ev.box_n)
                for j in range(n_patterns)]
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
        va = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=bets, idx=a,
                                             max_payout=max_payout) for j in range(n_patterns)])
        vb = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=bets, idx=b,
                                             max_payout=max_payout) for j in range(n_patterns)])
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


def make_flop_predict_fn(ev, mats_score: list, w_score: np.ndarray, mats_flop: list,
                         W_flop: np.ndarray, alpha: float):
    """`Evaluator.group_kfold_oof_generic(predict_fn, ...)`に渡す
    `predict_fn(train_idx, test_idx) -> (test_picks, chosen)`アダプタを作る。

    box3テンプレートの`fit_fn(train_idx) -> (w, pattern_idx)`(lobo_oof用の契約)とは
    引数・返り値とも異なる新規実装(レビュー指摘C2)。train_idxで
    `cost_weighted_rate`最大のflop重みパターンを選び、test_idxについて実際に
    `adjusted_picks`でpicksを作って返す。
    """
    import jra_eval as JE

    n_patterns = W_flop.shape[1]
    all_picks = [adjusted_picks(mats_score, w_score, mats_flop, W_flop[:, j], alpha, ev.box_n)
                for j in range(n_patterns)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)

    def predict_fn(train_idx, test_idx):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx)
                         for j in range(n_patterns)])
        best = int(np.argmax(vals))
        test_picks = [all_picks[best][i] for i in test_idx]
        return test_picks, best

    return predict_fn
