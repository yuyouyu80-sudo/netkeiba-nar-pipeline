# -*- coding: utf-8 -*-
"""Front2: アンサンブル(重みブレンド)ユーティリティ(2026-08-21新設)。box・axis共通。

`jra_signals.combine_signals()`/`score_race()`は重みをフラットなdict(またはNAMES順の
np.ndarray)として消費するだけなので、複数の候補重みベクトルを加重平均でブレンドした
1本のベクトルを作るだけで、コア関数側の変更は一切不要(2026-08-21のExplore調査で確認済み)。

top_N・類似度閾値のグリッド探索はしない(多重比較を増やさないため、既存の500パターン探索の
上にさらに多重比較を積み上げない設計方針)。top_N=8, max_cos_sim=0.9の1組に固定する。
"""
import numpy as np


def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    """2つの重みベクトルのコサイン類似度。いずれかがゼロベクトルの場合は0.0を返す。"""
    n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(w1, w2) / (n1 * n2))


def select_diverse_topn(W_POOL: np.ndarray, scores: np.ndarray, top_n: int = 8,
                        max_cos_sim: float = 0.9) -> list:
    """scores(500パターンのin-sampleスコア等)降順に貪欲選択し、既に選んだパターンいずれかとの
    コサイン類似度がmax_cos_sim超なら候補から除外する(似た重みばかり集めて多様性を失わない
    ため)。top_n件選べた時点、またはスコア降順の候補を使い切った時点で終了する
    (使い切った場合はtop_n未満になりうる)。戻り値はW_POOLの列indexのリスト(スコア降順)。"""
    order = np.argsort(-scores, kind="stable")
    selected: list = []
    for idx in order:
        if len(selected) >= top_n:
            break
        w = W_POOL[:, idx]
        if all(cosine_similarity(w, W_POOL[:, s]) <= max_cos_sim for s in selected):
            selected.append(int(idx))
    return selected


def blend_weights(W_POOL: np.ndarray, indices: list) -> np.ndarray:
    """指定した列indexの重みベクトル群を単純平均でブレンドする。W_POOLの各列は
    Dirichlet出力(sum=1)またはそれに準じる重みベクトルなので、単純平均で十分
    (加重平均が必要な特別な理由がない限りscore_weightedのような複雑化はしない)。"""
    if not indices:
        raise ValueError("indicesが空です")
    return W_POOL[:, indices].mean(axis=1)


def ensemble_fit_fn_factory(NAMES: list, score_fn, top_n: int = 8, max_cos_sim: float = 0.9):
    """lobo_oof/chronological_oofのfit_fn(train_idx) -> (w, pattern_id_tuple)契約に適合する
    アンサンブルfit_fnを作るファクトリ。score_fn(train_idx) -> (W_POOL, scores)という
    「学習側インデックスでの500パターンのスコア配列」を計算する関数を受け取る(呼び出し側が
    W_POOL・評価済みstake/returnをクロージャで持っているため、ここでは計算しない)。
    pattern_id_tupleは選択された列indexのソート済みタプル(LOBO退化検知に使う、
    同じ組み合わせが選ばれ続けていないかを見る)。"""
    def fit_fn(train_idx):
        W_POOL, scores = score_fn(train_idx)
        selected = select_diverse_topn(W_POOL, scores, top_n=top_n, max_cos_sim=max_cos_sim)
        blended = blend_weights(W_POOL, selected)
        return blended, tuple(sorted(selected))
    return fit_fn
