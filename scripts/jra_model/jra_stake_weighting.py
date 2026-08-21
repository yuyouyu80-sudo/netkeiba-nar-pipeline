# -*- coding: utf-8 -*-
"""Front3(2)(3): 確信度に基づくステーク配分(2026-08-21新設)。

既存の「高確信度Nレース/日」フィルタ(confidence_sweep_axis{5,4,3}.py等)は、日次グループ内で
確信度指標(gap_pct、スコア上位box_n位と次点のギャップをレース内スプレッドで正規化した値)の
上位N件だけを採用する0/1(採用/非採用)選抜だった。本モジュールはこれを連続的なステーク乗数へ
一般化する: 採用/非採用の二値ではなく、日次グループ内の確信度パーセンタイル順位に応じて
[lo, hi]区間の乗数を線形に割り当てる(確信度が高いレースほど多く賭け、低いレースほど
少なく賭ける)。

決済エンジン(jra_backtest.py/jra_axis_backtest.py)自体は変更しない。乗数は
jra_eval.Evaluator/jra_axis_eval.Evaluatorの`evaluate`/`block_bootstrap`/`block_bootstrap_diff`
に追加した`multipliers`引数(2026-08-21Front0)へレースごとの配列として渡すことで、
stake/returnへ事後的にelement-wise乗算される(数学的に等価、決済エンジンのキャッシュ構造は
壊さない)。

**注意**: JRAの確信度較正(topk_ladder、jra_confidence_calibrate.py)は現時点で統計的に
未証明(全K段階がbeats_trivial_baseline/min_blocks_okのいずれかのゲート未達)。よって本モジュールが
使う`gap_pct`は「較正済み的中確率」ではなく生のスコアギャップ順位であり、乗数の[lo,hi]自体も
Kelly基準のような理論的根拠を持つ値ではない、素朴な線形割当である点に留意すること
(将来課題: 較正が安定して以降、真のオッズベースKelly式サイジングへ発展させる)。
"""
import numpy as np
import pandas as pd

DEFAULT_LO = 0.5
DEFAULT_HI = 1.5


def multiplier_from_rank(conf_df: pd.DataFrame, group_col: str = "kaisai_date",
                         value_col: str = "gap_pct", lo: float = DEFAULT_LO,
                         hi: float = DEFAULT_HI) -> np.ndarray:
    """conf_df(各レース1行、group_col=グルーピングキー、value_col=確信度指標)を受け取り、
    conf_dfと同じ行順のnp.ndarrayでステーク乗数を返す。グループ内でvalue_colの昇順パーセンタイル
    順位(0=グループ最小, 1=グループ最大)を線形に[lo, hi]へ写像する。value_colがnp.infを含む
    場合(box_nがフィールド全体を覆う等）もrank()は正しく最大値として扱う。グループサイズが
    1件のみの日は乗数=(lo+hi)/2(中央値、順位が定義できないため)。
    """
    if not (0 <= lo <= hi):
        raise ValueError(f"lo<=hi かつ両方0以上である必要があります: lo={lo}, hi={hi}")

    def _rank_group(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n <= 1:
            return pd.Series((lo + hi) / 2.0, index=g.index)
        rank = g[value_col].rank(method="average", ascending=True)
        pct = (rank - 1.0) / (n - 1.0)
        return lo + pct * (hi - lo)

    out = conf_df.groupby(group_col, group_keys=False).apply(_rank_group)
    return out.reindex(conf_df.index).to_numpy(dtype=float)


def binary_from_topn(conf_df: pd.DataFrame, n_cut: int, group_col: str = "kaisai_date",
                     value_col: str = "gap_pct") -> np.ndarray:
    """既存の「高確信度Nレース/日」0/1フィルタと同一の選抜ロジックを、multipliers配列
    (1.0=採用/0.0=非採用)として返す薄いラッパー。multiplier_from_rank方式との比較用。"""
    conf_sorted = conf_df.sort_values(value_col, ascending=False, kind="stable")
    selected_idx = set(conf_sorted.groupby(group_col, group_keys=False).head(n_cut).index)
    return np.array([1.0 if i in selected_idx else 0.0 for i in conf_df.index], dtype=float)
