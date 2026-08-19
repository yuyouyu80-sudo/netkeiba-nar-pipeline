# -*- coding: utf-8 -*-
"""JRA/NAR共通の「確信度」較正ロジック — 単一の真実の源。

2026-08-12、地方(NAR)とJRAで確信度バッジが別々の統計量(JRA=較正なしの生スコア差、
NAR=LOBO較正済み実測的中率)を表示していたことが判明し、統一する過程で新設した。
統計学者・システムエンジニアの2専門家レビューを踏まえ、以下の設計にしている:

  * weights・blocks・hit列は必ず引数で渡す。モジュールグローバルへの暗黙依存は作らない
    (jra_signals.py/nar_signals.pyと同じ設計思想)。
  * 主手法はロジスティック回帰(Platt scaling、自由度2)。4分位バケット法は参考診断として
    残すが、実際にデプロイする較正表としては採用しない(JRA30ブロック・NAR100ブロック弱という
    規模では、バケット法はバケットあたりのhit率の95%CI幅が±10〜15ptに達し、実データで
    非単調な逆転(バケット間の大小が理論と逆)が確認されたため)。
  * バケット法を診断目的で使う場合も、経験ベイズ・シュリンケージ
    ((n*raw + shrink_k*overall) / (n+shrink_k))で小標本の極端値を全体平均へ引き寄せる。
    shrink_kはLOBOの内側でグリッドサーチする。
  * 「自明な基準(ブロック平均をそのまま予測)を上回るか」は、OOF Brier scoreの点推定の大小
    比較ではなく、ブロック単位ブートストラップ(復元抽出、レース単位ではなくブロック単位で
    抽出することで開催日×競馬場内の相関を尊重する)によるBrier差の95%CIで判定する
    (`beats_trivial_baseline = ci95_lo > MIN_CI_LO`)。
  * ブロック総数がMIN_BLOCKS_FOR_PCT未満の場合は、CIが有意であっても較正済み%を表示せず
    3段階(高/中/低)フォールバックにする(`min_blocks_ok`フラグ)。
"""
from typing import Sequence

import numpy as np
import pandas as pd

K_BUCKETS_DEFAULT = 4
SHRINK_K_GRID_DEFAULT = (10, 20, 40, 80)
MIN_BLOCKS_FOR_PCT = 60
MIN_CI_LO = 0.0
N_BOOT_DEFAULT = 2000


# --------------------------------------------------------------------------- gap特徴量
def gap_features(sorted_scores: np.ndarray, ladder_ks: Sequence[int]) -> dict:
    """降順ソート済みスコア配列から、gap_top2・gap_boundary_k(k in ladder_ks)を計算する。
    頭数不足やスコア差ゼロの場合は「箱が全頭を覆う」=最大確信として1.0を返す
    (gap_top2は0.0、これはNAR/JRA双方の既存実装を踏襲)。"""
    n = len(sorted_scores)
    spread = float(sorted_scores[0] - sorted_scores[-1]) if n > 0 else 0.0
    out = {
        "gap_top2": float((sorted_scores[0] - sorted_scores[1]) / spread)
        if (n > 1 and spread > 0) else 0.0,
        "spread": spread,
    }
    for k in ladder_ks:
        out[f"gap_boundary_{k}"] = (
            float((sorted_scores[k - 1] - sorted_scores[k]) / spread)
            if (n > k and spread > 0) else 1.0
        )
    return out


# --------------------------------------------------------------------------- 基礎統計
def _brier(pred: np.ndarray, hit: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred, dtype=float) - np.asarray(hit, dtype=float)) ** 2))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def _choose_feature_sign_safe(cand_results: dict) -> tuple:
    """OOF Brierが良い順に候補を並べ、Spearman相関が非負の中から最良を選ぶ。
    非負の候補が1つも無い場合のみ、全候補中の最良(符号無視)を警告付きで返す。"""
    ranked = sorted(cand_results.items(), key=lambda kv: kv[1]["oof_brier"])
    for feat, r in ranked:
        if r["spearman_with_hit"] >= 0:
            return feat, False
    return ranked[0][0], True


# --------------------------------------------------------------------------- bucket法(診断用)
def _bucket_table_shrunk(vals: np.ndarray, hits: np.ndarray, k_buckets: int, shrink_k: float):
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, k_buckets + 1)))
    edges = qs[1:-1]
    bucket_idx = np.digitize(vals, edges)
    overall = float(hits.mean())
    table = {}
    for bk in np.unique(bucket_idx):
        mask = bucket_idx == bk
        n = int(mask.sum())
        raw = float(hits[mask].mean())
        table[int(bk)] = (n * raw + shrink_k * overall) / (n + shrink_k)
    return edges, table, overall


def _lobo_oof_bucket(vals, hits, blocks, k_buckets, shrink_k) -> np.ndarray:
    oof = np.empty(len(vals))
    for b in sorted(set(blocks)):
        tr, te = blocks != b, blocks == b
        if tr.sum() < k_buckets:
            oof[te] = hits[tr].mean() if tr.sum() else hits.mean()
            continue
        edges, table, overall = _bucket_table_shrunk(vals[tr], hits[tr], k_buckets, shrink_k)
        test_bucket = np.digitize(vals[te], edges)
        oof[te] = [table.get(int(bk), overall) for bk in test_bucket]
    return oof


def _best_shrink_k_bucket(vals, hits, blocks, k_buckets, shrink_k_grid) -> tuple:
    """LOBOの内側でshrink_kをグリッドサーチし、OOF Brierを最小化する値を選ぶ。"""
    best_k, best_brier, best_oof = None, np.inf, None
    for sk in shrink_k_grid:
        oof = _lobo_oof_bucket(vals, hits, blocks, k_buckets, sk)
        b = _brier(oof, hits)
        if b < best_brier:
            best_k, best_brier, best_oof = sk, b, oof
    return best_k, best_brier, best_oof


# --------------------------------------------------------------------------- logistic法(主手法)
def _fit_logistic_1d(x: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 50) -> np.ndarray:
    """1変数ロジスティック回帰(切片+傾き)をNewton-Raphson(IRLS)でfitする。
    L2正則化(l2)により準分離(バケット的中率が0/1近くに偏る等)でも発散しない。
    sklearn等の追加依存を避けるためnumpyのみで実装する。"""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = p * (1 - p) + 1e-9
        grad = X.T @ (y - p) - l2 * beta
        H = -(X.T * w) @ X - l2 * np.eye(2)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return beta


def _predict_logistic_1d(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = beta[0] + beta[1] * x
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _lobo_oof_logistic(vals, hits, blocks) -> np.ndarray:
    oof = np.empty(len(vals))
    for b in sorted(set(blocks)):
        tr, te = blocks != b, blocks == b
        if hits[tr].std() == 0 or len(np.unique(vals[tr])) < 2:
            oof[te] = hits[tr].mean() if tr.sum() else hits.mean()
            continue
        beta = _fit_logistic_1d(vals[tr], hits[tr])
        oof[te] = _predict_logistic_1d(vals[te], beta)
    return oof


# --------------------------------------------------------------------------- ブートストラップCIゲート
def _block_bootstrap_brier_gain_ci(hits, trivial_oof, chosen_oof, blocks, n_boot, seed=0):
    """trivial_oofとchosen_oofのBrier差(trivial - chosen、正なら較正が勝つ)を
    ブロック単位ブートストラップ(復元抽出)で95%CI推定する。"""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(blocks)))
    by_block = {b: np.where(blocks == b)[0] for b in uniq}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_block[b] for b in sample])
        diffs[i] = _brier(trivial_oof[idx], hits[idx]) - _brier(chosen_oof[idx], hits[idx])
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p_le0 = float(np.mean(diffs <= 0))
    return lo, hi, p_le0


# --------------------------------------------------------------------------- 公開関数
def lobo_bucket_calibrate(
    df: pd.DataFrame,
    blocks: np.ndarray,
    hit_col: str,
    feature_candidates: Sequence[str] = ("gap_boundary_k",),
    sign_safe: bool = True,
    method: str = "logistic",
    k_buckets: int = K_BUCKETS_DEFAULT,
    shrink_k_grid: Sequence[float] = SHRINK_K_GRID_DEFAULT,
    hit_definition_label: str = None,
    n_boot: int = N_BOOT_DEFAULT,
    min_blocks_for_pct: int = MIN_BLOCKS_FOR_PCT,
    min_ci_lo: float = MIN_CI_LO,
    seed: int = 0,
) -> dict:
    """LOBO較正を行い、採用特徴量・較正パラメータ・ブートストラップCIゲート結果を返す。

    feature_candidatesが1つだけの場合(推奨: topk_ladderのように用途ごとに理論的動機のある
    特徴量を事前登録する場合)は、候補間の選択バイアス自体が発生しない。複数指定した場合は
    Spearman符号セーフな選択(NAR/JRA既存ロジックを踏襲)を行うが、ブロックbのOOF評価に
    ブロックbを含む全体集計が使われる点で軽微な選択バイアスが残る(候補が2つ程度なら実務上の
    影響は小さい)。
    """
    hits = df[hit_col].to_numpy(dtype=float)
    n_blocks_total = len(set(blocks))

    # 自明な基準(ブロック平均)のOOF
    trivial_oof = np.empty(len(df))
    for b in sorted(set(blocks)):
        tr, te = blocks != b, blocks == b
        trivial_oof[te] = hits[tr].mean() if tr.sum() else hits.mean()
    trivial_brier = _brier(trivial_oof, hits)

    # 候補特徴量ごとにOOF較正(診断: bucket法。主手法: logistic法)を評価
    cand_results = {}
    cand_oof = {}
    cand_extra = {}
    for feat in feature_candidates:
        vals = df[feat].to_numpy(dtype=float)
        if method == "bucket":
            best_sk, _, oof = _best_shrink_k_bucket(vals, hits, blocks, k_buckets, shrink_k_grid)
            cand_extra[feat] = {"shrink_k": best_sk}
        elif method == "logistic":
            oof = _lobo_oof_logistic(vals, hits, blocks)
            cand_extra[feat] = {}
        else:
            raise ValueError(f"unknown method: {method}")
        b = _brier(oof, hits)
        sp = _spearman(vals, hits)
        cand_results[feat] = {"oof_brier": b, "spearman_with_hit": sp}
        cand_oof[feat] = oof

    if sign_safe:
        chosen_feature, sign_warning = _choose_feature_sign_safe(cand_results)
    else:
        chosen_feature = min(cand_results, key=lambda f: cand_results[f]["oof_brier"])
        sign_warning = cand_results[chosen_feature]["spearman_with_hit"] < 0
    chosen_oof = cand_oof[chosen_feature]
    chosen_brier = cand_results[chosen_feature]["oof_brier"]

    ci_lo, ci_hi, p_le0 = _block_bootstrap_brier_gain_ci(
        hits, trivial_oof, chosen_oof, blocks, n_boot=n_boot, seed=seed)
    min_blocks_ok = n_blocks_total >= min_blocks_for_pct
    beats_trivial_baseline = ci_lo > min_ci_lo
    show_pct = bool(min_blocks_ok and beats_trivial_baseline)

    # 本番適用用に全データで最終fit(LOBOはあくまで手法検証用)
    vals_full = df[chosen_feature].to_numpy(dtype=float)
    if method == "bucket":
        shrink_k = cand_extra[chosen_feature]["shrink_k"]
        edges, table, overall = _bucket_table_shrunk(vals_full, hits, k_buckets, shrink_k)
        fit_params = {
            "type": "bucket", "shrink_k": shrink_k, "k_buckets": k_buckets,
            "edges": edges.tolist(), "bucket_rate": {str(k): v for k, v in table.items()},
            "overall_rate": overall,
        }
    else:
        beta = _fit_logistic_1d(vals_full, hits)
        fit_params = {"type": "logistic", "intercept": float(beta[0]), "slope": float(beta[1])}

    return {
        "n_races": len(df), "n_blocks_total": n_blocks_total,
        "overall_hit_rate_pct": round(float(hits.mean()) * 100, 1),
        "hit_definition_label": hit_definition_label,
        "method": method, "k_buckets": k_buckets if method == "bucket" else None,
        "candidates_tested": list(feature_candidates), "candidates": cand_results,
        "chosen_feature": chosen_feature, "chosen_feature_sign_warning": bool(sign_warning),
        "trivial_baseline_oof_brier": trivial_brier, "chosen_oof_brier": chosen_brier,
        "brier_gain_ci95": [ci_lo, ci_hi], "p_value_gain_le0": p_le0,
        "min_blocks_for_pct": min_blocks_for_pct, "min_blocks_ok": min_blocks_ok,
        "beats_trivial_baseline": beats_trivial_baseline, "show_pct": show_pct,
        "fit_params": fit_params,
    }


def apply_calibration(values: pd.Series, result: dict) -> pd.Series:
    """lobo_bucket_calibrateの戻り値(本番fit済みfit_params)を新しい値に適用する。
    show_pct=Falseの場合は呼び出し側で3段階フォールバック表示に切り替えること
    (このモジュールはパーセントの計算のみ担当し、表示方針は持たない)。"""
    fp = result["fit_params"]
    v = values.to_numpy(dtype=float)
    if fp["type"] == "logistic":
        pred = _predict_logistic_1d(v, np.array([fp["intercept"], fp["slope"]]))
        return pd.Series(pred * 100, index=values.index)
    edges = np.array(fp["edges"])
    bucket_rate = {int(k): v_ for k, v_ in fp["bucket_rate"].items()}
    idx = np.digitize(v, edges)
    pred = [bucket_rate.get(int(b), fp["overall_rate"]) for b in idx]
    return pd.Series(np.array(pred) * 100, index=values.index)


def tier_label(value: float, tercile_edges: tuple) -> str:
    """show_pct=Falseの場合の3段階フォールバック表示('低確信度'/'中確信度'/'高確信度')。
    tercile_edgesは直近データ内でのgap_boundary_k等の三分位境界(呼び出し側で算出)。"""
    lo, hi = tercile_edges
    if value >= hi:
        return "高確信度"
    if value <= lo:
        return "低確信度"
    return "中確信度"
