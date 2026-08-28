# -*- coding: utf-8 -*-
"""正則化(L1/L2)ロジスティック回帰による組合せ探索モジュール(2026-08-28、JRA Stage2)。
scripts/nar_model/nar_logistic.pyのJRA移植版。scikit-learnのLogisticRegressionを
フィッティング本体として使い、標準化・特徴量構築・OOF契約(jra_eval.Evaluator.
group_kfold_oof_genericのpredict_fn)はこのプロジェクト固有のロジックとして実装する。

既存 jra_signals.py / jra_eval.py / jra_dataset.py / jra_backtest.py はすべて無改造で
参照するのみ。

**NARとの設計差分(JRA Stage2計画のPhase J2、レビュー2のJRA移植提言を反映)**:
NAR Stage1は内側CV(train_idxをさらに5分割してC_gridを評価)によるネストCVだったが、
JRAは42ブロックしかなくNARの約140ブロックの3分の1以下のため、内側CVの評価ノイズは
理論上√(1643/246)≈2.6倍に拡大する。NAR側で実測済みの「C選択は事実上ノイズ
(Ridge全12グリッドの回収率幅0.69pt、Lassoは隣接格子間3.11ptの単一点スパイク、
いずれもブロックブートストラップCI半幅±3.8ptに埋もれる)」という問題は、ブロック数が
少ないJRAでは更に悪化すると予想される。よってJRAでは**ネストCV(内側CV)を行わない**:
`select_fixed_c()`で全データ1回だけの5分割CVによりC候補を参考評価し、事前に1点(または
少数)を固定して`make_fixed_c_fit_fn()`の外側group_kfold_oof_generic評価にのみ使う。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import jra_eval as JE


def race_row_ranges(mats: list) -> list:
    """各レースの行がrace_of_row内で占める(start, end)半開区間を返す(nar_logistic.pyと同一)。"""
    ranges = []
    pos = 0
    for m in mats:
        n = m["S"].shape[0]
        ranges.append((pos, pos + n))
        pos += n
    return ranges


def build_feature_matrix(mats: list, mats_names: list, names: list,
                         impute: str = "zero") -> tuple:
    """matsを縦積みしてX(全馬×len(names))を作る。戻り値: (X, race_of_row)。
    nar_logistic.build_feature_matrixと同一設計(mats_namesとnamesを分ける理由も同じ)。"""
    if impute != "zero":
        raise NotImplementedError(f"impute={impute!r} is not implemented")
    col_idx = [mats_names.index(n) for n in names]
    X_parts, race_of_row = [], []
    for ri, m in enumerate(mats):
        A = m["A"][:, col_idx]
        S = m["S"][:, col_idx]
        Xi = np.where(A > 0, S, np.nan)
        X_parts.append(Xi)
        race_of_row.append(np.full(Xi.shape[0], ri))
    return np.vstack(X_parts), np.concatenate(race_of_row)


def build_labels(races: list, actual: dict, label: str = "place3") -> np.ndarray:
    """build_feature_matrixと同じ行順の0/1ラベル配列。label="win"|"place3"(既定)|"place2"。
    JRAの払戻データはactual[race_id]["単勝"/"複勝"]がumaban単位キーの辞書
    (jra_dataset.parse_comboの単勝/複勝はint(combo_text)のためnar_dataset.pyと同型)。"""
    out = []
    for r in races:
        u = r["df"]["umaban"].astype(int).to_numpy()
        act = actual.get(r["race_id"], {})
        win_set = set(act.get("単勝", {}).keys())
        place_set = set(act.get("複勝", {}).keys())
        if label == "win":
            lab = np.isin(u, list(win_set)).astype(float)
        elif label == "place2":
            lab = (np.isin(u, list(place_set)) & ~np.isin(u, list(win_set))).astype(float)
        else:
            lab = np.isin(u, list(place_set)).astype(float)
        out.append(lab)
    return np.concatenate(out)


def standardize_fit(X: np.ndarray, train_rows: np.ndarray) -> dict:
    """学習行のみからmean/stdを算出(fold内完結)。std<=0の列は1.0にフォールバック。
    全NaN列は明示的にmean=0/std=1にフォールバックする(nar_logistic.py 2026-08-28修正と
    同一、RuntimeWarning回避+all_nan_cols診断フィールド。JRAはV4シグナル充足率94.5〜100%
    だが予想印は約半数のレースにしか無いため、fold次第では発火しうる想定で最初から導入)。"""
    Xt = X[train_rows]
    all_nan = np.all(np.isnan(Xt), axis=0)
    Xt_safe = Xt.copy()
    if all_nan.any():
        Xt_safe[:, all_nan] = 0.0
    mean = np.nanmean(Xt_safe, axis=0)
    std = np.nanstd(Xt_safe, axis=0)
    mean = np.where(all_nan, 0.0, mean)
    std = np.where(all_nan, 1.0, std)
    std = np.where(std > 1e-12, std, 1.0)
    return {"mean": mean, "std": std, "all_nan_cols": np.where(all_nan)[0].tolist()}


def standardize_apply(X: np.ndarray, params: dict) -> np.ndarray:
    """標準化した上でNaN(欠損シグナル)を0埋めする(=標準化後の平均相当、寄与ゼロ)。"""
    Z = (X - params["mean"]) / params["std"]
    return np.nan_to_num(Z, nan=0.0)


def fit_logistic(X: np.ndarray, y: np.ndarray, penalty: str, C: float) -> LogisticRegression:
    """sklearn.LogisticRegressionのラッパー(nar_logistic.fit_logisticと同一、sklearn 1.9.0の
    penalty= deprecation対応済み)。penalty="l1"はliblinear、"l2"はlbfgs。"""
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=2000)
    model.fit(X, y)
    return model


def score_from_model(X: np.ndarray, model: LogisticRegression) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def picks_from_scores(scores: np.ndarray, ranges: list, race_indices, box_n: int) -> list:
    """scores(build_feature_matrixの行順)からrace_indices順にレースごとの上位box_n頭の
    レース内行インデックスを切り出す(nar_logistic.picks_from_scoresと同一)。"""
    picks = []
    for ri in race_indices:
        start, end = ranges[ri]
        s = scores[start:end]
        k = min(box_n, len(s))
        picks.append(np.argsort(-s, kind="stable")[:k])
    return picks


def score_picks_logistic(mats: list, mats_names: list, model: LogisticRegression,
                         std_params: dict, box_n: int, names: list,
                         impute: str = "zero") -> list:
    """全レースについて、確定済みモデルでpicksを作る(本番反映・最終fit係数の適用に使う)。"""
    X, race_of_row = build_feature_matrix(mats, mats_names, names, impute=impute)
    Z = standardize_apply(X, std_params)
    scores = score_from_model(Z, model)
    return picks_from_scores(scores, race_row_ranges(mats), range(len(mats)), box_n)


def _fast_excess(ev_full: JE.Evaluator, cols: list, test_race_idx, test_picks: list) -> float:
    """idx分のレースだけでcost_weighted_rateの市場超過ptを計算する軽量版
    (nar_logistic._fast_excessと同一設計、s_sum/ms_sum==0時は-np.infを返す2026-08-28修正済み
    のバージョンを最初から採用)。jra_eval.Evaluator.settler.tablesを直接読む。"""
    s_sum = r_sum = ms_sum = mr_sum = 0
    tables = ev_full.settler.tables
    mkt_stake, mkt_ret = ev_full.mkt_stake, ev_full.mkt_ret
    for pos, ri in enumerate(test_race_idx):
        key = tuple(sorted(int(x) for x in test_picks[pos]))
        st, rt = tables[ri][key]
        s_sum += st[cols].sum()
        r_sum += rt[cols].sum()
        ms_sum += mkt_stake[ri, cols].sum()
        mr_sum += mkt_ret[ri, cols].sum()
    if not s_sum or not ms_sum:
        return -np.inf
    model = r_sum / s_sum * 100
    market = mr_sum / ms_sum * 100
    return model - market


def select_fixed_c(races: list, actual: dict, mats_all: list, mats_names: list, names: list,
                   method: str, C_grid, box_n: int, label: str = "place3",
                   n_folds: int = 5, seed: int = 0) -> dict:
    """JRA版のC選択(2026-08-28、レビュー2のJRA移植提言により内側ネストCVの代わりに採用)。
    全データを対象に1回だけn_folds分割CVを行い、C_grid各点の平均市場超過ptを参考値として
    返す。**この結果から選んだCは、その後のgroup_kfold_oof_generic評価(外側OOF)には
    再最適化として使わず、事前に固定した1点として渡すこと**(NARで確認済みの「C選択自体が
    ノイズ」問題を、JRAでは反復最適化しないことで最初から避ける設計)。
    戻り値: {"c_scores": {C: excess_pt, ...}, "best_c": C}(best_cは参考値、採用は呼び出し側の
    事前登録次第)。"""
    penalty = "l1" if method == "lasso" else "l2"
    X, race_of_row = build_feature_matrix(mats_all, mats_names, names)
    ranges = race_row_ranges(mats_all)
    y = build_labels(races, actual, label=label)
    blocks = JE.blocks_of(races)
    block_ids = sorted(set(blocks))

    ev_full = JE.Evaluator(races, actual, box_n=box_n)
    cols = [JE.JB.BET_TYPES.index(b) for b in JE.OBJ_BETS]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block_ids))
    cv_folds = [order[i::n_folds] for i in range(n_folds)]

    c_scores = {}
    for C in C_grid:
        fold_scores = []
        for fold in cv_folds:
            test_blocks = {block_ids[i] for i in fold}
            test_race_idx = np.array([i for i in range(len(races)) if blocks[i] in test_blocks])
            train_race_idx = np.array([i for i in range(len(races)) if blocks[i] not in test_blocks])
            train_rows = np.where(np.isin(race_of_row, train_race_idx))[0]
            y_train = y[train_rows]
            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                continue
            params = standardize_fit(X, train_rows)
            Z_train = standardize_apply(X[train_rows], params)
            model = fit_logistic(Z_train, y_train, penalty, C)
            Z_all = standardize_apply(X, params)
            scores = score_from_model(Z_all, model)
            test_picks = picks_from_scores(scores, ranges, test_race_idx, box_n)
            val = _fast_excess(ev_full, cols, test_race_idx, test_picks)
            if np.isfinite(val):
                fold_scores.append(val)
        c_scores[C] = float(np.mean(fold_scores)) if fold_scores else float("-inf")
    best_c = max(c_scores, key=c_scores.get)
    return {"c_scores": c_scores, "best_c": best_c}


def make_fixed_c_fit_fn(races: list, actual: dict, mats_all: list, mats_names: list,
                        names: list, method: str, C: float, box_n: int,
                        label: str = "place3"):
    """group_kfold_oof_generic用のpredict_fn(train_idx, test_idx) -> (test_picks, chosen)を
    返すファクトリ(JRA固定C版、ネストCVなし)。Cは事前に固定した1点(select_fixed_cの結果を
    参考に、呼び出し側が明示的に選ぶ)。chosen=C(固定値、fold間で常に同じになるのが正しい
    ——NARのRidge/Lassoのように内側CVでfold毎に違うCを選ぶ設計ではないことの確認に使う)。
    train_idxが単一クラスラベルしか持たない場合(理論上は起こりうるが、JRAの246レース規模
    ではplace3陽性率が1レースあたり3/12程度あるため現実的にはまず発生しない)は等重み
    フォールバックでpicksを作る(サイレントな失敗を避けるため警告をprintする)。"""
    penalty = "l1" if method == "lasso" else "l2"
    X, race_of_row = build_feature_matrix(mats_all, mats_names, names)
    ranges = race_row_ranges(mats_all)
    y = build_labels(races, actual, label=label)

    def predict_fn(train_idx: np.ndarray, test_idx: np.ndarray):
        train_rows = np.where(np.isin(race_of_row, train_idx))[0]
        y_train = y[train_rows]
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"[jra_logistic] 警告: train_idx({len(train_idx)}レース)がplace3単一クラス"
                  f"のためfit不能、等重みフォールバックでこのfoldを評価します。")
            equal_score = np.ones(len(names))
            X_test, _ = build_feature_matrix([mats_all[i] for i in test_idx], mats_names, names)
            X_test = np.nan_to_num(X_test, nan=0.0)
            scores = X_test @ equal_score
            test_ranges = race_row_ranges([mats_all[i] for i in test_idx])
            test_picks = picks_from_scores(scores, test_ranges, range(len(test_idx)), box_n)
            return test_picks, C
        params = standardize_fit(X, train_rows)
        Z_train = standardize_apply(X[train_rows], params)
        model = fit_logistic(Z_train, y_train, penalty, C)
        Z_all = standardize_apply(X, params)
        scores = score_from_model(Z_all, model)
        test_picks = picks_from_scores(scores, ranges, test_idx, box_n)
        return test_picks, C

    return predict_fn
