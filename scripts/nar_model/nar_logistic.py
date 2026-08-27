# -*- coding: utf-8 -*-
"""正則化(L1/L2)ロジスティック回帰による組合せ探索モジュール(2026-08-27、全ファクター
統合・高確信度選抜計画Phase2)。scikit-learnのLogisticRegressionをフィッティング本体として
使い、標準化・特徴量構築・OOF契約(nar_eval.Evaluator.group_kfold_oof_genericのpredict_fn)は
このプロジェクト固有のロジックとして実装する。

既存 nar_signals.py / nar_eval.py / nar_dataset.py / nar_backtest.py はすべて無改造で
参照するのみ。

設計判断:
  * 標準化はtrain側の平均・標準偏差のみで行い(fold内完結、make_priorsと同じ設計思想)、
    標準化してから欠損を0(=標準化後の平均)で埋める。combine()の「欠損は無視して残りで
    再配分」という思想に最も近い扱い。
  * 切片は正則化しない(sklearn既定の標準的な挙動)。
  * ラベルは既定で複勝的中(place3、1レース2-3頭が正例で最も標本が多く学習が安定)。
  * 内側CVもブロック単位(開催日×競馬場)。外側group_kfold_oof_genericのtrain_idxに
    含まれるブロックだけを対象にし、外側test側の情報が内側に漏れないようにする。
  * 内側評価は毎回BoxSettlerを再構築せず、box_n単位で一度だけ作ったEvaluatorに対して
    idx引数で部分評価する(nar_eval.Evaluator.evaluateのidxはレース単位の位置インデックス
    なのでそのまま使える)。C_grid×inner_folds×outer_foldsの掛け算で呼び出し回数が
    多くなるため、この最適化を必須とする。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import nar_eval as NE


def race_row_ranges(mats: list) -> list:
    """各レースの行がrace_of_row内で占める(start, end)半開区間を返す(build_feature_matrix
    はレース順に縦積みするため各レースの行は連続している)。picks_from_scoresの
    `race_of_row == ri`という全行スキャン(O(全行数)をレース数だけ繰り返す設計)を
    O(1)のスライスに置き換えるための事前計算(2026-08-27、内側CVが遅すぎたため追加)。"""
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

    mats_namesはmats(nar_signals.signal_matrices()の出力)が実際に持つ列の並び、namesは
    そこから使いたい列のサブセット(順序はnames順で出力される)。この2引数を分けるのは、
    box_n=5/4/3や複数のsearch呼び出し間で同じmats_all(例: NS.ALL_SIGNALS_V5の全40列)を
    使い回しつつ、呼び出しごとに異なる候補サブセット(names)へ安全に絞り込むため
    (namesをmats自体の構築に使うと、呼び出しごとにsignal_matrices()を再計算する必要が
    生じ非効率なため、事前に一度だけ全列で構築したmatsをここで絞り込む設計にした)。

    race_of_row[k]は行kが元々何番目のレース(matsのインデックス)由来かを示す
    (レースごとに行を束ね直すのに使う)。impute="zero"のみ対応(標準化後に0埋めするため、
    ここではNaNのまま返す — standardize_apply側で0埋めする)。"""
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
    """build_feature_matrixと同じ行順(races走査順)の0/1ラベル配列。
    label="win"(単勝的中umaban) | "place3"(複勝的中umaban、既定) | "place2"
    (複勝的中かつ単勝的中でないumabanで近似、place3よりラベルが粗いため既定はplace3)。"""
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
    """学習行のみからmean/stdを算出(fold内完結)。std<=0の列は1.0にフォールバックする
    (定数列の0除算回避、標準化後も値は0のままなので寄与ゼロという意味は保たれる)。

    2026-08-28追加(レビュー2指摘、JRA Stage2移植時の潜在バグ): train_rows側で全行NaNの
    列があると、np.nanmean/nanstdがRuntimeWarning付きでNaNを返し、standardize_applyの
    nan_to_numで最終的に0埋めされる(=そのfoldでそのシグナルが無言で無効化される)。
    最終結果(0寄与)自体は意図通りだが、NARでは全シグナルの充足率が94%以上のため発火
    しない一方、JRAでは疎なシグナルがinner train(約148レース)で全欠損になりうる。
    ここで明示的に検出してmean=0/std=1にフォールバックし(RuntimeWarningを回避)、
    all_nan_colsとして呼び出し側が診断できるようにする。"""
    Xt = X[train_rows]
    all_nan = np.all(np.isnan(Xt), axis=0)
    Xt_safe = Xt.copy()
    if all_nan.any():
        Xt_safe[:, all_nan] = 0.0  # nanmean/nanstdへのRuntimeWarning回避用ダミー値(下でmean/stdは上書き)
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
    """sklearn.LogisticRegressionのラッパー。penalty="l1"(liblinear)/"l2"(lbfgs)。
    Cは正則化強度の逆数(大きいほど正則化が弱い=弱いシグナルも残す)。

    実装メモ(2026-08-27、インストール時点でsklearn 1.9.0): このバージョンでは
    `penalty=`引数がdeprecatedになっており、`l1_ratio=`(0.0=L2/1.0=L1のelastic-net統一
    パラメータ)への移行が進んでいる。`penalty="l1"`をそのまま渡すと
    「penalty=l1 with l1_ratio=0.0」という不整合警告が出る(default l1_ratio=0.0のまま
    penaltyだけl1指定という食い違いのため)ため、ここでは`l1_ratio`を明示的に指定する
    新API経由で呼ぶ。lbfgsソルバーはl1_ratio=1.0(純L1)を扱えないため、L1はliblinear、
    L2はlbfgsを使う(挙動は実測で旧penalty=引数指定時と一致することを確認済み)。"""
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=2000)
    model.fit(X, y)
    return model


def score_from_model(X: np.ndarray, model: LogisticRegression) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def picks_from_scores(scores: np.ndarray, ranges: list, race_indices, box_n: int) -> list:
    """scores(build_feature_matrixの行順)からrace_indices順にレースごとの上位box_n頭の
    「レース内行インデックス」を切り出す(nar_eval.score_picksと同じ返り値の形)。

    rangesはrace_row_ranges(mats)の出力(各レースの行区間)。以前は`race_of_row == ri`で
    全行スキャンしていたが、レース数×全行数のオーダーで内側CVがボトルネックになったため
    (2026-08-27実測、縮小パラメータのスモークテストでも5分超)、O(1)のスライスに変更した。"""
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


def _fast_excess(ev_full: NE.Evaluator, cols: list, test_race_idx, test_picks: list) -> float:
    """idx分のレースだけでcost_weighted_rateの市場超過ptを計算する軽量版。

    nar_eval.Evaluator.evaluate(picks, idx=...)はpicks_fullを毎回「全レース分」構築して
    BoxSettler.returns_for()で全レースをスキャンしてからidxで絞る設計(idxが小さい部分集合
    でも計算量は全レース分かかる)。内側CV(C_grid×inner_folds×outer_folds)ではこの関数が
    大量に呼ばれるため、settler.tables/mkt_stake/mkt_retをidx分だけ直接読む本関数に
    置き換えて計算量をO(len(idx))に落とす(2026-08-27、実測でボトルネックと判明したため追加。
    ev_full.evaluate()と数値が一致することはPhase2実行前にテスト済み)。"""
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
    # 2026-08-28修正(レビュー2指摘、中程度〜JRA Stage2では重大): 元は s_sum==0 のとき
    # model=0.0(=excess=0-market)を返していた。「賭金が立たない」を「回収率0%」として
    # 扱うと、本来は評価不能な設定(退化したtest_race_idx部分集合など)がexcess=-market
    # というそれなりの値を持ってしまい、真に負のexcessしか出せない他の候補より不当に
    # 選ばれうる(argmaxのC選択・Dirichletパターン選択どちらでも起こりうる)。
    # NARでは内側fold(約200レース)で発火しないことを確認済みだが、JRA Stage2は
    # inner foldが約37レースまで縮むため現実的リスクとして-np.infに変更する
    # (「この候補は評価不能」として明示的にargmaxから排除する)。
    if not s_sum or not ms_sum:
        return -np.inf
    model = r_sum / s_sum * 100
    market = mr_sum / ms_sum * 100
    return model - market


def make_fit_fn(races: list, actual: dict, mats_all: list, mats_names: list, names: list,
                method: str, C_grid, box_n: int, label: str = "place3",
                inner_n_folds: int = 5, seed: int = 0):
    """nar_eval.Evaluator.group_kfold_oof_generic用のpredict_fn(train_idx, test_idx) ->
    (test_picks, chosen)を返すファクトリ。train_idx側のブロックをさらにinner_n_folds分割し、
    C_gridの各値を内側OOFの市場超過pt(複勝+ワイドのコスト加重回収率)で評価、best_Cを
    train_idx全体で再学習してtest_idxに適用する。chosenはbest_C(float、ハッシュ可能)。

    mats_allはnar_signals.signal_matrices(races, priors, mats_names)の出力(通常は
    ALL_SIGNALS_V5の全列で1回だけ構築したものを複数の探索で使い回す)。namesはそこから
    実際にモデルへ入力する列のサブセット(build_feature_matrixが絞り込む)。"""
    penalty = "l1" if method == "lasso" else "l2"
    X, race_of_row = build_feature_matrix(mats_all, mats_names, names)
    ranges = race_row_ranges(mats_all)
    y = build_labels(races, actual, label=label)
    blocks = NE.blocks_of(races)  # レース単位のブロックID

    ev_full = NE.Evaluator(races, actual, box_n=box_n)
    cols = [NE.NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]

    def inner_excess(test_race_idx: np.ndarray, test_picks: list) -> float:
        return _fast_excess(ev_full, cols, test_race_idx, test_picks)

    def fit_and_score(train_rows: np.ndarray, C: float):
        params = standardize_fit(X, train_rows)
        Z_train = standardize_apply(X[train_rows], params)
        model = fit_logistic(Z_train, y[train_rows], penalty, C)
        return model, params

    def predict_fn(train_idx: np.ndarray, test_idx: np.ndarray):
        train_rows = np.where(np.isin(race_of_row, train_idx))[0]
        train_blocks_all = sorted(set(blocks[train_idx]))

        rng = np.random.default_rng(seed + (int(train_idx.sum()) % 1_000_003))
        order = rng.permutation(len(train_blocks_all))
        inner_folds = [order[i::inner_n_folds] for i in range(inner_n_folds)]

        best_C, best_score = float(C_grid[0]), -np.inf
        for C in C_grid:
            inner_scores = []
            for fold in inner_folds:
                inner_test_blocks = {train_blocks_all[i] for i in fold}
                inner_test_race_idx = np.array(
                    [i for i in train_idx if blocks[i] in inner_test_blocks])
                inner_train_race_idx = np.array(
                    [i for i in train_idx if blocks[i] not in inner_test_blocks])
                if len(inner_test_race_idx) == 0 or len(inner_train_race_idx) == 0:
                    continue
                inner_train_rows = np.where(np.isin(race_of_row, inner_train_race_idx))[0]
                y_train = y[inner_train_rows]
                if y_train.sum() == 0 or y_train.sum() == len(y_train):
                    continue  # 単一クラスのみだとfit不能、このfoldはスキップ
                model, params = fit_and_score(inner_train_rows, C)
                Z_all = standardize_apply(X, params)
                scores = score_from_model(Z_all, model)
                test_picks = picks_from_scores(scores, ranges, inner_test_race_idx, box_n)
                inner_scores.append(inner_excess(inner_test_race_idx, test_picks))
            if inner_scores:
                mean_score = float(np.mean(inner_scores))
                if mean_score > best_score:
                    best_score, best_C = mean_score, float(C)

        model, params = fit_and_score(train_rows, best_C)
        Z_all = standardize_apply(X, params)
        scores = score_from_model(Z_all, model)
        test_picks = picks_from_scores(scores, ranges, test_idx, box_n)
        return test_picks, best_C

    return predict_fn
