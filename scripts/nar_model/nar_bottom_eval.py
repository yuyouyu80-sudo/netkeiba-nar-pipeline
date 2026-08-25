# -*- coding: utf-8 -*-
"""NAR「5位以下」予測モデルの評価基盤。既存 nar_eval.py / nar_backtest.py は無改造
(payoutの組合せ照合を使わない別タスクのため、並行モジュールとして新設)。

設計判断(Opus 5レビュー指摘の反映):
  * 的中判定はpayoutではなく着順ラベル(label_bottom)そのもの。BoxSettlerのような
    組合せ全列挙は不要で、「選んだbox_n頭のうちlabel_bottom=1だった数」を数えるだけの
    単純な行列演算(precision = Σhit / Σstake)。
  * 全シグナル欠損馬のスコアは+inf(=絶対に下位候補として選ばれない)にする。
    nar_eval.score_picksの-1e18センチネルをそのまま反転すると「最悪馬」として
    最優先選出されてしまうバグを踏むため。
  * 主指標は selection_optimism の true_edge_pt(前回NAR探索8/20の教訓: 保留率の
    大きい選択バイアス診断を優先する)。Nested LOBO OOFは併記するが、fold別argmax
    パターンのユニーク数を必ず追跡する(1〜2種類しか選ばれていない場合はin-sample評価と
    同値であり統計的検定に使わない)。
  * ブロック間比較(市場・系統間)はペア差分ブロックブートストラップにする
    (nar_eval.block_bootstrapの非ペア設計を踏襲しない。同一レース上の対比較にすることで
    共通ノイズを相殺し検出力を上げる)。

後方互換拡張(2026-08-25、K=3〜8スイープ対応):
  * Evaluator(labels=None, label_col="label_bottom"): labelsを渡せばdfを読まず直接使う
    (K別ラベルをdfに書き込まないための経路)。省略時は従来どおりdf[label_col]を読み、
    既存呼び出しの挙動は完全に維持される。
  * signal_label_correlation(..., labels=None, label_col="label_bottom")も同じパターン。
  * label_and_filter(races, k, box_n): field_size > max(k, box_n) でレースを絞り、
    (サブレースリスト, ラベル配列のリスト) を返す(dfへの列追加はしない)。
  * Evaluator.group_kfold_oof: 既存lobo_oof(leave-one-block-out、ブロック数が多いと
    学習集合がほぼ変わらずargmaxが実質固定される構造的欠陥が判明)とは別に追加する、
    held-out比率が有意に大きいグループK分割OOF。lobo_oofは無改造のまま残す。
"""
import numpy as np
import pandas as pd


def blocks_of(races: list) -> np.ndarray:
    """開催日×競馬場のブロックID(nar_eval.blocks_ofと同じ定義)。"""
    return np.array([f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races])


def market_picks_bottom(races: list, box_n: int) -> list:
    """市場ベースライン: 人気最下位(ninki番号最大)box_n頭。人気が欠損する馬は
    「不明」として最下位候補には選ばない(-1e18とし降順ソートで末尾に置く)。"""
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), -1e18, ninki)
        order = np.argsort(-key, kind="stable")  # 降順: ninki番号が大きい(人気薄)馬から
        k = min(box_n, len(order))
        picks.append(order[:k])
    return picks


def score_picks_bottom(mats: list, w: np.ndarray, box_n: int) -> list:
    """重みベクトルwでレースごとの下位box_n頭(スコアが低い=4着以内で終わりにくい馬)の
    行インデックスを返す。全シグナル欠損馬(den==0)はスコア+infとし、絶対に
    下位候補として選ばれないようにする。"""
    picks = []
    for m in mats:
        num = m["S"] @ w
        den = m["A"] @ w
        score = np.where(den > 0, num / den, np.inf)
        k = min(box_n, len(score))
        picks.append(np.argsort(score, kind="stable")[:k])  # 昇順: スコアが低い馬から
    return picks


def label_and_filter(races: list, k: int, box_n: int) -> tuple:
    """K=3〜8スイープ用: finish_pos_numericからKしきい値のラベルをその場で作り、
    field_size > max(k, box_n) でレースを絞る(box_n<=Kなレースで「全馬選出=市場と
    機械的に同値」という無情報行が混じる問題を、Kだけでなくbox_nも考慮して防ぐ)。

    dfへの列書き込みは行わない(K別ラベルを共有dfオブジェクトに書き込むと、複数のKパスが
    同じdfを参照している場合に上書き事故が起きるリスクがあるため)。ラベル配列は
    Evaluator(races, box_n, labels=...) に直接渡すこと。

    戻り値: (サブレースのリスト, 各レースに対応するラベルnp.ndarrayのリスト,
    元のracesリストにおけるインデックスのリスト)。3番目の要素は、Kに依存しない
    build_matrices()の結果(元のracesと1:1対応)をこのサブ集合にスライスするために使う
    (build_matricesの再利用最適化、Kごとに再計算しないための橋渡し)。
    """
    min_size = max(k, box_n)
    sub_races, sub_labels, sub_idx = [], [], []
    for i, r in enumerate(races):
        if r["field_size"] <= min_size:
            continue
        fp = r["df"]["finish_pos_numeric"].to_numpy(dtype=float)
        label = np.where(np.isnan(fp), np.nan, (fp >= k).astype(float))
        if np.isnan(label).all():
            continue
        sub_races.append(r)
        sub_labels.append(label)
        sub_idx.append(i)
    return sub_races, sub_labels, sub_idx


class Evaluator:
    """1つのレース集合に対する評価器。ラベル行列と市場ベンチマークを一度だけ作る。"""

    def __init__(self, races: list, box_n: int, labels: list = None,
                 label_col: str = "label_bottom"):
        self.races = races
        self.box_n = box_n
        if labels is not None:
            # K別ラベルを直接渡す経路(dfへの列追加を避ける、label_and_filter用)。
            self.labels = [np.asarray(l, dtype=float) for l in labels]
        else:
            self.labels = [r["df"][label_col].to_numpy(dtype=float) for r in races]
        self.field_sizes = np.array([len(l) for l in self.labels])
        self.blocks = blocks_of(races)
        self.block_ids = sorted(set(self.blocks))
        self.mkt_picks = market_picks_bottom(races, box_n)
        self.mkt_stake, self.mkt_return = self.hits_for(self.mkt_picks)

    def hits_for(self, picks: list):
        """stake[i] = 選んだ頭数(=min(box_n, field_size))、return[i] = そのうち
        実際にlabel_bottom=1だった数。"""
        n = len(picks)
        stake = np.empty(n, dtype=np.int64)
        ret = np.empty(n, dtype=np.int64)
        for i, idx in enumerate(picks):
            stake[i] = len(idx)
            ret[i] = int(self.labels[i][idx].sum()) if len(idx) else 0
        return stake, ret

    def precision(self, stake: np.ndarray, ret: np.ndarray, idx: np.ndarray = None) -> float:
        s = stake if idx is None else stake[idx]
        r = ret if idx is None else ret[idx]
        tot = s.sum()
        return float(r.sum() / tot * 100) if tot else 0.0

    def evaluate(self, picks: list, idx: np.ndarray = None) -> dict:
        st, rt = self.hits_for(picks)
        model = self.precision(st, rt, idx=idx)
        market = self.precision(self.mkt_stake, self.mkt_return, idx=idx)
        return {"model": model, "market": market, "excess": model - market,
                "stake": st, "return": rt}

    def recall_and_clean(self, picks: list) -> dict:
        """副次指標: recall(取りこぼし率の逆)・clean_box_rate(box全員的中率)。
        多重比較の対象にはしない、報告専用。"""
        st, rt = self.hits_for(picks)
        total_positive = np.array([int(l.sum()) for l in self.labels])
        recall = float(rt.sum() / total_positive.sum() * 100) if total_positive.sum() else 0.0
        clean = float((rt == st).mean() * 100)
        return {"recall_pct": recall, "clean_box_rate_pct": clean}

    # --------------------------------------------------------------- CV
    def lobo_oof(self, fit_fn, mats_all: list) -> dict:
        """Leave-one-block-out の out-of-fold 評価。

        fit_fn(train_idx) -> (w, chosen_pattern_index) を受け取る。chosen_pattern_indexは
        fold別argmaxのユニーク数を追跡するためのもの(1〜2種類しか選ばれない場合は
        in-sample評価と実質同値、という前回NAR探索の教訓への対応)。
        """
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
                score = np.where(den > 0, num / den, np.inf)
                k = min(self.box_n, len(score))
                picks[i] = np.argsort(score, kind="stable")[:k]
        result = {"picks": picks, **self.evaluate(picks)}
        result["fold_argmax_choices"] = choices
        result["fold_argmax_unique"] = len(set(choices))
        return result

    def group_kfold_oof(self, fit_fn, mats_all: list, n_folds: int = 8, seed: int = 13) -> dict:
        """ブロックをn_foldsグループに乱数分割するK分割out-of-fold評価(lobo_oofとは別に
        追加、lobo_oof自体は無改造のまま残す)。

        leave-one-block-outはブロック数が多い(NARは94〜138ブロック)と1ブロック抜いても
        学習集合がほぼ変わらず、fit_fnのargmaxが実質固定されてしまう構造的な欠陥がある
        (fold_argmax_uniqueが1〜2に張り付く)。ここではheld-out比率を意図的に大きくして
        (既定n_folds=8なら1/8=12.5%)、argmaxが実際に変動しうるようにする。
        """
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
                score = np.where(den > 0, num / den, np.inf)
                k = min(self.box_n, len(score))
                picks[i] = np.argsort(score, kind="stable")[:k]
        result = {"picks": picks, **self.evaluate(picks)}
        result["fold_argmax_choices"] = choices
        result["fold_argmax_unique"] = len(set(choices))
        result["n_folds"] = len(folds)
        return result

    # --------------------------------------------------------------- bootstrap
    def block_bootstrap(self, picks: list, n: int = 2000, seed: int = 11) -> dict:
        """ブロック単位・比推定量の絶対水準ブートストラップ(参考値)。"""
        st, rt = self.hits_for(picks)
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        out = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            s, r = st[idx].sum(), rt[idx].sum()
            out[k] = r / s * 100 if s else 0.0
        return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
                "hi": float(np.percentile(out, 97.5))}

    def paired_block_bootstrap(self, picks_a: list, picks_b: list, n: int = 2000,
                               seed: int = 41) -> dict:
        """レース単位でpicks_aとpicks_bのprecisionを対にした差分をブロックブートストラップ
        する(本命の比較手法。天候・その日の頭数構成といった共通ノイズを相殺する)。"""
        st_a, rt_a = self.hits_for(picks_a)
        st_b, rt_b = self.hits_for(picks_b)
        by_block = {b: np.where(self.blocks == b)[0] for b in self.block_ids}
        rng = np.random.default_rng(seed)
        ids = list(self.block_ids)
        diffs = np.empty(n)
        for k in range(n):
            chosen = rng.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            sa, ra = st_a[idx].sum(), rt_a[idx].sum()
            sb, rb = st_b[idx].sum(), rt_b[idx].sum()
            va = ra / sa * 100 if sa else 0.0
            vb = rb / sb * 100 if sb else 0.0
            diffs[k] = va - vb
        return {"mean": float(diffs.mean()), "lo": float(np.percentile(diffs, 2.5)),
                "hi": float(np.percentile(diffs, 97.5))}


def selection_optimism(ev: Evaluator, mats: list, W: np.ndarray, n_rep: int = 200,
                       seed: int = 99) -> dict:
    """「重みを選ぶ」という行為から得られる真の利得を測る(nar_eval.selection_optimismと
    同一設計、precision指標への適応版)。"""
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    all_picks = [score_picks_bottom(mats, W[:, j], ev.box_n) for j in range(W.shape[1])]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.hits_for(p)
        all_st.append(s)
        all_rt.append(r)
    rng = np.random.default_rng(seed)
    sel, unseen, unseen_mean = [], [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(ids))
        a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
        b = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
        va = np.array([ev.precision(all_st[j], all_rt[j], idx=a) for j in range(W.shape[1])])
        vb = np.array([ev.precision(all_st[j], all_rt[j], idx=b) for j in range(W.shape[1])])
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


# --------------------------------------------------------------------- walk-forward
def walk_forward_split(races: list):
    """開催日で時系列に前半/後半へ2分割したtrain_idx/test_idxを返す(簡易walk-forward用)。"""
    dates = sorted(set(r["kaisai_date"] for r in races))
    mid = dates[len(dates) // 2]
    train_idx = np.array([i for i, r in enumerate(races) if r["kaisai_date"] < mid])
    test_idx = np.array([i for i, r in enumerate(races) if r["kaisai_date"] >= mid])
    return train_idx, test_idx, mid


# --------------------------------------------------------------------- horse group split
def _horse_group_components(races: list) -> np.ndarray:
    """同じ馬が出走した複数レースを同じ連結成分にまとめる(Union-Find)。
    1レースには複数の馬がおり、それぞれ別レースでも出走しうるため、レース単位の
    単純な乱数分割では「馬IDのリーク」(同じ馬が train/test 両方に出現)を防げない。
    連結成分ごとfoldに割り当てることで、どの馬も片方のfoldにしか出現しないようにする。"""
    n = len(races)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    horse_to_race = {}
    for i, r in enumerate(races):
        if "horse_id" not in r["df"].columns:
            continue
        for h in r["df"]["horse_id"].astype(str).tolist():
            if h in horse_to_race:
                union(i, horse_to_race[h])
            else:
                horse_to_race[h] = i
    return np.array([find(i) for i in range(n)])


def horse_group_split(races: list, seed: int = 7):
    """馬ID単位で連結成分を作り、頭数ができるだけ均等になるよう貪欲に2グループへ分割する
    (感度分析用の簡易半分割)。"""
    comp = _horse_group_components(races)
    comp_ids = sorted(set(comp))
    rng = np.random.default_rng(seed)
    order = rng.permutation(comp_ids)
    sizes = {c: int((comp == c).sum()) for c in comp_ids}
    a_ids, b_ids, a_n, b_n = [], [], 0, 0
    for c in order:
        if a_n <= b_n:
            a_ids.append(c)
            a_n += sizes[c]
        else:
            b_ids.append(c)
            b_n += sizes[c]
    idx_a = np.where(np.isin(comp, a_ids))[0]
    idx_b = np.where(np.isin(comp, b_ids))[0]
    return idx_a, idx_b


# --------------------------------------------------------------------- diagnostics
def signal_label_correlation(entries: list, mats: list, names: list, labels: list = None,
                             label_col: str = "label_bottom") -> dict:
    """各シグナルとラベルのspearman相関を実測する(符号規約チェック用、Phase 0)。
    規約「高いほど4着以内で終わりやすい」が正しければ、ラベル(1=しきい値より下位)との
    相関は負になるはず。正の相関が出たシグナルは符号が意図と逆の可能性がある。

    labelsを渡せばK別ラベル配列を直接使う(dfを読まない)。省略時は従来どおり
    df[label_col]を読む(デフォルトlabel_col="label_bottom"で旧来の挙動を維持)。"""
    rows = []
    label_source = labels if labels is not None else [e["df"][label_col].to_numpy(dtype=float) for e in entries]
    for e, m, lab in zip(entries, mats, label_source):
        S, A = m["S"], m["A"]
        for i in range(len(lab)):
            row = {"label_bottom": lab[i]}
            for j, n in enumerate(names):
                row[n] = S[i, j] if A[i, j] > 0 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    corr = df.corr(method="spearman")["label_bottom"]
    return corr.drop("label_bottom").to_dict()
