# -*- coding: utf-8 -*-
"""新馬戦(デビュー戦)専用の着順予想モデル。

pattern29(predict.py)は過去走のスピード指数・着順履歴・脚質など「その馬自身の実戦成績」に
依存した10シグナル構成で、race_nameに「新馬|未勝利」を含むレースは最初から対象外にしている。
しかし新馬戦は出走馬全頭が今回が初出走のため、この前提が構造的に成り立たない。

実データ検証(scratchpad/inspect_shinba_features.py)により、既存10シグナルのうち
speed/form/style/apt/distanceの5つは新馬戦データでは全列空または定数(該当馬自身の過去走が
存在しないため)であることを確認済み。使用可能な5シグナル(jt/waku/train/sire/bms、いずれも
馬個体ではなく騎手・調教師・種牡馬・母父・枠番という「集団統計」ベースなので新馬でも成立する)
をpredict.pyから流用し、厩舎コメント評価(stable_comment_rating_code)を新規シグナルとして
追加した6シグナル構成にした。

class比較(CLASS_ORDINAL等)・DNF処理(DNF_CODES等)は新馬戦が単一クラスであることと
formシグナル自体が存在しないことから丸ごと不要なため移植していない。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
WINNER_JSON = DATA_DIR / "winner_shinba.json"
OUT_CSV = DATA_DIR / "predictions_shinba.csv"

TRAIN_RANK_MAP = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

# 厩舎コメント評価コード(netkeibaのIcon_Mark番号、"01"/"02"/"03")。実着順との相関を実測し
# 方向を確認済み(scratchpad/inspect_shinba_direction.py、160頭全件):
#   01 -> 平均着順5.35・勝率29.4%  02 -> 6.24・7.3%  03 -> 9.84・0%  (単調)
# 検証済みなのは「01が最良・03が最悪」という順序だけなので、線形マップのみ採用し
# 検証していない非線形の重み付け(例: 03を極端に低く見る等)は持ち込まない。
COMMENT_RATING_MAP = {"01": 3, "02": 2, "03": 1}

SHRINK_K = 12.0  # pattern29と同じempirical-Bayes疑似カウント(踏襲、変更する根拠なし)

# 暫定の均等重み(100パターン探索前のv0/プレースホルダ、レビュー用の初期予想生成にのみ使う)。
# 探索完了後は winner_shinba.json を実行時に読み込む方式に切り替わる(load_weights参照)。
# pattern29のWEIGHTSがハードコードのままwinner_v3.jsonと乖離していた既知のdriftを
# 再発させないため、predict.pyのようにWEIGHTS定数へは書き戻さない。
SHINBA_BASELINE_WEIGHTS = {
    "jt": 1 / 6, "waku": 1 / 6, "train": 1 / 6, "sire": 1 / 6, "bms": 1 / 6, "comment": 1 / 6,
}
SIGNAL_NAMES = list(SHINBA_BASELINE_WEIGHTS.keys())


def _num(series: pd.Series) -> pd.Series:
    """数値に変換。'-'/'--'等のプレースホルダは欠損扱い。"""
    cleaned = series.where(~series.astype(str).isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _pct(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False)
    cleaned = cleaned.where(~cleaned.isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _blend_minmax(*series_list) -> pd.Series:
    cols = [_minmax(s) for s in series_list]
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """一部のレースはオプションのテーブルが丸ごと欠落する(列自体が無い) - 全欠損として扱う。"""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _drop_scratched(df: pd.DataFrame) -> pd.DataFrame:
    """出走取消(オッズ/人気が'--')の馬を除外。"""
    odds = _num(df["bias_win_odds"])
    ninki = _num(df["bias_ninki"])
    return df[odds.notna() & ninki.notna()].reset_index(drop=True)


# 使用可能な5系統×レート種別。style/apt/distanceは新馬戦データで全列空/定数のため対象外
# (ca_running_style_*は"0%"/"0"固定、ca_speed_index_*は完全空、data_distance_slot1_*は
# 全馬"0%"/"0"固定 - いずれも「その馬自身の過去走条件付き統計」で新馬には存在しない)。
SHRINK_SPECS = {
    "jockey_win": ("ca_jockey_win_rate", "ca_jockey_runs"),
    "trainer_win": ("ca_trainer_win_rate", "ca_trainer_runs"),
    "waku_win": ("ca_waku_win_rate", "ca_waku_runs"),
    "sire_win": ("ca_sire_win_rate", "ca_sire_runs"),
    "sire_place3": ("ca_sire_place3_rate", "ca_sire_runs"),
    "sire_return": ("ca_sire_win_return_rate", "ca_sire_runs"),
    "bms_win": ("ca_broodmare_sire_win_rate", "ca_broodmare_sire_runs"),
    "bms_place3": ("ca_broodmare_sire_place3_rate", "ca_broodmare_sire_runs"),
    "bms_return": ("ca_broodmare_sire_win_return_rate", "ca_broodmare_sire_runs"),
}


def compute_priors(raw_dfs: list) -> dict:
    """EB縮約の事前値。pattern29の_PRIORSは70レース(通常戦)という別母集団のハードコード値
    であり新馬戦にはそのまま使えないため、新馬戦の対象母集団から都度計算する(現状14レース
    160頭、対象レースが増えるたびに自動的に更新される)。

    シニアエンジニアレビューで指摘・実測された問題を修正: 単純平均(頭数等分)だと
    runs<=2の極薄サンプル馬(160頭中44頭=27.5%、うち1戦1勝=100%のような外れ値を含む)が
    priorを不当に押し上げる(sire_winで実測+37%、bms_winで+24%の歪み)。「小標本を信頼し
    すぎない」ためのEB縮約なのに、縮約先のprior自体が小標本outlierに引っ張られては本末転倒
    なので、runsで露出加重(sum(rate*runs)/sum(runs))した平均に変更する。"""
    priors = {}
    for key, (rate_col, runs_col) in SHRINK_SPECS.items():
        rates = pd.concat([_pct(_col(df, rate_col)) for df in raw_dfs], ignore_index=True)
        runs = pd.concat([_num(_col(df, runs_col)) for df in raw_dfs], ignore_index=True).fillna(0.0)
        valid = rates.notna() & (runs > 0)
        total_runs = runs[valid].sum()
        priors[key] = float((rates[valid] * runs[valid]).sum() / total_runs) if total_runs > 0 else float(rates.mean(skipna=True))
    return priors


def _shrink(df: pd.DataFrame, key: str, priors: dict) -> pd.Series:
    """Empirical-Bayes縮約: サンプル数(runs)が少ないカテゴリ率をプールされた事前値へ
    引き寄せる(種牡馬runsの中央値は11程度でSHRINK_K=12と同水準のため、縮約が強く効く)。"""
    rate_col, runs_col = SHRINK_SPECS[key]
    rate = _pct(_col(df, rate_col))
    runs = _num(_col(df, runs_col)).fillna(0.0)
    prior = priors[key]
    return (rate.fillna(prior) * runs + prior * SHRINK_K) / (runs + SHRINK_K)


def build_shinba_signals(df: pd.DataFrame, priors: dict) -> dict:
    sig = {}

    # --- 騎手・調教師(縮約済み) ---
    jt_rate = pd.concat(
        [_shrink(df, "jockey_win", priors), _shrink(df, "trainer_win", priors)], axis=1
    ).mean(axis=1)
    sig["jt"] = _minmax(jt_rate)

    # --- 枠順バイアス(縮約済み) ---
    sig["waku"] = _minmax(_shrink(df, "waku_win", priors))

    # --- 調教評価 ---
    training = df["training_rank"].astype(str).str.strip().str.upper().map(TRAIN_RANK_MAP)
    sig["train"] = _minmax(training)

    # --- 種牡馬適性(縮約済み) ---
    sig["sire"] = _blend_minmax(
        _shrink(df, "sire_win", priors), _shrink(df, "sire_place3", priors), _shrink(df, "sire_return", priors)
    )

    # --- 母父適性(縮約済み) ---
    sig["bms"] = _blend_minmax(
        _shrink(df, "bms_win", priors), _shrink(df, "bms_place3", priors), _shrink(df, "bms_return", priors)
    )

    # --- 厩舎コメント評価(新規) ---
    # プロ予想家レビューで発見: 全馬が同じ評価コード(例: 202610020505は全馬"02")のレースは
    # min==maxで_minmax()がNaNを返しこのレースだけcomment信号が消える。これは既存のtotal_weight
    # 按分ロジック(score_shinba_race)がavail判定で自動的に吸収する設計どおりの挙動であり、
    # 個別の対処は不要(残り5シグナルの重みだけでそのレースを評価する)。
    rating = df["stable_comment_rating_code"].astype(str).str.strip().map(COMMENT_RATING_MAP)
    sig["comment"] = _minmax(rating)

    return sig


def score_shinba_race(df: pd.DataFrame, priors: dict, weights: dict) -> pd.DataFrame:
    df = df.copy()
    sig = build_shinba_signals(df, priors)

    total_score = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        if w <= 0:
            continue
        s = sig[name]
        avail = s.notna()
        total_score = total_score + s.fillna(0) * w
        total_weight = total_weight + avail.astype(float) * w

    df["_score"] = np.where(total_weight > 0, total_score / total_weight, np.nan)
    return df


def load_weights() -> tuple[dict, str]:
    """シニアエンジニアレビュー指摘対応: 読み込んだ重みのキー集合を検証せずに使うと、
    signal名の不一致がscore_shinba_race内のsig[name]でKeyErrorになり、main()のtry/exceptで
    レース単位のerroredに握りつぶされて原因が分かりにくくなる。fail-fastで即座に検知する。"""
    if WINNER_JSON.exists():
        weights = json.loads(WINNER_JSON.read_text(encoding="utf-8"))["weights"]
        source = "winner_shinba.json"
    else:
        weights = SHINBA_BASELINE_WEIGHTS
        source = "baseline equal-weight (winner_shinba.json not found yet)"

    if set(weights.keys()) != set(SIGNAL_NAMES):
        raise ValueError(
            f"weights key mismatch (source={source}): got {sorted(weights.keys())}, "
            f"expected {sorted(SIGNAL_NAMES)}"
        )
    return weights, source


def load_shinba_races() -> list:
    """race_results配下の各開催日CSVから、race_nameに「新馬」を含むレースだけを抽出する。
    「新馬|未勝利」ではなく「新馬」のみ - 未勝利戦は過去走を持つ馬が対象でありこのモデルの
    前提(過去走ゼロ)に合わないため意図的に除外する。"""
    entries = []
    results_dir = PROJECT_ROOT / "data" / "race_results" / "2026"
    for date_csv in sorted(results_dir.glob("2026*.csv")):
        date = date_csv.stem
        results = pd.read_csv(date_csv, dtype=str)
        races = results[["race_id", "race_name", "racecourse"]].drop_duplicates("race_id")
        shinba = races[races["race_name"].str.contains("新馬", na=False, regex=True)]
        for _, row in shinba.iterrows():
            entries.append({
                "race_id": row["race_id"], "kaisai_date": date,
                "racecourse": row["racecourse"], "race_name": row["race_name"],
            })
    return entries


def main():
    targets = load_shinba_races()
    print(f"target shinba races: {len(targets)}")

    raw = []
    missing = []
    for e in targets:
        path = PROJECT_ROOT / "data" / "newspaper" / f"{e['race_id']}.csv"
        if not path.exists():
            missing.append(e["race_id"])
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            missing.append(e["race_id"])
            continue
        df = _drop_scratched(df)
        if df.empty:
            missing.append(e["race_id"])
            continue
        raw.append({**e, "df": df})

    if not raw:
        print("no shinba races with usable newspaper data")
        return

    priors = compute_priors([e["df"] for e in raw])
    print("shrinkage priors:", {k: round(v, 2) for k, v in priors.items()})

    weights, weights_source = load_weights()
    print(f"weights ({weights_source}):", weights)

    all_predictions = []
    errored = []
    for e in raw:
        try:
            scored = score_shinba_race(e["df"], priors, weights)
            scored = scored.sort_values("_score", ascending=False, kind="stable").reset_index(drop=True)
            top5 = scored.head(5).copy()
            top5["race_id"] = e["race_id"]
            top5.insert(0, "pred_rank", range(1, len(top5) + 1))
            top5.insert(0, "kaisai_date", e["kaisai_date"])
            top5.insert(1, "racecourse", e["racecourse"])
            top5.insert(2, "race_name", e["race_name"])
            all_predictions.append(
                top5[["kaisai_date", "racecourse", "race_name", "race_id", "pred_rank", "waku", "umaban",
                      "horse_name", "bias_ninki", "bias_win_odds", "_score"]]
            )
        except Exception as exc:  # noqa: BLE001 - 1レースの異常で全体を止めない
            errored.append((e["race_id"], repr(exc)))
            continue

    if not all_predictions:
        print("no predictions produced")
        return

    result = pd.concat(all_predictions, ignore_index=True)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"predicted races: {result['race_id'].nunique()}")
    print(f"missing newspaper csv: {len(missing)} -> {missing}")
    print(f"errored races: {len(errored)} -> {errored}")


if __name__ == "__main__":
    main()
