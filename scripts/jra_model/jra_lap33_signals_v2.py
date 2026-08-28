# -*- coding: utf-8 -*-
"""「33ラップ理論」シグナル v2 — v1(`jra_lap33_signals.py`、別セッション作成、無改造で参照
するのみ)への拡張。プランレビュー(シニアエンジニア役・競馬予想家役サブエージェント、
2026-08-28)で指摘された、v1にまだ無い3点を追加する:

  1. **動的ペース補正**(競馬予想家レビュー致命的指摘): v1の「今回レースの参照値」は
     `jra_lap33_theory_table.py`のコース×距離×馬場の静的な理論値そのもの。実際にはレース固有の
     展開(逃げ・先行馬の頭数・比率)によって33ラップは大きく振れるはずで、静的な理論値だけでは
     理論が本来説明したい分散のほとんどを捨ててしまう。出馬表の`ca_running_style_category_label`
     から今回の逃げ・先行比率を求め、母集団の平均比率とのズレに応じて参照値を線形補正する。
  2. **芝/ダート・距離帯別に層別した型スコア**(競馬予想家レビュー重要指摘): v1は
     `history_index.past_starts`で取れる直近N走を距離・馬場を問わず一律に見ている。理論原文が
     示す実例(コンカラー)もダート限定であり、「ダートで瞬発力型に好走→芝でも同じ適性」という
     無条件の転用は理論の適用範囲を超える。今回と同じ(surface, 距離帯)の過去走のみを対象母集団
     とし、標本が不足する場合のみv1相当の層別無し版へフォールバックする(段階的縮退)。
  3. **新規性(独立性)チェック**(競馬予想家レビュー致命的指摘): 算出したシグナル値が既存の
     course/distance/style系シグナルと高相関(|r|>0.7、`jra_signal_gate_v4_2026_08_28.py`の
     `CORR_DROP_THRESHOLD`と同じ基準を流用)なら、実質的に既存シグナルの再現に過ぎないと判断
     できるようにする診断関数を用意する。

v1のファイル(`jra_lap33_signals.py`・`jra_lap33_theory_table.py`・`jra_lap33_theory_2026_08_28.py`
・`jra_lap33_signal_gate_2026_08_28.py`)はこのモジュールから import するだけで一切編集しない
(別セッションと同時に触ると衝突するため)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent

sys.path.insert(0, str(LIB_DIR))
import jra_history as JH  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402
import jra_signals as JS  # noqa: E402
from jra_lap33_theory_table import lookup as theory_lookup  # noqa: E402

N_LOOKBACK_V2 = 40  # 層別すると母数が減るため、v1(20)より広く取る
MIN_GROUP_N_V2 = 3  # 層別後の好走・凡走いずれかがこの本数未満なら「層別では型不明」
FRONT_STYLES = JS.RUNNING_STYLE_FRONT  # {"逃", "先"}(単一文字、jra_signals.py L102の実値に合わせる。
# 実データ確認: ca_running_style_category_labelは"逃げ"/"先行"ではなく"逃"/"先"/"差"/"追"の
# 単一文字。jra_signals.pyの既存RUNNING_STYLE_FRONTをそのまま再利用し、値の不一致を防ぐ。

# ペース補正係数: 「逃げ・先行比率が母集団平均よりX pt高い」ごとに参照値をどれだけ
# 持久力側(マイナス)に振るか。過大補正で符号が反転しないよう、理論値の代表的なレンジ
# (おおむね-3.5~+2.6、jra_lap33_theory_table.py参照)に対して穏当な値から始める。
# 経験的な最終値はmain()の診断(補正あり/無しの比較)で決める前提の暫定値。
PACE_ADJ_COEF = -2.0


# ===================================================================== 距離帯
def distance_band(distance_m: int) -> str:
    """距離帯(短距離~1400 / マイル1401~1700 / 中距離1701~2200 / 長距離2201~)。
    芝・ダート共通の粒度(v1のtheory_lookupが芝ダ別に値を持つため、surfaceは別軸で扱う)。"""
    if distance_m <= 1400:
        return "sprint"
    if distance_m <= 1700:
        return "mile"
    if distance_m <= 2200:
        return "middle"
    return "long"


# ===================================================================== 1. 動的ペース補正
def front_ratio(df: pd.DataFrame) -> float:
    """出走メンバー中の「逃げ・先行」比率(0~1)。列欠損時はnp.nan。"""
    col = "ca_running_style_category_label"
    if col not in df.columns:
        return np.nan
    labels = df[col].astype(str).str.strip()
    n = len(labels)
    if n == 0:
        return np.nan
    return float(labels.isin(FRONT_STYLES).sum()) / n


def population_front_ratio(races: list) -> float:
    """母集団全体(races)での平均逃げ・先行比率。補正のベースラインとして使う。"""
    ratios = [front_ratio(r["df"]) for r in races]
    ratios = [x for x in ratios if not np.isnan(x)]
    return float(np.mean(ratios)) if ratios else 0.5


def pace_adjusted_ref(base_ref: float, df: pd.DataFrame, baseline_ratio: float,
                      coef: float = PACE_ADJ_COEF) -> float:
    """base_ref(理論値) を今回メンバーの逃げ・先行比率で補正する。
    比率が母集団平均より高い(=先行争いが厳しくなりやすい=持久力寄りになりやすい)ほど
    マイナス方向に補正する(coef<0であることに注意、符号はmain()の診断で確認する)。"""
    ratio = front_ratio(df)
    if np.isnan(ratio) or base_ref is None:
        return base_ref
    return base_ref + coef * (ratio - baseline_ratio)


# ===================================================================== 2. 層別型スコア
def horse_type_score_stratified(
    horse_id: str, iso_date: str, history_index: "JH.HorseHistoryIndex", lap33_lookup: dict,
    race_meta: dict, target_surface: str, target_band: str,
    n_lookback: int = N_LOOKBACK_V2, min_group_n: int = MIN_GROUP_N_V2,
) -> tuple:
    """今回と同じ(surface, 距離帯)の過去走だけを対象に、v1と同じ「好走時平均33ラップ-凡走時
    平均33ラップ」を算出する。層別後の標本が不足する場合は v1(層別無し)の値へフォールバック
    する(段階的縮退、シニアエンジニアレビュー指摘: 標本不足で即0にすると死にシグナル化する
    リスクが高いため)。

    戻り値: (score, source) — source は "stratified" / "fallback_v1" / "neutral" のいずれか
    (診断のため、どの経路で値が決まったかを追跡できるようにする)。"""
    starts = history_index.past_starts(horse_id, iso_date, n=n_lookback)
    good, bad = [], []
    for s in starts:
        lap33 = lap33_lookup.get(s.race_id)
        if lap33 is None or (isinstance(lap33, float) and np.isnan(lap33)):
            continue
        meta = race_meta.get(s.race_id)
        if meta is None:
            continue
        _course, surface, distance_m = meta
        if surface != target_surface or distance_band(distance_m) != target_band:
            continue
        finish = pd.to_numeric(s.finish_pos, errors="coerce")
        if pd.isna(finish):
            continue
        (good if finish <= 3 else bad).append(lap33)

    if len(good) >= min_group_n and len(bad) >= min_group_n:
        return float(np.mean(good) - np.mean(bad)), "stratified"

    # 層別では標本不足 -> v1相当(層別無し、N_LOOKBACK=20)へフォールバック
    fallback = L33.horse_type_score(horse_id, iso_date, history_index, lap33_lookup,
                                    n_lookback=L33.N_LOOKBACK)
    return fallback, ("fallback_v1" if fallback != 0.0 else "neutral")


# ===================================================================== 統合: v2シグナル行列
def lap33_fit_v2_matrix(races: list, history_index: "JH.HorseHistoryIndex", lap33_lookup: dict,
                        race_meta: dict, baseline_ratio: float,
                        n_lookback: int = N_LOOKBACK_V2) -> dict:
    """race_id -> {"type_score": Series, "lap33_fit_v2": Series, "course_ref_adj": float,
    "source_counts": dict} 。v1のlap33_fit_matrixと同じ「1レース分のシグナル辞書」形式。"""
    out = {}
    for r in races:
        iso_date = L33.kaisai_date_to_iso(r["kaisai_date"])
        meta = race_meta.get(r["race_id"])
        base_ref = None if meta is None else theory_lookup(*meta)
        adj_ref = (None if base_ref is None
                  else pace_adjusted_ref(base_ref, r["df"], baseline_ratio))
        target_surface = None if meta is None else meta[1]
        target_band = None if meta is None else distance_band(meta[2])

        type_scores, sources = [], []
        for hid in r["df"]["horse_id"].astype(str):
            if target_surface is None:
                type_scores.append(0.0)
                sources.append("no_meta")
                continue
            score, source = horse_type_score_stratified(
                hid, iso_date, history_index, lap33_lookup, race_meta,
                target_surface, target_band, n_lookback=n_lookback)
            type_scores.append(score)
            sources.append(source)

        ts = pd.Series(type_scores, index=r["df"].index)
        if adj_ref is None:
            fit = pd.Series(np.nan, index=r["df"].index)
        else:
            fit = JS._minmax(ts * adj_ref)
        src_counts = pd.Series(sources).value_counts().to_dict()
        out[r["race_id"]] = {
            "type_score": ts, "lap33_fit_v2": fit,
            "course_ref": base_ref, "course_ref_adj": adj_ref, "source_counts": src_counts,
        }
    return out


# ===================================================================== 3. 新規性(独立性)チェック
CORR_DROP_THRESHOLD = 0.70  # jra_signal_gate_v4_2026_08_28.pyと同じ基準を流用


def novelty_check(fit_by_race: dict, races: list, priors_all: dict,
                  compare_signals=("course", "distance", "style", "apt")) -> pd.DataFrame:
    """lap33_fit_v2 と既存シグナルとの相関を計算する。|r|>CORR_DROP_THRESHOLD なら
    「実質的に既存シグナルの再現」と判定し、そのままPhase1ゲートに進めない判断材料にする。"""
    all_fit, all_existing = [], {name: [] for name in compare_signals}
    for r in races:
        fit = fit_by_race.get(r["race_id"], {}).get("lap33_fit_v2")
        if fit is None:
            continue
        current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
        sig = JS.compute_signals(r["df"], current_class, priors_all)
        all_fit.append(fit)
        for name in compare_signals:
            all_existing[name].append(sig.get(name, pd.Series(np.nan, index=r["df"].index)))

    fit_concat = pd.concat(all_fit)
    rows = []
    for name in compare_signals:
        existing_concat = pd.concat(all_existing[name])
        both = pd.concat([fit_concat, existing_concat], axis=1).dropna()
        both.columns = ["lap33_fit_v2", name]
        if len(both) < 10:
            rows.append({"signal": name, "n": len(both), "corr": np.nan, "flag_high_corr": False})
            continue
        corr = float(both["lap33_fit_v2"].corr(both[name]))
        rows.append({
            "signal": name, "n": len(both), "corr": corr,
            "flag_high_corr": bool(abs(corr) > CORR_DROP_THRESHOLD),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
    import jra_dataset as JD

    OUT_DIR = Path(
        r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
        r"\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("読み込み中...")
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()
    data = JD.load(rebuild=False)
    races = data["races"]
    priors_all = JS.make_priors([r["df"] for r in races])
    baseline_ratio = population_front_ratio(races)
    print(f"lap33_lookup: {len(lap33_lookup)}レース  race_meta: {len(race_meta)}レース  "
          f"対象母集団: {len(races)}レース  母集団平均逃げ先行比率: {baseline_ratio:.3f}")

    fit_v2 = lap33_fit_v2_matrix(races, history_index, lap33_lookup, race_meta, baseline_ratio)

    # --- 層別の効き具合診断(Phase0相当) ---
    all_sources = {}
    for v in fit_v2.values():
        for k, c in v["source_counts"].items():
            all_sources[k] = all_sources.get(k, 0) + c
    n_horses = sum(v["source_counts"].get(k, 0) for v in fit_v2.values() for k in v["source_counts"])
    print("\n=== 型スコアの決定経路(Phase0診断) ===")
    for k, c in sorted(all_sources.items(), key=lambda x: -x[1]):
        print(f"  {k}: {c}頭 ({c / n_horses * 100:.1f}%)")

    # --- v1 との比較(参考値のペース補正あり/無し) ---
    fit_v1 = L33.lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)
    v1_nonzero = sum(int((v["type_score"] != 0.0).sum()) for v in fit_v1.values())
    v2_stratified = all_sources.get("stratified", 0)
    print(f"\nv1(層別無し)で型判定できた馬: {v1_nonzero}/{n_horses}"
          f"({v1_nonzero / n_horses * 100:.1f}%)")
    print(f"v2で層別のまま型判定できた馬(stratified): {v2_stratified}/{n_horses}"
          f"({v2_stratified / n_horses * 100:.1f}%)")

    # --- 新規性チェック ---
    print("\n=== 新規性(独立性)チェック: 既存シグナルとの相関 ===")
    nc = novelty_check(fit_v2, races, priors_all)
    print(nc.to_string(index=False))
    nc.to_csv(OUT_DIR / "jra_lap33_v2_novelty_check.csv", index=False, encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'jra_lap33_v2_novelty_check.csv'}")
