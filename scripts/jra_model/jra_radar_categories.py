# -*- coding: utf-8 -*-
"""レーダーチャート面積着順予想シグナルの検証: 8カテゴリの軸構築・5集約方法・3順序定義。

背景・数学的根拠は計画ファイル(C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md)参照。
jra_signals.py・jra_lap33_signals.py は無改造、このファイルはサイドカーとして両者を呼び出す。

設計の要点:
  * 8カテゴリ(能力・調教評価/近走成績・調子/脚質・展開/コース・距離適性/血統適性/騎手・厩舎/
    予想印・専門家評価/33ラップ理論適合度)。凡走予測は今回の検証から完全に除外。
  * `脚質・展開`カテゴリには既存の脚質・持続力・展開系11シグナルに加え、新規`pace_fit`
    (netkeiba公式AI予想ペース区分×脚質)を含める。
  * `33ラップ理論適合度`は他カテゴリと異なりcompute_signals由来ではなく、
    jra_lap33_signals.lap33_fit_matrix()のtype_scoreを再利用し、この検証専用の後処理
    (type_score==0.0→NaN変換)を経てから軸化する。
  * 各馬・各カテゴリの軸値を1つの補完済み・レース内再正規化済み行列`R`にまとめ、
    線形平均・幾何平均(主検定)・最小値・上位3軸和・面積(3順序)の全アームがこの同一行列
    から計算される(アーム間の欠損処理・スケールの非対称性を排除する設計)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent

sys.path.insert(0, str(LIB_DIR))
import jra_eval as JE  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402
import jra_signals as JS  # noqa: E402

MIN_CATEGORIES = 4  # N_race < 4 のレースは評価対象から除外(計画セクション3-6)
AREA_EPS = 1e-6  # 幾何平均のlog(0)対策

# --------------------------------------------------------------------- カテゴリ定義

CATEGORY_SIGNAL_MAP = {
    "能力・調教評価": ["speed", "train"],
    "近走成績・調子": ["form", "margin", "timediff", "agari", "weight_trend", "class_drop"],
    "脚質・展開": [
        "style", "nige", "holdtime", "hold_just", "hold_wide",
        "corner4_position", "corner4_gap", "corner4_speedup",
        "corner3_position", "corner3_gap",
        "corner_transition_rank", "corner_transition_gap",
        "pace_fit",
    ],
    "コース・距離適性": ["course", "concerned", "distance", "apt", "waku", "interval", "kinryo"],
    "血統適性": ["sire", "bms", "ketto_training", "ketto_comment"],
    "騎手・厩舎": ["jt", "jockey_change", "jockey_owner", "prevjockey", "odds_jockey", "surf_jt"],
    "予想印・専門家評価": ["mark_honshi_score", "mark_cp_score", "mark_other_score"],
    # 特殊カテゴリ: compute_signalsではなくjra_lap33_signals由来。メンバー名は
    # ドキュメント目的のプレースホルダで、実際の計算はcompute_lap33_axis()が行う。
    "33ラップ理論適合度": ["lap33_axis"],
}
CATEGORIES = list(CATEGORY_SIGNAL_MAP.keys())

# pace_fit(新規シグナル)とlap33_axis(jra_lap33_signals由来の特殊軸)を除いた残りが
# 過不足なくALL_SIGNALS_V4(mark_composite_score除く)と一致することをここでassertする。
_SPECIAL = {"pace_fit", "lap33_axis"}
_all_members = {m for members in CATEGORY_SIGNAL_MAP.values() for m in members} - _SPECIAL
_expected = _all_members | {"mark_composite_score"}
assert _expected == set(JS.ALL_SIGNALS_V4), (
    "CATEGORY_SIGNAL_MAPとALL_SIGNALS_V4(mark_composite_score込み)が一致しない: "
    f"missing={set(JS.ALL_SIGNALS_V4) - _expected}  extra={_expected - set(JS.ALL_SIGNALS_V4)}"
)

# --------------------------------------------------------------------- 順序(面積専用)

ORDER_1 = [  # 主軸・物語的順序
    "能力・調教評価", "近走成績・調子", "脚質・展開", "33ラップ理論適合度",
    "コース・距離適性", "血統適性", "騎手・厩舎", "予想印・専門家評価",
]
ORDER_2 = [  # 3ブロック・ドメイン仮説(資質/条件噛み合わせ/人的要因)、探索的diagnostic
    "血統適性", "能力・調教評価", "コース・距離適性", "33ラップ理論適合度", "脚質・展開",
    "騎手・厩舎", "予想印・専門家評価", "近走成績・調子",
]
ORDER_NULL = [  # 真のnullチェック。順序1と辺集合が完全に重複しないよう機械的に選んだ順序
    "能力・調教評価", "血統適性", "脚質・展開", "予想印・専門家評価",
    "コース・距離適性", "近走成績・調子", "33ラップ理論適合度", "騎手・厩舎",
]


def _cyclic_edges(order: list) -> set:
    n = len(order)
    return {frozenset((order[i], order[(i + 1) % n])) for i in range(n)}


for _name, _order in (("ORDER_1", ORDER_1), ("ORDER_2", ORDER_2), ("ORDER_NULL", ORDER_NULL)):
    assert sorted(_order) == sorted(CATEGORIES), f"{_name}がCATEGORIESの並べ替えになっていない"

assert not (_cyclic_edges(ORDER_1) & _cyclic_edges(ORDER_NULL)), (
    "ORDER_1とORDER_NULLの隣接ペア(辺集合)が重複している - 真のnullチェックになっていない"
)

# --------------------------------------------------------------------- pace_fit(新規シグナル)

# H(ハイペース)なら差し・追込有利(+1)/逃げ・先行不利(-1)、S(スローペース)はその逆、
# M(ミドル)は全馬0(差別化情報なし)。
PACE_SIGN = {"H": 1.0, "M": 0.0, "S": -1.0}


def compute_pace_fit(df: pd.DataFrame) -> pd.Series:
    """race_pace_label(H/M/S、レース単位で全馬共通、fetch_corner_position.py 2026-08-29追加)と
    既存ca_running_style_category_label(jra_signals.styleと同じ列)から`pace_fit`を作る。
    race_pace_labelが欠損/未知のレースは全馬NaN(combine_signalsの重み再配分に委ねる)。
    Mペース(pace_sign=0)や全馬同一脚質のレースは、direction*signが全馬同値になり
    _minmaxのhi==lo安全設計で自動的に全NaN化される(=差別化情報なしとして扱う、他シグナルと
    同じ規約)。"""
    label = JS._col(df, "race_pace_label").astype(str).str.strip().str.upper()
    pace_sign = label.map(lambda s: PACE_SIGN.get(s, np.nan))
    if pace_sign.isna().all():
        return pd.Series(np.nan, index=df.index)
    sign = float(pace_sign.dropna().iloc[0])
    style_label = JS._col(df, "ca_running_style_category_label").astype(str).str.strip()
    direction = style_label.map(
        lambda s: 1.0 if s in JS.RUNNING_STYLE_CLOSE else (-1.0 if s in JS.RUNNING_STYLE_FRONT else 0.0)
    )
    return JS._minmax(direction * sign)


# --------------------------------------------------------------------- 33ラップ軸(この検証専用の後処理)

def compute_lap33_axis(race: dict, lap33_fit_all: dict) -> tuple:
    """jra_lap33_signals.lap33_fit_matrix()が返すtype_score/course_refから、この検証専用の
    後処理(type_score==0.0(型不明)をNaNに変換してから再度minmax)を経てlap33軸を作る。
    jra_lap33_signals.py本体(lap33_fit列自体)は使わず、type_score/course_refだけを
    再利用する(無改造)。戻り値: (axis: pd.Series, type_known_mask: pd.Series[bool])。"""
    df = race["df"]
    rec = lap33_fit_all.get(race["race_id"])
    if rec is None or rec["course_ref"] is None:
        return pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)
    ts = rec["type_score"].replace(0.0, np.nan)
    axis = JS._minmax(ts * rec["course_ref"])
    return axis, ts.notna()


# --------------------------------------------------------------------- 軸行列構築

def build_axis_matrix(races: list, priors: dict, history_index, lap33_lookup: dict,
                      race_meta: dict, category_map: dict = None) -> list:
    """races(jra_dataset形式)全体について、補完済み・レース内再正規化済みの軸行列`R`を
    1回だけ構築する。戻り値はracesと同じ順序・同じ長さのlist[dict]で、各要素:
      race_id, kaisai_date, racecourse,
      signals: compute_signals()の生辞書 + pace_fit + lap33_axis + lap33_type_known(診断用に保持),
      raw_axis: 補完・再正規化前のカテゴリ軸DataFrame(診断用、分散比較に使う),
      axis: 補完済み・レース内再minmax済みのカテゴリ軸DataFrame(集約計算はこちらを使う),
      imputed: axisと同shapeのbool DataFrame(Trueならそのセルは他馬平均で補完済み),
      dropped: {category: "missing"|"tied"}(このレースで使えなかったカテゴリとその理由),
      n_race: 生き残ったカテゴリ数。
    N_race<4のレースをこの時点では除外しない(除外は filter_min_categories() で行う、
    計画セクション3手順6: Evaluatorは除外後のracesリストで再構築する必要があるため、
    「どのレースが落ちたか」をraces側と対応付けたまま呼び出し側に委ねる)。

    category_map: 省略時はモジュール既定のCATEGORY_SIGNAL_MAP(8カテゴリ)を使う。
    カテゴリ統合案を検証する呼び出し側(例: jra_radar_axis_merge_search_2026_08_29.py)は、
    任意のカテゴリ分類(メンバーの再グルーピング)をここに渡せる。`lap33_axis`は
    sig辞書に格納済みのため、他カテゴリのメンバーと同列に混ぜて統合することも可能。
    """
    cat_map = category_map if category_map is not None else CATEGORY_SIGNAL_MAP
    categories = list(cat_map.keys())
    lap33_fit_all = L33.lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)

    records = []
    for race in races:
        df = race["df"]
        current_class = JS._class_ordinal(race["race_name"], JS.CLASS_ORDINAL)
        sig = JS.compute_signals(df, current_class, priors)
        sig["pace_fit"] = compute_pace_fit(df)
        lap33_axis, lap33_known = compute_lap33_axis(race, lap33_fit_all)
        sig["lap33_axis"] = lap33_axis
        sig["lap33_type_known"] = lap33_known  # 診断専用、カテゴリ計算には使わない

        raw_cols = {}
        for cat, members in cat_map.items():
            weights = {m: 1.0 for m in members}
            raw_cols[cat] = JS.combine_signals({m: sig[m] for m in members}, weights)
        raw_axis = pd.DataFrame(raw_cols, index=df.index)[categories]

        axis = pd.DataFrame(index=df.index, columns=categories, dtype=float)
        imputed = pd.DataFrame(False, index=df.index, columns=categories)
        dropped = {}
        for cat in categories:
            col = raw_axis[cat]
            if col.isna().all():
                dropped[cat] = "missing"
                continue
            mask = col.isna()
            filled = col.fillna(col.mean())
            renorm = JS._minmax(filled)
            if renorm.isna().all():
                # 補完後も全馬同値(タイ) - 欠損とは別理由として記録する。
                dropped[cat] = "tied"
                continue
            axis[cat] = renorm
            imputed[cat] = mask

        records.append({
            "race_id": race["race_id"], "kaisai_date": race["kaisai_date"],
            "racecourse": race["racecourse"],
            "signals": sig, "raw_axis": raw_axis, "axis": axis, "imputed": imputed,
            "dropped": dropped, "n_race": len(categories) - len(dropped),
        })
    return records


def filter_min_categories(races: list, records: list, min_categories: int = MIN_CATEGORIES) -> tuple:
    """N_race < min_categories のレースを races/records の両方から同じ順序で除外する
    (計画セクション3手順6、Evaluatorはこの除外後のracesリストで再構築すること)。"""
    keep = [i for i, rec in enumerate(records) if rec["n_race"] >= min_categories]
    return [races[i] for i in keep], [records[i] for i in keep]


def drop_categories(records: list, names: list) -> list:
    """指定カテゴリをすべてのレースから強制的に除外した新しいrecordsを返す(カバレッジ
    閾値割れ時のフォールバック用、例: 33ラップ理論適合度・pace_fitが事前登録した50%閾値を
    下回った場合に8軸/7軸版へ差し替える)。元のrecordsは変更しない。"""
    out = []
    for rec in records:
        axis = rec["axis"].drop(columns=[n for n in names if n in rec["axis"].columns])
        dropped = dict(rec["dropped"])
        for n in names:
            dropped.setdefault(n, "excluded_by_coverage_rule")
        out.append({**rec, "axis": axis, "dropped": dropped,
                   "n_race": len(CATEGORIES) - len(dropped)})
    return out


# --------------------------------------------------------------------- 集約方法(5種類)

def _score_from_axis(axis: pd.DataFrame, how: str) -> pd.Series:
    if how == "linear":
        return axis.mean(axis=1, skipna=True)
    if how == "geometric":
        logged = np.log(axis.astype(float) + AREA_EPS)
        return np.exp(logged.mean(axis=1, skipna=True))
    if how == "min":
        return axis.min(axis=1, skipna=True)
    if how == "top3sum":
        return axis.apply(lambda row: row.dropna().nlargest(3).sum(), axis=1)
    raise ValueError(f"unknown how={how!r}")


def _picks_from_scores(score: pd.Series, box_n: int) -> np.ndarray:
    vals = score.to_numpy(dtype=float)
    vals = np.where(np.isnan(vals), -1e18, vals)
    n = len(vals)
    return np.argsort(-vals, kind="stable")[: min(box_n, n)]


def _generic_picks(records: list, how: str, box_n: int) -> list:
    return [_picks_from_scores(_score_from_axis(rec["axis"], how), box_n) for rec in records]


def linear_picks(records: list, box_n: int = 5) -> list:
    return _generic_picks(records, "linear", box_n)


def geometric_mean_picks(records: list, box_n: int = 5) -> list:
    return _generic_picks(records, "geometric", box_n)


def min_picks(records: list, box_n: int = 5) -> list:
    return _generic_picks(records, "min", box_n)


def top3sum_picks(records: list, box_n: int = 5) -> list:
    return _generic_picks(records, "top3sum", box_n)


def radar_area_picks(records: list, order: list, box_n: int = 5) -> list:
    """固定順序`order`でレース内に生き残っている軸だけを巡回させ、隣接ペア積の和から
    面積を計算する(N=そのレースで生き残ったカテゴリ数。落ちたカテゴリはorderの相対順序を
    保ったまま単に飛ばす)。"""
    picks = []
    for rec in records:
        axis = rec["axis"]
        present = [c for c in order if c in axis.columns]
        n = len(present)
        vals = axis[present].to_numpy(dtype=float)
        shifted = np.roll(vals, -1, axis=1)
        area = 0.5 * np.sin(2 * np.pi / n) * (vals * shifted).sum(axis=1)
        picks.append(_picks_from_scores(pd.Series(area, index=axis.index), box_n))
    return picks


# --------------------------------------------------------------------- 無料診断(検定前、コスト0)

def diagnostic_winner_roundness(races: list, records: list, actual: dict) -> tuple:
    """勝ち馬(単勝の的中umaban)とそれ以外で、軸の最小値・最大値・標準偏差の分布を比較する
    (記述統計のみ)。戻り値: (明細DataFrame, group by is_winner の平均値DataFrame)。"""
    rows = []
    for race, rec in zip(races, records):
        win_map = actual.get(race["race_id"], {}).get("単勝", {})
        if not win_map:
            continue
        winner_umaban = int(next(iter(win_map)))
        umaban = pd.to_numeric(race["df"]["umaban"], errors="coerce")
        for idx in rec["axis"].index:
            vals = rec["axis"].loc[idx].dropna()
            if vals.empty:
                continue
            u = umaban.loc[idx]
            rows.append({
                "race_id": race["race_id"],
                "is_winner": bool(pd.notna(u) and int(u) == winner_umaban),
                "min_axis": float(vals.min()), "max_axis": float(vals.max()),
                "std_axis": float(vals.std(ddof=0)),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby("is_winner")[["min_axis", "max_axis", "std_axis"]].mean()
    return detail, summary


def diagnostic_pick_agreement(records: list, box_n: int = 4) -> dict:
    """幾何平均picksと線形平均picksの完全一致率(box_n)、および面積(順序1)picksと線形平均
    picksの完全一致率。一致率が極端に高い場合、後段の検定は「そもそも同じものを比較した」
    可能性がある点をレポート側で明記すること。"""
    lin = linear_picks(records, box_n)
    geo = geometric_mean_picks(records, box_n)
    area1 = radar_area_picks(records, ORDER_1, box_n)

    def _match_rate(a, b):
        if not a:
            return float("nan")
        return float(np.mean([set(x.tolist()) == set(y.tolist()) for x, y in zip(a, b)]))

    return {
        "geometric_vs_linear_exact_match_rate": _match_rate(geo, lin),
        "area_order1_vs_linear_exact_match_rate": _match_rate(area1, lin),
        "n_races": len(records),
    }


def diagnostic_axis_coverage(records: list) -> dict:
    """N_raceヒストグラム、カテゴリ別の脱落理由内訳、33ラップ・pace_fitのカバレッジ率
    (型判定/ペースラベル取得できた馬の割合)、再正規化前後の軸ごとの分散(プールしたstd)。"""
    n_race_hist = pd.Series([rec["n_race"] for rec in records]).value_counts().sort_index()

    drop_counts = {cat: {"missing": 0, "tied": 0} for cat in CATEGORIES}
    for rec in records:
        for cat, reason in rec["dropped"].items():
            drop_counts.setdefault(cat, {}).setdefault(reason, 0)
            drop_counts[cat][reason] += 1

    lap33_known = pd.concat([rec["signals"]["lap33_type_known"] for rec in records], ignore_index=True)
    pace_known = pd.concat(
        [rec["signals"]["pace_fit"].notna() if not rec["signals"]["pace_fit"].isna().all()
         else pd.Series([False] * len(rec["signals"]["pace_fit"]))
         for rec in records], ignore_index=True
    )

    raw_std = {cat: pd.concat([rec["raw_axis"][cat] for rec in records], ignore_index=True).std()
              for cat in CATEGORIES}
    final_std = {cat: pd.concat(
        [rec["axis"][cat] for rec in records if cat in rec["axis"].columns], ignore_index=True
    ).std() if any(cat in rec["axis"].columns for rec in records) else float("nan")
        for cat in CATEGORIES}

    return {
        "n_race_histogram": n_race_hist.to_dict(),
        "category_drop_counts": drop_counts,
        "lap33_type_known_rate": float(lap33_known.mean()) if len(lap33_known) else float("nan"),
        "pace_fit_known_rate": float(pace_known.mean()) if len(pace_known) else float("nan"),
        "raw_axis_std_by_category": raw_std,
        "final_axis_std_by_category": final_std,
    }


def diagnostic_block_count(races: list) -> int:
    return len(set(JE.blocks_of(races)))


def diagnostic_pace_style_correlation(records: list) -> float:
    """「脚質・展開」カテゴリ内でのpace_fit成分単体とstyle成分単体(既存pace_pressure項込み)の
    相関係数(Pearson)。相関が非常に高い場合、pace_fit追加が実質的に情報を増やしていない
    可能性がある点をレポート側で明記すること。"""
    pace = pd.concat([rec["signals"]["pace_fit"] for rec in records], ignore_index=True)
    style = pd.concat([rec["signals"]["style"] for rec in records], ignore_index=True)
    both = pd.concat([pace, style], axis=1, keys=["pace_fit", "style"]).dropna()
    if len(both) < 2:
        return float("nan")
    return float(both["pace_fit"].corr(both["style"]))


def run_all_diagnostics(races: list, records: list, actual: dict) -> dict:
    _, roundness_summary = diagnostic_winner_roundness(races, records, actual)
    return {
        "winner_roundness_summary": roundness_summary.to_dict() if not roundness_summary.empty else {},
        "pick_agreement_box4": diagnostic_pick_agreement(records, box_n=4),
        "axis_coverage": diagnostic_axis_coverage(records),
        "n_blocks": diagnostic_block_count(races),
        "pace_fit_style_correlation": diagnostic_pace_style_correlation(records),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
    import jra_dataset as JD

    print("読み込み中...")
    data = JD.load(rebuild=False)
    races, actual = data["races"], data["actual"]
    priors_all = JS.make_priors([r["df"] for r in races])
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()

    print(f"レース数(除外前): {len(races)}")
    print(f"カテゴリ数: {len(CATEGORIES)}  {CATEGORIES}")

    records = build_axis_matrix(races, priors_all, history_index, lap33_lookup, race_meta)
    races_f, records_f = filter_min_categories(races, records)
    print(f"N_race>={MIN_CATEGORIES}で残ったレース数: {len(races_f)}/{len(races)}")

    diag = run_all_diagnostics(races_f, records_f, actual)
    print("\n--- 無料診断 ---")
    for k, v in diag.items():
        print(f"{k}: {v}")
