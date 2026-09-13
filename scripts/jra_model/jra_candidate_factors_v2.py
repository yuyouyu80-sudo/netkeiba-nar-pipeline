# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-09-02、「JRAファクター検証データベースに予想ファクター充足度マップの
提案ファクターも追加してほしい」)による追加ファクター v2。

v1(`jra_candidate_factors_v1.py`)と同じ設計方針を踏襲する: 既存の本番スコア
(`jra_model_scoring.py`)・レーダー8カテゴリ(`jra_radar_categories.py`)は無改造、
このモジュールが完全に自己完結で計算する追加専用シグナル。build_factor_dataset.py から
呼ばれ、ファクター検証データベース(factor_database.json)の「新規候補ファクター v2」タイア
(jra_factor_registry.py)としてのみ使われる(predict_pattern29.py等の本番予想には一切
反映しない)。

対象8項目は「予想ファクター充足度マップ」で「計算のみで対応可」と判定された項目のうち、
実データ確認済みの列だけで安全に計算できるもの(自由記述テキストの分類ルール化・
コース形状の細かい事実断定・レース名からの性別限定判定など、根拠が薄くなりやすい項目は
今回見送った。理由はコード内コメント参照)。

データ源:
  - class_form_place_rate/class_challenge_flag: jra_candidate_factors_v1.build_history_index()
    が作るhorse_id別履歴(race_date昇順、point-in-time正しい)を再利用。
  - layoff_flag/second_after_layoff_flag/main_jockey_return_flag/bad_run_popularity_drop_flag:
    newspaper由来のdf自身の列(past1〜5_date/past1〜5_jockey/past1_finish/bias_ninki/
    bias_jockey)のみを使い、履歴インデックスへの参照は不要(このレースのCSV行だけで完結)。
  - kaisai_day_num: race_idの構造(YYYY+venue+kai+nichime+raceno)から機械的に算出
    (jra_candidate_factors_v1.kaisai_late_flag と同じ抽出ロジック、閾値を固定せず生値を返す)。

見送った項目とその理由(2026-09-02時点):
  - 直線の長さへの適性: 競馬場ごとの直線距離はコースによって内回り/外回りが複数存在し
    (新潟・京都・阪神等)、断定的な分類表を作ると実態と食い違うリスクが高いため見送り。
  - 牝馬限定戦か混合戦か: race_nameは"両津湾特別(2勝)"のようなレース名のみで性別限定の
    有無を含まないことを実データで確認(2026-08-30分race_resultsで全レース名を確認)。
    信頼できる列が無いため見送り。
  - 調教コメント・併せ馬コメントのテキスト分類系(手応え/行きっぷり等): 自由記述の分類
    ルール化は主観的判断が入り誤分類リスクが高いため、今回は見送り(将来の別ラウンド候補)。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent

sys.path.insert(0, str(LIB_DIR))
import jra_signals as JS  # noqa: E402

_HIST_COLS_NEEDED = {"race_date", "finish_num", "race_name"}  # v1のhist_idxが持つ列のうち使うもの

# past1〜5_dateは"2026.07.05"のようにドット区切り(race_results"2026-08-30"・
# kaisai_date"20260830"のいずれとも異なる第3の書式、実データ確認済み)。
_DATE_DOT_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")


def _parse_dot_date(s):
    """"2026.07.05" -> pandas.Timestamp。パース不能ならNaT。"""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if not _DATE_DOT_RE.match(s):
        return pd.NaT
    return pd.to_datetime(s, format="%Y.%m.%d", errors="coerce")


def kaisai_day_num(race_id: str):
    """race_id構造([8:10]=日目)から開催日目の生値を返す(kaisai_late_flagの精緻版、
    固定閾値7日ではなく任意の閾値で絞り込めるようにする)。"""
    if not isinstance(race_id, str) or len(race_id) != 12:
        return None
    try:
        return int(race_id[8:10])
    except ValueError:
        return None


def compute_class_factors(df: pd.DataFrame, race_name: str, kaisai_date: str, hist_idx: dict) -> dict:
    """現クラスでの実績(class_form_place_rate)・格上挑戦(class_challenge_flag)。
    v1のhist_idx(horse_id別、race_date="YYYYMMDD"昇順)をそのまま再利用する。"""
    horse_ids = JS._col(df, "horse_id")
    today_class = JS._class_ordinal(race_name, JS.CLASS_ORDINAL_V3EXT)

    place_rates, challenge_flags = [], []
    for i in df.index:
        hid = horse_ids.at[i]
        g = hist_idx.get(str(hid))
        if g is None or g.empty:
            place_rates.append(None)
            challenge_flags.append("no_history")
            continue
        hist = g[g["race_date"] < kaisai_date]
        if hist.empty:
            place_rates.append(None)
            challenge_flags.append("no_history")
            continue

        levels = hist["race_name"].map(lambda t: JS._class_ordinal(t, JS.CLASS_ORDINAL_V3EXT))
        if pd.isna(today_class):
            place_rates.append(None)
        else:
            sub = hist[levels >= today_class]
            if sub.empty:
                place_rates.append(None)
            else:
                n = len(sub)
                k = int((sub["finish_num"] <= 3).sum())
                place_rates.append(round(100.0 * k / n, 1))

        max_past = levels.max()
        if pd.isna(today_class) or pd.isna(max_past):
            challenge_flags.append("unknown")
        elif today_class > max_past:
            challenge_flags.append("up_challenge")
        else:
            challenge_flags.append("at_or_below_max")

    return {
        "class_form_place_rate": pd.Series(place_rates, index=df.index, dtype=object),
        "class_challenge_flag": pd.Series(challenge_flags, index=df.index, dtype=object),
    }


def compute_layoff_factors(df: pd.DataFrame, kaisai_date: str) -> dict:
    """鉄砲(休養明け初戦、layoff_flag)・叩き2戦目(second_after_layoff_flag)。
    df自身のpast1_date/past2_dateのみ使用(履歴インデックス不要)。"""
    kaisai_dt = pd.to_datetime(kaisai_date, format="%Y%m%d", errors="coerce")
    past1_date = JS._col(df, "past1_date").map(_parse_dot_date)
    past2_date = JS._col(df, "past2_date").map(_parse_dot_date)

    layoff, second_after = [], []
    for i in df.index:
        p1 = past1_date.at[i]
        if pd.isna(p1) or pd.isna(kaisai_dt):
            layoff.append("no_history" if pd.isna(p1) else "unknown")
        else:
            days = (kaisai_dt - p1).days
            if days >= 180:
                layoff.append("long_layoff_180d_plus")
            elif days >= 90:
                layoff.append("mid_layoff_90_179d")
            else:
                layoff.append("normal_interval")

        p2 = past2_date.at[i]
        if pd.isna(p1) or pd.isna(p2):
            second_after.append("insufficient_history")
        else:
            gap1 = (p1 - p2).days
            second_after.append("second_after_layoff" if gap1 >= 120 else "not_applicable")

    return {
        "layoff_flag": pd.Series(layoff, index=df.index, dtype=object),
        "second_after_layoff_flag": pd.Series(second_after, index=df.index, dtype=object),
    }


def compute_main_jockey_return(df: pd.DataFrame) -> pd.Series:
    """主戦騎手への手戻り(main_jockey_return_flag)。過去1〜5走のjockey列(フルネーム)の
    最頻出騎手を「主戦」と定義し、今回のbias_jockey(姓のみ等の短縮表記)が前方一致で
    その主戦騎手と同一人物とみなせるかを判定する(v1のjockey_switchと同じ前方一致判定を再利用、
    同姓の別人を誤って「同一」と判定するリスクは同様に残る)。"""
    cols = [f"past{n}_jockey" for n in range(1, 6) if f"past{n}_jockey" in df.columns]
    bias_jockey = JS._col(df, "bias_jockey")

    out = []
    for i in df.index:
        names = [str(df.at[i, c]).strip() for c in cols
                 if c in df.columns and pd.notna(df.at[i, c]) and str(df.at[i, c]).strip()]
        cj = bias_jockey.at[i] if i in bias_jockey.index else np.nan
        cj_s = str(cj).strip() if pd.notna(cj) else ""
        if not names or not cj_s:
            out.append("unknown")
            continue
        counts = pd.Series(names).value_counts()
        top_name, top_count = counts.index[0], int(counts.iloc[0])
        if top_count < 2:
            out.append("no_clear_main")
            continue
        is_match = top_name.startswith(cj_s) or cj_s.startswith(top_name)
        out.append("return_to_main" if is_match else "away_from_main")
    return pd.Series(out, index=df.index, dtype=object)


def compute_bad_run_popularity_drop(df: pd.DataFrame) -> pd.Series:
    """前走大敗による人気落ち(bad_run_popularity_drop_flag)。前走着順が悪く(10着以下)、
    かつ今回人気が低い(7番人気以下)組み合わせを検出する(「不利・馬場が原因の敗戦で
    人気だけ下がった」可能性がある馬、という仮説の材料。不利の有無自体は判定できない点に注意)。"""
    past1_finish = JS._num(JS._col(df, "past1_finish"))
    ninki = JS._num(JS._col(df, "bias_ninki"))
    out = []
    for i in df.index:
        pf, nk = past1_finish.at[i], ninki.at[i]
        if pd.isna(pf) or pd.isna(nk):
            out.append("unknown")
        elif pf >= 10 and nk >= 7:
            out.append("bad_run_then_low_pop")
        else:
            out.append("other")
    return pd.Series(out, index=df.index, dtype=object)


def compute_for_race(df: pd.DataFrame, race_name: str, kaisai_date: str, hist_idx: dict) -> dict:
    """レース1本ぶんのv2追加ファクターをdfのindexに揃えたSeries dictで返す。"""
    out = {}
    out.update(compute_class_factors(df, race_name, kaisai_date, hist_idx))
    out.update(compute_layoff_factors(df, kaisai_date))
    out["main_jockey_return_flag"] = compute_main_jockey_return(df)
    out["bad_run_popularity_drop_flag"] = compute_bad_run_popularity_drop(df)
    return out
