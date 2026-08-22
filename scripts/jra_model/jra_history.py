# -*- coding: utf-8 -*-
"""race_results アーカイブから「馬自身の過去走」を復元する層(2026-08-23新設、Step1再挑戦)。

馬柱(newspaper)CSVの past{i}_* 列と**同じ列名・同じ書式**のDataFrameを生成することだけに
責任を持つ(シグナル計算は一切行わない)。これにより jra_signals.compute_signals() を
無改造で再利用できる。

2026-08-22のOpus 5サブエージェントによる実データ検証で確定した対応関係(本モジュールの根拠):
  past{i}_beaten_by の括弧内 = 馬身ではなく「秒差」であり、
      自分の走破タイム − 勝ち馬の走破タイム(1着馬のみ 勝ちタイム − 2着タイム、負値)
      と完全一致する(n=13,494、相関1.0000000、最大絶対差3.4e-14)。
      → race_results.csv の margin列(日本語トークン+帯分数、増分表記、1着はNaN)を
        パースする必要は一切ない。
  past{i}_agari_3f          = last_3f          (完全一致、n=21,118)
  past{i}_finish            = finish_pos       (完全一致)
  past{i}_corner_positions  = passing_order    (完全一致)
  past{i}_race_class 相当   = race_name        ("由布院特別(2勝)"のようにクラスを内包し、
                                                 JS._class_ordinal の解決率100%)

同日2走が物理的に存在しないことは実測で確認済み(91,460行で重複0件、最短間隔6日)。
よって race_date の厳密不等号 `<` によるカットオフで自己参照リークは発生しない
(HorseHistoryIndex.__init__ でassertする)。

なお race_results.csv の margin列(日本語トークン: クビ/ハナ/アタマ/大/同着 + 帯分数)は
本線では使わない。検証専用として parse_margin_lengths() を残す(jra_history_validate_
2026_08_23.py のV11で「使わない判断が正しいこと」を裏付けるためだけに呼ばれる)。
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_signals as JS  # noqa: E402

N_PAST = 5

# --- 検証専用(本線では使わない、jra_history_validate_2026_08_23.pyのV11でのみ参照)。
# JRA公式の一般的な換算慣行: ハナ<アタマ<クビ<1/2<...<大差(10馬身超)。
MARGIN_TOKEN_LENGTHS = {
    "同着": 0.0, "ハナ": 0.05, "アタマ": 0.15, "クビ": 0.30, "大": 10.0,
}
_MIXED_FRACTION_RE = re.compile(r"^(\d+)\.(\d+)/(\d+)$")
_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")


def parse_margin_lengths(s) -> float:
    """【検証専用】race_results.csvのmargin列(馬身、増分表記) -> 数値。
    語彙は{クビ,ハナ,アタマ,大,同着}+帯分数("1.1/4"->1.25)+単純分数("3/4"->0.75)+整数
    のみで構成される(2025年103ファイル全件でパース失敗ゼロを確認済み)。"""
    if pd.isna(s):
        return np.nan
    t = str(s).strip()
    if t in MARGIN_TOKEN_LENGTHS:
        return MARGIN_TOKEN_LENGTHS[t]
    m = _MIXED_FRACTION_RE.match(t)
    if m:
        whole, num, den = m.groups()
        return float(whole) + float(num) / float(den)
    m = _FRACTION_RE.match(t)
    if m:
        num, den = m.groups()
        return float(num) / float(den)
    try:
        return float(t)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------- ロード・派生列
def load_results(years=("2024", "2025", "2026")) -> pd.DataFrame:
    """data/race_results/{year}/*.csv を全件連結する(data/race_results/nar/ は除外)。"""
    frames = []
    for y in years:
        ydir = RESULTS_DIR / y
        if not ydir.exists():
            continue
        for p in sorted(ydir.glob(f"{y}*.csv")):
            frames.append(pd.read_csv(p, dtype=str))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return add_derived_columns(df)


def parse_time_seconds(s) -> float:
    """"1:34.1" -> 94.1(分:秒.コンマ)。jra_signals._time_to_sec と同一パターン。DNFはNaN。"""
    m = re.match(r"^(\d+):(\d+\.?\d*)$", str(s).strip())
    if not m:
        return np.nan
    minutes, seconds = m.groups()
    return float(minutes) * 60.0 + float(seconds)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """sec / finish_num / is_dnf / beaten_by_sec / margin_len(検証専用) を付与する。"""
    df = df.copy()
    df["sec"] = df["time"].map(parse_time_seconds)
    text_finish = df["finish_pos"].astype(str)
    df["is_dnf"] = text_finish.isin(JS.DNF_CODES)
    df["finish_num"] = pd.to_numeric(df["finish_pos"], errors="coerce")

    df["beaten_by_sec"] = np.nan
    for race_id, g in df.groupby("race_id"):
        idx = g.index
        winners = g.index[(g["finish_num"] == 1) & g["sec"].notna()]
        if len(winners) == 0:
            continue
        winner_sec = df.loc[winners[0], "sec"]
        # 通常(単独1着): 勝者自身の値は 勝ちタイム-2着タイム(負値)。
        # 同着1着(稀、~0.15%): 次着(3着)を参照するのは不適切なため0.0とする。
        if len(winners) == 1:
            runner_up = g.index[(g["finish_num"] == 2) & g["sec"].notna()]
            if len(runner_up) > 0:
                runner_up_sec = df.loc[runner_up[0], "sec"]
                df.loc[winners[0], "beaten_by_sec"] = winner_sec - runner_up_sec
            else:
                df.loc[winners[0], "beaten_by_sec"] = 0.0
        else:
            df.loc[winners, "beaten_by_sec"] = 0.0
        others = idx.difference(winners)
        valid_others = others[g.loc[others, "sec"].notna()]
        df.loc[valid_others, "beaten_by_sec"] = df.loc[valid_others, "sec"] - winner_sec

    df["margin_len"] = df["margin"].map(parse_margin_lengths)
    return df


def is_starter(df: pd.DataFrame) -> pd.Series:
    """出走取消「取」・除外「除」を落とす(odds_finalが数値 かつ popularityが数値)。
    競走中止「中」は残す(賭けの対象として存在したため)。"""
    odds = pd.to_numeric(df["odds_final"], errors="coerce")
    ninki = pd.to_numeric(df["popularity"], errors="coerce")
    return odds.notna() & ninki.notna()


# --------------------------------------------------------------------- インデックス
@dataclass(frozen=True)
class PastStart:
    race_id: str
    race_date: str
    racecourse: str
    race_name: str
    finish_pos: str
    beaten_by_sec: float
    winner_name: str
    last_3f: str
    passing_order: str
    field_size: int


class HorseHistoryIndex:
    """horse_id -> race_date昇順のPastStartリスト。"""

    def __init__(self, results: pd.DataFrame):
        dup = results.groupby(["horse_id", "race_date"]).size()
        assert dup.max() <= 1, (
            "同一(horse_id, race_date)の重複出走が見つかりました。"
            "race_date粒度のカットオフ前提が崩れています。"
        )
        self._results = results
        self._by_horse: dict[str, list[PastStart]] = {}
        field_size = results.groupby("race_id")["horse_id"].transform("size")
        winner_name_map = (
            results[results["finish_num"] == 1]
            .drop_duplicates("race_id")
            .set_index("race_id")["horse_name"]
        )
        results = results.assign(field_size=field_size)
        for horse_id, g in results.groupby("horse_id"):
            g = g.sort_values("race_date")
            rows = []
            for _, r in g.iterrows():
                rows.append(PastStart(
                    race_id=r["race_id"], race_date=r["race_date"],
                    racecourse=r.get("racecourse", np.nan), race_name=r["race_name"],
                    finish_pos=r["finish_pos"], beaten_by_sec=r["beaten_by_sec"],
                    winner_name=winner_name_map.get(r["race_id"], np.nan),
                    last_3f=r["last_3f"], passing_order=r["passing_order"],
                    field_size=int(r["field_size"]),
                ))
            self._by_horse[str(horse_id)] = rows

    @property
    def min_date(self) -> str:
        return self._results["race_date"].min()

    @property
    def available_dates(self) -> set:
        return set(self._results["race_date"].unique())

    def past_starts(self, horse_id, before_date: str, n: int = N_PAST) -> list:
        """race_date < before_date を満たす直近n走を新しい順に返す(厳密不等号)。"""
        rows = self._by_horse.get(str(horse_id), [])
        out = [r for r in rows if r.race_date < before_date]
        return out[-n:][::-1]

    def n_prior(self, horse_id, before_date: str) -> int:
        """アーカイブ内の通算出走回数(履歴被覆の診断・バーンイン判定用)。"""
        rows = self._by_horse.get(str(horse_id), [])
        return sum(1 for r in rows if r.race_date < before_date)


# --------------------------------------------------------------------- 馬柱互換フレーム
def past_frame(index: HorseHistoryIndex, horse_ids, race_date: str, n: int = N_PAST) -> pd.DataFrame:
    """馬柱CSVと同じ列名のDataFrameを返す(index=horse_idsの順序に対応する連番)。
    過去走がn本に満たないスロットは全列NaN。"""
    cols = {}
    for i in range(1, n + 1):
        cols[f"past{i}_finish"] = []
        cols[f"past{i}_beaten_by"] = []
        cols[f"past{i}_agari_3f"] = []
        cols[f"past{i}_corner_positions"] = []
        cols[f"past{i}_field_size"] = []
        cols[f"past{i}_race_class"] = []
        cols[f"past{i}_date"] = []
        cols[f"past{i}_race_id"] = []

    for hid in horse_ids:
        starts = index.past_starts(hid, race_date, n=n)
        for i in range(1, n + 1):
            if i <= len(starts):
                s = starts[i - 1]
                cols[f"past{i}_finish"].append(s.finish_pos)
                if pd.isna(s.beaten_by_sec):
                    cols[f"past{i}_beaten_by"].append(np.nan)
                else:
                    name = s.winner_name if not pd.isna(s.winner_name) else ""
                    cols[f"past{i}_beaten_by"].append(f"{name}({s.beaten_by_sec:.1f})")
                cols[f"past{i}_agari_3f"].append(s.last_3f)
                cols[f"past{i}_corner_positions"].append(s.passing_order)
                cols[f"past{i}_field_size"].append(s.field_size)
                cols[f"past{i}_race_class"].append(s.race_name)
                cols[f"past{i}_date"].append(s.race_date)
                cols[f"past{i}_race_id"].append(s.race_id)
            else:
                for key in (f"past{i}_finish", f"past{i}_beaten_by", f"past{i}_agari_3f",
                            f"past{i}_corner_positions", f"past{i}_field_size",
                            f"past{i}_race_class", f"past{i}_date", f"past{i}_race_id"):
                    cols[key].append(np.nan)

    return pd.DataFrame(cols, index=range(len(horse_ids)))


def coverage_frame(index: HorseHistoryIndex, horse_ids, race_date: str) -> pd.DataFrame:
    """n_prior(通算出走回数) を返す(バーンイン判定・欠損指標ダミー用)。"""
    return pd.DataFrame({
        "n_prior": [index.n_prior(hid, race_date) for hid in horse_ids],
    }, index=range(len(horse_ids)))


if __name__ == "__main__":
    results = load_results()
    print(f"loaded {len(results)} rows, {results['race_id'].nunique()} races, "
          f"{results['horse_id'].nunique()} unique horses")
    idx = HorseHistoryIndex(results)
    print(f"index built: {len(idx._by_horse)} horses, min_date={idx.min_date}")
