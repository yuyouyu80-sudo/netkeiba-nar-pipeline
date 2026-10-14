# -*- coding: utf-8 -*-
"""クッション値の作業を再開するときに、保留(2026-10-14)のあと何がどれだけ溜まったかを見る。読むだけで何も書かない。

  python -X utf8 scripts/jra_model/cushion_2026_10_13/resume_status.py

見るもの(RESUME.md の「再開の条件」と対応):
- 当日朝の馬場情報(data/jra_baba_live): 取得日数、クッション値が取れた 競馬場×開催日 の数
- 開催回まとめのPDF(data/jra_baba): 最後の開催日
- 気象庁の観測(data/jma_amedas)と予報(data/jma_forecast): 最新の時刻・取得日数
- レース結果(data/race_results): 保留後の開催日数(前向きの確認に使える、まだ誰も見ていない期間)
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
PAUSED = "2026-10-14"


def live():
    fs = sorted(glob.glob(str(DATA / "jra_baba_live" / "*.csv")))
    if not fs:
        return "当日朝の馬場情報: まだ無い"
    d = pd.concat([pd.read_csv(f, dtype=str, encoding="utf-8-sig") for f in fs], ignore_index=True)
    c = d[d["cushion"].notna() & (d["cushion"] != "")]
    days = c["measure_time"].str[:10]
    after = c[days >= PAUSED]
    return (f"当日朝の馬場情報: 取得 {len(fs)} 日分({Path(fs[0]).stem}〜{Path(fs[-1]).stem})、"
            f"クッション値のある 競馬場×測定日 {c[['venue']].assign(d=days).drop_duplicates().shape[0]} 件"
            f"(うち保留後 {after[['venue']].assign(d=after['measure_time'].str[:10]).drop_duplicates().shape[0]} 件)")


def archive():
    last = None
    for f in glob.glob(str(DATA / "jra_baba" / "*" / "*.csv")):
        d = pd.read_csv(f, dtype=str, usecols=lambda c: c == "date")
        if len(d):
            m = d["date"].max()
            last = m if last is None or m > last else last
    return (f"開催回まとめのPDF: 最後の開催日 {last}(開催回が終わった翌平日に公開。"
            f"再開前に収集用の会話で fetch_jra_baba.py を回して、保留までの分を埋める)")


def weather():
    lasts = []
    for p in sorted((DATA / "jma_amedas").glob("[0-9][0-9]")):
        fs = sorted(p.glob("*.csv"))
        if fs:
            d = pd.read_csv(fs[-1], dtype=str, encoding="utf-8-sig", usecols=["datetime"])
            lasts.append(d["datetime"].max())
    fc = sorted((DATA / "jma_forecast").glob("*.csv"))
    s = f"気象庁の観測: 10競馬場の最新 {min(lasts) if lasts else '-'}〜{max(lasts) if lasts else '-'}"
    s += f" / 予報: {len(fc)} 日分" + (f"({fc[0].stem}〜{fc[-1].stem})" if fc else "")
    return s


def races():
    fs = sorted(glob.glob(str(DATA / "race_results" / "2026" / "2026*.csv")))
    days = [Path(f).stem for f in fs]
    after = [x for x in days if x >= PAUSED.replace("-", "")]
    return f"レース結果: 最後の開催日 {days[-1] if days else '-'}、保留後の開催日 {len(after)} 日"


def main():
    print(f"クッション値 再開前の確認(保留 {PAUSED})")
    for fn in (live, archive, weather, races):
        try:
            print("-", fn())
        except Exception as e:  # 1つ壊れても残りは出す
            print("-", fn.__name__, "読めない:", e)


if __name__ == "__main__":
    main()
