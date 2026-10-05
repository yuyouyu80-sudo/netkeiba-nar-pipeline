# -*- coding: utf-8 -*-
"""荒れ度分析 手順1-a: レース単位テーブルを作る(2011〜2026 JRA平地)。

入力(読み取りのみ): data/race_results/{year}/*.csv, data/payouts/{year}/*.csv
出力: analysis/arere_2026_10_05/out/races_all.csv.gz(1行1レース)

「荒れた」= 3連複払戻(100円あたり)が10,000円以上。同着で複数あれば最大値で判定。
障害戦(レース名に「障害」「ジャンプ」/J・G表記、または芝ダ欠損)は除外。
(「○○ジャンプS」は芝ダ列が「芝」で入っているため名前で除外する)
予測側に入れる列は発走前に分かるものだけ(レース条件・確定単勝オッズ)。
着順・通過順・上がり・当日馬体重は「事後」列(post_*)としてのみ保持し、モデルには使わない。
"""
import glob
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
         "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def category_of(name):
    s = str(name)
    if "新馬" in s:
        return "新馬"
    if "未勝利" in s:
        return "未勝利"
    return "一般"


def class_of(name):
    """旧表記(500万下/1000万下/1600万下)を新表記へ正規化。"""
    s = str(name)
    if "新馬" in s:
        return "新馬"
    if "未勝利" in s:
        return "未勝利"
    if "500万" in s or "1勝" in s:
        return "1勝"
    if "1000万" in s or "2勝" in s:
        return "2勝"
    if "1600万" in s or "3勝" in s:
        return "3勝"
    if re.search(r"\((GI|GII|GIII|重賞|G1|G2|G3)\)\s*$", s):
        return "重賞"
    if re.search(r"\((OP|L|オープン)\)\s*$", s) or "オープン" in s:
        return "OP"
    return "不明"


def age_cond(name):
    s = str(name)
    if "2歳" in s:
        return "2歳"
    if "3歳以上" in s or "4歳以上" in s:
        return "古馬混合"
    if "3歳" in s:
        return "3歳"
    return "不明"  # 特別戦名のみのレース(条件表記なし)


def dist_band(d):
    if d <= 1200:
        return "~1200"
    if d <= 1600:
        return "1300-1600"
    if d <= 2000:
        return "1700-2000"
    if d <= 2400:
        return "2100-2400"
    return "2500~"


def harville_p_cheap(p, thr=0.0075):
    """単勝の市場確率pからHarvilleで全3頭組の3着内確率を作り、
    推定3連複オッズ(=0.75/P)が100倍未満(=払戻1万円未満)になる組の確率合計を返す。
    戻り値: (P(安い組が来る), 推定オッズ最小の組の確率)"""
    n = len(p)
    if n < 3:
        return np.nan, np.nan
    idx = np.array(list(itertools.combinations(range(n), 3)))
    tot = np.zeros(len(idx))
    for perm in itertools.permutations(range(3)):
        a, b, c = idx[:, perm[0]], idx[:, perm[1]], idx[:, perm[2]]
        pa, pb, pc = p[a], p[b], p[c]
        tot += pa * pb / (1 - pa) * pc / (1 - pa - pb)
    return float(tot[tot >= thr].sum()), float(tot.max())


def odds_features(odds):
    o = np.sort(np.asarray(odds, float))
    n = len(o)
    q = 1.0 / o
    p = q / q.sum()
    ps = np.sort(p)[::-1]
    ent = float(-(p * np.log(p)).sum() / np.log(n))
    gini = float(np.abs(p[:, None] - p[None, :]).sum() / (2 * n * n * p.mean()))
    pc, pmax = harville_p_cheap(p)
    return dict(
        fav_odds=o[0], odds2=o[1] if n > 1 else np.nan, odds3=o[2] if n > 2 else np.nan,
        odds_gap12=np.log(o[1] / o[0]) if n > 1 else np.nan,
        odds_gap23=np.log(o[2] / o[1]) if n > 2 else np.nan,
        top3_share=float(ps[:3].sum()), entropy=ent, gini=gini,
        n_under10=int((o < 10).sum()), n_under5=int((o < 5).sum()),
        overround=float(q.sum()),
        harv_p_cheap=pc, harv_p_best=pmax,
    )


def load_payouts():
    rows = []
    for y in range(2011, 2027):
        for f in sorted(glob.glob(str(ROOT / "data" / "payouts" / str(y) / "*.csv"))):
            d = pd.read_csv(f, dtype=str)
            d = d[d["bet_type"] == "3連複"]
            rows.append(d[["race_id", "payout", "popularity"]])
    p = pd.concat(rows)
    p["payout"] = pd.to_numeric(p["payout"].str.replace(",", ""), errors="coerce")
    p["popularity"] = pd.to_numeric(p["popularity"], errors="coerce")
    g = p.groupby("race_id").agg(sanrenpuku=("payout", "max"), sanrenpuku_pop=("popularity", "max"),
                                 n_sanrenpuku_rows=("payout", "size"))
    return g


def race_record(rid, g):
    """1レース分の race_results 行(DataFrame)から、レース単位の1行(dict)を作る。障害戦・3頭未満は None。
    s6_race_scores.py(予想ページ表示用)からも使う。"""
    name = str(g["race_name"].iloc[0])
    surf = g["surface"].iloc[0]
    if "障害" in name or "ジャンプ" in name or re.search(r"JS\(", name) or re.search(r"\(JG", name) or surf not in ("芝", "ダ"):
        return None
    fin = g["finish_pos"].astype(str)
    started = g[~fin.isin(["取", "除"])].copy()
    n = len(started)
    if n < 3:
        return None
    odds = pd.to_numeric(started["odds_final"], errors="coerce")
    kin = pd.to_numeric(started["kinryo"], errors="coerce")
    sex = started["sex_age"].str[0]
    age = pd.to_numeric(started["sex_age"].str[1:], errors="coerce")
    # ハンデ戦の推定(元データにハンデ表記が無いため斤量から推定。見習い減量の区別ができない点に注意)
    k_adj = kin + np.where(sex == "牝", 2.0, 0.0)
    grp_rng = pd.DataFrame({"k": k_adj, "a": np.minimum(age, 4)}).groupby("a")["k"].agg(lambda s: s.max() - s.min())
    half = bool(((kin * 2) % 2 == 1).any())
    cls = class_of(name)
    hcap = cls not in ("新馬", "未勝利") and (half or (grp_rng.max() >= 6))
    dist = int(float(g["distance_m"].iloc[0]))
    fin_num = pd.to_numeric(started["finish_pos"].str.extract(r"^(\d+)")[0], errors="coerce")
    pop = pd.to_numeric(started["popularity"], errors="coerce")
    top3 = started[fin_num <= 3]
    rec = dict(
        race_id=rid, date=g["race_date"].iloc[0], year=int(g["race_date"].iloc[0][:4]),
        month=int(g["race_date"].iloc[0][5:7]), venue=VENUE.get(rid[4:6], rid[4:6]),
        kai=int(rid[6:8]), nichi=int(rid[8:10]), race_no=int(rid[10:12]),
        race_name=name, category=category_of(name), cls=cls, age_cond=age_cond(name),
        surface=surf, distance=dist, dist_band=dist_band(dist),
        weather=g["weather"].iloc[0], going=g["going"].iloc[0],
        n_horses=n, n_scratched=int(fin.isin(["取", "除"]).sum()),
        filly_only=bool((sex == "牝").all()), handicap_est=bool(hcap),
        odds_complete=bool(odds.notna().all()),
        # 事後列(モデルには使わない)
        post_n_dnf=int(fin.isin(["中", "失"]).sum()),
        post_top3_pop_max=float(pop[fin_num <= 3].max()) if len(top3) else np.nan,
        post_fav_finish=float(fin_num[pop == pop.min()].min()) if pop.notna().any() else np.nan,
    )
    if odds.notna().all() and (odds > 0).all():
        rec.update(odds_features(odds.to_numpy()))
    return rec


def main():
    pay = load_payouts()
    recs = []
    for y in range(2011, 2027):
        files = sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / "*.csv")))
        for f in files:
            d = pd.read_csv(f, dtype=str)
            for rid, g in d.groupby("race_id", sort=False):
                rec = race_record(rid, g)
                if rec is not None:
                    recs.append(rec)
        print(y, len(recs), flush=True)
    df = pd.DataFrame(recs)
    df = df.merge(pay, left_on="race_id", right_index=True, how="left")
    df["arere"] = (df["sanrenpuku"] >= 10000).astype("Int64")
    df.loc[df["sanrenpuku"].isna(), "arere"] = pd.NA
    df.to_csv(OUT / "races_all.csv.gz", index=False, encoding="utf-8")
    print("saved", len(df), "races; 3連複なし", int(df["sanrenpuku"].isna().sum()))
    print(df.groupby(["year"]).size().to_string())


if __name__ == "__main__":
    main()
