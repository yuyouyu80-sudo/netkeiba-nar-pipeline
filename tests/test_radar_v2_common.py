"""血統レーダー v2 共通処理(scripts/jra_model/pedigree_reports/radar_v2_common.py)のテスト(2026-10-06)。
合成データのみで、実データの読み込みはしない。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import radar_v2_common as C  # noqa: E402


def test_is_jump_catches_all_jump_name_styles():
    names = pd.Series(["障害3歳以上未勝利", "第25回中山大障害(JGI)", "牛若丸ジャンプS(OP)", "三木ホースランドJS(OP)",
                       "第20回阪神スプリングJ(JGII)", "イルミネーションJS(OP)",
                       "3歳未勝利", "第86回菊花賞(GI)", "ステイヤーズS(GII)", "ジャパンC(GI)", None])
    assert C.is_jump(names).tolist() == [True] * 6 + [False] * 5


def test_parse_finish_keeps_demotion_and_drops_non_finish():
    s = pd.Series(["1", "3(降)", "12(再)", "中", "除", "取", "失", None])
    out = C.parse_finish(s)
    assert out.iloc[:3].tolist() == [1, 3, 12]
    assert out.iloc[3:].isna().all()


def test_dedupe_prefers_race_number_match_then_min_race_id():
    df = pd.DataFrame({
        "horse_id": ["h1", "h1", "h2", "h2", "h3"],
        "race_date": ["2018-02-03"] * 5,
        "race_id": ["201805010101", "201805010104", "201805010202", "201805010203", "201805010301"],
        "race_number": ["4", "4", "9", "9", "1"],
        # 完全一致しない行(着順が違う)なので drift 除去では落ちず、追補1の規則で決まる
        "finish_pos": ["1", "2", "3", "4", "5"], "time": [None] * 5, "odds_final": [None] * 5,
        "jockey_name": ["a"] * 5, "race_name": ["x"] * 5, "racecourse": ["東京"] * 5,
    })
    out, meta = C.dedupe(df)
    assert set(out["race_id"]) == {"201805010104", "201805010202", "201805010301"}
    assert meta["addendum1_dropped"] == 2


def test_first_corner_priority_is_1_then_2_then_3():
    df = pd.DataFrame({"corner1_rank": [None, "3", None], "corner2_rank": ["5", "4", None],
                       "corner3_rank": ["7", "6", "8"]})
    rank, idx = C.first_corner(df)
    assert rank.tolist() == [5, 3, 8]
    assert idx.tolist() == [2, 1, 3]


def test_center_horse_values_weighted_mean_is_zero():
    hv = pd.DataFrame({"x": [0.1, -0.3, 0.5, 0.2], "n": [1, 3, 20, 7]})
    out = C.center_horse_values(hv)
    assert abs(np.average(out["xc"], weights=out["w"])) < 1e-12
    assert out["w"].tolist() == [1, 3, 5, 5]


def test_shrink_limits():
    anc = pd.DataFrame({"mean": [0.4, -0.2], "sw": [10.0, 3.0]})
    assert np.allclose(C.shrink(anc, 0.0), anc["mean"])
    parent = pd.Series([0.05, 0.05])
    assert np.allclose(C.shrink(anc, 1e12, parent), parent)


def test_selection_bias_positive_control():
    """真の距離適性が0の合成データで、ある距離帯で凡走した馬がその距離帯を離れる(=その帯の走が少ない)。
    走単位の中心化+馬単位の平均だと符号が偏るが、馬単位で集計した後の加重中心化では祖先間で偏らないこと。"""
    rng = np.random.default_rng(1)
    rows = []
    for anc in range(200):
        for h in range(30):
            hid = f"a{anc}h{h}"
            ability = rng.normal()
            first = ability + rng.normal()
            n_b = 1 if first < 0 else 6  # 凡走した馬は1回で離れる
            for i in range(n_b):
                perf = first if i == 0 else ability + rng.normal()
                rows.append((hid, f"s{anc}", perf))
    runs = pd.DataFrame(rows, columns=["horse_id", "sire", "c"])
    runs["c"] = runs["c"] - runs["c"].mean()  # 走単位の中心化(R−1の方式)
    hv = C.horse_values(runs, "c")
    ped = runs[["horse_id", "sire"]].drop_duplicates().rename(columns={"sire": "ped_S_horse_id"})
    # R−1方式: 馬単位で平均(重みなし)すると、全体が負に偏る
    hv_naive = hv.join(ped.set_index("horse_id"))
    naive = hv_naive.groupby("ped_S_horse_id")["x"].mean()
    assert (naive > 0).mean() < 0.2
    # v2方式: 馬単位で集計した後に加重中心化 → 祖先の正の割合が0.35〜0.65
    hvc = C.center_horse_values(hv)
    anc_tab = C.ancestor_table(hvc, ped, "ped_S_horse_id")
    share = (anc_tab["mean"] > 0).mean()
    assert 0.35 <= share <= 0.65
