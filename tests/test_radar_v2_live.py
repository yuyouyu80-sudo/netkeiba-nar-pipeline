"""血統レーダー v2 の R6(radar_v2_live.py、出走表から値を作る・前向きの記録)のテスト(2026-10-07)。合成データのみ。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import radar_v2_live as L  # noqa: E402
import radar_v2_reference as RF  # noqa: E402


def _synthetic_R(seed=0, n_days=60):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2020-01-04", periods=n_days, freq="7D")
    for i, d in enumerate(dates):
        for j in range(6):
            rows.append({"race_id": f"r{i:03d}{j}", "date": d, "L1": f"c{j % 3}|k{j % 2}|d{i % 4}", "L2": f"c{j % 3}|k{j % 2}",
                         "L3": f"s{j % 2}|b{j % 3}", "L4": f"s{j % 2}", "lift_front": rng.normal(0.02 * (j % 3), 0.1),
                         "pace_tier": ["速い", "平均", "遅い"][(i + j) % 3]})
    return pd.DataFrame(rows).set_index("race_id")


def test_req_new_matches_build_req_for_races_on_the_last_day():
    """その日より前のレースだけを使う req_new が、全期間を通した RF.build_req のその日のレースの値と一致する(全体・ペース別とも)。"""
    R = _synthetic_R()
    last = R["date"].max()
    X = R[R["date"] == last]
    prior = R[R["date"] < last]
    for tier in [None, "速い"]:
        Rt, Pt = R.copy(), prior.copy()
        if tier:
            # build_req は値の空の行を除いてから (key, 日付) で突き合わせる。当日のレースの値は自分のペース区分なら残す
            # (その日の行が残っていれば build_req も正しい)ので、比較は当日のレースの値を残したまま行う。
            Pt.loc[Pt["pace_tier"] != tier, "lift_front"] = np.nan
            Rt.loc[(Rt["pace_tier"] != tier) & (Rt["date"] < last), "lift_front"] = np.nan
        a = RF.build_req(Rt, "lift_front", 3.0).loc[X.index]
        b = L.req_new(Pt, X, "lift_front", 3.0)
        assert np.allclose(a["req"], b["req"]), tier
        assert (a["rule"] == b["rule"]).all()
        assert np.allclose(a["n_matched"], b["n_matched"])


def test_req_new_tier_uses_only_prior_races_of_that_tier():
    """ペース別の要求は、当日のレース自身のペース区分によらず、そのペースだった過去レースだけから作る(R5 の表示の不具合の再発防止)。"""
    R = _synthetic_R(seed=1)
    last = R["date"].max()
    X = R[R["date"] == last]
    P = R[R["date"] < last].copy()
    P.loc[P["pace_tier"] != "遅い", "lift_front"] = np.nan
    b = L.req_new(P, X, "lift_front", 3.0)
    # 当日の6レースはペース区分が混在しているが、どれも「遅い」の過去レースの集計から値が出る(全体平均との差 ≈ 0 に潰れない)
    assert (b["n_matched"] > 0).all()
    assert b["req"].abs().max() > 1e-3


def test_norm_name_and_class_labels():
    assert L.norm_name("Giant's Causeway (米)") == L.norm_name("Giant's  Causeway")
    assert L.norm_name("キタサンブラック") == "キタサンブラック"
    news = pd.DataFrame({"data_others_slot1_label": ["中2週", "連闘", "1ヶ月"], "data_others_slot2_label": ["２勝クラス", "２勝クラス", "nan"],
                         "data_others_slot3_label": ["56kg", "55kg", "２勝クラス"]})
    assert L.race_class_from_labels(news) == 2.0
    news2 = pd.DataFrame({"data_others_slot1_label": ["中2週"], "data_others_slot2_label": ["G3"]})
    assert L.race_class_from_labels(news2) == 5.0


def _fake_out(start_times, now_races):
    races = pd.DataFrame({"race_id": now_races, "racecourse": ["東京"] * len(now_races), "race_number": list(range(1, len(now_races) + 1)),
                          "start_time": start_times, "surface": ["芝"] * len(now_races), "distance": [1600] * len(now_races),
                          "class_ord": [1.0] * len(now_races), "skip": [None] * len(now_races)})
    f = pd.DataFrame([{"race_id": r, "horse_id": f"{r}_h{i}", "umaban_num": i + 1, "is_debut_race": False, "q_front": 0.1, "q_kire": -0.1,
                       "S": 0.1 * i, "F": 0.2, "S_own": 0.1, "F_own": 0.3, "F_front": 0.0, "F_kire": 0.0, "F_surf": 0.1, "F_db": 0.1, "F_deb": 0.0}
                      for r in now_races for i in range(5)])
    ent = f[["race_id", "horse_id"]].assign(ent_src="newspaper")
    ped = f[["horse_id"]].assign(ped_src="血統表")
    return {"races": races, "f": f, "rq": pd.DataFrame(index=now_races), "ent": ent, "ped": ped, "res": None, "b1": 0.29,
            "info": {"runs_source": "runs_all"}, "v3_g": {(r, h): {"良": 0.1} for r, h in zip(f["race_id"], f["horse_id"])}}


def test_record_first_record_wins_and_only_adds_new_races(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PROS", tmp_path)
    D = max(pd.Timestamp.today().normalize(), L.PROSPECTIVE_START)   # 今日(00:01 は過ぎていて 23:59 はまだ)
    out = _fake_out(["23:59", "23:59"], ["A", "B"])
    out["races"] = out["races"].iloc[:1]   # 1回目はレース A だけ(B の馬柱がまだ)
    L.record(D, out, None)
    rec = json.loads((tmp_path / f"{D.strftime('%Y%m%d')}.json").read_text(encoding="utf-8"))
    assert [r["race_id"] for r in rec["races"]] == ["A"] and rec["races"][0]["recorded_before_start"]
    first_at = rec["races"][0]["recorded_at"]
    out2 = _fake_out(["23:59", "00:01"], ["A", "B"])   # 2回目: A は既にある、B は発走時刻を過ぎている
    out2["f"].loc[out2["f"]["race_id"] == "A", "F"] = 9.9
    L.record(D, out2, None)
    rec = json.loads((tmp_path / f"{D.strftime('%Y%m%d')}.json").read_text(encoding="utf-8"))
    by = {r["race_id"]: r for r in rec["races"]}
    assert by["A"]["recorded_at"] == first_at and by["A"]["horses"][0]["F"] == 0.2   # 最初の記録が正本(上書きしない)
    assert by["B"]["recorded_before_start"] is False                                  # 発走後の記録は印が付く
    assert len(rec["runs"]) == 2


def test_record_refuses_days_before_prospective_start_or_with_results(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PROS", tmp_path)
    out = _fake_out(["23:59"], ["A"])
    for D, res in [(pd.Timestamp("2026-10-04"), None), (pd.Timestamp("2099-10-10"), pd.DataFrame({"race_id": ["A"]}))]:
        o = dict(out, res=res)
        try:
            L.record(D, o, None)
            raise AssertionError("拒否されるべき")
        except SystemExit:
            pass
    assert not list(tmp_path.glob("*.json"))
