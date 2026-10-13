# -*- coding: utf-8 -*-
"""finish_v4 F3(model.py)の小さな検査: 群の組み立て、派生の特徴量、オッズを読むモジュールに依存しないこと。"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FV4 = ROOT / "scripts" / "jra_model" / "finish_v4"
sys.path.insert(0, str(FV4))
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "pedigree_reports"))
import model as M  # noqa: E402


def test_spec_replaces_state_with_1d_when_dropped():
    full = M.spec(list(M.GROUPS))
    assert "k_mu" in full and "ab_mu" not in full
    no_state = M.spec([g for g in M.GROUPS if g != "芝ダ別の状態"])
    assert "ab_mu" in no_state and "k_mu" not in no_state
    assert len(set(full)) == len(full)


def test_derived_crosses():
    D = pd.DataFrame({"k_n": [0, 3, 6], "sire_m": [0.1, 0.2, 0.3], "dam_m": [0.5, 0.0, -0.2], "S_own": [1.0, 2.0, 3.0],
                      "kin": [1.0, 0.0, -1.0], "logd": np.log([1600.0, 2000.0, 1200.0]), "k_mu_gap": [0.2, 0.3, 0.4],
                      "k_new_s": [1.0, 0.0, 1.0], "rest_long": [1.0, 1.0, 1.0]} |
                     {z: [1.0, 2.0, 3.0] for z in M.F0.PED_Z} | {c: [1.0, 0.0, 1.0] for c in ["is_dirt", "class_n", "debut"]})
    d = M.derived(D)
    assert list(d["新×dam_m"]) == [0.5, 0.0, -0.0]
    assert np.isclose(d["斤量×距離"].iloc[0], 0.0)
    assert list(d["休み明け×経験"]) == [0.0, 0.0, 1.0]
    assert list(d["芝ダ差×初芝ダ"]) == [0.2, 0.0, 0.4]
    for f in M.SPECS["FULL"]:
        assert f in d.columns or f in ("ext", "short", "ext_amt", "F_own", "k_mu", "k_sd", "d_kin", "last_fig", "dist_dev", "dist_adev"), f


def _imports(path: Path) -> set:
    t = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return out


def test_model_modules_do_not_read_odds():
    """予想を作るモジュールは、オッズを読むモジュール(評価専用の物差し・現行の印)を import しない。"""
    banned = {"market_yardstick", "f0_market_eval", "f0_odds_proxy_check", "fmark_weekend", "load_score5"}
    for name in ["model.py", "ability_v2.py", "nonfinish.py", "pl_ext.py", "f0_dev_utils.py", "birth_data.py", "age_breeder.py", "finish_v4_live.py", "gstar_live.py", "g3_past5.py", "g1_decompose.py"]:
        assert not (_imports(FV4 / name) & banned), name


def test_final_guard_refuses_when_result_exists(tmp_path, monkeypatch):
    import pytest
    import finish_v4_eval as FE
    res = tmp_path / "final_result.json"
    res.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(FE, "MANIFEST", tmp_path / "missing_manifest.json")
    with pytest.raises(SystemExit) as e:
        FE.guard_final(res)
    msg = str(e.value)
    assert "結果が既にある" in msg and "prereg_manifest.json が無い" in msg


def test_final_modes_do_not_overlap_dev():
    import finish_v4_eval as FE
    assert max(FE.MODES["final_dry"]["years"]) <= 2020
    assert min(FE.MODES["final"]["years"]) == 2021 and max(FE.MODES["final"]["years"]) == 2025
