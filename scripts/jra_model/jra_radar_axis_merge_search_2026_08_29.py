# -*- coding: utf-8 -*-
"""現行8カテゴリをさらに統合(7/6/5軸)した場合に主検定(幾何平均vs線形平均、box4)の
結果がどう変わるかを確認するサイドカー(ユーザー依頼、2026-08-29)。

**位置づけ**: これは「軸を減らせば回収率が良くなるパターンを探す」ための事後選択探索
ではなく、あらかじめ決めた少数のドメイン的に妥当な統合案(4通り、8/7/6/5軸)を
同一の主検定にかけて並べるロバスト性チェックである。全結果を選ばずに開示する
(jra_radar_area_order_search_2026_08_29.pyで実演した通り、多数の中から最良を
選んで報告する行為自体が過学習を生むため、ここでは並べて見せるだけにとどめる)。

各統合の根拠:
  - 7軸: 能力・調教評価 + 近走成績・調子 → 「能力・調子」
    (静的な能力評価と直近成績はどちらも「馬の現在の地力・仕上がり」を表す)
  - 6軸: 上記 + コース・距離適性 + 33ラップ理論適合度 → 「コース・ラップ適性」
    (33ラップは特定コースの傾向適合度であり、コース・距離適性と同じ「コースとの相性」)
  - 5軸: 上記 + 血統適性 + 騎手・厩舎 → 「血統・人的」
    (両カテゴリに直接の因果的重なりは無く、圧縮トレンドの参考値として最も弱い統合)
"""
import sys
from pathlib import Path

import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_radar_categories as RC
import jra_signals as JS

sys.path.insert(0, str(Path(__file__).resolve().parent))

BOX_N = 4
BASE = RC.CATEGORY_SIGNAL_MAP


def _members(*cats):
    out = []
    for c in cats:
        out += BASE[c]
    return out


LEVELS = {
    "8軸(現行、既報告)": dict(BASE),
    "7軸(能力・調教評価+近走成績・調子)": {
        "能力・調子": _members("能力・調教評価", "近走成績・調子"),
        "脚質・展開": BASE["脚質・展開"],
        "コース・距離適性": BASE["コース・距離適性"],
        "血統適性": BASE["血統適性"],
        "騎手・厩舎": BASE["騎手・厩舎"],
        "予想印・専門家評価": BASE["予想印・専門家評価"],
        "33ラップ理論適合度": BASE["33ラップ理論適合度"],
    },
    "6軸(+コース・距離適性+33ラップ)": {
        "能力・調子": _members("能力・調教評価", "近走成績・調子"),
        "脚質・展開": BASE["脚質・展開"],
        "コース・ラップ適性": _members("コース・距離適性", "33ラップ理論適合度"),
        "血統適性": BASE["血統適性"],
        "騎手・厩舎": BASE["騎手・厩舎"],
        "予想印・専門家評価": BASE["予想印・専門家評価"],
    },
    "5軸(+血統適性+騎手・厩舎)": {
        "能力・調子": _members("能力・調教評価", "近走成績・調子"),
        "脚質・展開": BASE["脚質・展開"],
        "コース・ラップ適性": _members("コース・距離適性", "33ラップ理論適合度"),
        "血統・人的": _members("血統適性", "騎手・厩舎"),
        "予想印・専門家評価": BASE["予想印・専門家評価"],
    },
}


def main():
    print("データロード中...")
    data = jra_dataset.load(rebuild=False)
    races_all, actual = data["races"], data["actual"]
    priors_all = JS.make_priors([r["df"] for r in races_all])
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()

    print(f"{'レベル':32s} {'N':>5s} {'線形':>8s} {'幾何':>8s} {'差(pt)':>9s} {'95%CI':>22s} {'G1':>5s}")
    print("-" * 95)
    rows = []
    for name, cat_map in LEVELS.items():
        expected = set().union(*cat_map.values()) - {"lap33_axis", "pace_fit"}
        expected |= {"mark_composite_score"}
        assert expected == set(JS.ALL_SIGNALS_V4), f"{name}: カテゴリ完全性チェック失敗"

        records_all = RC.build_axis_matrix(
            races_all, priors_all, history_index, lap33_lookup, race_meta, category_map=cat_map
        )
        races, records = RC.filter_min_categories(races_all, records_all, min_categories=min(4, len(cat_map)))
        ev = JE.Evaluator(races, actual, box_n=BOX_N)
        lin = RC.linear_picks(records, BOX_N)
        geo = RC.geometric_mean_picks(records, BOX_N)
        eval_lin = ev.evaluate(lin)
        eval_geo = ev.evaluate(geo)
        diff = ev.block_bootstrap_diff(geo, lin)
        gate = "PASS" if diff["lo"] > 0 else "NO"
        print(f"{name:32s} {len(races):>5d} {eval_lin['excess']:>7.2f}p {eval_geo['excess']:>7.2f}p "
              f"{eval_geo['excess']-eval_lin['excess']:>+8.2f}p  "
              f"[{diff['lo']:>+6.2f},{diff['hi']:>+6.2f}]  {gate:>5s}")
        rows.append({
            "level": name, "n_categories": len(cat_map), "n_races": len(races),
            "linear_excess_pt": eval_lin["excess"], "geometric_excess_pt": eval_geo["excess"],
            "diff_pt": eval_geo["excess"] - eval_lin["excess"],
            "ci_lo": diff["lo"], "ci_hi": diff["hi"], "gate_pass": diff["lo"] > 0,
        })

    print("\n結論: 上記は最良の1つを選ぶための探索ではなく、全水準を並べて開示している。")
    print("軸を統合しても主検定の結論(G1=NO)が覆るかどうかを確認する目的のみに使うこと。")

    import json
    out_dir = Path(
        r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
        r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jra_radar_axis_merge_search_2026_08_29_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote jra_radar_axis_merge_search_2026_08_29_result.json")


if __name__ == "__main__":
    main()
