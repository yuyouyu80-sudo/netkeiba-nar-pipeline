# -*- coding: utf-8 -*-
"""Stage 0-B(3列目改良プラン、2026-09-16): 3連複フォーメーション列別検証ツールの母集団
(data2_races.json、319レース)とfactor_database.json側の通常戦母集団(314レース)の
対称差分を確認し、原因が障害戦であることを実データで裏付ける。あわせて
pred_rank_box4(factor_database側)とrank4(data2_races.json側)の一致率、および
priorsの出所(build_data2.py=winner_*.json内のpriors / build_factor_dataset.py=
MS.compute_priors_for_population())の違いを記録する。

決定事項(ユーザー指示、2026-09-16): 障害戦はStage 0-Cの恒久データセットから除外する。
"""
import json
import pathlib
import re

PROJECT_ROOT = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
RESCUE_DIR = DATA_DIR / "_rescue_2026_09_16"

HURDLE_RE = re.compile("障害")

data2 = json.loads((RESCUE_DIR / "data2_races.json").read_text(encoding="utf-8"))
fd = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))

data2_race_ids = {r["race_id"] for r in data2}
fd_all_race_ids = {r["race_id"]: r for r in fd["races"]}
fd_normal_race_ids = {rid for rid, r in fd_all_race_ids.items() if r["race_type"] == "normal"}

out = []
out.append(f"data2_races.json race count: {len(data2_race_ids)}")
out.append(f"factor_database.json total race count: {len(fd_all_race_ids)}")
out.append(f"factor_database.json normal(通常戦) race count: {len(fd_normal_race_ids)}")

diff_data2_minus_fd = data2_race_ids - fd_normal_race_ids
diff_fd_minus_data2 = fd_normal_race_ids - data2_race_ids
out.append(f"data2側にあってfactor_database(normal)に無い: {len(diff_data2_minus_fd)}件")
out.append(f"factor_database(normal)にあってdata2側に無い: {len(diff_fd_minus_data2)}件")

# 差分レースのrace_nameを確認し、障害戦(レース名に「障害」を含む)かどうかを判定
data2_by_id = {r["race_id"]: r for r in data2}
n_hurdle = 0
n_other = 0
for rid in sorted(diff_data2_minus_fd):
    rname = data2_by_id[rid]["race_name"]
    is_hurdle = bool(HURDLE_RE.search(rname))
    if is_hurdle:
        n_hurdle += 1
    else:
        n_other += 1
    # factor_database側での分類(存在すれば)も併記
    fd_rtype = fd_all_race_ids.get(rid, {}).get("race_type", "(factor_databaseに存在しない)")
    out.append(f"  {rid}  race_name={rname!r}  hurdle={is_hurdle}  fd_race_type={fd_rtype}")

out.append(f"差分{len(diff_data2_minus_fd)}件のうち障害戦={n_hurdle}件, それ以外={n_other}件")
out.append(f"仮説確認: 差分は全て障害戦か -> {n_hurdle == len(diff_data2_minus_fd) and n_other == 0}")

# ------------------------------------------------------------ pred_rank_box4 vs rank4 一致率
out.append("")
out.append("=== pred_rank_box4(factor_database) vs rank4(data2_races.json) 一致率 ===")
common_race_ids = data2_race_ids & fd_normal_race_ids
n_horse_total = 0
n_horse_match = 0
n_horse_mismatch = 0
mismatch_examples = []
for rid in sorted(common_race_ids):
    d2_horses = {h["umaban"]: h for h in data2_by_id[rid]["horses"]}
    fd_horses = {h["umaban"]: h for h in fd_all_race_ids[rid]["horses"]}
    for umaban, fh in fd_horses.items():
        dh = d2_horses.get(umaban)
        if dh is None:
            continue
        pr_fd = fh.get("pred_rank_box4")
        pr_d2 = dh.get("rank4")
        if pr_fd is None or pr_d2 is None:
            continue
        n_horse_total += 1
        try:
            match = int(pr_fd) == int(pr_d2)
        except (TypeError, ValueError):
            match = False
        if match:
            n_horse_match += 1
        else:
            n_horse_mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(f"    {rid} umaban={umaban} pred_rank_box4={pr_fd} rank4={pr_d2}")

out.append(f"比較対象馬数: {n_horse_total}")
out.append(f"一致: {n_horse_match} ({100.0 * n_horse_match / n_horse_total:.2f}%)" if n_horse_total else "比較対象0件")
out.append(f"不一致: {n_horse_mismatch}")
if mismatch_examples:
    out.append("不一致の例(最大10件):")
    out.extend(mismatch_examples)

# ------------------------------------------------------------ priors来歴の記録
out.append("")
out.append("=== priorsの出所(コード読み取りによる確認) ===")
out.append("build_data2.py (data2_races.json生成元): winner_v3.json/winner_box4.json/winner_box3.json")
out.append("  内の \"priors\" フィールドをそのまま JMS.score_race() へ渡している(L40-43)。")
out.append("build_factor_dataset.py (factor_database.json生成元): MS.compute_priors_for_population()")
out.append("  で通常戦246レースの df から動的に再計算した priors を使っている(L72-76)。")
out.append("決定: 現行公開中ツール(formation_tool_v3.html)の再現・後継が目的のため、")
out.append("  Stage 0-Cでは build_data2.py と同じ winner_*.json 内の priors を使用する。")

out.append("")
out.append("=== 決定事項(ユーザー指示 2026-09-16) ===")
out.append("障害戦はStage 0-Cの恒久データセットから除外する。")
out.append("これによりStage 0-Cの母集団はfactor_database.json側のnormal(通常戦)314レースと")
out.append("完全一致する設計にでき、factor_database側に障害戦の290atom評価を追加する必要はない。")

result_text = "\n".join(out)
(DATA_DIR / "jra_formation_population_diff_2026_09_16_result.txt").write_text(result_text, encoding="utf-8")
print(result_text)
