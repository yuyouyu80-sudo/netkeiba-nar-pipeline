# -*- coding: utf-8 -*-
"""既存Artifact「3連複フォーメーション列別検証ツール」(627a1a36)のRACESデータを、
2026-09-15に再構築された5つの低回収率/消し材料台帳(単勝462/複勝的中率5%未満242/
ワイド407/馬連的中率3%未満2017/3連複的中率3%未満1786)を使ってt/f/w/ur/srマークだけ
再計算する。b(BOX4)・s(score5=BOX5)・fp(着順)・pay(払戻)は元のまま変更しない。

方式: 各台帳のパターン(selected: {group_id: option_id})を、jra_ledger_search_*.py群と
同じベクトル化atom_mask方式でOR結合する(horse_qualifies()を馬×パターンの全組合せで
逐次呼ぶと計算量的に非現実的なため)。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_factor_registry as FR  # noqa: E402

ORIG_HTML = Path(r"C:\Users\yuyou\.claude\projects\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\tool-results\artifact-627a1a36-1789201126-7b04.html")
OUT_JSON = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\formation_tool_new_races.json")
OUT_LOG = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\rebuild_formation_tool_log.txt")

LEDGER_FILES = {
    "t": (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "単勝"),
    "w": (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "ワイド"),
    "f": (DATA_DIR / "jra_ledger_search_fukusho_hitrate_2026_09_15_result.json", "複勝"),
    "ur": (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "馬連"),
    "sr": (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "3連複"),
}


def log(msg):
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    OUT_LOG.write_text("", encoding="utf-8")

    log("元Artifact HTMLからRACESブロブを抽出中...")
    html = ORIG_HTML.read_text(encoding="utf-8")
    m = re.search(r"const RACES = (\[.*?\]);\s*\n", html, re.S)
    if not m:
        raise RuntimeError("RACES定義が見つからない")
    orig_races = json.loads(m.group(1))
    log(f"  元RACES: {len(orig_races)}レース")

    log("factor_database.json読み込み中...")
    fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
    fdb_races_by_id = {r["race_id"]: r for r in fdb["races"]}

    # 元RACESに登場する(race_id, umaban)の集合に対応するfactor_database側horse dictを収集
    flat_horses = []
    horse_index = {}  # (race_id, umaban) -> index into flat_horses
    missing = 0
    for r in orig_races:
        rid = r["id"]
        fr = fdb_races_by_id.get(rid)
        if fr is None:
            missing += len(r["h"])
            continue
        by_umaban = {h["umaban"]: h for h in fr["horses"]}
        for h in r["h"]:
            fh = by_umaban.get(h["u"])
            if fh is None:
                missing += 1
                continue
            horse_index[(rid, h["u"])] = len(flat_horses)
            flat_horses.append(fh)
    log(f"  現行factor_databaseと突合できた馬: {len(flat_horses)}頭 (未突合: {missing}頭)")

    log("atom_maskを構築中(290atom)...")
    atom_masks = {}
    for gid, group in FR.FACTOR_GROUPS.items():
        for opt in group["options"]:
            key = (gid, opt["id"])
            atom_masks[key] = np.array(
                [FR.evaluate_option(h, group, opt) for h in flat_horses], dtype=bool)
    log(f"  atom数: {len(atom_masks)}")

    n = len(flat_horses)
    mark_arrays = {}
    summary = {}
    for mark_key, (path, bt) in LEDGER_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        all_rows = data["bet_types"][bt]["rows"]
        zero_rows = [r for r in all_rows if r.get("hit_rate_pct", 0.0) == 0.0]
        rows = zero_rows if zero_rows else all_rows
        cum = np.zeros(n, dtype=bool)
        for row in rows:
            sel = row["selected"]
            pat_mask = np.ones(n, dtype=bool)
            for gid, oid in sel.items():
                pat_mask &= atom_masks[(gid, oid)]
            cum |= pat_mask
        mark_arrays[mark_key] = cum
        summary[mark_key] = {
            "bet_type": bt, "total_patterns": len(all_rows), "used_patterns": len(rows),
            "used_is_zero_subset": bool(zero_rows), "n_horses_marked": int(cum.sum()), "n_horses": n,
        }
        log(f"  {mark_key}({bt}, 全{len(all_rows)}件中hit_rate0%={len(zero_rows)}件を使用): "
            f"該当馬 {int(cum.sum())}/{n} ({cum.sum()/n*100:.1f}%)")

    log("新RACES組み立て中...")
    new_races = []
    for r in orig_races:
        rid = r["id"]
        new_h = []
        for h in r["h"]:
            idx = horse_index.get((rid, h["u"]))
            nh = dict(h)  # b, s, fp, u, n はそのまま維持
            if idx is None:
                nh["t"] = nh["f"] = nh["w"] = nh["ur"] = nh["sr"] = False
            else:
                for mk in ("t", "f", "w", "ur", "sr"):
                    nh[mk] = bool(mark_arrays[mk][idx])
            new_h.append(nh)
        nr = dict(r)
        nr["h"] = new_h
        new_races.append(nr)

    OUT_JSON.write_text(json.dumps(new_races, ensure_ascii=False), encoding="utf-8")
    log(f"書き出し完了: {OUT_JSON} ({OUT_JSON.stat().st_size/1024/1024:.2f}MB)")

    summary_path = OUT_JSON.with_name("formation_tool_mark_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"サマリ書き出し完了: {summary_path}")


if __name__ == "__main__":
    main()
