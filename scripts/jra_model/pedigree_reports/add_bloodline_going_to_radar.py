# -*- coding: utf-8 -*-
"""race_radar_data.json の各レースに、系統適性指数(馬場状態別)を追記する(2026-10-03新設)。

入力: race_radar_data.json(build_race_radar_data.py) と bloodline_going_index.json(build_bloodline_going_index.py)
出力: race_radar_data.json の各レースに "bl_going" を追加(何度実行しても同じ結果になる)。

bl_going の中身:
- pool: 指数を引いたプール(サーフェス|距離帯)
- ref[role][going]: 同条件の過去レース上位3着馬(延べ)の系統指数の平均(レース基準線に使う)
- horses[horse_id][role][going]: 今日の出走馬の父系統/母父系統/父父系統の指数(分類できない馬は無し)
- table[role]: 表示用(系統×馬場状態の指数とN)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_race_radar_data as B  # noqa: E402

SCRATCH = B.SCRATCH
PROF = B.PROF
DATA_PATH = SCRATCH / "race_radar_data.json"
INDEX_PATH = SCRATCH / "bloodline_going_index.json"
GOINGS = ["良", "稍重", "重", "不良"]
KEYS = ["全"] + GOINGS  # "全"=馬場を問わない系統の総合効果(縮約済み)。馬場別の上乗せは検証で再現しなかったため既定はこちら
ROLE_NAME_FIELD = {"sire": "sire_name", "bms": "bms_name", "ss": "ss_name"}
MIN_TABLE_N = 300  # 表に載せる系統の、プール内(全馬場合算)の最小走数


def main():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    idx_all = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    bmap = json.loads(B.BLOODLINE_MAP_PATH.read_text(encoding="utf-8"))
    index = idx_all["index"]

    for rno, d in payload["races"].items():
        info = d["race_info"]
        surface = info["surface"]
        bucket = PROF.distance_bucket(float(info["distance"]))

        pool_cells = {role: index.get(role, {}).get(surface, {}).get(bucket, {}) for role in ROLE_NAME_FIELD}

        def lookup(role, going, name):
            cat = bmap.get(name) if name else None
            if cat is None:
                return None, None
            if going == "全":  # どの馬場のセルにも同じ総合効果"main"が入っている
                for g in GOINGS:
                    cell = pool_cells[role].get(g, {}).get(cat)
                    if cell:
                        return cat, cell["main"]
                return cat, None
            cell = pool_cells[role].get(going, {}).get(cat)
            return cat, (cell["idx"] if cell else None)

        # --- レース基準: 同条件の過去レース上位3着馬(延べ)の平均指数 ---
        ref = {}
        for role, field in ROLE_NAME_FIELD.items():
            ref[role] = {}
            for g in KEYS:
                vals = []
                for r in d["matched_races"]:
                    for h in r["top3"]:
                        _cat, v = lookup(role, g, h.get(field))
                        if v is not None:
                            vals.append(v)
                ref[role][g] = {"mean": round(sum(vals) / len(vals), 2) if vals else None, "n": len(vals)}

        # --- 今日の出走馬 ---
        horses = {}
        for h in (d.get("today") or {}).get("horses", []):
            hid = h["horse_id"]
            ent = {}
            for role, field in ROLE_NAME_FIELD.items():
                cat = bmap.get(h.get(field)) if h.get(field) else None
                ent[role] = {"cat": cat, "idx": {g: lookup(role, g, h.get(field))[1] for g in KEYS}}
            horses[hid] = ent

        # --- 表示用テーブル ---
        table = {}
        for role in ROLE_NAME_FIELD:
            cells = index.get(role, {}).get(surface, {}).get(bucket, {})
            cats = {}
            for g in GOINGS:
                for cat, c in cells.get(g, {}).items():
                    cats.setdefault(cat, {"n_total": 0})["n_total"] += c["n"]
            rows = []
            for cat, meta in cats.items():
                if meta["n_total"] < MIN_TABLE_N:
                    continue
                by_going = {g: cells.get(g, {}).get(cat) for g in GOINGS}
                main = next((c["main"] for c in by_going.values() if c), None)
                by_going["全"] = {"idx": main, "n": meta["n_total"]}
                rows.append({"cat": cat, "n_total": meta["n_total"], "by_going": by_going})
            rows.sort(key=lambda r: -(r["by_going"]["全"]["idx"] if r["by_going"]["全"]["idx"] is not None else -99))
            table[role] = rows

        pool_races = {g: idx_all["pool_races"].get(f"{surface}|{bucket}|{g}", 0) for g in GOINGS}
        d["bl_going"] = {"pool": f"{surface}|{bucket}", "pool_label": f"{'ダート' if surface == 'ダ' else '芝'}・{bucket}",
                         "pool_races": pool_races, "ref": ref, "horses": horses, "table": table}
    payload["bl_going_meta"] = {"k": idx_all["k"], "split_year": idx_all["split_year"],
                                "n_runs": idx_all["n_runs"], "validation": idx_all["validation"]}
    DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {DATA_PATH} ({len(payload['races'])}レースに系統適性指数を追記)")


if __name__ == "__main__":
    main()
