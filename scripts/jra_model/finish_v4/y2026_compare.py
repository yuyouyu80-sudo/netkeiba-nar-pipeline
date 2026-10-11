# -*- coding: utf-8 -*-
"""finish_v4: 2026年の層(事前登録 §7、記述のみ)— 現行の印と、同じレース・同じ点数で比べる(2026-10-11)。

- 現行の印: col1_special の前向きの精算 fmark_ledger.csv(発走前に記録された weekend_fmark の分だけ、通常戦)。
- 完成形: y2026_utils.parquet(確認と同じ処理を 2026年に当てた効用、係数は 2014〜2025年)。各レースで、現行の印と同じ点数 k だけ
  組の確率の高い順に買う版と、主ルール20点の版。払戻は data/payouts(同着は全部の当たり組)。
- レース数が少ない(数十)ため、区間も合否も出さない。
出力: data/jra_pipeline/finish_v4/y2026_compare.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import sanrenpuku_rules as SR  # noqa: E402

LEDGER = F0.C.PROJECT_ROOT / "data" / "jra_pipeline" / "col1_special" / "fmark_ledger.csv"


def main():
    L = pd.read_csv(LEDGER, dtype={"race_id": str, "date": str})
    U = pd.read_parquet(F0.OUT_DIR / "y2026_utils.parquet")
    lam = json.loads((F0.OUT_DIR / "y2026_lam.json").read_text(encoding="utf-8"))
    pay = F0.load_sanrenpuku([2026])
    rows = []
    for r in L.itertuples():
        g = U[U["race_id"] == r.race_id].sort_values("umaban_num")
        if g.empty or r.race_id not in pay:
            continue
        um = g["umaban_num"].astype(int).tolist()
        u = g["u_FINAL"].to_numpy(float)
        rec = {"date": r.date, "race_id": r.race_id, "race_name": r.race_name, "cur_points": int(r.points), "cur_hit": int(r.hit),
               "cur_ret": int(r.ret)}
        for lab, rule in [("k", f"N{int(r.points)}"), ("N20", "N20")]:
            t = SR.race_tickets(um, u, lam, rule)["tickets"]
            s = SR.settle(t, pay[r.race_id])
            rec[f"fin_{lab}_points"], rec[f"fin_{lab}_hit"], rec[f"fin_{lab}_ret"] = s["n"], int(s["hit"]), s["ret"]
        rows.append(rec)
    R = pd.DataFrame(rows)
    out = {"n_races": int(len(R)), "dates": sorted(R["date"].unique().tolist()), "races": rows, "summary": {}}
    for who, pts, hit, ret in [("現行の印", "cur_points", "cur_hit", "cur_ret"), ("完成形(点数をそろえる)", "fin_k_points", "fin_k_hit", "fin_k_ret"),
                               ("完成形(20点)", "fin_N20_points", "fin_N20_hit", "fin_N20_ret")]:
        out["summary"][who] = {"points": int(R[pts].sum()), "hit_rate": round(float(R[hit].mean()), 4),
                               "roi": round(float(R[ret].sum() / (100 * R[pts].sum())), 4), "ret": int(R[ret].sum())}
    (F0.OUT_DIR / "y2026_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
