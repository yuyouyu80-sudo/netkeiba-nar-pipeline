# -*- coding: utf-8 -*-
"""netkeibaの「血統ビーム出馬表」が持つ父系統色分け(bias_sire_bloodline/bias_dam_sire_bloodline、
サンデーサイレンス系/ターントゥ系/ノーザンダンサー系/ナスルーラ系/ネイティヴダンサー系/
ハンプトン系/セントサイモン系/その他の8分類)は、種牡馬個体に紐づく固定属性(レースごとに
変わらない)。2026年分のnewspaper CSV(816ファイル、bias_sire_bloodline列は実測100%網羅)から
{馬名: 系統}のマップを構築し、pedigree_commentary_profile.csvの全祖先(sire/bms/dam)に
名前で突き合わせる。出力は今後の再利用のためJSON化する。"""
import glob
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"  # 中間生成物・キャッシュの置き場(旧: セッション固有のscratchpad)
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_PATH = SCRATCH / "bloodline_name_map.json"


def main():
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "newspaper" / "*.csv")))
    name_map = {}
    for f in files:
        try:
            df = pd.read_csv(
                f, dtype=str, encoding="utf-8",
                usecols=["bias_sire", "bias_sire_bloodline", "bias_dam_sire", "bias_dam_sire_bloodline"],
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue
        for _, r in df.iterrows():
            if pd.notna(r["bias_sire"]) and pd.notna(r["bias_sire_bloodline"]) and r["bias_sire_bloodline"]:
                name_map.setdefault(r["bias_sire"], r["bias_sire_bloodline"])
            if pd.notna(r["bias_dam_sire"]) and pd.notna(r["bias_dam_sire_bloodline"]) and r["bias_dam_sire_bloodline"]:
                name_map.setdefault(r["bias_dam_sire"], r["bias_dam_sire_bloodline"])

    print(f"収集した馬名→系統マップ: {len(name_map)}件")
    counts = pd.Series(list(name_map.values())).value_counts()
    print(counts.to_dict())

    OUT_PATH.write_text(json.dumps(name_map, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
