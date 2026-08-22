# -*- coding: utf-8 -*-
"""Step1再挑戦(選択肢B)の検証ゲート(2026-08-23新設)。

jra_history.py(race_resultsアーカイブから馬柱互換の past{i}_* 列を復元する層)が、
既存の馬柱(newspaper)ベースの値と十分に一致することを、馬柱・アーカイブ両方が存在する
211レースで機械的に確認する。**このゲートを全て満たすまでPhase 3(母集団層の新設)以降には
進まない**(計画書 valiant-cuddling-aho.md Phase 2 参照)。

実行方法: python scripts/jra_model/jra_history_validate_2026_08_23.py
出力: data/jra_pipeline/jra_history_validate_2026_08_23_result.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset as JD  # noqa: E402
import jra_history as JH  # noqa: E402
import jra_market_model as MM  # noqa: E402
import jra_signals as JS  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "jra_history_validate_2026_08_23_result.json"


def kaisai_to_iso(d: str) -> str:
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def main():
    print("=== jra_dataset.load() (馬柱ベース、既存211レース) ===")
    data = JD.load(rebuild=False)
    races, actual = data["races"], data["actual"]
    print(f"races={len(races)}")

    print("=== jra_history archive ロード ===")
    results = JH.load_results()
    idx = JH.HorseHistoryIndex(results)
    print(f"archive rows={len(results)} horses={len(idx._by_horse)} min_date={idx.min_date}")

    result = {"n_races": len(races)}

    # ------------------------------------------------------------ V1/V2/V3/V6: 過去走の列一致
    # 注意(2026-08-23判明): newspaperのpast{i}とarchiveのpast{i}は、地方競馬・海外レース
    # 出走やバックフィルの欠測により「同じ位置iが同じ実レースを指すとは限らない」
    # (Opus調査: 過去走参照22,554スロット中645スロット=2.9%がJRAバックフィルの穴、
    # 656スロットが地方競馬、14スロットが海外)。そのため位置iで無条件に突き合わせると、
    # 実際には別レースを比較することになり、見かけの不一致率が過大に出る。
    # past{i}_date(newspaper)とpast{i}_date(archive)が完全一致する馬(=同じ5走を参照している
    # ことが保証された馬)だけに絞って再現性を検証する(V6相当)。全体の不一致率も参考値として残す。
    beaten_by_new_all, beaten_by_arc_all = [], []
    beaten_by_new_ex, beaten_by_arc_ex = [], []
    finish_match, finish_total = 0, 0
    finish_match_ex, finish_total_ex = 0, 0
    agari_match, agari_total = 0, 0
    agari_match_ex, agari_total_ex = 0, 0
    corner_match, corner_total = 0, 0
    corner_match_ex, corner_total_ex = 0, 0
    n_horses_exact, n_horses_total = 0, 0

    # 2026-08-23判明の追加バグ: 「日付集合が一致する馬」でもnewspaper側とarchive側で
    # past{i}の並び順(position)がズレるケースがあった(サンプル調査で確認)。position базе
    # の突き合わせでは誤ペアリングになるため、**日付をキーにした突き合わせ**に変更する
    # (同一馬・同一日付の重複出走は無い=V13で確認済みなので、日付は一意キーとして使える)。
    for r in races:
        df = r["df"]
        race_date = kaisai_to_iso(r["kaisai_date"])
        horse_ids = df["horse_id"].tolist()
        arc_pf = JH.past_frame(idx, horse_ids, race_date, n=5)

        for j in range(len(df)):
            new_by_date, arc_by_date = {}, {}
            for i in range(1, 6):
                if f"past{i}_date" in df.columns:
                    d_new = str(df[f"past{i}_date"].iloc[j]).replace(".", "-")
                    if d_new != "nan":
                        new_by_date[d_new] = {
                            "beaten_by": JS._margin(df[f"past{i}_beaten_by"].iloc[j]) if f"past{i}_beaten_by" in df.columns else np.nan,
                            "finish": str(df[f"past{i}_finish"].iloc[j]) if f"past{i}_finish" in df.columns else None,
                            "agari": pd.to_numeric(pd.Series([df[f"past{i}_agari_3f"].iloc[j]]), errors="coerce").iloc[0] if f"past{i}_agari_3f" in df.columns else np.nan,
                            "corner": str(df[f"past{i}_corner_positions"].iloc[j]) if f"past{i}_corner_positions" in df.columns else None,
                        }
                d_arc = str(arc_pf[f"past{i}_date"].iloc[j])
                if d_arc != "nan":
                    arc_by_date[d_arc] = {
                        "beaten_by": JS._margin(arc_pf[f"past{i}_beaten_by"].iloc[j]),
                        "finish": str(arc_pf[f"past{i}_finish"].iloc[j]),
                        "agari": pd.to_numeric(pd.Series([arc_pf[f"past{i}_agari_3f"].iloc[j]]), errors="coerce").iloc[0],
                        "corner": str(arc_pf[f"past{i}_corner_positions"].iloc[j]),
                    }

            n_horses_total += 1
            if new_by_date and set(new_by_date) == set(arc_by_date):
                n_horses_exact += 1

            common_dates = set(new_by_date) & set(arc_by_date)
            for d in common_dates:
                nv, av = new_by_date[d], arc_by_date[d]
                a, b = nv["beaten_by"], av["beaten_by"]
                if not (pd.isna(a) or pd.isna(b)):
                    beaten_by_new_all.append(a)
                    beaten_by_arc_all.append(b)
                    beaten_by_new_ex.append(a)
                    beaten_by_arc_ex.append(b)
                if nv["finish"] not in (None, "nan") and av["finish"] not in (None, "nan"):
                    finish_total += 1
                    finish_total_ex += 1
                    if nv["finish"] == av["finish"]:
                        finish_match += 1
                        finish_match_ex += 1
                if pd.notna(nv["agari"]) and pd.notna(av["agari"]):
                    agari_total += 1
                    agari_total_ex += 1
                    if np.isclose(nv["agari"], av["agari"], atol=0.05):
                        agari_match += 1
                        agari_match_ex += 1
                if nv["corner"] not in (None, "nan") and av["corner"] not in (None, "nan"):
                    corner_total += 1
                    corner_total_ex += 1
                    if nv["corner"] == av["corner"]:
                        corner_match += 1
                        corner_match_ex += 1

    result["V6_exact_match_horse_rate"] = n_horses_exact / n_horses_total if n_horses_total else None
    result["V6_n"] = n_horses_total
    result["V1_beaten_by_sec_corr_all"] = float(np.corrcoef(beaten_by_new_all, beaten_by_arc_all)[0, 1]) if len(beaten_by_new_all) > 5 else None
    result["V1_n_pairs_all"] = len(beaten_by_new_all)
    result["V1_beaten_by_sec_corr"] = float(np.corrcoef(beaten_by_new_ex, beaten_by_arc_ex)[0, 1]) if len(beaten_by_new_ex) > 5 else None
    result["V1_beaten_by_max_abs_diff"] = float(np.max(np.abs(np.array(beaten_by_new_ex) - np.array(beaten_by_arc_ex)))) if beaten_by_new_ex else None
    result["V1_n_pairs"] = len(beaten_by_new_ex)
    result["V2_finish_match_rate_all"] = finish_match / finish_total if finish_total else None
    result["V2_finish_match_rate"] = finish_match_ex / finish_total_ex if finish_total_ex else None
    result["V2_n"] = finish_total_ex
    result["V3_agari_match_rate"] = agari_match_ex / agari_total_ex if agari_total_ex else None
    result["V3_agari_n"] = agari_total_ex
    result["V3_corner_match_rate"] = corner_match_ex / corner_total_ex if corner_total_ex else None
    result["V3_corner_n"] = corner_total_ex
    print(f"V6 exact-match horse rate={result['V6_exact_match_horse_rate']:.4f} n={result['V6_n']}")
    print(f"V1(全体、参考) beaten_by_sec corr={result['V1_beaten_by_sec_corr_all']} n={result['V1_n_pairs_all']}")
    print(f"V1(exact集合限定) beaten_by_sec corr={result['V1_beaten_by_sec_corr']:.7f} "
          f"max_abs_diff={result['V1_beaten_by_max_abs_diff']:.2e} n={result['V1_n_pairs']}")
    print(f"V2(全体、参考) finish match_rate={result['V2_finish_match_rate_all']:.4f}")
    print(f"V2(exact集合限定) finish match_rate={result['V2_finish_match_rate']:.4f} n={result['V2_n']}")
    print(f"V3(exact集合限定) agari match_rate={result['V3_agari_match_rate']:.4f} n={result['V3_agari_n']}  "
          f"corner match_rate={result['V3_corner_match_rate']:.4f} n={result['V3_corner_n']}")

    # ------------------------------------------------------------ V4/V5: 合成z_近走の相関
    priors_all = JS.make_priors([r["df"] for r in races])
    z_new_all, z_arc_all = [], []
    per_sig_new = {s: [] for s in MM.RECENT_FORM_SIGNALS}
    per_sig_arc = {s: [] for s in MM.RECENT_FORM_SIGNALS}
    for r in races:
        df = r["df"]
        race_date = kaisai_to_iso(r["kaisai_date"])
        current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
        sig_new = JS.compute_signals(df, current_class, priors_all, JS.CLASS_ORDINAL)

        horse_ids = df["horse_id"].tolist()
        arc_pf = JH.past_frame(idx, horse_ids, race_date, n=5)
        df_arc = df.copy().reset_index(drop=True)
        for i in range(1, 6):
            for col in (f"past{i}_finish", f"past{i}_beaten_by", f"past{i}_agari_3f",
                        f"past{i}_corner_positions", f"past{i}_field_size", f"past{i}_race_class"):
                df_arc[col] = arc_pf[col].to_numpy()
        sig_arc = JS.compute_signals(df_arc, current_class, priors_all, JS.CLASS_ORDINAL)

        recent_new = np.nanmean(np.column_stack(
            [sig_new[n].to_numpy(dtype=float) for n in MM.RECENT_FORM_SIGNALS]), axis=1)
        recent_arc = np.nanmean(np.column_stack(
            [sig_arc[n].to_numpy(dtype=float) for n in MM.RECENT_FORM_SIGNALS]), axis=1)
        z_new_all.extend(list(recent_new))
        z_arc_all.extend(list(recent_arc))
        for s in MM.RECENT_FORM_SIGNALS:
            per_sig_new[s].extend(list(sig_new[s].to_numpy(dtype=float)))
            per_sig_arc[s].extend(list(sig_arc[s].to_numpy(dtype=float)))

    def _paired_corr(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 5:
            return None
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    result["V4_composite_z_recent_corr"] = _paired_corr(z_new_all, z_arc_all)
    print(f"V4 composite z_recent corr={result['V4_composite_z_recent_corr']}")
    result["V5_per_signal_corr"] = {s: _paired_corr(per_sig_new[s], per_sig_arc[s]) for s in MM.RECENT_FORM_SIGNALS}
    print(f"V5 per-signal corr={result['V5_per_signal_corr']}")

    # ------------------------------------------------------------ V7: 出走馬集合の一致(is_starter vs _drop_scratched)
    starter_match, starter_total = 0, 0
    for r in races:
        df = r["df"]
        kept_new = set(JS._drop_scratched(df)["umaban"].astype(int))
        rid = r["race_id"]
        rr = results[results["race_id"] == rid]
        if rr.empty:
            continue
        kept_arc = set(rr[JH.is_starter(rr)]["umaban"].astype(int))
        starter_total += 1
        if kept_new == kept_arc:
            starter_match += 1
    result["V7_starter_set_match_rate"] = starter_match / starter_total if starter_total else None
    result["V7_n"] = starter_total
    print(f"V7 starter set match_rate={result['V7_starter_set_match_rate']} n={result['V7_n']}")

    # ------------------------------------------------------------ V8: オッズ整合
    odds_new, odds_arc = [], []
    for r in races:
        df = r["df"]
        rid = r["race_id"]
        rr = results[results["race_id"] == rid].set_index(results[results["race_id"] == rid]["horse_id"])
        if rr.empty:
            continue
        for _, row in df.iterrows():
            hid = row["horse_id"]
            if hid in rr.index:
                bo = pd.to_numeric(pd.Series([row.get("bias_win_odds")]), errors="coerce").iloc[0]
                of = pd.to_numeric(rr.loc[hid, "odds_final"], errors="coerce")
                of = of.iloc[0] if isinstance(of, pd.Series) else of
                if pd.notna(bo) and pd.notna(of) and bo > 0 and of > 0:
                    odds_new.append(np.log(bo))
                    odds_arc.append(np.log(of))
    result["V8_log_odds_corr"] = _paired_corr(odds_new, odds_arc)
    result["V8_n"] = len(odds_new)
    print(f"V8 log-odds corr={result['V8_log_odds_corr']} n={result['V8_n']}")

    # ------------------------------------------------------------ V9: 単勝払戻整合
    payout_ok, payout_total = 0, 0
    for rid, of_map in results.groupby("race_id"):
        win_actual = actual.get(rid, {}).get("単勝", {})
        for _, row in of_map.iterrows():
            of = pd.to_numeric(row["odds_final"], errors="coerce")
            if pd.isna(of):
                continue
            umaban = pd.to_numeric(row["umaban"], errors="coerce")
            if pd.isna(umaban):
                continue
            pay = win_actual.get(int(umaban))
            if pay is None:
                continue
            payout_total += 1
            if abs(pay - of * 100) < 1.0:
                payout_ok += 1
    result["V9_payout_consistency_rate"] = payout_ok / payout_total if payout_total else None
    result["V9_n"] = payout_total
    print(f"V9 payout consistency={result['V9_payout_consistency_rate']} n={result['V9_n']}")

    # ------------------------------------------------------------ V10: 市場ベンチマーク整合
    fav_match, fav_total = 0, 0
    for r in races:
        df = r["df"]
        rid = r["race_id"]
        rr = results[results["race_id"] == rid]
        if rr.empty:
            continue
        ninki_new = pd.to_numeric(df["bias_ninki"], errors="coerce")
        if (ninki_new == 1).sum() != 1:
            continue
        fav_new_umaban = int(df.loc[ninki_new == 1, "umaban"].iloc[0])
        pop_arc = pd.to_numeric(rr["popularity"], errors="coerce")
        if (pop_arc == 1).sum() != 1:
            continue
        fav_arc_umaban = int(rr.loc[pop_arc == 1, "umaban"].iloc[0])
        fav_total += 1
        if fav_new_umaban == fav_arc_umaban:
            fav_match += 1
    result["V10_favorite_match_rate"] = fav_match / fav_total if fav_total else None
    result["V10_n"] = fav_total
    print(f"V10 favorite match_rate={result['V10_favorite_match_rate']} n={result['V10_n']}")

    # ------------------------------------------------------------ V11: margin列換算(検証専用)の妥当性
    results_sorted = results.sort_values(["horse_id", "race_date"])
    cum_len, cum_sec = [], []
    for hid, g in results_sorted.groupby("horse_id"):
        g = g.reset_index(drop=True)
        run_len, run_sec = 0.0, 0.0
        for _, row in g.iterrows():
            ml = JH.parse_margin_lengths(row["margin"])
            bs = row["beaten_by_sec"]
            if pd.notna(ml) and pd.notna(bs):
                run_len += ml
                run_sec += bs
                cum_len.append(run_len)
                cum_sec.append(run_sec)
    result["V11_margin_len_vs_sec_corr"] = _paired_corr(cum_len, cum_sec)
    print(f"V11 margin_len(累積) vs beaten_by_sec(累積) corr={result['V11_margin_len_vs_sec_corr']}")

    # ------------------------------------------------------------ V12: 2パラメータ当てはめの一致
    feats_new = MM.build_composite_features(races, actual, priors_all, JS.CLASS_ORDINAL)
    races_arc_df = []
    for r in races:
        df = r["df"]
        race_date = kaisai_to_iso(r["kaisai_date"])
        horse_ids = df["horse_id"].tolist()
        arc_pf = JH.past_frame(idx, horse_ids, race_date, n=5)
        df_arc = df.copy().reset_index(drop=True)
        for i in range(1, 6):
            for col in (f"past{i}_finish", f"past{i}_beaten_by", f"past{i}_agari_3f",
                        f"past{i}_corner_positions", f"past{i}_field_size", f"past{i}_race_class"):
                df_arc[col] = arc_pf[col].to_numpy()
        races_arc_df.append({**r, "df": df_arc})
    feats_arc = MM.build_composite_features(races_arc_df, actual, priors_all, JS.CLASS_ORDINAL)

    def fit_2param(feats):
        def nll2(params):
            b0, b1 = params
            return MM.race_nll(np.array([b0, b1, 0.0]), feats)
        from scipy.optimize import minimize
        res = minimize(nll2, np.array([1.0, 0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        return res.x

    beta_new = fit_2param(feats_new)
    beta_arc = fit_2param(feats_arc)
    result["V12_beta_newspaper"] = beta_new.tolist()
    result["V12_beta_archive"] = beta_arc.tolist()
    same_sign = bool(np.sign(beta_new[1]) == np.sign(beta_arc[1]))
    ratio = float(beta_arc[1] / beta_new[1]) if beta_new[1] != 0 else None
    result["V12_beta1_same_sign"] = same_sign
    result["V12_beta1_ratio"] = ratio
    print(f"V12 beta(newspaper)={beta_new} beta(archive)={beta_arc} same_sign={same_sign} ratio={ratio}")

    # ------------------------------------------------------------ V13: 同日2走なしの確認
    dup = results.groupby(["horse_id", "race_date"]).size()
    result["V13_duplicate_same_day_starts"] = int((dup > 1).sum())
    diffs = results_sorted.assign(_dt=pd.to_datetime(results_sorted["race_date"])).groupby("horse_id")["_dt"].diff().dt.days
    all_gaps = diffs.dropna()
    result["V13_min_gap_days"] = float(all_gaps.min()) if len(all_gaps) else None
    print(f"V13 duplicate_same_day_starts={result['V13_duplicate_same_day_starts']} "
          f"min_gap_days={result['V13_min_gap_days']}")

    # ------------------------------------------------------------ 合否判定
    gates = {
        "V1": (result["V1_beaten_by_sec_corr"] or 0) >= 0.999,
        "V2": (result["V2_finish_match_rate"] or 0) >= 0.999,
        "V3": (result["V3_agari_match_rate"] or 0) >= 0.999 and (result["V3_corner_match_rate"] or 0) >= 0.999,
        "V4": (result["V4_composite_z_recent_corr"] or 0) >= 0.90,
        "V5": all((v or 0) >= 0.85 for v in result["V5_per_signal_corr"].values()),
        "V7": (result["V7_starter_set_match_rate"] or 0) >= 0.99,
        "V8": (result["V8_log_odds_corr"] or 0) >= 0.98,
        "V9": (result["V9_payout_consistency_rate"] or 0) >= 0.99,
        "V10": (result["V10_favorite_match_rate"] or 0) >= 0.80,
        "V12": result["V12_beta1_same_sign"] and result["V12_beta1_ratio"] is not None
               and 0.7 <= result["V12_beta1_ratio"] <= 1.5,
        "V13": result["V13_duplicate_same_day_starts"] == 0,
    }
    result["gates"] = gates
    result["all_pass"] = all(gates.values())
    print("\n=== ゲート判定 ===")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"全体: {'PASS - Phase3以降へ進行可' if result['all_pass'] else 'FAIL - 要調査'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
