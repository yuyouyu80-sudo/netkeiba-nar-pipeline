# -*- coding: utf-8 -*-
"""コーナー通過順位の実測(passing_order)とシミュレーションの一致度を測る新指標m6。

計画(ステップ1)に基づく。トークン分割・タイ許容ロジックは今後sim_runner_lib.pyへ
共通化する前提でここに実装するが、既存の_parse_passing_positions()
(recalibrate_running_style.py 43-59行目)とは役割が異なる(あちらは最初/最終コーナーの
脚質集団平均較正専用、こちらは全コーナー・馬個体単位のfootrule評価専用)。

コーナー距離は対称オーバル近似で算出する(エンジニアレビュー承認済みの設計):
    4角 = D_TOTAL - HOME_STRETCH_M            (=sim_runner_lib.race_metrics()のd_stretch_entryと同一)
    3角 = 4角 - CORNER_LEN_M
    2角 = 3角 - HOME_STRETCH_M                 (向正面をホームストレッチ長で近似、精度は3・4角より低い)
    1角 = 2角 - CORNER_LEN_M
実測passing_orderのトークンは末尾(4角側)から対応づける(短距離レースでは1角・2角の
トークンが無い=直近2コーナーのみ記録、というJRAの表記慣行に合わせる)。

実測トークンはタイ(同着集団番号、例 "12-12-10-4")を含む密順位表記であるため、
average-rank(タイ集団は平均順位)でシム側と実測側の両方を正規化し、footruleを計算する
(厳密なsorted()一意順位だと、タイ集団内のシム側の並び順で恣意的な誤差が生じるため)。
"""
import sys

import numpy as np

import horse_pair_sim as hp

STYLES = ["逃", "先", "差", "追"]


def corner_distances(distance_m: float, home_stretch_m: float, corner_len_m: float):
    """距離が長い順(4角→3角→2角→1角)にコーナー距離を返す。0未満は含めない。"""
    labels_dists = []
    d4 = distance_m - home_stretch_m
    d3 = d4 - corner_len_m
    d2 = d3 - home_stretch_m
    d1 = d2 - corner_len_m
    for label, d in (("4角", d4), ("3角", d3), ("2角", d2), ("1角", d1)):
        if d is not None and d >= 0:
            labels_dists.append((label, d))
        else:
            break
    return labels_dists


def parse_passing_order_tokens(passing_order):
    """'11-11-10-4' -> [11, 11, 10, 4] のようにint配列で返す。欠測はNone。"""
    if passing_order is None:
        return None
    s = str(passing_order).strip()
    if not s or s.lower() == "nan":
        return None
    tokens = [t for t in s.split("-") if t.strip()]
    out = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out or None


def _rankdata_average(values: dict) -> dict:
    """タイを平均順位で扱うランク化(scipy.stats.rankdata(method='average')相当、
    scipy依存を避けるための自前実装)。"""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


def footrule_avg_rank(rank_a: dict, rank_b: dict):
    """footrule_norm()のaverage-rank対応版(タイを許容する)。
    2026-08-13エンジニアレビュー指摘: 正規化分母がsim_runner_lib.footrule_norm()の
    整数除算(n*n//2)と食い違っていた(浮動小数除算 n*n/2.0)。m1/m5とm6を横並びで
    比較・報告する際の尺度不整合を避けるため、footrule_norm()と同じ整数除算に統一する。"""
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None
    raw = sum(abs(rank_a[k] - rank_b[k]) for k in common)
    max_possible = (n * n) // 2
    return raw / max_possible if max_possible > 0 else None


def race_m6(states, real_df, distance_m, home_stretch_m, corner_len_m, is_straight):
    """1レース分のm6(コーナー別footrule)を返す。states: hp.simulate()の戻り値。
    real_df: そのrace_idの実測行(umaban/passing_order列必須)。

    戻り値: {
      "corners": {"4角": {"footrule": 0.12, "n_matched": 11, "estimated_footrule": .., ...}, ...},
      "overall_footrule": 平均(コーナー横断), "overall_footrule_no1_2": 3角・4角のみの平均,
    }
    直線コース(is_straight)はコーナーが無いためNoneを返す。
    """
    if is_straight:
        return None

    real = real_df.copy()
    real_by_umaban = {}
    for _, r in real.iterrows():
        tokens = parse_passing_order_tokens(r.get("passing_order"))
        if tokens:
            real_by_umaban[int(r["umaban"])] = tokens

    if not real_by_umaban:
        return None

    max_tokens = max(len(v) for v in real_by_umaban.values())
    corners = corner_distances(distance_m, home_stretch_m, corner_len_m)[:max_tokens]
    if not corners:
        return None

    is_estimated = {s.umaban: bool(s.baseline.is_estimated) for s in states}

    result_corners = {}
    footrule_values = []
    footrule_34 = []
    for idx, (label, d) in enumerate(corners):
        # tokensは末尾(4角側)から対応: idx=0->4角->末尾要素(-1)、idx=1->3角->(-2) ...
        real_at_corner = {}
        for u, tokens in real_by_umaban.items():
            if len(tokens) > idx:
                real_at_corner[u] = tokens[-(idx + 1)]
        if len(real_at_corner) < 2:
            continue

        sim_time_at_corner = {}
        for s in states:
            if s.umaban not in real_at_corner:
                continue
            sim_time_at_corner[s.umaban] = hp.time_at_distance(s, max(0.0, d))

        common = sorted(set(sim_time_at_corner) & set(real_at_corner))
        if len(common) < 2:
            continue

        sim_rank = _rankdata_average({u: sim_time_at_corner[u] for u in common})
        real_rank = _rankdata_average({u: real_at_corner[u] for u in common})
        fr = footrule_avg_rank(sim_rank, real_rank)
        if fr is None:
            continue

        est_common = [u for u in common if is_estimated.get(u)]
        non_est_common = [u for u in common if not is_estimated.get(u)]

        def _stratified_footrule(subset):
            """部分集合「内」で順位を再計算してからfootruleを取る(全馬中の順位をそのまま
            使うと、分母(n^2/2、nは部分集合サイズ)と分子(全馬レンジの順位差)のスケールが
            食い違い1.0を超えてしまうため)。"""
            if len(subset) < 2:
                return None
            sub_sim_rank = _rankdata_average({u: sim_time_at_corner[u] for u in subset})
            sub_real_rank = _rankdata_average({u: real_at_corner[u] for u in subset})
            return footrule_avg_rank(sub_sim_rank, sub_real_rank)

        result_corners[label] = {
            "footrule": fr,
            "n_matched": len(common),
            "estimated_footrule": _stratified_footrule(est_common),
            "n_estimated": len(est_common),
            "non_estimated_footrule": _stratified_footrule(non_est_common),
            "n_non_estimated": len(non_est_common),
        }
        footrule_values.append(fr)
        if label in ("3角", "4角"):
            footrule_34.append(fr)

    if not footrule_values:
        return None

    return {
        "corners": result_corners,
        "overall_footrule": float(np.mean(footrule_values)),
        "overall_footrule_3_4": float(np.mean(footrule_34)) if footrule_34 else None,
    }


if __name__ == "__main__":
    print("corner_passing_metrics.py: import only, use race_m6() from another script", file=sys.stderr)
