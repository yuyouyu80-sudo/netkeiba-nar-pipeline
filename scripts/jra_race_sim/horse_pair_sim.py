# -*- coding: utf-8 -*-
"""1番ブレトワルダ・2番カレンワッツアップの2頭による、相互作用ありの前向き
シミュレーション。perceive(知覚)→decide(決定)→apply(適用)の3段階に分離し、
知覚は必ず「1ティック前のスナップショット」からのみ行う(同一ティック内の
更新順序に結果が依存するバグを構造的に防ぐため)。

メカニズム(全てmeters基準。SVG描画変換はhorse_run_pair.html側で行う):
  1. 内側進路変更: ホームストレッチ手前で、内側(レール方向)に他馬がおらず
     縦方向5m以上の隙間があれば、レールへ向けて横移動する(距離短縮のため)。
  2. ドラフティング: 直前(縦6m以内・横2.5m未満)に他馬がいれば、
     スタミナ消耗率をDRAFT_FACTOR倍に軽減する。
  3. ホームストレッチのブロック回避: 残り距離が353.9m(実測)ではなく
     HOME_STRETCH_M(トラックジオメトリから逆算した実測値、sim_geometry参照)
     以下になったら、横移動はもう距離短縮にならない。前方5m未満に他馬がいれば
     左右どちらかに5m以上の隙間があればそちらへ回避、なければ減速して追従する。
  4. 先頭馬のペース配分: ベースラインスタミナ曲線からの乖離(サープラス)を
     追走馬と比較し、劣勢ならペースを落とす(ヒステリシス+速度フロアつき)。
     キック区間(2026-08-09〜: horse_baseline.build_curve()のkick_start_d、
     3コーナー相当地点)以降は発動しない(「スタミナ・スピードを使い切る」という
     要望と矛盾するため)。以前はHOME_STRETCH_M基準だったが、新しいラストスパート
     モデルでキック区間がホームストレッチより広くなったため切り替えた。
"""
import json
from dataclasses import dataclass, field

import numpy as np

import sim_geometry as sg
import horse_baseline as hb

DT = 0.05  # s

GAP_LONGITUDINAL_M = 3.0                  # ホームストレッチで「前に馬がいて回避を検討するか」のブロック判定の縦方向の閾値
# 2026-08-08、ユーザー指定値に変更(苗場特別1レースのfootrule較正値から、8日分の
# 実測との突き合わせで検証するため)。旧値: INSIDE_CUT=2.0, STRETCH_CLEAR=1.0,
# LATERAL_OVERLAP=2.5, LATERAL_GAP_FOR_AVOID=5.0。
GAP_LONGITUDINAL_INSIDE_CUT_M = 1.5       # 内側への進路変更(ホームストレッチ手前)に必要な縦方向の隙間
GAP_LONGITUDINAL_STRETCH_CLEAR_M = 1.0    # ホームストレッチで回避先レーンが「空いている」と判定する縦方向の隙間
LATERAL_OVERLAP_M = 1.5        # これ未満なら「同じレーンにいる」とみなす(馬体幅相当)
LATERAL_GAP_FOR_AVOID_M = 3.0  # ホームストレッチでの左右回避に必要な横方向の隙間
MAX_LATERAL_SPEED_MPS = 1.8    # 横移動の最大速度(瞬間移動を禁止するため)
DRAFT_GAP_M = 6.0              # ドラフト(風よけ)効果が働く縦方向の距離
DRAFT_FACTOR = 0.95            # ドラフト時のスタミナ消耗率倍率

PACE_HOLD_FACTOR = 0.97            # 先頭馬がペースを落とす時の速度倍率
PACE_SURPLUS_MARGIN_START = -8.0   # サープラス差(pt)がこれ以下で減速開始
PACE_SURPLUS_MARGIN_STOP = -3.0    # サープラス差(pt)がこれ以上に回復したら解除
PACE_SPEED_FLOOR_RATIO = 0.85      # ベース速度に対する絶対下限

LANE_MIN_M = 0.0
TRACK_WIDTH_M = 22.0  # 新潟ダート想定の目安(未実測の推定値、頭数の異なる全レース共通で流用)
PAIR = tuple(range(1, 16))  # シミュレーション対象の馬(馬番)。レースごとにsimulate_one_race.pyが上書きする
# 直線コース(コーナーが存在しないコース、例: 新潟の芝1000m専用直線コース)専用フラグ。
# Trueの場合、apply()/simulate()はlaps・lap_t・is_in_corner等「周回するオーバル」前提の
# 計算を一切行わない(circumference_mが物理的に存在しないため)。simulate_one_race.pyが
# レースごとに上書きする。この場合HOME_STRETCH_Mはsimulate_one_race.py側でD_TOTAL以上に
# 設定し、レース全体を「ホームストレッチ挙動(ブロック回避・ペース配分は発動しない)」
# 扱いにする(コーナーが無いレーン移動には距離短縮効果が無いため、内側への進路変更を
# 狙う意味自体が無い、という簡略化)。
IS_STRAIGHT_COURSE = False
# 出走ゲートの初期位置: 実際の競馬は馬番=内側から外側へのゲート順とほぼ一致し、
# 1頭あたり約1mの幅が確保される。「1番と5番の差が指定の5mになる」よう
# GATE_SPACING_M=1.25m/番に校正している(1mちょうどだと1〜5番の差は4mになるため)。
GATE_SPACING_M = 1.25


def gate_start_lane_m(umaban):
    return (umaban - 1) * GATE_SPACING_M

ENABLE_DRAFT = True
ENABLE_INSIDE_CUT = True
ENABLE_STRETCH_AVOID = True
ENABLE_PACE_CONTROL = True


@dataclass
class HorseState:
    umaban: int
    baseline: object
    d_rail: float = 0.0        # レール基準の距離(公式距離。速度・スタミナ曲線はこれで駆動)
    ground_distance: float = 0.0  # 実際に地面を移動した距離(外を回った分、d_railより多くなる)
    lane_m: float = 0.0
    stamina: float = 100.0
    finished: bool = False
    finish_time: float = None
    pace_holding: bool = False
    log: list = field(default_factory=list)


@dataclass
class Decision:
    target_speed: float
    target_lane: float


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_states(pair=PAIR):
    return [HorseState(umaban=u, baseline=hb.load_horse(u), lane_m=gate_start_lane_m(u))
            for u in pair]


def longitudinal_gap(me, other):
    """other が me よりどれだけ前にいるか(正なら前、負なら後ろ)。"""
    return other.d_rail - me.d_rail


def lateral_gap(me, other):
    return abs(other.lane_m - me.lane_m)


def blocking_horse_ahead(me, others):
    """縦5m未満・横2.5m未満で自分の前にいる馬のうち、最も近い1頭を返す(いなければNone)。"""
    candidates = [o for o in others
                  if 0 < longitudinal_gap(me, o) < GAP_LONGITUDINAL_M and lateral_gap(me, o) < LATERAL_OVERLAP_M]
    if not candidates:
        return None
    return min(candidates, key=lambda o: longitudinal_gap(me, o))


def inside_move_blocked(me, others):
    """内側(レール方向)へ動こうとした時、他馬の誰かがその進路上(自分より内側・
    縦4m以内)にいて塞いでいないか。"""
    for other in others:
        if other.lane_m >= me.lane_m - 0.05:
            continue  # otherは自分より内側にいない → この馬は進路を塞がない
        if abs(longitudinal_gap(me, other)) < GAP_LONGITUDINAL_INSIDE_CUT_M:
            return True
    return False


def draft_factor(me, others):
    if not ENABLE_DRAFT:
        return 1.0
    for other in others:
        gap = longitudinal_gap(me, other)  # 正: otherが前 = meがドラフトを受ける
        if 0 < gap < DRAFT_GAP_M and lateral_gap(me, other) < LATERAL_OVERLAP_M:
            return DRAFT_FACTOR
    return 1.0


def decide_pace_control(me, others):
    """先頭馬のペース配分。直後を追走する馬(最もd_railが大きい後続馬)と比較する。
    戻り値: (target_speed_multiplier, new_pace_holding)。"""
    if not ENABLE_PACE_CONTROL or any(o.d_rail >= me.d_rail for o in others):
        return 1.0, False  # 先頭でなければ発動しない
    chaser = max(others, key=lambda o: o.d_rail)
    surplus_self = me.stamina - me.baseline.stamina_baseline(me.d_rail)
    surplus_other = chaser.stamina - chaser.baseline.stamina_baseline(chaser.d_rail)
    deficit = surplus_self - surplus_other
    holding = me.pace_holding
    if holding:
        if deficit >= PACE_SURPLUS_MARGIN_STOP:
            holding = False
    else:
        if deficit <= PACE_SURPLUS_MARGIN_START:
            holding = True
    mult = PACE_HOLD_FACTOR if holding else 1.0
    return mult, holding


def lane_is_clear(target_lane, me, others):
    for other in others:
        if abs(other.lane_m - target_lane) < LATERAL_OVERLAP_M and abs(longitudinal_gap(me, other)) < GAP_LONGITUDINAL_STRETCH_CLEAR_M:
            return False
    return True


def pick_avoid_lane(me, others):
    """ホームストレッチで前方をふさがれた時、左右どちらに逃げるか決める(全馬との
    隙間を確認する)。どちらも無理ならNoneを返す(減速して追従するしかない)。"""
    inward = me.lane_m - LATERAL_GAP_FOR_AVOID_M
    outward = me.lane_m + LATERAL_GAP_FOR_AVOID_M
    candidates = []
    if inward >= LANE_MIN_M and lane_is_clear(inward, me, others):
        candidates.append(inward)
    if outward <= TRACK_WIDTH_M and lane_is_clear(outward, me, others):
        candidates.append(outward)
    # 内側優先(進路が空いていれば距離的に不利にはならない、僅かでも合理的な選択)
    candidates.sort(key=lambda c: abs(c - LANE_MIN_M))
    return candidates[0] if candidates else None


def decide(me, others, is_final_stretch):
    base_speed = me.baseline.speed(me.d_rail)
    # 2026-08-09: ペース配分の除外判定と、進路(内側カット/ブロック回避)の切り替え判定を分離した。
    # 新しいラストスパートモデルでキック区間(3コーナー相当、kick_start_d)がホームストレッチ
    # (is_final_stretch、HOME_STRETCH_M基準)よりゴールから遠い側まで広がるようになったため、
    # 両方をis_final_stretchのままにすると、前傾化したキック立ち上がりの一部が「まだペース配分
    # (先頭馬の減速判定)が有効な区間」と重なってしまう(kick_start_dはHOME_STRETCH_M+
    # CORNER_LEN_M地点なので常にD_TOTAL-HOME_STRETCH_M以下=is_final_stretch開始より先に来る)。
    # 進路の切り替え(内側カット/ブロック回避)は従来通りis_final_stretch基準のまま変更しない。
    # kick_start_dはKICK_START_MIN_M=300mで下限クランプされるため、理論上は
    # (HOME_STRETCH_Mが長く距離が短いコースで)D_TOTAL-HOME_STRETCH_Mを上回りうる
    # (シニアエンジニアレビューで指摘、279レースでの実発火は未確認だが安全側に対処する)。
    # その場合でも「ホームストレッチではペース配分を発動しない」という設計原則を必ず守るため、
    # is_final_stretchも併せて判定する(kick_start_d側の不変条件だけに依存しない)。
    is_past_kick_start = me.d_rail >= me.baseline.curve["kick_start_d"]

    if not is_past_kick_start and not is_final_stretch:
        pace_mult, new_holding = decide_pace_control(me, others)
        me_pace_holding_next = new_holding
        target_speed = max(base_speed * pace_mult, base_speed * PACE_SPEED_FLOOR_RATIO)
    else:
        me_pace_holding_next = False  # 3コーナー(kick_start_d)以降・ホームストレッチではペース配分を発動しない
        target_speed = base_speed

    if not is_final_stretch:
        if ENABLE_INSIDE_CUT and me.lane_m > LANE_MIN_M and not inside_move_blocked(me, others):
            target_lane = max(LANE_MIN_M, me.lane_m - MAX_LATERAL_SPEED_MPS * DT)
        else:
            target_lane = me.lane_m
    else:
        target_lane = me.lane_m
        if ENABLE_STRETCH_AVOID:
            blocker = blocking_horse_ahead(me, others)
            if blocker is not None:
                avoid_lane = pick_avoid_lane(me, others)
                if avoid_lane is not None:
                    target_lane = avoid_lane
                else:
                    target_speed = min(target_speed, blocker.baseline.speed(blocker.d_rail))

    return Decision(target_speed, target_lane), me_pace_holding_next


def apply(state, decision, laps, start_t, t):
    lane_delta = clamp(decision.target_lane - state.lane_m,
                        -MAX_LATERAL_SPEED_MPS * DT, MAX_LATERAL_SPEED_MPS * DT)
    state.lane_m = clamp(state.lane_m + lane_delta, LANE_MIN_M, TRACK_WIDTH_M)

    if IS_STRAIGHT_COURSE:
        cf = 1.0  # 直線コースはコーナーが無いため、レーンによる距離増加は常にゼロ
    else:
        lap_t = sg.lap_t_at_distance(state.d_rail, start_t, laps)
        in_corner = sg.is_in_corner(lap_t)
        cf = sg.CORNER_R_M / (sg.CORNER_R_M + state.lane_m) if in_corner else 1.0

    if state.d_rail < hb.GATE_ACCEL_DIST_M:
        # ゲート加速ランプ区間(v(d)=v_seam*sqrt(d/GATE_ACCEL_DIST_M)、d=0で速度0の特異点)は
        # 「速度×dt」で前進させる通常のオイラー法だと、dtをどれだけ細かくしても収束しない
        # (数十msでも約1秒分の誤差が残ることを検証で確認)。この区間だけは厳密解
        # d(t) = k*t^2 (k = v_seam^2/(4*GATE_ACCEL_DIST_M)) を直接使う。tは全馬が同時に
        # 動き出すこのシミュレーションでは経過時間そのもの。区間はごく短い(2m)ため、
        # その間のコーナー係数cfは定数として扱う近似で誤差は無視できる。
        v_seam = state.baseline.speed(hb.GATE_ACCEL_DIST_M)
        k = v_seam * v_seam / (4.0 * hb.GATE_ACCEL_DIST_M)
        new_d_rail = min(hb.GATE_ACCEL_DIST_M, k * t * t)
        d_rail_increment = max(0.0, new_d_rail - state.d_rail)
        ground_increment = d_rail_increment / cf
        state.d_rail = new_d_rail
        state.ground_distance += ground_increment
    else:
        ground_increment = decision.target_speed * DT   # 実際に走った距離(レーンによる補正なし)
        d_rail_increment = ground_increment * cf          # レール基準の進み(コーナーでレーン分だけ目減り)
        state.d_rail = min(hb.D_TOTAL, state.d_rail + d_rail_increment)
        state.ground_distance += ground_increment
    return d_rail_increment, ground_increment


def simulate(t_max=200.0, verbose=False):
    states = make_states(PAIR)
    if IS_STRAIGHT_COURSE:
        laps, start_t = 0.0, 0.0  # 直線コースでは未使用(apply()側でIS_STRAIGHT_COURSE分岐によりスキップ)
    else:
        laps = hb.D_TOTAL / sg.CIRCUMFERENCE_M
        start_t = sg.start_lap_t(sg.G["finish_lap_t"], laps)

    t = 0.0
    for s in states:
        s.log.append((t, s.d_rail, s.lane_m, s.baseline.speed(s.d_rail), s.stamina, s.ground_distance))

    n_steps = int(t_max / DT)
    for _ in range(n_steps):
        if all(s.finished for s in states):
            break
        snapshot = [HorseState(**{**s.__dict__, "log": []}) for s in states]  # 1ティック前のコピー

        decisions = []
        for i, me in enumerate(states):
            others = snapshot[:i] + snapshot[i + 1:]
            is_final = sg.is_final_stretch(me.d_rail, hb.D_TOTAL)
            if me.finished:
                decisions.append((Decision(0.0, me.lane_m), me.pace_holding))
            else:
                decisions.append(decide(me, others, is_final))

        t += DT
        for i, me in enumerate(states):
            decision, next_holding = decisions[i]
            others = snapshot[:i] + snapshot[i + 1:]
            if not me.finished:
                d_rail_increment, ground_increment = apply(me, decision, laps, start_t, t)
                dfactor = draft_factor(me, others)
                # スタミナは「実際に走った距離(ground_increment)」に対して課金する。外を回ると
                # 同じd_rail進行でもground_incrementが大きくなり(コーナーでR/(R+L)分だけ余計に
                # 走る必要があるため)、内側の馬より早くスタミナを消耗する。以前はd_rail基準で
                # 課金しておりこの効果が欠落していた(タイムの遅れだけがワイド走行のペナルティに
                # なっていたが、実際のレースではスタミナ消耗の増加も伴う)。
                me.stamina -= me.baseline.effort(me.d_rail) * dfactor * ground_increment
                me.pace_holding = next_holding
                if me.d_rail >= hb.D_TOTAL - 1e-9:
                    me.finished = True
                    me.finish_time = t
            me.log.append((t, me.d_rail, me.lane_m, decision.target_speed, me.stamina, me.ground_distance))

    return states


def time_at_distance(state, d_target):
    """状態のログ(d_railは単調増加)から、d_railが最初にd_targetへ達した時刻を線形補間で
    求める。ゴール後は他馬が完走するまでログがd_railを固定したまま続く(フリーズ)ため、
    「最後のログ」ではなく必ず二分探索で最初にd_targetへ到達したインデックスを探す
    (でないと完走後に足踏みしている間の時刻を誤って返してしまう)。"""
    log = state.log
    if d_target <= log[0][1]:
        return log[0][0]
    lo, hi = 0, len(log) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if log[mid][1] < d_target:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return log[0][0]
    a, b = log[lo - 1], log[lo]
    span = b[1] - a[1]
    f = (d_target - a[1]) / span if span > 0 else 0.0
    return a[0] + (b[0] - a[0]) * f


def leader_lap_table(states, d_total, step=200):
    """200mごとに、その地点に最初に到達した(=その時点の1位の)馬の通過タイムと
    区間(ラップ)タイムを求める。実際のレースのラップタイム表と同じ形式。"""
    marks = list(range(step, int(d_total) + 1, step))
    rows = []
    prev_t = 0.0
    for d_mark in marks:
        t_leader, umaban_leader = min(
            (time_at_distance(s, d_mark), s.umaban) for s in states
        )
        rows.append({
            "distance": d_mark, "umaban": umaban_leader,
            "cumulative": round(t_leader, 2), "split": round(t_leader - prev_t, 2),
        })
        prev_t = t_leader
    return rows


def export_json(states, out_path, every=2):
    """every=2 (dt=0.05sの2ステップごと=0.1s間隔)に間引いてJSONへ出力。"""
    horses = {}
    for s in states:
        pts = []
        for i, (t, d, lane, v, stamina, ground_d) in enumerate(s.log):
            if i % every == 0 or i == len(s.log) - 1:
                pts.append([round(t, 3), round(d, 2), round(lane, 3), round(v, 3), round(stamina, 2), round(ground_d, 2)])
        # s.log[-1][0]は全馬共有の最終ログ時刻(一番遅く完走した馬の時刻)なので、
        # 早く完走した馬には使えない(time_at_distanceと同じ「フリーズしたログ」問題)。
        # 必ず各馬自身のfinish_timeを使う。
        finish_time = s.finish_time if s.finish_time is not None else s.log[-1][0]
        horses[str(s.umaban)] = {
            "name": s.baseline.name, "waku": s.baseline.waku,
            "isEstimated": s.baseline.is_estimated, "totalTime": round(finish_time, 2), "pts": pts,
            # 脚質(逃/先/差/追。過去走データが無い馬はNone)。物理演算(build_curve()の
            # running_style引数)が実際に使った値そのものであり、実測結果の後付けマージ
            # (build_venue_artifact.pyのget_actual_for_race())からは取得しない
            # (シミュレーション入力と表示値がズレるリスクを避けるため、UI/UXレビュー指摘)。
            "runningStyle": s.baseline.running_style,
        }
    payload = {"distance": hb.D_TOTAL, "metersPerUnit": sg.METERS_PER_UNIT,
               "trackWidthM": TRACK_WIDTH_M, "horses": horses,
               "leaderLapTable": leader_lap_table(states, hb.D_TOTAL, step=200),
               # HTML側のJSが独自にジオメトリ値を持つと(既存のhorse_run_pair.htmlで
               # 329.75というPython側physics_geometry導入前の旧derivation値がJS側に
               # 残ってしまい353.9という実際の値とズレていた実例がある)、レースごとに
               # 別途値を渡し忘れる/ズレるリスクがあるため、Pythonが実際に使った幾何値を
               # そのままJSON経由でHTML側へ渡し、単一の真実源にする。
               "circumferenceM": (None if IS_STRAIGHT_COURSE else sg.CIRCUMFERENCE_M),
               "homeStretchM": sg.HOME_STRETCH_M,
               "isStraightCourse": IS_STRAIGHT_COURSE}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return payload


def summarize(states):
    for s in states:
        d_rails = [p[1] for p in s.log]
        lanes = [p[2] for p in s.log]
        monotone = all(d_rails[i] <= d_rails[i + 1] + 1e-6 for i in range(len(d_rails) - 1))
        in_bounds = all(LANE_MIN_M - 1e-6 <= l <= TRACK_WIDTH_M + 1e-6 for l in lanes)
        finish_time = s.finish_time if s.finish_time is not None else s.log[-1][0]
        surplus = s.stamina - s.baseline.stamina_baseline(s.d_rail)
        print("umaban=%d %-10s finish=%.2fs (solo=%.2fs, +%.2fs)  final_stamina=%.1f%%  "
              "surplus_vs_baseline=%.1fpt  lane[min=%.2f,max=%.2f]  d_rail_monotone=%s  lane_in_bounds=%s  "
              "ground_distance=%.1fm(+%.1fm vs 公式1800m)" % (
                  s.umaban, s.baseline.name, finish_time, s.baseline.total_time_solo,
                  finish_time - s.baseline.total_time_solo, s.stamina, surplus,
                  min(lanes), max(lanes), monotone, in_bounds,
                  s.ground_distance, s.ground_distance - hb.D_TOTAL))


if __name__ == "__main__":
    states = simulate()
    summarize(states)
    out = export_json(states, r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir\horse_pair_sim.json")
    print("exported horses:", list(out["horses"].keys()), "pts per horse:", len(next(iter(out["horses"].values()))["pts"]))
