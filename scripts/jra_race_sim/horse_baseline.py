# -*- coding: utf-8 -*-
"""馬ごとの速度・スタミナカーブ(HorseBaseline)を組み立てるモジュール。

**序盤0-1200m(shape_seg1)は実際のレース(苗場特別・ブレトワルダ、ユーザー提供の実測200m
ラップ)に基づいて較正済み**。実測ラップの200m区間平均速度は
[16.00, 18.35, 16.13, 15.15, 15.87, 15.87] m/sで、序盤200-400mにピーク、600-800mで
いったん落ち込み、800m以降で巡航速度に収束するという「ゲートで遅い」とは逆の形状だった。
区間中心(100,300,...,1100m)を通る線形補間で実測を再現している。all_horses_curve.py
(未修正のまま残置、このシミュレーション作業では使わない)にも同型のshape_seg1較正前の
バージョンがある。

**2026-08-09以降、ラストスパート(kick_start_d〜ゴール)はshape_seg1とは別の、
スタミナ収支・最高速度駆動の新モデル(build_curve()内、solve_kick_r()等)を使う**。
旧モデル(上がり3Fタイムへのcurve-fit)は撤廃した。詳細は`BASE_RATE_ABS`定義前後の
コメントとメモリ(project_kick_model_redesign_2026_08_09.md)を参照。
"""
import numpy as np
import pandas as pd

import sim_geometry as sg

CSV_PATH = r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir\naeba_potential.csv"
D_TOTAL = 1800.0
SEG1_LEN = 1200.0
SEG2_LEN = 600.0
N = 901  # 2m刻み

# 実測200mラップ(苗場特別・ブレトワルダ)の区間平均速度を、区間中心の距離に対する
# 「巡航速度(1000-1200m区間)比」として正規化した値。200-400m区間(18.35m/s)が
# 全区間中で最速=最高速度で、300mビンがその代表点。0-200m区間(16.00m/s)は
# 「静止状態からスタートダッシュで最高速度まで加速する途中の平均」と解釈し、
# 独立した速度プラトーとしては扱わない(下記の3段階モデルを参照)。
SEG1_BIN_CENTERS_M = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])
_real_seg_speeds = np.array([18.35, 16.13, 15.15, 15.87, 15.87])
SEG1_BIN_RATIO = _real_seg_speeds / _real_seg_speeds[-1]
PEAK_RATIO = SEG1_BIN_RATIO[0]  # 最高速度比(200-400m区間の実測値、全区間中の最速)

# 序盤(0-200m)を3段階でモデル化する:
#  1. [0, GATE_ACCEL_DIST_M]: ゲートを出た直後、静止状態(0m/s)から加速する等加速度運動
#     v(d) = v_seam*sqrt(d/GATE_ACCEL_DIST_M)。d=0で厳密に速度0。
#  2. [GATE_ACCEL_DIST_M, DASH_PEAK_DIST_M]: 「スタートダッシュ」でさらに加速し、
#     DASH_PEAK_DIST_M地点で最高速度(PEAK_RATIO)に一度だけ到達する。
#  3. [DASH_PEAK_DIST_M, SEG1_LEN]: 実測ビン(300,500,700,900,1100m)を線形補間。
#     DASH_PEAK_DIST_M=200は300mビンと同値(最高速度)なのでここで連続に繋がる。
GATE_ACCEL_DIST_M = 20.0
DASH_PEAK_DIST_M = 200.0
DASH_SEAM_RATIO = 16.00 / 15.87  # GATE_ACCEL_DIST_M地点の速度比(旧モデルの0-100mプラトー値を流用)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def shape_seg1(d):
    d = np.asarray(d, dtype=float)
    # [DASH_PEAK_DIST_M, SEG1_LEN]: 実測ビンの線形補間(範囲外はクランプ)。1200m地点は
    # SEG1_BIN_RATIO[-1]=1.0でshape_seg2(u=0, taper)=1.0と連続になる。
    base = np.interp(d, SEG1_BIN_CENTERS_M, SEG1_BIN_RATIO)
    # [GATE_ACCEL_DIST_M, DASH_PEAK_DIST_M]: スタートダッシュでDASH_SEAM_RATIOから
    # PEAK_RATIOへ滑らかに立ち上がる(単調・一度だけピークに到達)。
    dash_u = np.clip((d - GATE_ACCEL_DIST_M) / (DASH_PEAK_DIST_M - GATE_ACCEL_DIST_M), 0.0, 1.0)
    dash = DASH_SEAM_RATIO + (PEAK_RATIO - DASH_SEAM_RATIO) * smoothstep(dash_u)
    # [0, GATE_ACCEL_DIST_M]: 0からDASH_SEAM_RATIOへのsqrtランプ(等加速度運動)。
    ramp = DASH_SEAM_RATIO * np.sqrt(np.clip(d, 0.0, GATE_ACCEL_DIST_M) / GATE_ACCEL_DIST_M)
    return np.where(d < GATE_ACCEL_DIST_M, ramp, np.where(d < DASH_PEAK_DIST_M, dash, base))


def _style_key(style):
    """RUNNING_STYLE_POSITION_SCOREの有効キー("逃"/"先"/"差"/"追")ならそのまま、
    それ以外(None・NaN・未知のラベル)はNoneに正規化する。"""
    return style if style in RUNNING_STYLE_POSITION_SCORE else None


def style_multiplier(d, style):
    """脚質による巡航区間の速度形状補正。d<MULTIPLIER_ONSET_DIST_M(=DASH_PEAK_DIST_M=200、
    ゲートダッシュ+スタート直後の共通フェーズ)またはstyle未設定/NaN/未知ラベルでは、
    RUNNING_STYLE_K1の値によらず無条件に1.0(補正なし)を返す明示分岐にしている
    (「position_score=0.5で結果的に1.0になる」という算術的な相殺には頼らない設計。
    _gate_ramp_time()がshape_seg1のd<GATE_ACCEL_DIST_M区間の値を前提にした独立の解析解を
    持つため、この区間の値を脚質で動かすと時間不変性が崩れる。シニアエンジニアレビュー指摘)。"""
    d = np.asarray(d, dtype=float)
    key = _style_key(style)
    if key is None:
        return np.ones_like(d)
    score = RUNNING_STYLE_POSITION_SCORE[key]  # 0=先頭寄り・1=後方寄り
    strength = RUNNING_STYLE_K1 * (0.5 - score)  # 逃(score<0.5)は正=速め、追(score>0.5)は負=抑えめ
    u = np.clip((d - MULTIPLIER_ONSET_DIST_M) / MULTIPLIER_ONSET_RAMP_M, 0.0, 1.0)
    return 1.0 + strength * smoothstep(u)


def shape_seg1_for_style(d, style):
    """shape_seg1(d)に脚質補正を掛けたもの。style=None(NaN)またはRUNNING_STYLE_K1=0のとき、
    style_multiplier()が恒等的に1.0を返すため、shape_seg1(d)とビット単位で一致する。"""
    return shape_seg1(d) * style_multiplier(d, style)


def _gate_ramp_time(v_cruise):
    """v(d) = v_cruise*DASH_SEAM_RATIO*sqrt(d/GATE_ACCEL_DIST_M) の0〜GATE_ACCEL_DIST_Mの
    解析解時間。台形則(離散数値積分)はこのsqrt特異点付近で収束が遅く、n=4000程度では
    無視できない誤差(実測で0.3秒超)が出ることが判明したため、この区間だけは厳密解を使う。"""
    v_seam = v_cruise * DASH_SEAM_RATIO
    return 2.0 * GATE_ACCEL_DIST_M / v_seam


def seg1_time_for_v(v_cruise, n=4000):
    # ゲートランプ区間[0, GATE_ACCEL_DIST_M]は解析解、残り[GATE_ACCEL_DIST_M, SEG1_LEN]
    # (スタートダッシュ+実測ビン)は特異点を含まないので通常の台形則で十分正確。
    t_ramp = _gate_ramp_time(v_cruise)
    dd = np.linspace(GATE_ACCEL_DIST_M, SEG1_LEN, n)
    t_rest = np.trapezoid(1.0 / (v_cruise * shape_seg1(dd)), dd)
    return t_ramp + t_rest


# 2026-08-09: ラストスパートを「3コーナー起点・ゴールでスタミナ30%前後着地・最高速度は
# スピード指数_直近5走平均を参考」という新モデルへ全面置き換え(ユーザー指定・全面置き換え
# 方式で確定)。旧モデル(shape_seg2/solve_taper、実測or推定の上がり3Fタイムへcurve-fit)は
# 撤廃した。t_l3f(実測or推定)は物理フィットには使わず、HorseBaseline.t_l3f_input経由で
# 検証用の参考値としてのみ残す。診断: 五頭連峰特別(202604020408、新潟8R芝1800m)で
# baseline(単走)が実測より平均+1.12秒遅く、279レース全体でも「相互作用寄与
# (m2_error-ctrl_solo_error)」が279/279レース(100%)で正——horse_pair_sim.py側の
# 「速度は常にbaseline.speed()が上限」という構造が主因で今回の変更では反転しない見込みだが、
# baseline自体を実測に近づけることで絶対誤差は縮小しうる。詳細は
# project_kick_model_redesign_2026_08_09.md(メモリ)参照。

KICK_START_MIN_M = 300.0   # shape_seg1の実測ビン下端(SEG1_BIN_CENTERS_M[0])。これ未満は実測根拠が無い
MIN_KICK_LEN_M = 150.0     # 防御的下限(279レースの実ジオメトリでは発火しない、Plan agent検証済み)
TARGET_FINAL_STAMINA = 30.0  # ゴール到達時の目標スタミナ残り(ユーザー指定の固定値、旧40+0.4*stamina_indexを置換)
KICK_EFFORT_EXPONENT = 2.0   # effort_kick=(v/v_start)^exponent。出力~v^3・単位距離effort~v^2という
                              # 物理的近似に基づき固定し、較正対象にしない(自由定数を絞る方針)。
R_MIN, R_MAX = 0.80, 3.0     # v_peak/v_start の二分探索範囲。
# 2026-08-09追記: R_MIN=0.3(v_peakがv_startの30%まで減速可)は根拠のない仮値で、
# 実際には279レース中56.6%でsolve_kick_r()がr=0.3に完全飽和し、ゴール前に速度が
# 壊滅的に落ちる(先頭馬ラップ実測比較で終盤200mが実測12.8秒→sim28.2秒等)非現実的な
# 形状を生んでいた(ユーザー指摘: 「先頭馬ラップタイムを参考に修正」「上がり3Fの乖離も激しい」、
# 上がり3F誤差は3604頭平均+12.06秒・中央値+16.31秒という壊滅的な値だった)。
# recalibrate_kick_rmin.py(data/lap_times実測+コース幾何からkick_start_d地点の実測速度に
# 対するゴール直前200m区間の実測速度比を279レース全件で算出)で較正: 全体
# min=0.817・p1=0.820・p5=0.861・p10=0.878・中央値0.968(ダート0.946/芝0.994、
# むしろ芝はほぼ減速無しが標準)。R_MIN=0.80はこの実測レンジの下限(0.817)よりわずかに
# 保守的(279レース外の未知パターン・区間平均による近似誤差の余地)だが、旧R_MIN=0.3が
# 実測の最も極端なケースの2.5倍以上の減速を許していたことに比べれば大幅に現実的。
#
# 2026-08-18追記: JRA実況動画フレーム(8/1・8/2の17レース、視覚読み取り+二重読取で
# 再現性確認済み)を使い、KICK_START_MIN_M・DASH_PEAK_DIST_M・R_MIN/R_MAX・
# KICK_EFFORT_EXPONENTの再較正を試みたが(探索用12レース+確認用5レースの二段階、
# 詳細はvaliant-cuddling-aho.mdプラン参照)、いずれも有意・一貫した改善候補は
# 見つからなかった。R_MIN/R_MAXとKICK_EFFORT_EXPONENTは2026-08-15の感度診断で
# 既に構造的に較正不能(capped分岐がほぼ全馬を占める)と判明済み。KICK_START_MIN_Mは
# 17レース全件のkick_start_d自然値(379.5〜1179.5m)が現行floor=300.0mを一度も
# binding しておらず、floorを引き上げても改善傾向が出なかった。DASH_PEAK_DIST_Mは
# 探索フェーズで100〜150mにわずかな改善傾向が見えたが、確認用データでは100は
# 効果ゼロ(同値)・150はむしろ悪化と、探索フェーズのシグナルが再現されなかった
# (探索/確認の分離で選択バイアスを検出できた例)。5パラメータとも変更なしで確定。

# BASE_RATE_ABS: 巡航区間([0,kick_start_d))の絶対スタミナ消費率(%/m)。実測ラベルが無いため
# recalibrate_kick_base_rate.py でfootrule/タイムMAEを最小化するグリッドサーチ+
# 訓練70レース/検証(holdout)70レース分割(4件に1件ずつ、重複無し)で較正する(1個だけの自由定数)。
#
# 初回探索[0.003,0.005,0.008,0.012,0.018,0.025]は全候補でfootrule/timeMAEが完全に同一の
# 値になるという異常を検知した(_diag_base_rate_range.pyで原因調査、シニアエンジニア
# レビューでも独立に同じ問題を指摘): この範囲ではsolve_kick_r()が求めるv_peakが常に
# max_speed_capを超えてしまい、capped分岐(v_cruiseをmax_speed_cap基準に再計算)に
# 常に落ちるため、BASE_RATE_ABSの値自体が結果に反映されていなかった(TARGET_FINAL_STAMINA=30も
# 事実上機能せず、final_stamina中央値が実測48〜83%と30から大きく乖離)。
# [0.035,0.040,0.045,0.050,0.055]→[0.060,0.065]と範囲を上げて再探索し、0.060(footrule
# train 0.5916・holdout 0.5974)を採用していた。
#
# 2026-08-09追記(第2版・現行値): この0.060はR_MIN=0.3(非現実的な仮値)を前提にした較正
# だったと判明。実際には279レース中56.6%でr=R_MIN=0.3に完全飽和し、ゴール前に速度が
# 壊滅的に落ちる形状になっていた(上がり3F誤差3604頭平均+12.06秒・中央値+16.31秒、
# ユーザー指摘「先頭馬ラップタイムを参考に修正」「上がり3Fの乖離も激しい」で発覚)。
# R_MINを実測ベースの0.80へ引き上げた(定義箇所のコメント参照)ことでBASE_RATE_ABSの
# 効き方が根本的に変わるため再較正した。目的関数もfootrule/timeMAEだけでは形状の破綻を
# 検知できなかった反省から、上がり3F絶対誤差(m3)を主指標に追加(recalibrate_kick_base_rate.py)。
# 候補[0.025,0.030,0.035,0.040,0.045,0.050]のtrain(70レース)でl3fMAEは0.025/0.030=1.437秒
# (ほぼ同値、max_speed_capによるcapped分岐が支配的でBASE_RATE_ABSの影響が小さい領域)→
# 0.035で同じく1.437秒(僅差ながら最小)→0.040以降1.827秒→2.573秒→2.810秒と急速に悪化
# (R_MIN=0.80への飽和が増えるため)。0.035をholdout(別70レース)で再検証しl3fMAE=1.554秒・
# footrule=0.5960・timeMAE=1.699秒と良好であることを確認し、0.035を採用。
BASE_RATE_ABS = 0.0350

# 脚質(逃げ/先行/差し/追込)による位置取り・キック強度補正(2026-08-XX)。レバー1(位置取り、
# shape_seg1_for_style経由)とレバー2(キック強度、TARGET_FINAL_STAMINA経由)の2つが共有する
# 「position_score」軸(0=隊列先頭寄り・1=後方寄り)。recalibrate_running_style.pyのPhase 0で
# 279レース中オーバルコース275レース・3543頭(race_potential_*.csvの脚質列×data/race_results
# のpassing_order、最初/最終コーナー通過順を(token-1)/(field_size-1)で正規化)を実測して
# 決定した値(較正で再フィットする対象ではなく固定値。自由パラメータはK1・K2の2個のみに絞る方針)。
RUNNING_STYLE_POSITION_SCORE = {"逃": 0.2593, "先": 0.3788, "差": 0.5375, "追": 0.7039}

# 上がり3F相対順位の実測ターゲット(0=最速、参考値。同スクリプトPhase 0で算出)。
# 差0.4562(最速)・先0.5223・追0.5328・逃0.6011(最遅)——「追込が上がり3Fで最も脚を使う」
# わけではなく、実際には差しが最速で追込は先行とほぼ同水準、という実測結果(ユーザー提示の
# 一般統計とは「差し・追込のどちらが上か」の部分で食い違うが、実測データを優先する)。
RUNNING_STYLE_L3F_RANK_TARGET = {"逃": 0.6011, "先": 0.5223, "差": 0.4562, "追": 0.5328}

# 2026-08-11: K1/K2較正の経緯(値の符号がプランの当初想定と逆になった理由を含む)。
#
# 第1ラウンド(逐次較正、Stage A: K2=0固定でK1、Stage B: K1固定でK2)では
# STAGE_A_CANDIDATES=[0.0,0.5,1.0,1.5,2.0,3.0](非負のみ)でK1=0.00(補正なし)が
# loss最小という結果になった。シニアエンジニアレビュー+build_curve()の直接検算で、
# これは探索範囲が非負に偏っていたための誤りと判明: build_curve()内でs0
# (=脚質補正後のshape_seg1_for_style(kick_start_d))がk_kick(=(D_TOTAL-kick_start_d)*
# kick_time_integral(r)/s0)の分母に入るため、K1>0で逃げ馬のs0を局所的に上げると
# k_kick・k_cruiseが両方縮小し、時間不変性を保つよう再計算されたv_cruiseが逆に下がり、
# 区間積分であるkickStartTime(=k_cruise/v_cruise)では符号が反転する(瞬時値v_startは
# 意図通りの符号で動くが、隊列内の到達順位という要件2そのものは逆行する)。
# 負のK1を試すとloss 0.38→0.25(K1=-0.5)とただちに改善したため、探索範囲を
# [-1.0,-0.75,-0.5,-0.35,-0.2,-0.1,0.0,0.5,1.0]に組み直しK1=-0.35(train loss=0.1292、
# holdout=0.1642)を得た。
#
# 続けてK1=-0.35固定でStage Bを再実行すると、今度はK2=0.00が最良(K2を上げるほど
# 単調悪化)という同型の片側探索結果になった。原因はK1のl3fへの波及(r自体はK1に
# 依存しないが、v_startの変化を通じキック区間の速度連続性に効くため)で、K1=-0.35の
# 時点で既にK2=0でも逃/追がtarget方向へオーバーシュートしていた。負のK2を試すと
# K2=-30.00でloss 0.4777→0.0744まで改善した。
#
# さらにK2=-30固定でK1を再チェックすると最適値が探索境界(-0.5)からも動いておらず、
# K1↔K2の双方向の波及(K1はs0経由、K2はr経由、いずれもk_kickへ独立に効く)が
# 逐次1変数較正では収束しないほど強いと判断し、K1×K2の2次元グリッドサーチに切替えた
# (position損失とl3f損失を同一シミュレーション結果から同時に計算するevaluate_joint()、
# combined_loss=position_loss+l3f_loss)。粗いグリッド([-1.5,-1.2,-0.9,-0.6,-0.3,0.0]x
# [-45,-30,-15,0])・細かいグリッド([-0.40,-0.35,-0.30,-0.25,-0.20]x[-20,-15,-10])の
# 両方でK1=-0.30/K2=-15.00が隣接候補すべてを下回る安定した内部最適値と確認
# (train combined=0.3271、holdout combined=0.3991、近傍の損失曲面もなだらか)。
#
# 結果として、K1・K2ともプランが当初想定した符号(「逃は目標スタミナ高め=キック弱め」等)
# とは逆になった。ただし観測される脚質別のposition_score・l3f順位(build_curve()の内部
# パラメータの符号ではなく実際のシミュレーション結果)は実測ターゲットと良く一致しており、
# このプロジェクトが一貫して採用してきた「パラメータの物語より観測結果の実測整合性を優先する」
# 方針(上がり3Fの脚質順位でユーザー提示統計より実測データを優先した判断と同型)に従い、
# 符号が直感と異なることを理由に不採用とはしない。詳細はrunning_style_stage_a_report.json・
# running_style_stage_b_report.json・running_style_joint_report.json(scratchpad)参照。
RUNNING_STYLE_K1 = -0.55  # レバー1(位置取り)の効き目。符号は「逃が前方化」の直感と逆(上記コメント参照)
RUNNING_STYLE_K2 = 0.0  # レバー2(キック強度)。2026-08-15較正でこのレバーは実質不要と判明(下記コメント参照)
# 2026-08-13更新(旧値: K1=-0.30, K2=-15.0): sim_geometry.is_in_corner()の固定比率バグ
# (計画ステップ0)を修正した後、脚質集団平均較正(recalibrate_running_style.py --phase joint)の
# combined_loss基準ではholdoutで旧値の方が良く見えたが、これは粗い代理指標(脚質群平均の
# position/L3F順位)であり、実際のレース精度とは相関が弱いと判明した。着順footrule(m1)・
# 走破タイムMAE(m2)・直線入りfootrule(m5)・コーナー通過順位footrule(m6、新設)という、
# 実際のレース結果に対する直接指標では train・holdout の両方かつ全指標でK1=-0.35/K2=-10.0が
# 一貫して優れていた(m6: train +1.6%, holdout +2.8%改善)。
#
# エンジニアレビュー指摘を受け、この2候補だけの比較が過学習でないか、holdout 69レースの
# ペア差分・ペアブートストラップ95%CI(significance_test_k1k2.py、既存verify_val_significance.py
# と同じ手法)で統計的有意性を検証した。結果: **m6(このステップの目的指標)のみ有意に改善**
# (mean_diff=-0.0160, 95%CI=[-0.0244,-0.0073]、0をまたがない。改善39レース/悪化16レース/同14)。
# m1/m2/m5は95%CIが0をまたぎ有意差なし(=改善も悪化もしていないと判断、悪化はしていない)。
# 象徴レース(新潟8R五頭連峰特別)ではm6が悪化した(0.6224→0.6531)が、これは上記69レース中の
# 自然なばらつき(悪化16件のうちの1件)であり、全体の統計的有意性を覆すものではないと判断した。
# 詳細: scratchpad/significance_test_k1k2.log・significance_test_k1k2_result.json、
# evaluate_m6_batch.log・m6_batch_evaluation_result.json参照。
# 【今後の課題】今回はcombined_lossグリッド15点中2点のみをm1/m2/m5/m6で比較した。他候補
# (特にK1=-0.25近辺)も同様に評価し、-0.35/-10.0が真に安定した近傍最適かは未確認。
#
# 2026-08-15更新(旧値: K1=-0.35, K2=-10.0): 上記【今後の課題】に対応。ユーザーから
# 「展開予想の答え合わせ(8/1・8/2)で途中順位が実際の結果と大きく異なる」という指摘を受け、
# K1/K2近傍のグリッドサーチを実施(プラン: valiant-cuddling-aho.md、シニアエンジニア・
# UI/UXデザイナー2専門家レビュー反映済み)。エンジニアレビュー指摘を踏まえ、単一holdoutでの
# 「選んでから検定する」多重比較を避けるため、train(oval_ids[0::4]、69レース)で候補選抜→
# これまで較正・検定に一度も使っていない独立プール(oval_ids[1::4]、新holdout、69レース)で
# 最終検定、という二段階構成にした(旧holdout oval_ids[2::4]は8/13で既に意思決定に使用済みの
# ため参考値のみ)。
#
# Stage A(train、粗いグリッド9点→境界で単調改善が続いたため2回追加でグリッドを外側へ拡張、
# 計22候補評価): K1が負に大きいほど・K2が0に近いほどtrain上のm6が単調に改善し続け、
# K1=-0.55/K2=0.0で頭打ち(m6 train=0.5636、旧値train=0.5901)。この「境界まで単調改善」
# という挙動自体は過学習を疑うべき典型的な兆候(エンジニアレビュー指摘)のため、より穏健な
# 候補(K1=-0.45/K2=-5.0)も含め2候補をStage Bで検証した。
#
# Stage B(新holdout、Bonferroni補正後97.5%CI、N_BOOT=2000、SEED=20260815): 2候補とも
# 「m6が有意に改善し、m1/m2/m5が有意に悪化していない」という採用基準を満たした。
# K1=-0.55/K2=0.0: m6 mean_diff=-0.0333, 97.5%CI=[-0.0478,-0.0200](有意に改善)、
# m5 mean_diff=-0.0227, 97.5%CI=[-0.0427,-0.0031](有意に改善)、m2も有意に改善、
# m1は97.5%CI=[-0.0232,+0.0075]で有意差なし(=悪化していない)。旧holdout(参考、95%CI)でも
# 同方向で再現(m6 mean_diff=-0.0252、m1以外すべて有意改善)。K1=-0.45/K2=-5.0はm6のみ有意改善
# (m5は有意差なし)で、K1=-0.55/K2=0.0の方が全指標で同等以上に良かったため、こちらを採用した。
# 「境界まで単調改善」という懸念は、2つの独立したholdout(train選抜には一切使っていない
# oval_ids[1::4]と、8/13で使用済みのoval_ids[2::4])の両方で同方向・同程度の効果が再現した
# ことで相当程度打ち消されたと判断(単一splitへの過学習なら通常ここまで綺麗には再現しない)。
#
# K2=0.0は「レバー2(キック強度による脚質補正)が実質不要」という意味で、8/11のK1/K2較正
# コメントが前提としていた「2レバー必要」というモデルの物語自体を変える結果になるが、
# このプロジェクトの一貫した方針(パラメータの物語より観測結果の実測整合性を優先)に従い、
# 直感との食い違いを理由に不採用とはしない。
#
# 詳細: scratchpad/step2_stageA_result.csv(22候補のtrain評価)、
# step2_stageB_result.json(2候補×新旧holdoutの検定結果)、step1_diagnostic_result.csv
# (旧race_json vs 現行パラメータの診断、パラメータ変更前の切り分け)参照。
#
# 【今後の課題】今回もK1<-0.55・K2>0の方向をさらに追った場合に改善が続くかは未確認
# (境界での打ち切り)。次回較正時に近傍を再確認すること。

MULTIPLIER_ONSET_DIST_M = DASH_PEAK_DIST_M  # =200.0。この距離未満はstyle_multiplierが恒等的に1.0
MULTIPLIER_ONSET_RAMP_M = 300.0             # onsetからこの距離だけかけて定常値まで滑らかに立ち上がる(固定値、較正対象外)


def style_target_final_stamina(style):
    """脚質によるTARGET_FINAL_STAMINA補正(レバー2)。style未設定/NaN/未知ラベルは
    無条件にTARGET_FINAL_STAMINA(補正なし)を返す明示分岐(style_multiplierと同じ設計方針)。
    score<0.5(逃・先寄り)は目標スタミナを高めに(キック弱め)、score>0.5(差・追寄り)は
    低めに(キック強め)動かす。レバー1のstrength=K1*(0.5-score)と同じ符号規約。"""
    key = _style_key(style)
    if key is None:
        return TARGET_FINAL_STAMINA
    score = RUNNING_STYLE_POSITION_SCORE[key]
    return TARGET_FINAL_STAMINA + RUNNING_STYLE_K2 * (0.5 - score)

# 「スピード指数_直近5走平均」→最高速度(m/s)の較正定数(recalibrate_kick_peak_speed.py、
# 279レース3604頭・実測上がり3Fとの回帰、n=1729(芝)/1293(ダート)、R²は芝0.049/ダート0.047と
# 低い(この特徴量単体では終盤ペースの分散の5%程度しか説明しない、既知の限界)。
# ブートストラップ95%CIは芝[0.0052,0.0176]・ダート[0.0060,0.0167]で傾きの符号は安定。
_KICK_SPEED_IDX_VALID_RANGE = (0.0, 105.0)  # 実測range-252〜106のうち負値・105超を異常値としてNaN化
_KICK_SPEED_IDX_FALLBACK_MEDIAN = 74.0       # 異常値/NaN(16.1%)時のフォールバック(全体中央値)
_KICK_PEAK_SPEED_TURF_COEF = (16.0658, 0.011323)  # (切片, 傾き) m/s、y=a+b*speed_idx
_KICK_PEAK_SPEED_DIRT_COEF = (14.8432, 0.011142)
_KICK_PEAK_TO_AVG_RATIO = 1.0298  # 実測上がり3F(600m平均)→瞬間ピーク相当への上方スケール(lap_times実測、中央値)

IS_STRAIGHT_COURSE = False  # simulate_one_race.py等がレースごとに上書きする(horse_pair_sim.pyの同名フラグと同期)

_KICK_START_D = None  # レースごとに_init_kick_geometry()で設定するキャッシュ
_K_CRUISE_BY_STYLE = None  # {"逃":..,"先":..,"差":..,"追":..,None:..} 巡航区間([0,_KICK_START_D))
# 所要時間の「v_cruise=1」相当値(closed-form用)。脚質によって巡航速度形状(shape_seg1_for_style)が
# 変わるため、per-horseではなくper-style(実質5パターン)のdictにしてある。同じ脚質の馬は
# 同じ値を共有するため計算コストは1レースあたり最大5回で無視できる(_K_CRUISE単一スカラー
# だった旧設計をレース単位キャッシュのまま拡張、per-horse再計算は不要という設計判断)。


def kick_start_distance():
    """3コーナー相当(最終コーナー区間の入口)のd_rail。ゴールまでの残り距離が
    HOME_STRETCH_M+CORNER_LEN_M になる地点(sim_geometryの簡易オーバル=直線2本+コーナー2つの
    うち、最終コーナー区間が実際の3-4コーナーを合成的に表す)。直線コース(IS_STRAIGHT_COURSE)は
    「3コーナー」という概念が存在しないため対象外とし、従来通りD_TOTAL-600.0を使う。"""
    if IS_STRAIGHT_COURSE:
        raw = D_TOTAL - 600.0
    else:
        raw = D_TOTAL - (sg.HOME_STRETCH_M + sg.CORNER_LEN_M)
    return float(min(max(raw, KICK_START_MIN_M), D_TOTAL - MIN_KICK_LEN_M))


def seg1_time_for_v_upto(v_cruise, d_upper, n=4000, style=None):
    """seg1_time_for_v()の一般化: [0, d_upper](d_upper<=SEG1_LEN想定)の所要時間。
    ゲートランプ区間の解析解+台形則という構成はseg1_time_for_v()と同じ。styleは
    shape_seg1_for_style()経由で使う(style=Noneはshape_seg1(dd)とビット単位で一致)。"""
    t_ramp = _gate_ramp_time(v_cruise)
    dd = np.linspace(GATE_ACCEL_DIST_M, d_upper, n)
    t_rest = np.trapezoid(1.0 / (v_cruise * shape_seg1_for_style(dd, style)), dd)
    return t_ramp + t_rest


def _init_kick_geometry():
    """レースごとに1回だけ呼ぶ(D_TOTAL・IS_STRAIGHT_COURSE・sim_geometryのジオメトリを
    設定した直後、simulate_one_race.py等が呼ぶ)。build_curve()はこのキャッシュを毎回参照する。"""
    global _KICK_START_D, _K_CRUISE_BY_STYLE
    _KICK_START_D = kick_start_distance()
    _K_CRUISE_BY_STYLE = {
        style: seg1_time_for_v_upto(1.0, _KICK_START_D, style=style)
        for style in (*RUNNING_STYLE_POSITION_SCORE.keys(), None)
    }


def _kick_g(u, r):
    """キック区間の速度形状(v_start=1相当)。u=0でv_start、u=1でv_peak=rに滑らかに遷移。"""
    return 1.0 + (r - 1.0) * smoothstep(u)


def _kick_time_integral(r, n=2000):
    """J(r)=∫[0,1] du/g(u,r)。v_start=1・区間長=1相当でのキック区間所要時間。"""
    uu = np.linspace(0.0, 1.0, n)
    return float(np.trapezoid(1.0 / _kick_g(uu, r), uu))


def _kick_effort_integral(r, n=2000):
    """I(r)=∫[0,1] g(u,r)^exponent du。rについて単調増加(rが大きい=速いほど消費が増える)。"""
    uu = np.linspace(0.0, 1.0, n)
    return float(np.trapezoid(_kick_g(uu, r) ** KICK_EFFORT_EXPONENT, uu))


def solve_kick_r(kick_start_d, base_rate_abs, target_final_stamina=None):
    """スタミナ収支だけからr=v_peak/v_startを解く(v_cruiseに依存しない二分探索、
    solve_taper()と同型)。stamina_at_kick_start-target_final_staminaが負になる
    (巡航だけで既に目標を割り込む)場合は下限1.0ptで防御し、r=R_MIN側に張り付かせる。
    target_final_stamina省略時はTARGET_FINAL_STAMINA(脚質補正なし)を使う(後方互換)。"""
    target_final_stamina = TARGET_FINAL_STAMINA if target_final_stamina is None else target_final_stamina
    kick_len = D_TOTAL - kick_start_d
    stamina_at_kick_start = 100.0 - base_rate_abs * kick_start_d
    budget = max(1.0, stamina_at_kick_start - target_final_stamina)
    target_i = budget / (base_rate_abs * kick_len)
    lo, hi = R_MIN, R_MAX
    if _kick_effort_integral(lo) >= target_i:
        return lo
    if _kick_effort_integral(hi) <= target_i:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if _kick_effort_integral(mid) < target_i:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _solve_v_cruise_capped(kick_start_d, s0, v_peak_abs, t_total, k_cruise, n=2000, n_bisect=60):
    """v_peakを絶対値v_peak_absに固定した場合のv_cruiseを二分探索で解く(v_startがv_cruiseに
    依存するため、この場合だけキック区間の所要時間がv_cruiseに対して非双曲的になる)。
    k_cruiseは呼び出し元(build_curve)が該当styleの_K_CRUISE_BY_STYLE[key]を渡す
    (このレースの実測88.2%がcapped分岐を通るため、_K_CRUISEをdict化した際にここが
    モジュールグローバルを直接参照したままだと型エラーになる、シニアエンジニアレビュー指摘)。"""
    kick_len = D_TOTAL - kick_start_d
    uu = np.linspace(0.0, 1.0, n)

    def total_time(vc):
        v_start = vc * s0
        v_u = v_start + (v_peak_abs - v_start) * smoothstep(uu)
        t_kick = np.trapezoid(kick_len / v_u, uu)
        return k_cruise / vc + t_kick

    lo, hi = 0.5, 30.0  # m/s、物理的に十分広いレンジ
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        if total_time(mid) > t_total:
            lo = mid  # まだ遅い(所要時間が長すぎる)→ v_cruiseを上げる
        else:
            hi = mid
    return (lo + hi) / 2


def _max_speed_from_index(speed_idx_recent5, is_dirt):
    lo, hi = _KICK_SPEED_IDX_VALID_RANGE
    si = speed_idx_recent5
    if pd.isna(si) or si < lo or si > hi:
        si = _KICK_SPEED_IDX_FALLBACK_MEDIAN
    a, b = _KICK_PEAK_SPEED_DIRT_COEF if is_dirt else _KICK_PEAK_SPEED_TURF_COEF
    return (a + b * si) * _KICK_PEAK_TO_AVG_RATIO


def build_curve(t_total, t_l3f, stamina_index, max_speed_cap, running_style=None,
                 target_final_stamina_offset=0.0):
    """t_l3f・stamina_indexはキック区間の形状にはもう使わない(2026-08-09の全面置き換えで、
    上がり3Fタイムへのcurve-fitと`final_stamina=40+0.4*stamina_index`を撤廃したため)。
    呼び出し元(HorseBaseline.t_l3f_input、検証CSV)との互換のため引数としては残す。
    max_speed_cap: _max_speed_from_index()で求めた、このレース・この馬でのキック最高速度(m/s)。
    running_style: "逃"/"先"/"差"/"追"またはNone(脚質不明、補正なし)。RUNNING_STYLE_K1=K2=0の
    ときはrunning_styleの値によらず常に旧来の(脚質補正なしの)挙動とビット単位で一致する。
    target_final_stamina_offset: 2026-08-14追加(ステップB・モンテカルロ・アンサンブル用)。
    style_target_final_stamina()の結果に加算する馬個別のオフセット(pt)。既定0.0は従来の
    挙動とビット単位で一致する。monte_carlo_ensemble.pyが、その馬の「上がり3F_安定度_
    標準偏差(秒)」由来のスプレッドから試行ごとに{-spread,0,+spread}を渡す想定
    (この秒→pt換算係数は較正済みではなく暫定のヒューリスティック、詳細は同スクリプトの
    コメント参照)。"""
    kick_start_d = _KICK_START_D
    target_final_stamina = style_target_final_stamina(running_style) + target_final_stamina_offset
    r = solve_kick_r(kick_start_d, BASE_RATE_ABS, target_final_stamina)
    s0 = float(shape_seg1_for_style(np.array([kick_start_d]), running_style)[0])  # 巡航形状のkick_start_d地点での値(v_cruise=1相当)
    k_cruise = _K_CRUISE_BY_STYLE[_style_key(running_style)]
    k_kick = (D_TOTAL - kick_start_d) * _kick_time_integral(r) / s0
    v_cruise = (k_cruise + k_kick) / t_total

    v_start = v_cruise * s0
    v_peak = r * v_start
    capped = bool(v_peak > max_speed_cap)
    if capped:
        v_cruise = _solve_v_cruise_capped(kick_start_d, s0, max_speed_cap, t_total, k_cruise)
        v_start = v_cruise * s0
        v_peak = max_speed_cap

    dist_grid = np.linspace(0, D_TOTAL, N)
    seg1_mask = dist_grid < kick_start_d
    speed_grid = np.empty(N)
    speed_grid[seg1_mask] = v_cruise * shape_seg1_for_style(dist_grid[seg1_mask], running_style)
    u = np.clip((dist_grid[~seg1_mask] - kick_start_d) / (D_TOTAL - kick_start_d), 0.0, 1.0)
    speed_grid[~seg1_mask] = v_start + (v_peak - v_start) * smoothstep(u)

    # 時間グリッド(ゲート加速ランプは解析解、それ以降は台形則。既存の理由と同じ:
    # d=0でv=0の特異点近傍は粗いグリッドの台形則だと系統的に誤差が出るため)。
    time_grid = np.zeros(N)
    v_seam = v_cruise * DASH_SEAM_RATIO
    in_ramp = dist_grid <= GATE_ACCEL_DIST_M + 1e-9
    time_grid[in_ramp] = 2.0 * np.sqrt(GATE_ACCEL_DIST_M * dist_grid[in_ramp]) / v_seam
    start_i = int(np.sum(in_ramp))
    for i in range(start_i, N):
        dd = dist_grid[i] - dist_grid[i - 1]
        v_avg = 0.5 * (speed_grid[i] + speed_grid[i - 1])
        time_grid[i] = time_grid[i - 1] + dd / v_avg

    # スタミナ(絶対単位、事後リスケール無し)。旧モデルの`100-(100-final_stamina)*(cum/total)`は
    # cum(D_TOTAL)=totalとなる正規化のためeffort形状によらず必ずfinal_staminaに一致してしまい、
    # 「スタミナ30%になるよう速度を決める」という因果関係を作れない(Plan agentが数式的に発見、
    # 自分でも算術検算して確認済み)。そのため事後リスケールを撤廃し、絶対単位のeffortをそのまま
    # 積分してstamina_gridを作る(ゴールでの値はr(=solve_kick_r)の解を通じて間接的にしか
    # TARGET_FINAL_STAMINAに一致しない。max_speed_capで頭打ちの場合はTARGET_FINAL_STAMINAより
    # 高い値で着地する)。
    effort = np.empty(N)
    effort[seg1_mask] = BASE_RATE_ABS
    ratio = speed_grid[~seg1_mask] / max(v_start, 1e-9)
    effort[~seg1_mask] = BASE_RATE_ABS * ratio ** KICK_EFFORT_EXPONENT
    cum = np.zeros(N)
    for i in range(1, N):
        dd = dist_grid[i] - dist_grid[i - 1]
        cum[i] = cum[i - 1] + 0.5 * (effort[i] + effort[i - 1]) * dd
    # solve_kick_r()はR_MIN=0.3(最大減速)まで下げても収支が合わない(巡航区間だけで
    # 既にスタミナの大半を使っている)ケースを弾けないため、0%を下限にクランプする
    # (BASE_RATE_ABSが大きすぎる候補で発生することをrecalibrate_kick_base_rate.pyの
    # 事前診断で確認済み)。
    stamina_grid = np.maximum(0.0, 100.0 - cum)

    if capped:
        # capped時はv_cruiseを再計算したため、solve_kick_r()が返した元のrはもはや
        # 実際のv_peak/v_startと一致しない(診断値としてズレるため再計算する)。
        r = v_peak / max(v_start, 1e-9)

    return {
        "v_cruise": v_cruise, "v_peak": v_peak, "r": r, "kick_start_d": kick_start_d, "capped": capped,
        "t_total_check": time_grid[-1],
        "dist_grid": dist_grid, "speed_grid": speed_grid, "effort_grid": effort,
        "time_grid": time_grid, "stamina_grid": stamina_grid,
        "final_stamina": float(stamina_grid[-1]),
    }


class HorseBaseline:
    """build_curve()の離散グリッドを線形補間し、距離dの連続関数として使えるようにする。"""

    def __init__(self, umaban, name, t_total, t_l3f, stamina_index, waku, max_speed_cap, is_estimated=False,
                 running_style=None, target_final_stamina_offset=0.0):
        self.umaban = umaban
        self.name = name
        self.waku = waku
        self.is_estimated = is_estimated
        self.running_style = running_style  # "逃"/"先"/"差"/"追"またはNone(脚質不明)
        self.curve = build_curve(t_total, t_l3f, stamina_index, max_speed_cap, running_style,
                                  target_final_stamina_offset)
        self.total_time_solo = float(self.curve["time_grid"][-1])
        self.t_l3f_input = float(t_l3f)  # 単走推定の上がり3F入力値(実測 or 推定)。2026-08-09以降は
        # キック区間のフィット対象ではなく検証用の参考値としてのみ使う。

    def speed(self, d_rail):
        d_rail = min(max(d_rail, 0.0), D_TOTAL)
        return float(np.interp(d_rail, self.curve["dist_grid"], self.curve["speed_grid"]))

    def effort(self, d_rail):
        """距離1mあたりの実際のスタミナ消費率(絶対%/m)。2026-08-09以降、effort_gridは
        既に絶対単位(事後リスケール無し)なのでそのまま返す。ドラフティング係数はこの
        戻り値にさらに掛ける。"""
        d_rail = min(max(d_rail, 0.0), D_TOTAL)
        return float(np.interp(d_rail, self.curve["dist_grid"], self.curve["effort_grid"]))

    def stamina_baseline(self, d_rail):
        """相互作用が無い場合、この距離でこの残量のはず、というベースライン。"""
        d_rail = min(max(d_rail, 0.0), D_TOTAL)
        return float(np.interp(d_rail, self.curve["dist_grid"], self.curve["stamina_grid"]))


_regression_cache = None

# 3パラメータ(切片+speed_idx+agari_avg)の回帰に対し、実測データがある馬(have)が
# これ未満しかいないレースでは回帰を使わない(中央値のみにフォールバックする)。
# 8/2全35レースへの拡張時、新潟12R(202604020412、have=3=パラメータ数と同数で
# 事実上の厳密内挿=残差ゼロの縮退フィット)で、have範囲外(speed_idx=94.1等)への
# 外挿により回帰係数が暴走しt_seg1が負値(v_cruiseが負=物理的に無意味な速度)になり、
# d_railが単調増加しない致命的なバグとして顕在化した。苗場特別はhave=9で回帰係数が
# 十分安定しており(この閾値未満にならない)、この変更による挙動差は無い。
_MIN_REGRESSION_N = 6

# have=0(このレースの出走馬に実測ペースデータが1頭も無い)場合の最終フォールバック定数。
# 8/2全35レースへの拡張時、2歳新馬・(2/3歳)未勝利戦の多くでhave=0になることが判明した
# (持続タイムはnetkeiba側で一定以上のレース実績がある馬にしか算出されないため)。
# median自体が存在しない(空集合のmedian=NaN)ため、中央値シュリンクも使えない。
# 代わりに「代表的な巡航速度(苗場特別の実測に近い値)を基準に、このレース内の相対的な
# 総合スピード指数(0-100)で加減する」という粗い近似で、少なくとも物理的に妥当な
# (正の・順位に意味のある)速度曲線を作る。
#
# 2026-08-08: sim_vs_actual検証レポートで発覚した系統誤差を受け、新馬戦330頭分の実測
# (recalibrate_baseline_fallback.py、data/race_results+data/newspaperから再実行可能)で
# 芝/ダート別に再較正(第1版、芝/ダートの2区分・中央値)。同日中のサブエージェントレビューで
# 「ダート新馬の残差」が実は距離依存(distance_mとimplied_cruise_vの相関: ダートr=-0.90、
# 芝r=-0.68)であると判明し、距離項を加えた線形モデル(芝/ダート共通の傾き+is_dirtで切片
# シフト)をLOO(leave-one-race-out)で検証(全体MAE-28.0%、ダート-41.0%、芝-22.3%、
# いずれも現行=2区分中央値比)。
#
# ただし↑はLOOも8/2実測検証(35レース)も両方とも較正標本と同じ「新馬」クラスのみで
# 効果を確認したもの。8/2実測検証で全頭フォールバックだった10レースを芝/ダートで個別に
# 見ると、ダートは3レース全て改善(未勝利1件含む)した一方、芝は新馬3レース全て改善した
# 半面、未勝利4レースのうち距離1800m以上の3レースでbiasがむしろ悪化(相互作用物理を
# 含まないctrl_solo列でも同じ傾向を確認済みのため、シミュレーション側でなくこの
# フォールバック定数自体の問題)。芝は新馬でのみ検証された距離項を未勝利へ外挿すると
# 悪化するため不採用とし、芝は第1版の中央値(16.17/35.60)に据え置く。ダートは新馬・
# 未勝利いずれの実測でも改善したため第2版(距離項)を採用する。詳細は
# project_jra_baseline_fallback_recalibration.mdの追記3を参照。
_FALLBACK_CRUISE_V_TURF = 16.17  # m/s、芝は中央値のまま据え置き
_FALLBACK_L3F_SEC_TURF = 35.60   # 秒、同上
_FALLBACK_CRUISE_V_DIRT_COEF = (18.628 - 0.307, -0.1483)  # (切片, dist_m/100の係数)
_FALLBACK_L3F_SEC_DIRT_COEF = (33.550 + 5.083, 0.1390)     # 同上

# 2026-08-08: have=1〜5(そのレース内に実測`持続タイム`データを持つ馬=have馬が1〜5頭だけいる、
# _MIN_REGRESSION_N未満)のフォールバックを較正。従来はこのケースで対象馬全員に一律
# median_t_seg1/median_t_l3fを返すのみで、対象馬自身のspeed_idx/agari_avgを一切参照しない
# (同一レース内の推定対象馬が完全に同一の速度曲線になる)欠陥があった。8日間279レース中
# 121レース(43%)がhave=1〜5に該当し、着順footrule(正規化)平均は0.6432と、have=0
# (全頭fallback、下のspeed_factorで個別差別化する分岐)の0.5692より悪いという逆説的な
# 結果になっていた(実測検証で発覚、race_id=202601010409の7頭で予測上がり3Fが約35.4秒に
# 完全一致していた例などから発見)。
#
# 対応: レース内の「切片」(そのレースの絶対的な速さの水準)はそのレース自身のhave馬の
# 中央値のまま(少数標本でも中央値なら頑健)とし、「傾き」(speed_idxの1点差が何秒に
# 相当するか)だけ全レースをプールして一度だけ較正した固定値を使う部分プーリング方式。
# speed_idxは「レース内相対0-100指数」なのでレースをまたいで生の値を比較する意味は無いが、
# 「speed_idxが1点高い馬は、そのレースの中央値に対してどれだけ速いか」という関係性
# (傾き)自体は多数のレースに共通する安定した量、という考え方。
#
# 較正方法(`recalibrate_need_estimate_slope.py`、再実行可能): 8日間279レースのうち
# have>=2の206レース・1224頭を対象に、have馬をレース内偏差(値−そのレースのhave馬平均)に
# 変換してプールし、speed_idx偏差に対するt_seg1・t_l3f偏差の原点通過回帰(切片は構造的に
# 厳密0)で傾きを推定。LORO(leave-one-race-out、206回)で安定性を確認済み(傾きの
# 変動係数: seg1=2.2%・l3f=0.8%、非常に安定)。ただしこれは「傾きの推定値自体が母数
# 1224頭でブレない」ことを示すのみで、「speed_idxがt_seg1をよく説明する」ことは意味しない
# (実際の相関係数はseg1=-0.123・l3f=-0.321と、l3f側はまずまず・seg1側は弱い。サブエージェント
# レビューで「変動係数の低さ=高精度」と誤読しないよう指摘された点)。それでも符号は正しく、
# 何も調整しない現状(=推定対象馬が完全に同一の速度曲線になる)よりは確実に改善する。
# 芝/ダート別の傾きにはある程度の差がある
# (dirt seg1が約3.7倍)が、have=0フォールバックでの教訓(新馬限定較正の未勝利への
# 外挿失敗)を踏まえ、標本を分割する強い根拠が無い限り最初から分割しない方針とし、
# 単一プール値を採用する。
#
# 3パラメータ回帰(coef_seg1)をそのままhave=1〜5へ適用する案は、実測検証(疑似have=3で
# blowup率0.47%)で過去のバグ(_MIN_REGRESSION_N導入の原因、have=3での回帰暴走・
# t_seg1負値化)を再現することを確認したため不採用。
_NEED_ESTIMATE_SLOPE_SEG1 = -0.006654  # 秒/speed_idx点(レース内偏差の原点通過回帰)
_NEED_ESTIMATE_SLOPE_L3F = -0.015789   # 同上


def _fallback_cruise_v(dist_m, is_dirt):
    if is_dirt:
        a, b = _FALLBACK_CRUISE_V_DIRT_COEF
        return a + b * (dist_m / 100.0)
    return _FALLBACK_CRUISE_V_TURF


def _fallback_l3f_sec(dist_m, is_dirt):
    if is_dirt:
        a, b = _FALLBACK_L3F_SEC_DIRT_COEF
        return a + b * (dist_m / 100.0)
    return _FALLBACK_L3F_SEC_TURF


# simulate_one_race.py / solo_baseline_for_race.py が CSV_PATH 等と同様にレース開始時に
# 上書きするモジュール属性("芝" or "ダート")。距離依存モデルのis_dirt判定に使う。
# SURFACE未設定(旧来呼び出し等)時はis_dirt=0(芝側)扱いにする(標本の大半(290/330)を
# 占め、旧flat定数にも近い安全側のため)。
SURFACE = None


def _get_regression():
    """all_horses_curve.py と同一の回帰+中央値シュリンク方式(実測9頭からT_SEG1・
    T_L3Fを推定)。推定値フォールバックが必要な馬(5番ナイトスラッガー等)向けに、
    all_horses_curve.pyの118〜150行目の計算式をそのまま複製している(値は不変)。"""
    global _regression_cache
    if _regression_cache is not None:
        return _regression_cache
    df = pd.read_csv(CSV_PATH)
    have_mask = df["持続タイム_今回距離帯(秒)"].notna() & df["持続タイム_今回距離帯_上がり3F(秒)"].notna()
    have = df[have_mask].copy()
    have["t_seg1"] = have["持続タイム_今回距離帯(秒)"] - have["持続タイム_今回距離帯_上がり3F(秒)"]
    have["t_l3f"] = have["持続タイム_今回距離帯_上がり3F(秒)"]
    if len(have) == 0:
        _regression_cache = (None, None, None, None, None)
        return _regression_cache
    median_t_seg1 = float(have["t_seg1"].median())
    median_t_l3f = float(have["t_l3f"].median())
    median_speed_idx = float(have["総合スピード指数(0-100,場内相対)"].median())
    if len(have) < _MIN_REGRESSION_N:
        _regression_cache = (None, None, median_t_seg1, median_t_l3f, median_speed_idx)
        return _regression_cache
    X = np.column_stack([
        np.ones(len(have)),
        have["総合スピード指数(0-100,場内相対)"].to_numpy(),
        have["上がり3F_平均_過去5走(秒)"].to_numpy(),
    ])
    coef_seg1, _, _, _ = np.linalg.lstsq(X, have["t_seg1"].to_numpy(), rcond=None)
    coef_l3f, _, _, _ = np.linalg.lstsq(X, have["t_l3f"].to_numpy(), rcond=None)
    _regression_cache = (coef_seg1, coef_l3f, median_t_seg1, median_t_l3f, median_speed_idx)
    return _regression_cache


def _estimate_times(speed_idx, agari_avg):
    coef_seg1, coef_l3f, median_t_seg1, median_t_l3f, median_speed_idx = _get_regression()
    if median_t_seg1 is None:  # have=0(中央値すら存在しない): 代表巡航速度+相対指数で近似
        si = 50.0 if pd.isna(speed_idx) else speed_idx
        speed_factor = 0.85 + 0.30 * (si / 100.0)  # 指数50→1.0倍、100→1.15倍、0→0.85倍
        is_dirt = 1.0 if SURFACE == "ダート" else 0.0
        fallback_cruise_v = _fallback_cruise_v(D_TOTAL, is_dirt)
        fallback_l3f_sec = _fallback_l3f_sec(D_TOTAL, is_dirt)
        t_seg1 = SEG1_LEN / (fallback_cruise_v * speed_factor)
        t_l3f = fallback_l3f_sec if pd.isna(agari_avg) else float(agari_avg)
        return t_seg1, t_l3f
    if coef_seg1 is None:
        # have=1〜5(_MIN_REGRESSION_N未満、回帰は使わないが中央値は存在する): 対象馬自身の
        # speed_idxで中央値を加減する(_NEED_ESTIMATE_SLOPE_*較正の経緯は定数定義を参照)。
        # 個別speed_idxが欠測なら「無調整」(=中央値そのまま)がデフォルトになるようmedian_
        # speed_idx自体にフォールバックする(havemask側でhave馬のspeed_idx欠測は無いことを
        # 確認済みだが、need馬側は約16%欠測するためこのガードは実際に発火する)。
        si = median_speed_idx if pd.isna(speed_idx) else speed_idx
        t_seg1 = median_t_seg1 + _NEED_ESTIMATE_SLOPE_SEG1 * (si - median_speed_idx)
        t_l3f = median_t_l3f + _NEED_ESTIMATE_SLOPE_L3F * (si - median_speed_idx)
        # 定数を第1引数にする(max(t, 1.0)ではなくmax(1.0, t)): NaNとの比較は常にFalseに
        # なるため、tがNaNの場合max(t,1.0)はNaNのまま素通りしてしまう(サブエージェント
        # レビューで発見)。現状median_speed_idxがNaNになることは無い(have馬のspeed_idx
        # 欠測は実測0%)が、将来欠測が発生した場合に無言でNaNを伝播させない安全側の書き方。
        t_seg1 = max(1.0, t_seg1)
        t_l3f = max(1.0, t_l3f)
        return t_seg1, t_l3f
    shrink = 0.5
    pred_seg1 = coef_seg1[0] + coef_seg1[1] * speed_idx + coef_seg1[2] * agari_avg
    pred_l3f = coef_l3f[0] + coef_l3f[1] * speed_idx + coef_l3f[2] * agari_avg
    t_seg1 = median_t_seg1 + shrink * (pred_seg1 - median_t_seg1)
    t_l3f = median_t_l3f + shrink * (pred_l3f - median_t_l3f)
    # 物理的に正であることを保証する最終防御(通常は発火しない。have不足レースでの
    # 回帰暴走はhave_mask側の閾値ガードで既に防いでいるが、万一に備えた下限)。
    # 定数を第1引数にする理由は上のhave=1〜5分岐と同じ(NaN対策、サブエージェントレビューで発見)。
    t_seg1 = max(1.0, t_seg1)
    t_l3f = max(1.0, t_l3f)
    return t_seg1, t_l3f


# 2026-08-14追加(ステップB・モンテカルロ・アンサンブル用): {umaban: offset_pt}。
# run_race()/run_race_mc()が試行ごとに設定・リセットするモジュールグローバル
# (CSV_PATH等と同じ「レースごとに明示上書き」パターン)。空dict(既定)は
# 全馬offset=0.0=従来の挙動とビット単位で一致する。
STAMINA_OFFSET_OVERRIDE = {}


def load_horse(umaban):
    """naeba_potential.csvからHorseBaselineを構築する。実測の持続タイムがある馬は
    それを直接使い、無い馬(推定対象)はall_horses_curve.pyと同じ回帰+シュリンク方式で
    推定する。"""
    df = pd.read_csv(CSV_PATH)
    row = df[df["馬番"] == umaban].iloc[0]
    stamina_raw = row["総合スタミナ指数(0-100,場内相対)"]
    # 距離帯別実績が全馬とも欠測のレース(未勝利戦等)ではminmax_norm列が全てNaNになり得るため、
    # 中立値50にフォールバックする(NaNのままだとJSON上のスタミナ%表示がNaNになるだけで
    # 物理シミュレーション自体は壊れないが、表示のため防御しておく)。
    stamina_index = 50.0 if pd.isna(stamina_raw) else float(stamina_raw)
    if pd.notna(row["持続タイム_今回距離帯(秒)"]) and pd.notna(row["持続タイム_今回距離帯_上がり3F(秒)"]):
        t_total = float(row["持続タイム_今回距離帯(秒)"])
        t_l3f = float(row["持続タイム_今回距離帯_上がり3F(秒)"])
        is_estimated = False
    else:
        speed_idx = row["総合スピード指数(0-100,場内相対)"]
        agari_avg = row["上がり3F_平均_過去5走(秒)"]
        t_seg1, t_l3f = _estimate_times(speed_idx, agari_avg)
        t_total = t_seg1 + t_l3f
        is_estimated = True
    is_dirt = (SURFACE == "ダート")
    max_speed_cap = _max_speed_from_index(row["スピード指数_直近5走平均"], is_dirt)
    style_raw = row.get("脚質")  # 列が無いCSV(旧スナップショット等)ではNone、NaNならNoneに正規化
    running_style = None if pd.isna(style_raw) else str(style_raw)
    stamina_offset = STAMINA_OFFSET_OVERRIDE.get(int(umaban), 0.0)
    return HorseBaseline(umaban, row["馬名"], t_total, t_l3f, stamina_index, int(row["枠番"]),
                          max_speed_cap, is_estimated, running_style, stamina_offset)


if __name__ == "__main__":
    h1 = load_horse(1)
    h2 = load_horse(2)
    for h in (h1, h2):
        print(h.umaban, h.name, "solo_total_time=%.2fs" % h.total_time_solo,
              "final_stamina=%.1f%%" % h.curve["stamina_grid"][-1])
