# scripts/jra_race_sim/ — JRA展開予想シミュレーション

着順予想(`scripts/jra_model/`・`scripts/nar_model/`)とは別種の、物理シミュレーションによる
「展開予想」(各馬の相互作用込みの走行位置・速度をコース幾何から再現するモデル)一式。
2026-08-18、セッション固有scratchpadから永続化した(詳細な移行経緯は
`C:\Users\yuyou\.claude\plans\valiant-cuddling-aho.md`参照)。

## 構成

- **このディレクトリ直下**: シミュレーションエンジン本体
  (`horse_baseline.py`/`horse_pair_sim.py`/`sim_geometry.py`/`race_potential.py`/
  `sim_runner_lib.py`/`corner_passing_metrics.py`/`course_specs.py`/`simulate_one_race.py`)。
  1レース分のシミュレーションを行う中核ライブラリ。
- **`video_positions/`**: JRA実況動画フレームから読み取った馬位置データ
  (`data/video_positions/`)の処理パイプライン(サンプリング計画・検証ゲート・運用ノウハウ)。
  新しい動画フレームバッチを取り込む定型フローは
  `.claude/skills/jra-video-position-batch/SKILL.md`参照。
- **`publish/`**: コースアニメーション公開ページ(競馬場ごとの全レース閲覧ページ)の生成・
  再生成スクリプト。「publish」とだけ言及すると他の処理と混同しうるので、コミット
  メッセージ等では`scripts/jra_race_sim/publish/`とフルパスで書くこと。
- **`race_json/`・`race_json_display/`・`_workdir/`**: エンジン+入力データから再生成可能な
  派生物(`.gitignore`対象、コミットしない)。`race_json/`は較正の判断根拠として使う
  確定済みデータで、新規開催日以外は上書きしない(`publish/generate_race_json.py`参照)。
  `race_json_display/`は公開ページ表示専用の差分再生成コピー
  (`publish/regenerate_race_json_display.py`参照)。
