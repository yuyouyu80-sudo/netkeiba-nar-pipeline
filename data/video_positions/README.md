# data/video_positions/ — 動画由来の馬位置データ

JRA実況動画フレームから視覚読み取りした馬の位置データ。処理パイプライン・運用ノウハウは
`scripts/jra_race_sim/video_positions/METHODOLOGY.md`、新規バッチの取り込み手順は
`.claude/skills/jra-video-position-batch/SKILL.md`参照。

## ファイル

- `{race_id}.csv`: 1レース分の読み取り結果。列:
  `t_sec,checkpoint,umaban,rank_official,source,confidence`
  (`rank_official`という列名だが値は動画読み取りによる推定順位。`source`は`video_frame_read`
  固定、`confidence`は`medium(vision-read)`固定)。
- `{race_id}_notes.txt`: 読み取り時のオフセット確認・異常検知・信頼度に関する自由記述メモ。
- `_manifest.csv`: レースごとの登録状況。列: `race_id,date,venue,race_name,source_folder,
  offset,n_kick_valid,n_dash_valid,status,added_at`。

  **設計上の注記**: `data/_manifest/scraped_race_ids.csv`(他パイプラインの冪等性管理)が
  「1試行=1行の追記専用イベントログ」型なのに対し、こちらは「レース1行を状態遷移で
  上書き」型(意図的な設計選択)。status遷移の過程はgit diffでのみ追跡可能。

  | status | 意味 | 次アクション |
  |---|---|---|
  | pending | 登録済み・読み取り未着手 | `sampling_plan.py`実行→読み取りエージェント起動 |
  | reading | 読み取りエージェント起動済み・完了待ち | 完了待ち、失敗時は該当race_idのみ個別再起動 |
  | validated | `validation_gate.py`通過(除外区間以外は健全) | 較正・分析に利用可 |
  | needs_review | 要確認フラグ(または二重読取で乖離) | 人間がnotesを確認し、再読み取りor除外区間追加でvalidatedへ手動遷移 |

- `exclusions.csv`: 読み取り不能・UIアーティファクト等により除外すべき区間の記録。列:
  `race_id,t_start,t_end,detected_by,judged_as,reason`。
  - 離散点は`t_start=t_end`(1点1行)、連続区間は実値、開区間(以降すべて)は`t_end`空欄。
  - `detected_by`: `automatic`(validation_gate.pyの自動検出)/`manual`(二重読取・目視investigation)
  - `judged_as`: `genuine`(実際の動きと判定、除外不要)/`artifact`(UI起因、除外推奨)
