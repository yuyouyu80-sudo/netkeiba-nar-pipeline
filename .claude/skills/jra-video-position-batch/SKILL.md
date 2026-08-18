---
name: jra-video-position-batch
description: JRA実況動画から切り出したフレーム画像(「実際の位置」フォルダ)を使って、新しい開催日・レースの馬位置データを読み取り・検証・data/video_positions/へ確定するスキル。ユーザーが動画フレームフォルダを用意し「動画から位置を読み取ってほしい」「展開予想の較正データを追加したい」等と依頼した際に使う。
---

# jra-video-position-batch

JRA実況動画フレーム(ユーザーが1秒おきに切り出した画像)から馬の「実際の位置」を読み取り、
`data/video_positions/{race_id}.csv`として確定するまでの定型フロー。読み取り手法・検証設計の
詳細な根拠は`scripts/jra_race_sim/video_positions/METHODOLOGY.md`を参照(このSKILL.mdの各
ステップから該当節番号でリンクする)。

## 前提条件

- 動画フレーム画像フォルダが、以下の命名規則でユーザーのローカルに用意されていること:
  `シミュレーション結果と実際の結果動画画像\{月}月{日}日{開催場}{R}R{レース名}\実際の位置\
  frame_XXXX.png`(このリポジトリの外、git管理対象外)。
- `data/race_results/2026/{date}.csv`(公式着順・通過順位)がそのレース分取得済みであること
  (検証ゲートで使う)。
- `scripts/jra_race_sim/race_json_display/{race_id}.json`(シミュレーション結果、
  `sampling_plan.py`のキック開始時刻推定に使う)が生成済みであること。無ければ先に
  `scripts/jra_race_sim/publish/generate_race_json.py`(または
  `regenerate_race_json_display.py`)で生成する。
- **1レースあたり読み取りに数時間規模かかる**(背景実行のサブエージェント前提)。複数レースを
  一度に依頼された場合も、必ず1レース1エージェントに分割する(METHODOLOGY.md §5)。

## 手順

1. **race_id特定・manifest登録**
   対象日の`data/race_results/2026/{date}.csv`からrace_idを特定し、
   `data/video_positions/_manifest.csv`に新規行を追加する(status=pending)。
   **冪等性チェック**: 既に該当race_idが`_manifest.csv`にあり、statusが
   `validated`または`needs_review`ならスキップする(重複登録しない)。`pending`/`reading`で
   止まっている場合はそこから再開する。

2. **サンプリング計画の算出**
   `python scripts/jra_race_sim/video_positions/sampling_plan.py --races-json <対象レース一覧> --out <出力先>`
   を実行し、各race_idの`kick_ts`/`dash_ts`(読み取り対象タイムスタンプ)を得る。
   `_manifest.csv`のstatusを`reading`に更新する。

3. **動画フレーム読み取り(1レース1エージェント)**
   各race_idごとに、背景実行のサブエージェントを1体ずつ起動する。エージェントへの指示には
   必ず以下を含める:
   - オフセット実測の方法(METHODOLOGY.md §2)
   - 密集読み取り時のクロップ・ズーム手法(METHODOLOGY.md §3)
   - 異常検知時の追加投資・genuine/artifact判定(METHODOLOGY.md §4)
   - 出力スキーマ: `t_sec,checkpoint,umaban,rank_official,source,confidence`
     (`data/video_positions/{race_id}.csv`として保存)
   - 公式結果・通過順位は一切参照しないこと(読み取りの独立性を保つ)
   - 途中で打ち切る場合でも、読み終えた分は必ずCSV+notesとして書き出すこと

   **失敗時の回復**(METHODOLOGY.md §6): 「failed」通知を受けたら、まず出力ファイルの
   存在・完全性を確認する→quota起因ならリセット時刻を確認後に再起動→transcript resumeが
   使えれば`SendMessage`→使えなければ該当race_idのみ新規agentで再起動(部分ファイルが
   あれば再利用・検証してから続きを読む指示を含める)。

4. **検証ゲート**
   `python scripts/jra_race_sim/video_positions/validation_gate.py --manifest data/video_positions/_manifest.csv`
   を実行する(METHODOLOGY.md §7)。`flagged_windows`が出た区間・`best_footrule`が高い
   (目安0.25超)レースは、notesファイルを確認しつつ人間が目視でgenuine/artifactを判定する。
   artifactと判定した区間は`data/video_positions/exclusions.csv`に追記する
   (`race_id,t_start,t_end,detected_by,judged_as,reason`。離散点は`t_start=t_end`、
   開区間は`t_end`空欄)。判定が終わったレースは`_manifest.csv`のstatusを`validated`に、
   要再確認のまま残るレースは`needs_review`に更新する。

5. **二重読み取りチェック**(METHODOLOGY.md §8)
   母集団は**今回新規追加した分を優先**し、無作為2〜3レースを選ぶ。ただし今回の新規追加が
   2レース以下の場合は、`_manifest.csv`の累積プール(既存の`validated`分を含む)から補って
   2〜3レースを確保する(少数バッチ追加時に二重読み取りが実質スキップされるのを防ぐ)。
   元の読み取り手法(クロップ・ズーム)を再現できる専任サブエージェントに、既存CSV/notesを
   見せずに独立読み取りを依頼し、一致率を確認する。大きく乖離する場合は該当レースの
   読み取り手順自体を疑い、必要なら読み直す。

6. **確定・コミット**
   `data/video_positions/{race_id}.csv`・`{race_id}_notes.txt`・`_manifest.csv`・
   `exclusions.csv`の変更をgit commitする(コード変更とは分けてデータコミットとする)。

## 完了報告フォーマット

対象レース数・成功/要確認件数・検出した除外区間の件数を簡潔に報告する。例:
「対象5レース中5レース読み取り完了、検証ゲート通過4件・要確認1件(race_id=...、理由=...)、
除外区間を2件検出しexclusions.csvに追記」。
