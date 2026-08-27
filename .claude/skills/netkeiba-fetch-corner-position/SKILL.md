---
name: netkeiba-fetch-corner-position
description: 指定した開催日(YYYYMMDD)または単一race_idについて、netKeibaのnewspaper.htmlに含まれる「AI展開」3・4コーナー位置取り予想を取得し、既存のdata/newspaper/{race_id}.csvへcorner4_rank/corner4_gap_pct/corner4_gap_lengths/corner4_speedup/corner3_rank/corner3_gap_pct/corner3_gap_lengthsを追加・上書きする。ユーザーが「3コーナー位置」「4コーナー位置」「展開予想」「AI展開」の取得・更新を依頼した際に使う。事前にnetkeiba-fetch-newspaperで対象race_idのCSVが生成済みであることが前提。
---

# netkeiba-fetch-corner-position

このワークスペースで、`scripts/fetch_corner_position.py` を使って開催日1日分(または単一race_id)
の3・4コーナー位置取り予想(netkeibaの「AI展開」ウィジェット)データを取得し、既存の馬柱(newspaper)
CSVへ追加するスキル。

対象データはnewspaper.html自体に(JS実行不要で)埋め込まれているため、mark_list.htmlと違い
Playwrightは不要。既存の`fetch_newspaper_html()`(requestsベース)を再利用する軽量版。

## 前提条件

- 対象race_idについて `scripts/fetch_newspaper.py` で `data/newspaper/{race_id}.csv` が
  **既に生成済み**であること。未生成のrace_idはスキップされる(新規行は作らず、既存行に列を
  追加・上書きするだけの軽量版、`refresh_bias.py`/`fetch_marks.py`と同じ設計)。
- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。
- `pip install -r requirements.txt` 済みであること(Playwrightは不要)。

## 手順

1. **取得の実行**
   - 開催日1日分(JRA、既定):
     ```
     python scripts/fetch_corner_position.py --date {YYYYMMDD}
     ```
   - 単一race_id:
     ```
     python scripts/fetch_corner_position.py --race-id {race_id}
     ```
   - NAR:
     ```
     python scripts/fetch_corner_position.py --date {YYYYMMDD} --circuit nar
     ```
   出力: 既存 `data/newspaper/{race_id}.csv` への列追加・上書き。
   - `corner4_rank`: 4コーナー時点の推定順位(1=先頭、同着は同順位)
   - `corner4_gap_pct`: 先頭からの差(netkeiba側の座標スケール、0=先頭)
   - `corner4_gap_lengths`: 上記を馬身換算した近似値(柵の間隔≈1馬身という換算)
   - `corner4_speedup`: 4コーナーでの加速マーク(▶)の数(0〜3)
   - `corner3_rank` / `corner3_gap_pct` / `corner3_gap_lengths`: 3コーナー時点の同種データ
     (2026-08-27追加)。加速マーク(▶)は4コーナー時点のみ描画される仕様のため
     `corner3_speedup`列は無い。

2. **結果確認**
   - `--date` 実行時は最後に `refreshed corner3/corner4 position: {更新頭数}/{総頭数} horses
     across {成功レース数}/{全体レース数} races for {YYYYMMDD}` のサマリ行が出力される。
   - スキップ・失敗は `skipped (...)` / `Failed (N): ...` として別途出力される。

3. **スキップの扱い**
   - 「既存newspaper CSVが無い」でスキップされたrace_idは、先に`netkeiba-fetch-newspaper`
     スキルを実行してから再実行する。
   - 「AI展開データが無い」は仕様通りの正常系(全レースにこの機能があるわけではないことを
     実データで確認済み)であり、再実行しても解消しない。

## 注意点

- コーナーの内部識別子と実際の表示ラベルがズレている: `#CornerSwitch`のタブは
  `id="Corner01"`(表示は"スタート後")、`id="Corner02"`(表示は"3コーナー")、
  `id="Corner03"`(表示は"4コーナー")の3つのみ(2026-08-27確認)。1コーナー・2コーナー
  単独のデータはnetkeiba側に存在しないため取得不可。
- 先頭判定: 馬アイコンのスプライトが基準状態で左向きに走っており、反時計回り
  (AntiClockwise)コースはCSSの`transform: scale(-1,1)`で見た目だけ左右反転しているため、
  生データの`left:0%`は常に先頭を表す(データの意味自体はコースの回り方に無関係)。
- 縦方向(横方向・柵からの距離)のデータはこのウィジェットには実質存在しない(表示上の
  重なり回避のためだけの値であることを実データで確認済み)。取得しているのは先頭からの
  縦方向(進行方向)の距離のみ。
- netkeiba側が取消馬反映前の古い座標を`//`行コメントとして残したまま、直後に有効な最新版を
  置いていることがある(NAR実データで確認済み)。パーサー側で行コメントを除去してから
  解析するため通常は意識不要。
- 馬身換算の詳細・導出根拠は `src/netkeiba_pipeline/parsers/corner_position_parser.py` の
  モジュールdocstringおよび `CORNER4_LENGTH_PCT_PER_HORSE` を参照。
