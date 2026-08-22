#!/bin/bash
# Step2(2026-08-22): 過去データのバックフィル(race_results/payoutsのみ)。
#
# Opus 5サブエージェント調査(2026-08-21)を受け、market_model(Step1)のCI幅を狭めるための
# データ量拡大策。馬柱(newspaper)はnetkeiba側の集計列(ca_*/data_*等)が取得時点で計算された
# 値の可能性が高く、過去に遡って取得すると未来情報が混入する(look-ahead bias)リスクがある
# ため、本バックフィルの対象には含めない。race_results/payoutsは確定情報(結果・払戻)なので
# リークしない。
#
# 対象: 2024-08-22〜2026-08-21の土日(JRAは主に土日開催のため候補を土日に絞る。平日祝日開催は
# 拾えないが、run_pilot.pyはmanifestベースで既取得race_idを自動スキップするため、後日
# 個別日付を追加で流しても安全)。
set -uo pipefail
cd "$(dirname "$0")/.."

DATES_FILE=/tmp/jra_backfill_dates_2026_08_22.txt

python -c "
import datetime
start = datetime.date(2024, 8, 22)
end = datetime.date(2026, 8, 21)
d = start
while d <= end:
    if d.weekday() in (5, 6):  # Sat=5, Sun=6
        print(d.strftime('%Y%m%d'))
    d += datetime.timedelta(days=1)
" | tr -d '\r' > "$DATES_FILE"

total=$(wc -l < "$DATES_FILE")
echo "対象候補日数(2024-08-22〜2026-08-21の土日): $total"

n=0
while IFS= read -r d; do
  d="${d%$'\r'}"
  n=$((n + 1))
  echo "=== [$n/$total] $d ==="
  python scripts/run_pilot.py --date "$d" --circuit jra
done < "$DATES_FILE"

echo "=== バックフィル完了 ==="
