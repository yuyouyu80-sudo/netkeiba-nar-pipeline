# 血統レーダー v2 のスタミナの軸の前向きの記録 P3(radar_v2_stamina_live.py --record、記述のみ)をタスクスケジューラへ登録する(2026-10-07新設)。
# 使い方(PowerShell): powershell -ExecutionPolicy Bypass -File scripts\jra_model\pedigree_reports\register_radar_v2_stamina_task.ps1
# 解除:               Unregister-ScheduledTask -TaskName "netkeiba_radar_v2_stamina_record" -Confirm:$false
#
# - 毎週 土・日・月 の 09:25 と 12:05 に起動(事前登録 RADAR_V2_PREREG_2026_10_06.md 追補7。追補6 の記録の5分後)。
#   race_names(watch_odds_auto.py、08:30)と馬柱が出た後。12:05 は朝に馬柱が無かったレースを足すため。
# - 開催が無い日・race_names や馬柱がまだ無い日は何もせず正常終了する。記録はレースごとに最初の1回が正本で、
#   発走時刻を過ぎたレースは「発走後」の印が付く(P3 は記述のみ、追補6 の P1・P2 の記録には触れない)。
# - 追補6 の記録と同じ .venv_dl の python で実行する(torch は使わない)。ログ: logs\radar_v2_stamina_live_{date}.log
# - ログオン中のユーザーとして実行(パスワードを保存しない)。ログオフ中・電源オフ中は動かない。
$ErrorActionPreference = "Stop"
$TaskName = "netkeiba_radar_v2_stamina_record"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $Root ".venv_dl\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw ".venv_dl が見つかりません: $Python" }

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-u scripts\jra_model\pedigree_reports\radar_v2_stamina_live.py --date-today --record" -WorkingDirectory $Root
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday,Monday -At 09:25
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday,Monday -At 12:05
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -Hidden `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($t1, $t2) `
    -Settings $settings -Principal $principal `
    -Description "JRA開催日に血統レーダー v2 のスタミナの軸の前向きの記録 P3(発走前、記述のみ)を作る" -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List TaskName, NextRunTime
