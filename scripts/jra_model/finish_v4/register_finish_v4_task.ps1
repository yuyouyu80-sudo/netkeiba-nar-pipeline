# 着順予想 finish_v4 の前向きの記録 P4(finish_v4_live.py --date-today --record)をタスクスケジューラへ登録する(2026-10-11新設)。
# 使い方(PowerShell): powershell -ExecutionPolicy Bypass -File scripts\jra_model\finish_v4\register_finish_v4_task.ps1
# 解除:               Unregister-ScheduledTask -TaskName "netkeiba_finish_v4_record" -Confirm:$false
#
# - 毎週 土・日・月 の 09:30 と 12:10 に起動(事前登録 PREREG_FINISH_V4.md §8)。race_names(08:30)と馬柱が出た後。
#   12:10 は朝に馬柱が無かったレースを足すため。開催が無い日・出走表がまだ無い日は何もせず正常終了する。
# - 記録はレースごとに最初の1回が正本で、発走時刻を過ぎたレースは recorded_before_start=false になる(評価に使わない)。
# - .venv_dl の pythonw.exe で実行する(python.exe は Windows Terminal の画面ごと止まる事例があったため。メモリ参照)。
#   ログ: logs\finish_v4_live_{date}.log
# - ログオン中のユーザーとして実行(パスワードを保存しない)。ログオフ中・電源オフ中は動かない。
$ErrorActionPreference = "Stop"
$TaskName = "netkeiba_finish_v4_record"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $Root ".venv_dl\Scripts\pythonw.exe"
if (-not (Test-Path $Python)) { throw ".venv_dl の pythonw.exe が見つかりません: $Python" }

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-u scripts\jra_model\finish_v4\finish_v4_live.py --date-today --record" -WorkingDirectory $Root
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday,Monday -At 09:30
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday,Monday -At 12:10
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -Hidden `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($t1, $t2) `
    -Settings $settings -Principal $principal `
    -Description "JRA開催日に着順予想 finish_v4 の完成形の3連複20点の買い目を発走前に記録する(P4、前向きの記録)" -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List TaskName, NextRunTime
