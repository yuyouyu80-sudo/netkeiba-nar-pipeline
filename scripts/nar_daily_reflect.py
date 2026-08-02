# -*- coding: utf-8 -*-
"""NAR(地方競馬)の日次反映チェーンを1コマンドで通しで実行する。

fetch_newspaper(馬柱取得) → run_pilot(確定結果取得、失敗は非fatal) →
predict_pattern29 --circuit nar(見出し生成) → refresh_bias(オッズ再取得) →
predict_top5_nar(予想生成) → fetch_quick_result_nar(当日速報、失敗は非fatal) →
box_return_nar → confidence_sweep_baseline/box4/box3_nar → confidence_per_race_nar →
build_artifact_nar(レポート再構築) → git add/commit/push、の順に実行する。

Claude CodeのArtifact公開機能(claude.ai)は対話的セッションからしか呼び出せないため
このスクリプトの対象外。実行後、生成された data/nar_pipeline/prediction_report_nar.html
をClaude Codeセッション側でArtifactとして再公開すること。

使い方:
    python scripts/nar_daily_reflect.py --date 20260802
    python scripts/nar_daily_reflect.py --date 20260802 --verify-date 20260801
    python scripts/nar_daily_reflect.py --date 20260802 --skip-fetch --skip-results
"""
import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9))


def run(cmd: list[str], allow_fail: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NARの日次反映チェーン(馬柱取得→結果取得→予想生成→回収率/確信度再計算"
        "→レポート再構築→git push)を通しで実行する。"
    )
    parser.add_argument(
        "--date", default=datetime.now(JST).strftime("%Y%m%d"),
        help="馬柱取得・予想生成の対象日(YYYYMMDD、既定: 実行日のJST日付)",
    )
    parser.add_argument(
        "--verify-date", default=None,
        help="結果取得(run_pilot)の対象日(YYYYMMDD、既定: --dateと同じ)。"
        "前日分の結果を検証したい場合に指定する",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="fetch_newspaper.pyをスキップする")
    parser.add_argument("--skip-results", action="store_true", help="run_pilot.pyをスキップする")
    parser.add_argument("--no-push", action="store_true", help="git commit/pushを行わない")
    args = parser.parse_args()

    date = args.date
    verify_date = args.verify_date or date
    py = sys.executable

    if not args.skip_fetch:
        run([py, "scripts/fetch_newspaper.py", "--date", date, "--circuit", "nar"])

    if not args.skip_results:
        # 結果がまだnetkeiba側に掲載されていない(当日未終了・翌日反映待ち)のは正常系のため非fatal
        run([py, "scripts/run_pilot.py", "--date", verify_date, "--circuit", "nar"], allow_fail=True)

    run([py, "scripts/predict_pattern29.py", "--circuit", "nar", "--date", date])
    run([py, "scripts/refresh_bias.py", "--date", date, "--circuit", "nar"])
    run([py, "scripts/predict_top5_nar.py", "--date", date])
    run([py, "scripts/fetch_quick_result_nar.py", "--date", date], allow_fail=True)

    run([py, "scripts/nar_model/box_return_nar.py"])
    run([py, "scripts/nar_model/confidence_sweep_baseline_nar.py"])
    run([py, "scripts/nar_model/confidence_sweep_box4_nar.py"])
    run([py, "scripts/nar_model/confidence_sweep_box3_nar.py"])
    run([py, "scripts/nar_model/confidence_per_race_nar.py"])
    run([py, "scripts/build_artifact_nar.py"])

    if not args.no_push:
        run(["git", "add", "data/nar_pipeline", "data/newspaper/nar", "data/_manifest/scraped_race_ids.csv"])
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if staged.returncode == 0:
            print("\ncommit対象の変更なし。")
        else:
            run(["git", "commit", "-m", f"NAR pipeline data update ({date})"])
            run(["git", "fetch", "origin"], allow_fail=True)
            merge_rc = run(["git", "merge", "origin/main", "--no-edit"], allow_fail=True)
            if merge_rc != 0:
                print(
                    "\n!!! git merge に失敗しました(コンフリクトの可能性があります)。"
                    "git status を確認し、手動でコンフリクトを解消してから "
                    "git push してください。自動pushはスキップします。"
                )
                return
            run(["git", "push"])

    print("\n完了。レポート: data/nar_pipeline/prediction_report_nar.html")
    print("Artifact再公開はこのスクリプトの対象外(Claude Codeセッション側で手動再公開すること)。")


if __name__ == "__main__":
    main()
