import pandas as pd

from src.netkeiba_pipeline.storage import paths, writer


def test_write_race_result_is_idempotent_per_race_id(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULTS_DIR", tmp_path / "race_results")

    df_v1 = pd.DataFrame([{"race_id": "202406010101", "horse_id": "1", "finish_pos": "1"}])
    writer.write_race_result(df_v1, "20240106", "202406010101")

    other_race_df = pd.DataFrame([{"race_id": "202406010102", "horse_id": "2", "finish_pos": "1"}])
    writer.write_race_result(other_race_df, "20240106", "202406010102")

    # Re-writing the same race_id (simulating a retry) must replace, not duplicate.
    df_v2 = pd.DataFrame([{"race_id": "202406010101", "horse_id": "1", "finish_pos": "2"}])
    writer.write_race_result(df_v2, "20240106", "202406010101")

    result = pd.read_csv(paths.race_result_csv_path("20240106"), dtype=str)
    assert len(result) == 2
    row_101 = result[result["race_id"] == "202406010101"].iloc[0]
    assert row_101["finish_pos"] == "2"


def test_write_race_result_circuit_nar_uses_separate_path(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULTS_DIR", tmp_path / "race_results")
    monkeypatch.setattr(paths, "PAYOUTS_DIR", tmp_path / "payouts")

    jra_df = pd.DataFrame([{"race_id": "202406010101", "horse_id": "1", "finish_pos": "1"}])
    writer.write_race_result(jra_df, "20260726", "202406010101")  # circuit omitted -> jra

    nar_df = pd.DataFrame([{"race_id": "202630072601", "horse_id": "1", "finish_pos": "1"}])
    writer.write_race_result(nar_df, "20260726", "202630072601", circuit="nar")

    nar_payouts = pd.DataFrame([{"race_id": "202630072601", "bet_type": "単勝", "rank": 1}])
    writer.write_payouts(nar_payouts, "20260726", "202630072601", circuit="nar")

    # Same kaisai_date, two different circuits -> two different files, JRA path unprefixed.
    jra_path = paths.race_result_csv_path("20260726")
    nar_path = paths.race_result_csv_path("20260726", circuit="nar")
    assert jra_path != nar_path
    assert jra_path == tmp_path / "race_results" / "2026" / "20260726.csv"
    assert nar_path == tmp_path / "race_results" / "nar" / "2026" / "20260726.csv"

    assert set(pd.read_csv(jra_path, dtype=str)["race_id"]) == {"202406010101"}
    assert set(pd.read_csv(nar_path, dtype=str)["race_id"]) == {"202630072601"}
    assert paths.payout_csv_path("20260726", circuit="nar").exists()


def test_manifest_skip_logic(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "MANIFEST_PATH", tmp_path / "_manifest" / "scraped_race_ids.csv")

    assert writer.is_already_scraped("202406010101") is False

    writer.mark_scraped("202406010101", status="success")
    assert writer.is_already_scraped("202406010101") is True

    writer.mark_scraped("202406010102", status="failed")
    assert writer.is_already_scraped("202406010102") is False
