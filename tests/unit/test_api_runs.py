import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from radarsim.api.main import app  # noqa: E402

client = TestClient(app)


def test_create_run_returns_active_tracks_and_metrics() -> None:
    response = client.post("/runs", json={"scenario": "scenarios/basic/two_targets.yaml"})

    assert response.status_code == 200
    body = response.json()
    assert body["seed"] == 42
    assert body["metrics"]["detection_probability"] == 1.0
    active = [track for track in body["tracks"] if track["status"] == "ACTIVE"]
    assert len(active) == 2
    assert len(body["ground_truth"]) == 2
    assert {truth["target_id"] for truth in body["ground_truth"]} == {"target-A", "target-B"}


def test_create_run_high_threshold_returns_no_tracks_and_null_rmse() -> None:
    response = client.post(
        "/runs",
        json={"scenario": "scenarios/basic/two_targets.yaml", "snr_threshold_db": 1000.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tracks"] == []
    assert body["metrics"]["position_rmse"] is None


def test_create_run_missing_scenario_returns_404() -> None:
    response = client.post("/runs", json={"scenario": "scenarios/does_not_exist.yaml"})
    assert response.status_code == 404


def test_create_run_invalid_request_body_returns_422() -> None:
    response = client.post("/runs", json={})
    assert response.status_code == 422
