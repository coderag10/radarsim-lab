from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from radarsim.api.main import app  # noqa: E402
from radarsim.api.routes.scenarios import get_scenario  # noqa: E402

client = TestClient(app)

_VALID_SCENARIO = """\
simulation:
  duration: 2.0
  timestep: 1.0
  seed: 7
targets:
  - id: t1
    model: constant_velocity
    position: [0.0, 0.0]
    velocity: [1.0, 0.0]
"""


def test_list_scenarios_includes_two_targets_fixture() -> None:
    response = client.get("/scenarios")

    assert response.status_code == 200
    paths = [entry["path"] for entry in response.json()]
    assert any("two_targets.yaml" in path for path in paths)


def test_get_scenario_detail() -> None:
    response = client.get("/scenarios/scenarios/basic/two_targets.yaml")

    assert response.status_code == 200
    body = response.json()
    assert body["duration"] == 5.0
    assert body["timestep"] == 1.0
    assert body["seed"] == 42
    assert body["num_targets"] == 2


def test_get_scenario_not_found_returns_404() -> None:
    response = client.get("/scenarios/scenarios/does_not_exist.yaml")
    assert response.status_code == 404


def test_get_scenario_malformed_raises_422(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("targets: []\n")

    with pytest.raises(HTTPException) as exc_info:
        get_scenario(str(bad))

    assert exc_info.value.status_code == 422


def test_list_scenarios_skips_malformed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RADARSIM_SCENARIOS_DIR", str(tmp_path))
    (tmp_path / "good.yaml").write_text(_VALID_SCENARIO)
    (tmp_path / "bad.yaml").write_text("targets: []\n")

    response = client.get("/scenarios")

    assert response.status_code == 200
    paths = [entry["path"] for entry in response.json()]
    assert any("good.yaml" in path for path in paths)
    assert not any("bad.yaml" in path for path in paths)
