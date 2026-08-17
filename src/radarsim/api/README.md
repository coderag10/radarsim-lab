# api

FastAPI service exposing simulation runs over HTTP. `POST /runs` is a
thin JSON wrapper around `radarsim.cli.run.run_scenario` — the exact
same tested pipeline (`World` → `RadarModel` → `ThresholdDetector` →
`Tracker` → `metrics`) the CLI uses, just JSON in/out instead of
argv/stdout.

## Endpoints

- `GET /health` — liveness check
- `GET /scenarios` — list scenarios found under the scenarios root
- `GET /scenarios/{path}` — one scenario's metadata without running it
- `POST /runs` — run a scenario, return tracks + metrics (see `radarsim.api.schemas.run.RunRequest`/`RunResponse` for the exact shape)

Interactive docs (Swagger UI) are auto-generated at `/docs` once the server is running.

## Running it

```bash
uv sync --extra dev --extra api
uvicorn radarsim.api.main:app --reload
# or: uv run python -m radarsim.api.main
```

`api` is an optional dependency group — code and tests here degrade gracefully without it (`tests/unit/test_api_*.py` skip via `pytest.importorskip("fastapi")`, and `mypy` treats `fastapi`/`pydantic` imports as `Any` rather than failing when the extra isn't installed; see the `[[tool.mypy.overrides]]` entry in `pyproject.toml`).

## Not implemented (yet)

- WebSocket/streaming endpoint for live per-timestep updates — deferred until `dashboard/` (Phase 8c) exists to define what shape of streaming it actually needs, rather than guessing at a protocol now
- Scenario upload — only scenarios already on disk can be run; accepting arbitrary uploaded YAML is a bigger validation/security surface, not needed yet
- Authentication — this is a local research tool, not a multi-tenant service
