# api

FastAPI application exposing simulation runs (scenario upload, run
execution, live results via WebSocket) to the `dashboard/` frontend.

This is a **Phase 7** deliverable — see `docs/ARCHITECTURE.md` "Phased
build order". Nothing here is implemented until the simulation core,
signal processing, detection, tracking, and metrics phases are done
and there's something real to expose.

Planned layout once started:

```
api/
├── main.py       # FastAPI app instance, startup/shutdown
├── routes/       # HTTP + WebSocket route handlers
└── schemas/      # pydantic request/response models
```
