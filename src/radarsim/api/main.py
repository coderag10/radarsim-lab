from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from radarsim import __version__
from radarsim.api.routes import runs, scenarios

app = FastAPI(
    title="radarsim-lab API",
    description="HTTP API for running radar simulation scenarios.",
    version=__version__,
)

# Scoped to local dev origins -- the known, real future consumer is
# Phase 8c's dashboard dev server on a different localhost port, not
# a public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenarios.router)
app.include_router(runs.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
