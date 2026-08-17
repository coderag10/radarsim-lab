"""FastAPI service exposing simulation runs over HTTP.

Thin HTTP layer over already-tested logic: `POST /runs` is a JSON
wrapper around `radarsim.cli.run.run_scenario`, the same function the
CLI uses. See `radarsim.api.main` for the app instance, `.routes` for
handlers, `.schemas` for request/response models.
"""
