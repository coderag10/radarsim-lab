from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from radarsim.api.schemas.run import RunRequest, RunResponse
from radarsim.cli.run import run_scenario

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    """Run a scenario end-to-end -- the same `cli.run.run_scenario` the CLI uses."""
    try:
        result = run_scenario(
            Path(request.scenario),
            radar_position=np.array([request.radar_x, request.radar_y]),
            noise_std=np.array(request.noise_std),
            reference_rcs=request.reference_rcs,
            reference_range=request.reference_range,
            reference_snr_db=request.reference_snr_db,
            snr_threshold_db=request.snr_threshold_db,
        )
    except OSError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RunResponse.from_run_result(result)
