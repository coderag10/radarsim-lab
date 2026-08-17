/**
 * Typed client for the radarsim-lab API.
 *
 * These interfaces are hand-written to mirror `src/radarsim/api/schemas/`
 * exactly (no codegen) -- keep them in sync manually if the Python
 * schemas change:
 *   - schemas/scenario.py  -> ScenarioSummary
 *   - schemas/run.py       -> RunRequest, RunResponse, SensorInfo,
 *                              GroundTruthOut, TrackOut, MetricsOut
 */

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export interface ScenarioSummary {
  path: string
  duration: number
  timestep: number
  seed: number
  num_targets: number
}

export interface RunRequest {
  scenario: string
  radar_x?: number
  radar_y?: number
  noise_std?: [number, number, number]
  reference_rcs?: number
  reference_range?: number
  reference_snr_db?: number
  snr_threshold_db?: number
}

export interface SensorInfo {
  id: string
  position: [number, number]
}

export interface GroundTruthOut {
  target_id: string
  position: [number, number]
}

export type TrackStatus = 'TENTATIVE' | 'ACTIVE' | 'LOST'

export interface TrackOut {
  track_id: string
  status: TrackStatus
  position: [number, number]
  velocity: [number, number]
}

export interface MetricsOut {
  detection_probability: number
  position_rmse: number | null
}

export interface RunResponse {
  scenario: string
  duration: number
  timestep: number
  seed: number
  num_targets: number
  sensor: SensorInfo
  ground_truth: GroundTruthOut[]
  tracks: TrackOut[]
  metrics: MetricsOut
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      return String((body as { detail: unknown }).detail)
    }
  } catch {
    // response wasn't JSON -- fall through to the generic message below
  }
  return response.statusText
}

export async function listScenarios(): Promise<ScenarioSummary[]> {
  const response = await fetch(`${API_BASE_URL}/scenarios`)
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return (await response.json()) as ScenarioSummary[]
}

export async function runScenario(request: RunRequest): Promise<RunResponse> {
  const response = await fetch(`${API_BASE_URL}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return (await response.json()) as RunResponse
}
