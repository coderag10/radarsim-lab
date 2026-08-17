import { useState } from 'react'
import type { RunRequest } from '../api'

export interface RunParams {
  radarX: number
  radarY: number
  snrThresholdDb: number
}

interface RunFormProps {
  scenario: string | null
  params: RunParams
  onParamsChange: (params: RunParams) => void
  onRun: (request: RunRequest) => void
  running: boolean
}

export function RunForm({ scenario, params, onParamsChange, onRun, running }: RunFormProps) {
  const [localParams, setLocalParams] = useState(params)

  const update = (patch: Partial<RunParams>) => {
    const next = { ...localParams, ...patch }
    setLocalParams(next)
    onParamsChange(next)
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    if (!scenario) return
    onRun({
      scenario,
      radar_x: localParams.radarX,
      radar_y: localParams.radarY,
      snr_threshold_db: localParams.snrThresholdDb,
    })
  }

  return (
    <form className="run-form" onSubmit={handleSubmit}>
      <label className="field">
        <span className="field-label">Radar X (m)</span>
        <input
          className="field-input"
          type="number"
          step="1"
          value={localParams.radarX}
          onChange={(event) => update({ radarX: Number(event.target.value) })}
        />
      </label>
      <label className="field">
        <span className="field-label">Radar Y (m)</span>
        <input
          className="field-input"
          type="number"
          step="1"
          value={localParams.radarY}
          onChange={(event) => update({ radarY: Number(event.target.value) })}
        />
      </label>
      <label className="field">
        <span className="field-label">SNR threshold (dB)</span>
        <input
          className="field-input"
          type="number"
          step="1"
          value={localParams.snrThresholdDb}
          onChange={(event) => update({ snrThresholdDb: Number(event.target.value) })}
        />
      </label>
      <button className="run-button" type="submit" disabled={!scenario || running}>
        {running ? 'Running…' : 'Run'}
      </button>
    </form>
  )
}
