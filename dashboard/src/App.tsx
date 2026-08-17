import { useEffect, useState } from 'react'
import { ApiError, listScenarios, runScenario } from './api'
import type { RunResponse, ScenarioSummary } from './api'
import { ScenarioSelector } from './components/ScenarioSelector'
import { RunForm } from './components/RunForm'
import type { RunParams } from './components/RunForm'
import { RadarView } from './components/RadarView'
import { TracksTable } from './components/TracksTable'
import { MetricsPanel } from './components/MetricsPanel'
import './App.css'

const DEFAULT_PARAMS: RunParams = { radarX: 0, radarY: 0, snrThresholdDb: 0 }

function App() {
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([])
  const [scenariosError, setScenariosError] = useState<string | null>(null)
  const [scenariosLoading, setScenariosLoading] = useState(true)

  const [selectedScenario, setSelectedScenario] = useState<string | null>(null)
  const [runParams, setRunParams] = useState<RunParams>(DEFAULT_PARAMS)

  const [result, setResult] = useState<RunResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listScenarios()
      .then((fetched) => {
        if (cancelled) return
        setScenarios(fetched)
        if (fetched.length > 0) setSelectedScenario(fetched[0].path)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setScenariosError(error instanceof ApiError ? error.message : 'Failed to load scenarios')
      })
      .finally(() => {
        if (!cancelled) setScenariosLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleRun: React.ComponentProps<typeof RunForm>['onRun'] = (request) => {
    setRunning(true)
    setRunError(null)
    runScenario(request)
      .then(setResult)
      .catch((error: unknown) => {
        setRunError(error instanceof ApiError ? error.message : 'Failed to run scenario')
      })
      .finally(() => setRunning(false))
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Radar Lab</h1>
        <p className="app-subtitle">Run a scenario and view tracking results against ground truth.</p>
      </header>

      <section className="app-controls">
        {scenariosLoading && <p className="app-status">Loading scenarios…</p>}
        {scenariosError && <p className="app-status app-status-error">{scenariosError}</p>}
        {!scenariosLoading && !scenariosError && (
          <>
            <ScenarioSelector scenarios={scenarios} selected={selectedScenario} onSelect={setSelectedScenario} />
            <RunForm
              scenario={selectedScenario}
              params={runParams}
              onParamsChange={setRunParams}
              onRun={handleRun}
              running={running}
            />
          </>
        )}
        {runError && <p className="app-status app-status-error">{runError}</p>}
      </section>

      {result && (
        <section className="app-results">
          <RadarView groundTruth={result.ground_truth} tracks={result.tracks} sensor={result.sensor} />
          <div className="app-results-side">
            <MetricsPanel metrics={result.metrics} />
            <TracksTable tracks={result.tracks} />
          </div>
        </section>
      )}

      {!result && !running && !scenariosLoading && !scenariosError && (
        <p className="app-status">Select a scenario and click Run to see results.</p>
      )}
    </div>
  )
}

export default App
