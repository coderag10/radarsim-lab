import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'
import type { RunResponse, ScenarioSummary } from './api'

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')
  return {
    ...actual,
    listScenarios: vi.fn(),
    runScenario: vi.fn(),
  }
})

const scenario: ScenarioSummary = {
  path: 'scenarios/basic/two_targets.yaml',
  duration: 5,
  timestep: 1,
  seed: 42,
  num_targets: 2,
}

const runResponse: RunResponse = {
  scenario: scenario.path,
  duration: 5,
  timestep: 1,
  seed: 42,
  num_targets: 2,
  sensor: { id: 'radar-1', position: [0, 0] },
  ground_truth: [{ target_id: 'target-A', position: [20, 25] }],
  tracks: [{ track_id: 'track-0', status: 'ACTIVE', position: [20.03, 25.15], velocity: [1.87, -1.0] }],
  metrics: { detection_probability: 1, position_rmse: 0.13 },
}

describe('App', () => {
  beforeEach(() => {
    vi.mocked(api.listScenarios).mockResolvedValue([scenario])
    vi.mocked(api.runScenario).mockResolvedValue(runResponse)
  })

  it('loads scenarios, runs the selected one, and renders results', async () => {
    render(<App />)

    await screen.findByRole('option', { name: /two_targets\.yaml/ })

    await userEvent.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() =>
      expect(api.runScenario).toHaveBeenCalledWith(expect.objectContaining({ scenario: scenario.path })),
    )

    expect((await screen.findAllByText('track-0')).length).toBeGreaterThan(0)
    expect(screen.getByText('100.0%')).toBeInTheDocument()
    expect(screen.getByText('0.130')).toBeInTheDocument()
  })

  it('shows an error message when the run fails', async () => {
    vi.mocked(api.runScenario).mockRejectedValue(new api.ApiError('scenario not found', 404))
    render(<App />)

    await screen.findByRole('option', { name: /two_targets\.yaml/ })
    await userEvent.click(screen.getByRole('button', { name: /run/i }))

    expect(await screen.findByText('scenario not found')).toBeInTheDocument()
  })
})
