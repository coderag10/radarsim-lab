import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, listScenarios, runScenario } from './api'

describe('listScenarios', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns parsed scenarios on success', async () => {
    const scenarios = [{ path: 'a.yaml', duration: 5, timestep: 1, seed: 42, num_targets: 2 }]
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(scenarios) }),
    )

    const result = await listScenarios()

    expect(result).toEqual(scenarios)
  })

  it('throws ApiError with the response detail on failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: () => Promise.resolve({ detail: 'scenario not found' }),
      }),
    )

    await expect(listScenarios()).rejects.toMatchObject(
      new ApiError('scenario not found', 404),
    )
  })
})

describe('runScenario', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the request body and returns the parsed response', async () => {
    const response = {
      scenario: 'a.yaml',
      duration: 5,
      timestep: 1,
      seed: 42,
      num_targets: 1,
      sensor: { id: 'radar-1', position: [0, 0] },
      ground_truth: [],
      tracks: [],
      metrics: { detection_probability: 1, position_rmse: null },
    }
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(response) })
    vi.stubGlobal('fetch', fetchMock)

    const result = await runScenario({ scenario: 'a.yaml' })

    expect(result).toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/runs'),
      expect.objectContaining({ method: 'POST' }),
    )
  })
})
