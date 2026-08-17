import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ScenarioSelector } from './ScenarioSelector'
import type { ScenarioSummary } from '../api'

const scenarios: ScenarioSummary[] = [
  { path: 'scenarios/basic/two_targets.yaml', duration: 5, timestep: 1, seed: 42, num_targets: 2 },
  { path: 'scenarios/other.yaml', duration: 10, timestep: 0.5, seed: 1, num_targets: 3 },
]

describe('ScenarioSelector', () => {
  it('renders an option for every scenario', () => {
    render(<ScenarioSelector scenarios={scenarios} selected={scenarios[0].path} onSelect={vi.fn()} />)

    expect(screen.getByRole('option', { name: /two_targets\.yaml/ })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /other\.yaml/ })).toBeInTheDocument()
  })

  it('calls onSelect when a different scenario is chosen', async () => {
    const onSelect = vi.fn()
    render(<ScenarioSelector scenarios={scenarios} selected={scenarios[0].path} onSelect={onSelect} />)

    await userEvent.selectOptions(screen.getByRole('combobox'), scenarios[1].path)

    expect(onSelect).toHaveBeenCalledWith(scenarios[1].path)
  })

  it('shows a placeholder and disables the select when there are no scenarios', () => {
    render(<ScenarioSelector scenarios={[]} selected={null} onSelect={vi.fn()} />)

    expect(screen.getByRole('combobox')).toBeDisabled()
    expect(screen.getByText('No scenarios found')).toBeInTheDocument()
  })
})
