import type { ScenarioSummary } from '../api'

interface ScenarioSelectorProps {
  scenarios: ScenarioSummary[]
  selected: string | null
  onSelect: (path: string) => void
  disabled?: boolean
}

export function ScenarioSelector({ scenarios, selected, onSelect, disabled }: ScenarioSelectorProps) {
  return (
    <label className="field">
      <span className="field-label">Scenario</span>
      <select
        className="field-input"
        value={selected ?? ''}
        onChange={(event) => onSelect(event.target.value)}
        disabled={disabled || scenarios.length === 0}
      >
        {scenarios.length === 0 && <option value="">No scenarios found</option>}
        {scenarios.map((scenario) => (
          <option key={scenario.path} value={scenario.path}>
            {scenario.path} ({scenario.num_targets} targets, {scenario.duration}s)
          </option>
        ))}
      </select>
    </label>
  )
}
