import type { MetricsOut } from '../api'
import './MetricsPanel.css'

interface MetricsPanelProps {
  metrics: MetricsOut
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <div className="metrics-panel">
      <h3 className="panel-title">Metrics</h3>
      <div className="metrics-panel-grid">
        <div className="stat-tile">
          <span className="stat-tile-label">Detection probability</span>
          <span className="stat-tile-value">{(metrics.detection_probability * 100).toFixed(1)}%</span>
        </div>
        <div className="stat-tile">
          <span className="stat-tile-label">Position RMSE</span>
          <span className="stat-tile-value">
            {metrics.position_rmse !== null ? metrics.position_rmse.toFixed(3) : 'N/A'}
          </span>
        </div>
      </div>
    </div>
  )
}
