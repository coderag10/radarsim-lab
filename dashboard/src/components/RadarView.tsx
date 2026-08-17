import { useMemo, useState } from 'react'
import type { GroundTruthOut, SensorInfo, TrackOut, TrackStatus } from '../api'
import './RadarView.css'

interface RadarViewProps {
  groundTruth: GroundTruthOut[]
  tracks: TrackOut[]
  sensor: SensorInfo
}

// Status is a state, not an identity -- uses the reserved status palette
// (dataviz skill), never the categorical series colors. Mode-invariant
// by design (see references/palette.md).
const STATUS_COLOR: Record<TrackStatus, string> = {
  ACTIVE: 'var(--status-good)',
  TENTATIVE: 'var(--status-warning)',
  LOST: 'var(--status-critical)',
}

const STATUS_LABEL: Record<TrackStatus, string> = {
  ACTIVE: 'Active',
  TENTATIVE: 'Tentative',
  LOST: 'Lost',
}

const WIDTH = 560
const HEIGHT = 420
const PADDING = 48
const TICK_COUNT = 5

interface Tooltip {
  x: number
  y: number
  title: string
  lines: string[]
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  if (lo === hi) return [lo]
  const step = (hi - lo) / count
  return Array.from({ length: count + 1 }, (_, i) => lo + step * i)
}

export function RadarView({ groundTruth, tracks, sensor }: RadarViewProps) {
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)

  const { scaleX, scaleY, xTicks, yTicks } = useMemo(() => {
    const xs = [sensor.position[0], ...groundTruth.map((g) => g.position[0]), ...tracks.map((t) => t.position[0])]
    const ys = [sensor.position[1], ...groundTruth.map((g) => g.position[1]), ...tracks.map((t) => t.position[1])]
    const xMin = Math.min(...xs)
    const xMax = Math.max(...xs)
    const yMin = Math.min(...ys)
    const yMax = Math.max(...ys)
    const xPad = Math.max((xMax - xMin) * 0.2, 5)
    const yPad = Math.max((yMax - yMin) * 0.2, 5)
    const xLo = xMin - xPad
    const xHi = xMax + xPad
    const yLo = yMin - yPad
    const yHi = yMax + yPad

    const sx = (x: number) => PADDING + ((x - xLo) / (xHi - xLo)) * (WIDTH - 2 * PADDING)
    // flip Y so +Y (north) renders at the top, matching how the scenario
    // targets' velocities are described (world frame, not SVG's down-is-positive)
    const sy = (y: number) => HEIGHT - PADDING - ((y - yLo) / (yHi - yLo)) * (HEIGHT - 2 * PADDING)

    return {
      scaleX: sx,
      scaleY: sy,
      xTicks: niceTicks(xLo, xHi, TICK_COUNT),
      yTicks: niceTicks(yLo, yHi, TICK_COUNT),
    }
  }, [groundTruth, tracks, sensor])

  const showTooltip = (event: React.MouseEvent<SVGGElement>, title: string, lines: string[]) => {
    const bounds = event.currentTarget.ownerSVGElement?.getBoundingClientRect()
    if (!bounds) return
    setTooltip({ x: event.clientX - bounds.left, y: event.clientY - bounds.top, title, lines })
  }
  const hideTooltip = () => setTooltip(null)

  return (
    <div className="radar-view">
      <h3 className="radar-view-title">Radar view</h3>
      <svg
        className="radar-view-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Ground truth and tracked target positions relative to the sensor"
      >
        {yTicks.map((tick) => (
          <line
            key={`grid-y-${tick}`}
            className="radar-view-gridline"
            x1={PADDING}
            x2={WIDTH - PADDING}
            y1={scaleY(tick)}
            y2={scaleY(tick)}
          />
        ))}
        {xTicks.map((tick) => (
          <line
            key={`grid-x-${tick}`}
            className="radar-view-gridline"
            x1={scaleX(tick)}
            x2={scaleX(tick)}
            y1={PADDING}
            y2={HEIGHT - PADDING}
          />
        ))}

        <line
          className="radar-view-axis"
          x1={PADDING}
          x2={WIDTH - PADDING}
          y1={HEIGHT - PADDING}
          y2={HEIGHT - PADDING}
        />
        <line className="radar-view-axis" x1={PADDING} x2={PADDING} y1={PADDING} y2={HEIGHT - PADDING} />
        {xTicks.map((tick) => (
          <text key={`xt-${tick}`} className="radar-view-tick" x={scaleX(tick)} y={HEIGHT - PADDING + 18}>
            {Math.round(tick)}
          </text>
        ))}
        {yTicks.map((tick) => (
          <text
            key={`yt-${tick}`}
            className="radar-view-tick"
            x={PADDING - 10}
            y={scaleY(tick) + 4}
            textAnchor="end"
          >
            {Math.round(tick)}
          </text>
        ))}
        <text className="radar-view-axis-label" x={WIDTH / 2} y={HEIGHT - 6} textAnchor="middle">
          X (m)
        </text>
        <text
          className="radar-view-axis-label"
          x={14}
          y={HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${HEIGHT / 2})`}
        >
          Y (m)
        </text>

        {/* Sensor */}
        <g
          onMouseMove={(event) => showTooltip(event, sensor.id, [`(${sensor.position[0]}, ${sensor.position[1]})`])}
          onMouseLeave={hideTooltip}
        >
          <path
            className="radar-view-sensor"
            d={sensorPath(scaleX(sensor.position[0]), scaleY(sensor.position[1]))}
          />
          <circle cx={scaleX(sensor.position[0])} cy={scaleY(sensor.position[1])} r={12} fill="transparent" />
          <text
            className="radar-view-label"
            x={scaleX(sensor.position[0]) + 12}
            y={scaleY(sensor.position[1]) - 10}
          >
            {sensor.id}
          </text>
        </g>

        {/* Ground truth */}
        {groundTruth.map((truth) => {
          const x = scaleX(truth.position[0])
          const y = scaleY(truth.position[1])
          return (
            <g
              key={truth.target_id}
              onMouseMove={(event) =>
                showTooltip(event, truth.target_id, [
                  'Ground truth',
                  `(${truth.position[0].toFixed(2)}, ${truth.position[1].toFixed(2)})`,
                ])
              }
              onMouseLeave={hideTooltip}
            >
              <path className="radar-view-truth" d={crosshairPath(x, y)} />
              <circle cx={x} cy={y} r={12} fill="transparent" />
              <text className="radar-view-label" x={x + 10} y={y + 16}>
                {truth.target_id}
              </text>
            </g>
          )
        })}

        {/* Tracks */}
        {tracks.map((track) => {
          const x = scaleX(track.position[0])
          const y = scaleY(track.position[1])
          return (
            <g
              key={track.track_id}
              onMouseMove={(event) =>
                showTooltip(event, track.track_id, [
                  STATUS_LABEL[track.status],
                  `pos (${track.position[0].toFixed(2)}, ${track.position[1].toFixed(2)})`,
                  `vel (${track.velocity[0].toFixed(2)}, ${track.velocity[1].toFixed(2)})`,
                ])
              }
              onMouseLeave={hideTooltip}
            >
              <circle className="radar-view-track-ring" cx={x} cy={y} r={8} />
              <circle cx={x} cy={y} r={6} fill={STATUS_COLOR[track.status]} />
              <circle cx={x} cy={y} r={12} fill="transparent" />
              <text className="radar-view-label" x={x + 10} y={y - 10}>
                {track.track_id}
              </text>
            </g>
          )
        })}
      </svg>

      {tooltip && (
        <div className="radar-view-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
          <div className="radar-view-tooltip-title">{tooltip.title}</div>
          {tooltip.lines.map((line) => (
            <div key={line} className="radar-view-tooltip-line">
              {line}
            </div>
          ))}
        </div>
      )}

      <div className="radar-view-legend">
        <span className="radar-view-legend-item">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path className="radar-view-truth" d={crosshairPath(7, 7)} />
          </svg>
          Ground truth
        </span>
        <span className="radar-view-legend-item">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path className="radar-view-sensor" d={sensorPath(7, 7)} />
          </svg>
          Sensor
        </span>
        {(Object.keys(STATUS_LABEL) as TrackStatus[]).map((status) => (
          <span className="radar-view-legend-item" key={status}>
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <circle cx="7" cy="7" r="6" fill={STATUS_COLOR[status]} />
            </svg>
            {STATUS_LABEL[status]}
          </span>
        ))}
      </div>
    </div>
  )
}

function crosshairPath(x: number, y: number, size = 6): string {
  return `M ${x - size} ${y} L ${x + size} ${y} M ${x} ${y - size} L ${x} ${y + size}`
}

function sensorPath(x: number, y: number, size = 8): string {
  return `M ${x} ${y - size} L ${x + size} ${y + size} L ${x - size} ${y + size} Z`
}
