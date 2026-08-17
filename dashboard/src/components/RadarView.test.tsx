import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RadarView } from './RadarView'
import type { GroundTruthOut, SensorInfo, TrackOut } from '../api'

const sensor: SensorInfo = { id: 'radar-1', position: [0, 0] }
const groundTruth: GroundTruthOut[] = [
  { target_id: 'target-A', position: [20, 25] },
  { target_id: 'target-B', position: [-15, 25] },
]
const tracks: TrackOut[] = [
  { track_id: 'track-0', status: 'ACTIVE', position: [20.03, 25.15], velocity: [1.87, -1.0] },
  { track_id: 'track-1', status: 'TENTATIVE', position: [-15.1, 25.02], velocity: [1.07, 2.07] },
]

describe('RadarView', () => {
  it('renders a label for the sensor, every ground truth point, and every track', () => {
    render(<RadarView groundTruth={groundTruth} tracks={tracks} sensor={sensor} />)

    expect(screen.getByText('radar-1')).toBeInTheDocument()
    expect(screen.getByText('target-A')).toBeInTheDocument()
    expect(screen.getByText('target-B')).toBeInTheDocument()
    expect(screen.getByText('track-0')).toBeInTheDocument()
    expect(screen.getByText('track-1')).toBeInTheDocument()
  })

  it('renders the legend with ground truth, sensor, and all three statuses', () => {
    render(<RadarView groundTruth={groundTruth} tracks={tracks} sensor={sensor} />)

    expect(screen.getByText('Ground truth')).toBeInTheDocument()
    expect(screen.getByText('Sensor')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByText('Tentative')).toBeInTheDocument()
    expect(screen.getByText('Lost')).toBeInTheDocument()
  })

  it('renders one filled track circle per track', () => {
    const { container } = render(<RadarView groundTruth={groundTruth} tracks={tracks} sensor={sensor} />)

    const trackRings = container.querySelectorAll('.radar-view-track-ring')
    expect(trackRings).toHaveLength(tracks.length)
  })

  it('handles an empty scenario without crashing', () => {
    render(<RadarView groundTruth={[]} tracks={[]} sensor={sensor} />)
    expect(screen.getByText('radar-1')).toBeInTheDocument()
  })
})
