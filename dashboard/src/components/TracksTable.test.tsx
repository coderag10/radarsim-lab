import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TracksTable } from './TracksTable'
import type { TrackOut } from '../api'

const tracks: TrackOut[] = [
  { track_id: 'track-0', status: 'ACTIVE', position: [20.03, 25.15], velocity: [1.87, -1.0] },
  { track_id: 'track-1', status: 'LOST', position: [-15.1, 25.02], velocity: [1.07, 2.07] },
]

describe('TracksTable', () => {
  it('renders one row per track with id, status, position, and velocity', () => {
    render(<TracksTable tracks={tracks} />)

    expect(screen.getByText('track-0')).toBeInTheDocument()
    expect(screen.getByText('ACTIVE')).toBeInTheDocument()
    expect(screen.getByText('track-1')).toBeInTheDocument()
    expect(screen.getByText('LOST')).toBeInTheDocument()
    expect(screen.getByText('(20.03, 25.15)')).toBeInTheDocument()
  })

  it('shows an empty-state message when there are no tracks', () => {
    render(<TracksTable tracks={[]} />)

    expect(screen.getByText(/no tracks/i)).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
