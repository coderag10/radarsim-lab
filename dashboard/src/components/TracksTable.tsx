import type { TrackOut, TrackStatus } from '../api'
import './TracksTable.css'

interface TracksTableProps {
  tracks: TrackOut[]
}

const STATUS_CLASS: Record<TrackStatus, string> = {
  ACTIVE: 'status-badge status-active',
  TENTATIVE: 'status-badge status-tentative',
  LOST: 'status-badge status-lost',
}

export function TracksTable({ tracks }: TracksTableProps) {
  if (tracks.length === 0) {
    return (
      <div className="tracks-table">
        <h3 className="panel-title">Tracks</h3>
        <p className="tracks-table-empty">No tracks -- try a lower SNR threshold.</p>
      </div>
    )
  }

  return (
    <div className="tracks-table">
      <h3 className="panel-title">Tracks</h3>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Status</th>
            <th>Position (m)</th>
            <th>Velocity (m/s)</th>
          </tr>
        </thead>
        <tbody>
          {tracks.map((track) => (
            <tr key={track.track_id}>
              <td>{track.track_id}</td>
              <td>
                <span className={STATUS_CLASS[track.status]}>{track.status}</span>
              </td>
              <td className="tabular">
                ({track.position[0].toFixed(2)}, {track.position[1].toFixed(2)})
              </td>
              <td className="tabular">
                ({track.velocity[0].toFixed(2)}, {track.velocity[1].toFixed(2)})
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
