# dashboard

React + TypeScript frontend: select a scenario, run it against the live API, and view the result — ground truth vs. tracked positions on a 2D radar view, a tracks table, and detection/RMSE metrics.

Vite + React 19 + TypeScript. No charting library — the radar view is a hand-rolled SVG scatter (a handful of points; a full charting library would be dependency weight for no benefit), styled per this project's `dataviz` design conventions (status colors for track state, categorical/neutral marks for identity, light+dark mode aware).

## Running it

Needs the API running (see [`../src/radarsim/api/README.md`](../src/radarsim/api/README.md)):

```bash
# terminal 1, from the repo root
uv run python -m radarsim.api.main

# terminal 2, from this directory
npm install
npm run dev
```

Open `http://localhost:5173`. To point at an API running somewhere other than `http://localhost:8000`, copy `.env.example` to `.env.local` and set `VITE_API_BASE_URL`.

## Scripts

- `npm run dev` — dev server with HMR
- `npm run build` — type-check (`tsc -b`) + production build
- `npm run test` — Vitest (unit + component tests, React Testing Library)
- `npm run lint` — oxlint

## Structure

```
src/
├── api.ts              # typed fetch client -- interfaces hand-mirror
│                          src/radarsim/api/schemas/ (kept in sync manually,
│                          no codegen: see api.ts's own header comment)
├── App.tsx              # top-level wiring: fetch scenarios, run, loading/error states
└── components/
    ├── ScenarioSelector.tsx
    ├── RunForm.tsx
    ├── RadarView.tsx     # the 2D scatter (ground truth / tracks / sensor)
    ├── TracksTable.tsx
    └── MetricsPanel.tsx
```

## Not implemented (yet)

Live/animated playback -- the API only returns the final state of a run, not a per-timestep stream. That needs a WebSocket endpoint on the backend first (deliberately deferred in Phase 8b until there was a real consumer to define the shape); this dashboard is "run and view the result," not "watch it happen."
