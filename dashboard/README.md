# dashboard

React + TypeScript visualization frontend (radar view, tracks panel,
range-Doppler map, filter-comparison charts).

This is a **Phase 7** deliverable — see `../docs/ARCHITECTURE.md`
"Phased build order". It starts only once the backend pipeline
(`src/radarsim/{core,targets,radar,signals,detection,tracking,fusion,metrics}`)
produces something worth visualizing, and talks to it through
`src/radarsim/api/`.
