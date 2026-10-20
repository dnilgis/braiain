# BRAIAIN — The S&P of AI

The AI buildout, measured. A single-screen board of sourced, dated metrics on
the AI economy — capital, compute, power, infrastructure, emergent effects,
and society — anchored by the BRAIAIN Buildout Index.

Live at [braiain.com](https://braiain.com).

## Rules the board runs on

- Nothing appears without a named public source and a date.
- Ticking figures are arithmetic on filed numbers (a filed pace divided to the
  second, or a labeled pace delta from a dated statement) — never measurements.
- Flows tick. Counts tick only as labeled pace deltas. Shares never tick.
- Anything unsourceable gets cut, and the methodology page says what was cut.

## Stack

- Vanilla HTML/CSS/JS, one page plus `methodology.html`
- `data.json` is the canonical data record
- Hosted on GitHub Pages

## Refresh cadence

Re-based around each big-4/NVIDIA earnings print (~8x/year), minimum
quarterly. Every re-base updates `data.json`, both HTML files, and the
changelog in `methodology.html`.

Full sources, formulas, and the changelog: [braiain.com/methodology.html](https://braiain.com/methodology.html)
