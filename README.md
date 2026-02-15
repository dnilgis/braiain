# BRAIAIN

**The Voight-Kampff Test for the AI Age**

A free daily game where you decide: real photo or AI-generated fake? 10 rounds, pro tips, streak tracking. New challenge every day at [braiain.com](https://braiain.com).

## How It Works

Each day, 10 new images are served. You guess **REAL** or **AI** for each one. After every round you get the answer, global accuracy stats, and a detection tip. At the end, share your score Wordle-style.

## Stack

- **Frontend:** Vanilla HTML / CSS / JS — single `index.html` file
- **Hosting:** GitHub Pages
- **Backend:** Cloudflare Worker + KV (vote tracking, global stats)
- **Content:** Manually curated daily challenges via `daily.json`

## Setup

1. Clone the repo
2. Push to GitHub Pages — the game works immediately with hardcoded fallback data
3. (Optional) Deploy `worker.js` to Cloudflare Workers with a KV binding to enable global stats
4. Update `daily.json` daily with new rounds

## Features

- No signup, no ads, no tracking beyond anonymous vote counts
- Works offline with fallback data
- Keyboard shortcuts (← REAL, → AI)
- Mobile-friendly with haptic feedback
- WCAG AA contrast compliant
- Wordle-style share grid (🟢🔴🟢🟢...)
- Classroom-ready — runs on Chromebooks, no login walls

## Files

```
index.html     — the entire game
daily.json     — today's challenge data
worker.js      — Cloudflare Worker (stats API)
og-image.png   — social share image
```

## How AI Is Used

BRAIAIN uses AI at several points in its pipeline:

- **Game content:** Each daily challenge includes AI-generated images sourced and curated manually. Players try to distinguish these from real photographs pulled from Pexels and Unsplash.
- **Tagline generation:** Google Gemini generates round taglines and descriptive text for the daily challenge metadata.
- **Development:** Claude (Anthropic) is used extensively as a coding partner for building and iterating on the game, curator tool, worker, and outreach materials.
- **Promotional content:** Gemini video generation has been used for creating demo and promotional videos.
- **Content curation:** A custom curator tool assists with sourcing and organizing images, but final selection is always manual — no automated content makes it into the live game without human review.

AI does **not** make gameplay decisions, moderate content, or interact with players. All challenge quality is human-controlled.

## License

MIT

