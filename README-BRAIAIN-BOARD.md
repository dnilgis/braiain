# BRAIAIN board — deploy and refresh

New site for braiain.com: a sourced, dated dashboard of the AI buildout.
Everything is in the browser — no terminal, no build step.

## What's in this zip

| File | What it is |
|---|---|
| `index.html` | The board. Replaces the old game homepage. |
| `methodology.html` | Every figure's source, date, formula, and confidence. New file. |
| `data.json` | The canonical data record behind the board. New file. |
| `og-image.png` | The share card (replaces the game's). Screenshots and link previews show the Index. |
| `README-BRAIAIN-BOARD.md` | This file. Doesn't need to be uploaded, but harmless if it is. |

## Deploy (GitHub web UI)

1. Open github.com/dnilgis/braiain.
2. Click **Add file → Upload files**.
3. Drag all files from this zip in.
4. Commit to `main`. GitHub Pages redeploys on its own; give it a minute or two.
5. Open braiain.com in a private window and confirm the board loads and the
   "$ since you opened this page" line is counting.

The old game files (challenges/, images/, daily.json, guide.html, faq.html,
educators.html, etc.) are untouched — they just aren't linked from the new
homepage. Delete them whenever you want; the board doesn't use them.

## One edit for you to make (optional, one line)

The footer says "This board runs on one sponsor. Taking inquiries." with no
contact link, because that's your call. To add one, edit `index.html` in the
GitHub web editor, find that sentence, and wrap it:
`<a href="mailto:YOUR-ADDRESS">Taking inquiries.</a>`

## Refreshing the board (quarterly, or after each earnings week)

The board is honest because every number carries an as-of date — it does NOT
go stale invisibly, but it should be re-based roughly 8 times a year
(each big-4 + NVIDIA earnings print) or at minimum quarterly.

The simple way: start a session with Claude and say "re-base the braiain
board" — the refresh is: re-run the source audit, update `data.json`, update
the figures and dates in both HTML files, bump the changelog, re-verify.
Guardrail that session should obey: reject any figure that moved more than
3x or less than 0.5x against the prior quarter unless the source explains it.

## Deliberately NOT done in v1 (and why)

- **US data-center map** — a legal coordinate dataset exists (OpenStreetMap,
  ODbL, 1,816 US features) but pulling and verifying it properly didn't fit
  this pass. No pins beat approximate pins. v1.1 candidate.
- **Live market caps / stock prices** — every free aggregator checked has
  unclear or restrictive republication terms. Watchlist shows sourced events
  only.
- **AGI crowd forecast** — Metaculus's terms require written permission for
  commercial reuse. If you want it: email legal@metaculus.com describing the
  board, and it can be added as a labeled panel on a yes.
- **Free embeddable counters** (the backlink engine) and per-counter OG
  images — v1.1, after the board has a launch week.
- **Payments** — no sponsor checkout yet; first sponsor is a conversation,
  not a Stripe link, until there's traffic proof.

## Launch plan (from the panel, worth doing in week one)

1. Post to Hacker News / r/dataisbeautiful with one number as the title
   (e.g., "The AI buildout is running at $25,082 per second").
2. Send the stat pack (index level + 3 sector stats + methodology link) to
   AI newsletters: Ben's Bites, The Neuron, TLDR AI.
3. Re-post at every earnings re-base: "Buildout Index re-based: +X%."
