# BRAIAIN

A parody AI lab with a real, very small language model. brAIAIn is trained only on
a twelve-line founding text (`data/genesis.txt`, the Ten Instructions) plus one
sentence per paying customer. Every paid sentence ships a new model version with
release notes, a paper and a status incident.

Everything below is done in a browser. No terminal.

## How it works

- `scripts/train.py` runs every 10 minutes in GitHub Actions (`.github/workflows/train.yml`).
  It reads paid Stripe checkouts, screens each sentence, retrains every version,
  writes the notes, and commits. GitHub Pages republishes the site.
- The website rebuilds any version in the visitor's browser from the public files in
  `data/` and chats with it (`assets/braiain.js`). Nothing is sent to a server.
- `api/worker.js` is the real API, run as a free Cloudflare Worker.
- The model is a word trigram model: it counts which word follows each pair of
  words and writes by weighted dice. Each version is also trained on a few passages
  written by the version before it, and the newest paid sentence counts four times.

## Teaser mode (what's live now)

While `paymentLink` in `assets/config.js` is empty, the site is in teaser mode:
the buy buttons become **Join the waitlist**, and a countdown runs to `opensAt`
(the same moment for every visitor). When the countdown ends and there is still
no payment link, it says "any minute now". To move the opening, edit `opensAt`.

### Turn on the waitlist box (2 minutes, free)
1. Go to formspree.io → sign up → **New form** → name it `braiain waitlist`.
2. Copy the form endpoint (looks like `https://formspree.io/f/abcdwxyz`).
3. In the repo, edit `assets/config.js` → paste it into `waitlistEndpoint` → commit.
Sign-ups then arrive in your Formspree inbox. Until this is set, the box is hidden
and a line says sign-ups open soon, so no email is ever typed into a box that goes nowhere.
The site promises one email when training opens; send it from Formspree's export.

## One-time setup

Do steps 1 to 6 in Stripe **test mode** first (toggle at the top of the Stripe
dashboard). Step 7 switches to real money.

### 1. Let the Action save its work
GitHub repo → **Settings** → **Actions** → **General** → *Workflow permissions* →
**Read and write permissions** → **Save**.

### 2. Stripe product and Payment Link
1. Stripe → **Product catalog** → **Add product**. Name: `One sentence`. Price: `$5.00`, one time. Save.
2. Stripe → **Payment Links** → **New** → pick that product.
3. Under **Options**, turn on **Add custom fields**:
   - Field 1: type **Text**, label `Your sentence (up to 255 characters)`. Required.
   - Field 2: type **Text**, label `Credit as (optional)`. Mark it optional.
   The trainer finds them by the words *sentence* and *credit* in the label, so keep those words.
4. Under **After payment**, choose **Don't show confirmation page** → redirect to
   `https://www.braiain.com/thanks.html`.
5. **Create link**. Copy the link (starts with `https://buy.stripe.com/`).
   Also copy its ID from the link's detail page (starts with `plink_`).

### 3. Stripe key that can only read checkouts
Stripe → **Developers** → **API keys** → **Create restricted key**.
Name it `braiain-trainer`. Set **Checkout Sessions** to **Read**. Leave everything
else as **None**. Create, and copy the key (starts with `rk_test_`).

### 4. Give the keys to GitHub
GitHub repo → **Settings** → **Secrets and variables** → **Actions**.
- **Secrets** tab → **New repository secret**:
  - `STRIPE_SECRET_KEY` = the restricted key.
  - `ANTHROPIC_API_KEY` = an Anthropic API key (console.anthropic.com → API Keys).
    Optional. Without it, release notes come from built-in templates.
- **Variables** tab → **New repository variable**:
  - `STRIPE_PAYMENT_LINK` = the `plink_` ID. This makes the trainer ignore any
    other sales on your Stripe account.

### 5. Turn on the buy buttons
In the repo, open `assets/config.js` → pencil icon → paste the Payment Link
between the quotes on the `paymentLink` line → **Commit changes**.

### 6. Test it
1. Open braiain.com → **Feed the model** → pay with card `4242 4242 4242 4242`,
   any future date, any CVC.
2. GitHub repo → **Actions** → **Train brAIAIn** → **Run workflow** (or wait up to ~10 min).
3. When the run is green, wait a minute for Pages, then reload the Models page.
   brAIAIn-1 should be there with your sentence.

### 7. Go live
Switch Stripe to live mode and repeat steps 2 and 3 (live product, live link, live
restricted key `rk_live_`). Replace the `STRIPE_SECRET_KEY` secret, the
`STRIPE_PAYMENT_LINK` variable and the link in `assets/config.js`.
Test purchases are never mixed in, because test and live keys see different data.

### 8. The API (optional, free)
1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Create Worker** →
   name it `braiain-api` → **Deploy**.
2. **Edit code** → delete what's there → paste all of `api/worker.js` → **Deploy**.
3. Copy the Worker address (`https://braiain-api.<you>.workers.dev`) into the
   `apiBase` line of `assets/config.js` and commit. The API page then shows it as live.

## Running it

- **Remove a sentence:** find its id (like `s00012`) in `data/corpus.json`, add it on
  its own line to `data/removed.txt`, commit. The Action retrains every version
  without it, blanks it from the public files, and replaces that version's notes
  with a recall notice. (It still exists in git history.)
- **Refunds:** sentences that fail screening are listed in `data/state.json` under
  `rejected`, with the last 6 characters of the Stripe checkout id. No text is
  stored. Find it in Stripe → **Payments** and refund.
- **Block more words:** add them to `data/blocklist.txt`, one per line.
- **Change the price:** change the Stripe price, then the `price` line in `assets/config.js`.

## Limits worth knowing

- GitHub runs scheduled jobs late sometimes, so a new version can take longer than 10 minutes.
- GitHub pauses scheduled jobs in a repo with no commits for 60 days. The Action
  commits a heartbeat once a day to keep that from happening.
- At most 25 new sentences are processed per run; the rest wait for the next run.
