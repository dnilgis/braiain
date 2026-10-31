# BRAIAIN

A parody AI lab with a real, very small language model. brAIAIn is trained only on
a twelve-line founding text (`data/genesis.txt`, the Ten Instructions) plus one
sentence per paying customer. Every approved sentence ships a new model version with
release notes, a paper and a status incident.

Everything below is done in a browser. No terminal.

Code, model and training data are public domain (CC0, see `LICENSE`).

## How it works

- `scripts/train.py` runs every 10 minutes in GitHub Actions (`.github/workflows/train.yml`),
  and right away whenever you edit `data/approved.txt`, `declined.txt` or `removed.txt`.
  It reads paid Stripe checkouts, skips refunded ones, screens each sentence, holds it
  for your approval, retrains every version, writes the notes, and commits. GitHub
  Pages republishes the site.
- The website rebuilds any version in the visitor's browser from the public files in
  `data/` and chats with it (`assets/braiain.js`). Nothing is sent to a server.
- `api/worker.js` is the API, run as a free Cloudflare Worker (optional).
- The model is a word trigram model: it counts which word follows each pair of words
  and writes by weighted dice. Each version is also trained on a few passages written
  by the version before it (screened first), and the newest sentence counts four times.

## The three tiers

| Tier | Price | Buyer gets |
|---|---|---|
| Contributor | $5 | one sentence, a numbered release, credit or Anonymous |
| Certified | $15 | the above, plus a printable certificate (`/certificate.html?v=N`) and a thank-you in the paper |
| Naming rights | $50 | the above, plus a codename for that version, reviewed like the sentence |

The trainer tells the tiers apart by the amount paid, so each tier is its own Payment
Link with its own price.

## Before launch: fill these in

`assets/config.js` (pencil icon on GitHub, then **Commit changes**):
- `operator`: your LLC's legal name, exactly as registered. It appears on the Terms and
  Privacy pages. Stripe expects the seller to be identifiable.
- `paymentLink`, `paymentLinkCertificate`, `paymentLinkNaming`: the three
  `https://buy.stripe.com/...` links. **Buying stays closed until `opensAt` even if these
  are filled in early**, so paste them in ahead of time and the site flips on its own
  at 9:00 pm Central.

## Teaser mode

Until `opensAt` has passed and a payment link exists, the buy buttons say **Join the
waitlist** and a countdown runs. If the countdown ends before a link is set, it says
"opening shortly" and the waitlist stays up.

The waitlist goes to Formspree (`waitlistEndpoint`). The site promises one email when
training opens, then deletes the list. When you send it, include a mailing address
and a line like "you will not hear from us again" (US anti-spam rules expect both),
then delete the submissions in Formspree.

## One-time Stripe setup

Do steps 1 to 5 in Stripe **test mode** first (toggle at the top of the dashboard).
Step 6 rehearses the whole flow. Step 7 switches to real money.

### 1. Let the Action save its work
Already done and verified: the Action has committed to this repo.

### 2. Products and Payment Links
1. Stripe → **Product catalog** → **Add product** three times:
   `One sentence` $5.00, `One sentence with certificate` $15.00, `Name a version` $50.00. One time, USD.
2. Stripe → **Payment Links** → **New**, once per product. On each:
   - **Options → Add custom fields:**
     - Text field, label `Your sentence`, required. Set the maximum length to 255 if offered.
     - Text field, label `Credit as (optional)`, optional.
     - On the $50 link only: text field, label `Version codename`, required.
     The trainer finds fields by the words *sentence*, *credit* and *codename* in the
     label, so keep those words.
   - **Require customers to accept your terms of service**, if offered, pointing to
     `https://www.braiain.com/terms.html`.
   - **Custom text / description:** "Your sentence and credit are public and reviewed before
     publishing. Refunded if not approved. Parody project; the model is real."
   - **After payment:** **Don't show confirmation page** → redirect to
     `https://www.braiain.com/thanks.html`.
3. Copy each link (`https://buy.stripe.com/...`).
4. Stripe → **Settings → Public details**: statement descriptor `BRAIAIN`.
5. Stripe → **Settings → Customer emails**: turn on successful payment receipts.
   Stripe → **Settings → Team and security → Notifications**: email me about successful
   payments, so you know when something is waiting for review.

### 3. A Stripe key that can only read
Stripe → **Developers → API keys → Create restricted key**, name `braiain-trainer`.
Set to **Read**: **Checkout Sessions**, **Payment Links**, **PaymentIntents**, **Charges**.
Everything else **None**. (Payment Links lets the trainer match the links in `config.js` to
their checkouts. PaymentIntents and Charges let it skip refunded and disputed payments.)

### 4. Give the keys to GitHub
Repo → **Settings → Secrets and variables → Actions**.
- **Secrets:** `STRIPE_SECRET_KEY` = the restricted key. Optional: `ANTHROPIC_API_KEY`
  (otherwise release notes come from built-in templates).
- No variable needed: the trainer reads only the Payment Links listed in `config.js`.
  (Optional override: a variable `STRIPE_PAYMENT_LINK` with the `plink_` IDs, comma-separated.)

If a payment link is in `config.js` but the secret is missing, the Action fails red on
purpose, so you notice before buyers do.

### 5. Put the test links in `config.js`
Paste the three test links and commit. Set `opensAt` a few minutes in the past while
testing, then put it back.

### 6. Rehearse the whole thing (test mode)
1. Buy each tier with card `4242 4242 4242 4242`, any future date, any CVC.
2. Repo → **Actions → Train brAIAIn → Run workflow**. Open the run's log: it lists each
   payment as `awaiting review pi=XXXXXXXX tier=... flags=...`. Status shows the count.
3. Approve one (see below), decline one, and refund the declined one in Stripe.
4. Check the Models page, the model card, and `/certificate.html?v=N` for the certificate tier.
5. To clear test data afterward: add the test versions' ids to `data/removed.txt`.
   (Or ask for a reset before launch.)

### 7. Go live
Switch Stripe to live mode and repeat steps 2 to 5 with live products, links and a live
restricted key (`rk_live_`). Replace the secret, the variable and the three links.
Put `opensAt` back to the launch time. Make one real $5 purchase with your own card,
decline it, and refund it, to prove the live path end to end.

## Reviewing sentences

Clean sentences publish on their own, usually within 10 minutes of payment.

A sentence the filter flags (a possible full name, an address or long number, a threat
word, an advert, a possible minor) waits for you instead. You get an email at your own Gmail
with the sentence, the credit, the codename, and an 8-character code:

- **Publish it:** open the `approved.txt` link in the email, add the code on a new line, commit.
- **Decline it:** add the code to `declined.txt` instead, commit, and refund the payment in Stripe.

The email needs two repo secrets: `GMAIL_ADDRESS` (your Gmail) and `GMAIL_APP_PASSWORD`
(a Google app password, made at myaccount.google.com/apppasswords). Without them, flagged
sentences still wait, but no email is sent; the Action log lists them.

Pending text is never written to the public repo. To hold every sentence for review,
add a repo variable `REQUIRE_APPROVAL` = `true` (Settings → Secrets and variables →
Actions → Variables). `false` holds nothing.

## Automatic announcements

After each run, `scripts/announce.py` posts every new version to X and Bluesky and copies
the newest data to the Hugging Face Space and model repo. Each part switches on when its
secrets exist (Settings → Secrets and variables → Actions):

- **X:** `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_SECRET`. X charges per post
  (about $0.20 for a post with a link). Fund the X developer account first.
- **Bluesky:** `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` (Bluesky → Settings → App passwords). Free.
- **Hugging Face:** `HF_TOKEN`, a token with write access to the braiain-lab repos. Free.

Nothing posts history: a channel switched on later starts from the next new version.
Removed versions are never posted, but a post already made stays up; delete it by hand.

## Running it

- **Remove a sentence** (abuse, a buyer's request, a refund or dispute after publishing):
  add its corpus id (like `s00012`) to `data/removed.txt` and commit. Every version is
  retrained without it and its notes become a recall notice. It still exists in git history.
  The Action log tells you the id if a published payment was later refunded or disputed.
- **Block more words:** add them to `data/blocklist.txt`, one per line.
- **Change prices:** change the Stripe prices, then the price lines in `assets/config.js`.

## Limits worth knowing

- GitHub can run scheduled jobs late. After approving, the push trigger usually runs
  within a minute; if not, click **Run workflow**.
- GitHub pauses schedules in a repo with no commits for 60 days. A daily heartbeat
  commit prevents that.
- At most 25 new sentences are trained per run; the rest wait for the next run.
- Every run retrains every version from scratch, so a run takes longer as sentences
  accumulate (seconds at a few hundred; plan an incremental trainer before about a thousand).
