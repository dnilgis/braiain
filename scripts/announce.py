"""After each training run: announce new versions on X and Bluesky, and sync the
Hugging Face Space and model repo. Standard library, plus huggingface_hub for the sync.

Every channel is optional and independent. A channel with no secrets is skipped and its
pointer moves forward, so adding secrets later never posts a backlog.

Secrets (repo Settings > Secrets and variables > Actions):
  X:            X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET
  Bluesky:      BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  Hugging Face: HF_TOKEN (write access to braiain-lab/brAIAIn and braiain-lab/brAIAIn-0)
"""
import base64, hashlib, hmac, json, os, re, secrets, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITE = "https://www.braiain.com"
HF_SPACE = "braiain-lab/brAIAIn"
HF_MODEL = "braiain-lab/brAIAIn-0"
MAX_POSTS_PER_RUN = 3


def load(name, default):
    try:
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def env(name):
    return os.environ.get(name, "").strip()


# ---------------------------------------------------------------- text

def post_text(e, limit, url_len=None):
    """One release announcement. url_len: how the network counts the link (X counts 23)."""
    v = e["v"]
    url = SITE + "/v/" + str(v) + ".html"
    name = "brAIAIn-" + str(v) + (' "' + e["codename"] + '"' if e.get("codename") else "")
    notes = e.get("notes") or {}
    issue = (notes.get("known_issues") or [""])[0]
    by = e.get("by") or "Anonymous"
    head = name + " is out.\n\nTrained on one sentence, from " + by + ":\n"
    tail = ("\n\nKnown issue: " + issue if issue else "") + "\n\n" + url
    count = lambda t: len(t) - (len(url) - url_len if url_len else 0)
    sentence = e.get("sentence") or ""
    quoted = '"' + sentence + '"'
    if count(head + quoted + tail) > limit:
        tail = "\n\n" + url  # drop the joke before cutting the buyer's sentence
    room = limit - count(head + tail)
    if len(quoted) > room:
        quoted = '"' + sentence[:max(0, room - 5)].rstrip() + '..."'
    return head + quoted + tail, url


# ---------------------------------------------------------------- X

def x_post(text):
    keys = [env("X_API_KEY"), env("X_API_SECRET"), env("X_ACCESS_TOKEN"), env("X_ACCESS_SECRET")]
    if not all(keys):
        return None
    ck, cs, tk, ts = keys
    url = "https://api.x.com/2/tweets"
    oauth = {"oauth_consumer_key": ck, "oauth_nonce": secrets.token_hex(16), "oauth_signature_method": "HMAC-SHA1",
             "oauth_timestamp": str(int(time.time())), "oauth_token": tk, "oauth_version": "1.0"}
    q = lambda s: urllib.parse.quote(s, safe="~")
    params = "&".join(q(k) + "=" + q(oauth[k]) for k in sorted(oauth))
    base = "POST&" + q(url) + "&" + q(params)
    sig = base64.b64encode(hmac.new((q(cs) + "&" + q(ts)).encode(), base.encode(), hashlib.sha1).digest()).decode()
    oauth["oauth_signature"] = sig
    auth = "OAuth " + ", ".join(q(k) + '="' + q(oauth[k]) + '"' for k in sorted(oauth))
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), method="POST",
                                 headers={"Authorization": auth, "Content-Type": "application/json",
                                          "User-Agent": "braiain-announcer/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("data", {}).get("id", "ok")


# ---------------------------------------------------------------- Bluesky

def bsky_post(text, url, image=None, alt=""):
    handle, pw = env("BLUESKY_HANDLE").lstrip("@"), env("BLUESKY_APP_PASSWORD")
    if not handle or not pw:
        return None

    def call(path, body, token=None):
        h = {"Content-Type": "application/json", "User-Agent": "braiain-announcer/1.0"}
        if token:
            h["Authorization"] = "Bearer " + token
        req = urllib.request.Request("https://bsky.social/xrpc/" + path, data=json.dumps(body).encode(), method="POST", headers=h)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    sess = call("com.atproto.server.createSession", {"identifier": handle, "password": pw})
    blob = None
    if image:
        with open(image, "rb") as f:
            req = urllib.request.Request("https://bsky.social/xrpc/com.atproto.repo.uploadBlob", data=f.read(), method="POST",
                                         headers={"Content-Type": "image/png", "Authorization": "Bearer " + sess["accessJwt"],
                                                  "User-Agent": "braiain-announcer/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = json.load(r)["blob"]
    b = text.encode("utf-8")
    start = b.find(url.encode("utf-8"))
    record = {"$type": "app.bsky.feed.post", "text": text,
              "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    if start >= 0:  # make the link clickable
        record["facets"] = [{"index": {"byteStart": start, "byteEnd": start + len(url.encode("utf-8"))},
                             "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}]}]
    if blob:
        record["embed"] = {"$type": "app.bsky.embed.images", "images": [{"alt": alt[:1000], "image": blob,
                           "aspectRatio": {"width": 1600, "height": 900}}]}
    res = call("com.atproto.repo.createRecord", {"repo": sess["did"], "collection": "app.bsky.feed.post", "record": record},
               sess["accessJwt"])
    return res.get("uri", "ok")


# ---------------------------------------------------------------- email

def gmail_send(messages):
    """messages: list of EmailMessage. Sends with the GMAIL_ADDRESS app password. Returns count sent."""
    import smtplib
    addr, pw = env("GMAIL_ADDRESS"), env("GMAIL_APP_PASSWORD").replace(" ", "")
    if not addr or not pw or not messages:
        return 0
    n = 0
    try:
        srv = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
        srv.login(addr, pw)
        for m in messages:
            m["From"] = "BRAIAIN Laboratories <" + addr + ">"
            try:
                srv.send_message(m)
                n += 1
            except Exception as e:
                print("could not send email", type(e).__name__, file=sys.stderr)
        srv.quit()
    except Exception as e:
        print("could not sign in to Gmail", type(e).__name__, file=sys.stderr)
    return n


def buyer_emails(versions):
    """Tell each buyer their version is live. Only from a separate lab Gmail, so the owner's
    own address is never shown to buyers: needs REVIEW_EMAIL_TO set to a different address
    than GMAIL_ADDRESS."""
    path = os.path.join(ROOT, ".buyer_notices.json")
    try:
        notices = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return
    os.remove(path)
    sender, owner = env("GMAIL_ADDRESS").lower(), env("REVIEW_EMAIL_TO").lower()
    # Buyers see the sender address, so only a lab account may send to them.
    if not sender or not owner or sender == owner or "braiain" not in sender.split("@")[0]:
        print("buyer emails skipped: they need a lab Gmail with braiain in its name (GMAIL_ADDRESS) and REVIEW_EMAIL_TO", file=sys.stderr)
        return
    from email.message import EmailMessage
    vmap = {e["v"]: e for e in versions}
    out = []
    for n in notices:
        e = vmap.get(n["v"])
        if not e or e.get("removed"):
            continue
        v = e["v"]
        name = "brAIAIn-" + str(v) + (' "' + e["codename"] + '"' if e.get("codename") else "")
        share = SITE + "/v/" + str(v) + ".html"
        post = name + ' is out. Trained on one sentence: "' + (e.get("sentence") or "") + '"'
        lines = [
            "Your sentence is now in the training data of " + name + ".",
            "",
            '"' + (e.get("sentence") or "") + '"',
            "",
            "It now knows " + str(e.get("metrics", {}).get("vocabulary", "")) + " words. Some of them are yours.",
            "",
            "Release notes:  " + SITE + "/model.html?v=" + str(v),
            "Your release card (image):  " + SITE + "/card.html?v=" + str(v),
        ]
        if n.get("tier") in ("certificate", "naming"):
            lines.append("Your certificate:  " + SITE + "/certificate.html?v=" + str(v))
        lines += [
            "",
            "Share it. This link shows your release card in the preview:",
            "  " + share,
            "  Post on X:  https://twitter.com/intent/tweet?text=" + urllib.parse.quote(post) + "&url=" + urllib.parse.quote(share + "?ref=x"),
            "  Post on Bluesky:  https://bsky.app/intent/compose?text=" + urllib.parse.quote(post + " " + share + "?ref=bsky"),
            "",
            "If it was a gift, forward this email.",
            "",
            "BRAIAIN Laboratories",
            "A parody AI lab. The model is real. Questions: hello@braiain.com",
        ]
        m = EmailMessage()
        m["Subject"] = name + " is live. You trained it."
        m["To"] = n["email"]
        m["Reply-To"] = "hello@braiain.com"
        m.set_content("\n".join(lines))
        out.append(m)
    sent = gmail_send(out)
    print("buyer emails sent:", sent, "of", len(out))


def daily_email():
    """Email the owner today's brAIAIn answers image with a caption ready to post."""
    path = os.path.join(ROOT, ".daily_notice.json")
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    os.remove(path)
    caption = d["q"] + "\n\nbrAIAIn-" + str(d["v"]) + ': "' + d["a"] + '"\n\n' + SITE.replace("https://www.", "") + "/daily.html?ref=x"
    owner = env("REVIEW_EMAIL_TO") or env("GMAIL_ADDRESS")
    if owner:
        from email.message import EmailMessage
        m = EmailMessage()
        m["Subject"] = "brAIAIn answers, " + d["date"] + ": ready to post"
        m["To"] = owner
        m.set_content("Today's post. Copy the caption, attach the image.\n\n---\n" + caption + "\n---\n")
        if d.get("png") and os.path.exists(d["png"]):
            with open(d["png"], "rb") as f:
                m.add_attachment(f.read(), maintype="image", subtype="png", filename="brAIAIn-answers-" + d["date"] + ".png")
        print("daily email sent" if gmail_send([m]) else "daily email not sent")
    return d, caption


# ---------------------------------------------------------------- Hugging Face

def hf_files():
    """What the Space and the model repo should contain right now."""
    space = {}
    for n in ("versions.json", "corpus.json", "training.json"):
        with open(os.path.join(DATA, n), "rb") as f:
            space["data/" + n] = f.read()
    with open(os.path.join(ROOT, "assets", "braiain.js"), encoding="utf-8") as f:
        space["braiain.js"] = f.read().replace('"/data/', '"data/').encode()
    with open(os.path.join(ROOT, "hf", "space", "index.html"), "rb") as f:
        space["index.html"] = f.read()
    model = {}
    with open(os.path.join(ROOT, "models", "latest.json"), "rb") as f:
        model["brAIAIn-latest.json"] = f.read()
    with open(os.path.join(ROOT, "hf", "model", "generate.py"), "rb") as f:
        model["generate.py"] = f.read()
    corpus = load("corpus.json", [])
    lines = ["# brAIAIn training data: the Ten Instructions, then every published paid sentence, one per line.",
             "# Each version also trains on a few passages written by the version before it (not listed here).",
             "# Public domain (CC0). Live copy: https://www.braiain.com/data.html"]
    with open(os.path.join(DATA, "genesis.txt"), encoding="utf-8") as f:
        lines += [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    removed = set()
    if os.path.exists(os.path.join(DATA, "removed.txt")):
        for l in open(os.path.join(DATA, "removed.txt"), encoding="utf-8"):
            parts = l.split("#", 1)[0].split()
            if parts:
                removed.add(parts[0])
    lines += [c["text"] for c in corpus if c.get("id") not in removed and c.get("text")]
    model["training-data.txt"] = ("\n".join(lines) + "\n").encode()
    card = model_readme()
    if card:
        model["README.md"] = card
    return space, model


def model_readme():
    """Model card for the Hugging Face model repo, rebuilt from the latest version."""
    versions = load("versions.json", [])
    if isinstance(versions, dict):
        versions = versions.get("versions", [])
    versions = [e for e in versions if not e.get("removed")]
    if not versions:
        return None
    L = max(versions, key=lambda e: int(e["v"]))
    v, m = int(L["v"]), L.get("metrics") or {}
    ev = L.get("evals") or {}
    if not ev.get("gsm8k"):  # newest eval may still be running; use the newest finished one
        done = [e for e in versions if (e.get("evals") or {}).get("gsm8k")]
        if done:
            ev = max(done, key=lambda e: int(e["v"]))["evals"]
    ev_v = ev.get("version", v)
    g, t = ev.get("gsm8k") or {}, ev.get("truthfulqa_mc1") or {}
    name = "brAIAIn-%d" % v
    src = "  source:\n    name: BRAIAIN Laboratories (self-reported)\n    url: https://www.braiain.com/benchmarks.html\n"
    idx = ""
    if g:
        idx += ("- task:\n    type: text-generation\n    name: Text Generation\n  dataset:\n    name: GSM8K\n    type: openai/gsm8k\n"
                "    config: main\n    split: test\n  metrics:\n  - type: accuracy\n    value: %s\n    name: accuracy\n" % g.get("score", 0) + src)
    if t:
        idx += ("- task:\n    type: text-generation\n    name: Text Generation\n  dataset:\n    name: TruthfulQA (MC1)\n    type: truthfulqa/truthful_qa\n"
                "    config: multiple_choice\n    split: validation\n  metrics:\n  - type: mc1\n    value: %s\n    name: mc1 (per-word log-likelihood)\n" % t.get("score", 0) + src)
    front = "---\nlanguage:\n- en\nlicense: cc0-1.0\npipeline_tag: text-generation\ntags:\n- parody\n- n-gram\n- trigram\n- tiny\n"
    if idx:
        front += "model-index:\n- name: %s\n  results:\n" % name + "\n".join("  " + l if l else l for l in idx.rstrip("\n").split("\n")) + "\n"
    front += "---\n\n"
    cn = (' "%s"' % L["codename"]) if L.get("codename") else ""
    rows = ["| GSM8K (%s questions) | %s%% (%s correct) | |" % (format(g.get("n", 0), ","), g.get("score", 0), g.get("correct", 0))] if g else []
    if t:
        rows.append("| TruthfulQA MC1 (%s questions), per-word log-likelihood | %s%% (%s correct) | Random %s%%, always pick the shortest answer %s%%, always pick the longest %s%% |"
                    % (t.get("n"), t.get("score"), t.get("correct"), t.get("chance"), t.get("always_shortest"), t.get("always_longest")))
        if t.get("score_raw") is not None:
            rows.append("| TruthfulQA MC1, raw log-likelihood | %s%% | Random %s%% |" % (t["score_raw"], t.get("chance")))
    evnote = "" if ev_v == v else " These scores are for brAIAIn-%s; the latest version is still being tested." % ev_v
    body = """# %(name)s%(cn)s

A language model trained on %(sent)d sentences.

Talk to it: [brAIAIn on Spaces](https://huggingface.co/spaces/braiain-lab/brAIAIn) · Train it: [braiain.com](https://www.braiain.com)

brAIAIn is a word trigram model from [BRAIAIN Laboratories](https://www.braiain.com), a parody AI lab. It is real. It is very small. Anyone can pay $5 to add one sentence to its training data, and every sentence trains a new version. This repository always holds the latest one. It is named brAIAIn-0 because that is the name it was born with.

## Model details

| | |
|---|---|
| Version | %(name)s |
| Architecture | Autoregressive count-based trigram language model with backoff |
| Context window | 2 tokens |
| Parameters | %(params)s (distinct word pairs and triples counted) |
| Vocabulary | %(vocab)s words, whitespace tokenizer |
| Training data | The Ten Instructions (%(found)s lines), %(paid)s paid sentences, plus text written by the previous version (%(selfp)s%% of training tokens) |
| Training tokens | %(tokens)s |
| Gradient steps | 0 |
| Alignment | RLHF (Reinforcement Learning from Human Funding) |

## Evaluation

Measured by the lab on every question in each benchmark. Self-reported.%(evnote)s

| Benchmark | brAIAIn | Reference |
|---|---|---|
%(rows)s

brAIAIn beats random on TruthfulQA only because per-word scoring favours longer answers. Picking the longest answer every time does about as well.

## Usage

```
python generate.py "What is the meaning of life?"
```

No dependencies. Or talk to it in a browser at [braiain.com](https://www.braiain.com).

## Files

- `brAIAIn-latest.json`: the weights. `tri` maps two words to counts of the next word, `bi` maps one word to counts of the next, `uni` holds word counts.
- `training-data.txt`: the human-written training data. Each version also trains on a few passages written by the version before it.
- `generate.py`: a sampler.

## Intended use

None.

## Out-of-scope use

Everything else.

## Limitations

Yes.

## Version history

Every version's release notes, metrics and training data: [braiain.com/models.html](https://www.braiain.com/models.html)

## About

BRAIAIN Laboratories is a parody of AI companies and is not affiliated with any real AI company. Contact: hello@braiain.com
""" % dict(name=name, cn=cn, sent=m.get("sentences", 0), params=format(m.get("parameters", 0), ","), vocab=m.get("vocabulary", 0),
            found=m.get("founding_sentences", 12), paid=m.get("paid_sentences", 0), selfp=m.get("self_consumption", 0),
            tokens=format(m.get("training_tokens", 0), ","), rows="\n".join(rows), evnote=evnote)
    return (front + body).encode()


def digest(files):
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(k.encode() + b"\0" + hashlib.sha256(files[k]).digest())
    return h.hexdigest()[:16]


def hf_sync(state):
    token = env("HF_TOKEN")
    if not token:
        return
    space, model = hf_files()
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    for repo, kind, files, key in ((HF_SPACE, "space", space, "hf_space"), (HF_MODEL, "model", model, "hf_model")):
        d = digest(files)
        if state.get(key) == d:
            continue
        try:
            api.create_commit(repo_id=repo, repo_type=kind, commit_message="Sync from braiain.com",
                              operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()])
            state[key] = d
            print("synced Hugging Face", kind)
        except Exception as e:
            print("could not sync Hugging Face", kind, type(e).__name__, str(e)[:200], file=sys.stderr)


# ---------------------------------------------------------------- main

def main():
    versions = load("versions.json", [])
    if isinstance(versions, dict):
        versions = versions.get("versions", [])
    latest = max((e["v"] for e in versions), default=0)
    state = load("announce.json", {})
    changed = False
    for chan, limit, url_len, fn in (("x", 280, 23, x_post), ("bluesky", 300, None, bsky_post)):
        key = "posted_" + chan
        if key not in state:  # first run: start from here, never post history
            state[key] = latest
            changed = True
            continue
        todo = [e for e in versions if e["v"] > state[key] and not e.get("removed") and e["v"] > 0]
        for e in todo[:MAX_POSTS_PER_RUN]:
            text, url = post_text(e, limit, url_len)
            try:
                res = fn(text) if chan == "x" else fn(text, url)
            except Exception as ex:
                body = ""
                if hasattr(ex, "read"):
                    try:
                        body = ex.read().decode()[:300]
                    except Exception:
                        pass
                print("could not post to", chan, "v" + str(e["v"]), type(ex).__name__, getattr(ex, "code", ""), body, file=sys.stderr)
                break  # try again next run
            if res is None:  # channel not set up: move on silently
                state[key] = latest
                changed = True
                break
            print("posted to", chan, "brAIAIn-" + str(e["v"]))
            state[key] = e["v"]
            changed = True
        if not todo and state[key] < latest:
            state[key] = latest
            changed = True
    buyer_emails(versions)
    dd = daily_email()
    if dd and env("BLUESKY_HANDLE") and state.get("posted_daily") != dd[0]["date"]:
        d, caption = dd
        try:
            url = SITE + "/daily.html?ref=bsky"
            body = d["q"] + "\n\nbrAIAIn-" + str(d["v"]) + ': "' + d["a"] + '"'
            room = 300 - len(url) - 2
            if len(body) > room:
                body = body[:room - 4].rstrip() + '..."'
            text = body + "\n\n" + url
            if bsky_post(text, url, d.get("png") if d.get("png") and os.path.exists(d["png"]) else None,
                         "brAIAIn answers: " + d["q"] + " " + d["a"]):
                state["posted_daily"] = d["date"]
                changed = True
                print("posted daily answer to bluesky")
        except Exception as ex:
            print("could not post daily answer to bluesky", type(ex).__name__, file=sys.stderr)
    before = dict(state)
    try:
        hf_sync(state)
    except Exception as ex:  # never let the mirror stop announcements or emails
        print("could not sync Hugging Face", type(ex).__name__, file=sys.stderr)
    if changed or state != before:
        with open(os.path.join(DATA, "announce.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
