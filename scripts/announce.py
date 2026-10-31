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
    url = SITE + "/model.html?v=" + str(v)
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

def bsky_post(text, url):
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
    b = text.encode("utf-8")
    start = b.find(url.encode("utf-8"))
    record = {"$type": "app.bsky.feed.post", "text": text,
              "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    if start >= 0:  # make the link clickable
        record["facets"] = [{"index": {"byteStart": start, "byteEnd": start + len(url.encode("utf-8"))},
                             "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}]}]
    res = call("com.atproto.repo.createRecord", {"repo": sess["did"], "collection": "app.bsky.feed.post", "record": record},
               sess["accessJwt"])
    return res.get("uri", "ok")


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
    removed = {l.split("#")[0].strip() for l in open(os.path.join(DATA, "removed.txt"), encoding="utf-8")} if os.path.exists(os.path.join(DATA, "removed.txt")) else set()
    lines += [c["text"] for c in corpus if c.get("id") not in removed]
    model["training-data.txt"] = ("\n".join(lines) + "\n").encode()
    return space, model


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
    before = dict(state)
    hf_sync(state)
    if changed or state != before:
        with open(os.path.join(DATA, "announce.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
