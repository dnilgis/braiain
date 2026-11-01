"""Share pages, release-card images and the daily "brAIAIn answers" post.

Runs in GitHub Actions after train.py, with the site served at http://localhost:8000
and Google Chrome available (both set up by the workflow). Writes:
  v/N.html     a small page per version with its own link preview, redirecting to the model card
  v/N.png      that version's release card (rendered from /card.html)
  data/daily.json, daily/DATE.png   one question a day, answered by the newest version
  .daily_notice.json   private, for announce.py to email the owner (not committed)
"""
import hashlib, html, json, os, random, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITE = "https://www.braiain.com"
LOCAL = os.environ.get("CARDS_BASE", "http://localhost:8000")
DAILY_HOUR = 7  # Central time; the daily answer is made at the first run after this
CARD_BUDGET = 3 * 60  # seconds of card rendering per run; the rest wait for the next run


def load(name, default):
    try:
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", os.environ.get("CHROME", "")):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def shot(url, out, size="1600,900"):
    exe = chrome()
    if not exe:
        print("no Chrome found; images skipped", file=sys.stderr)
        return False
    tmp = out + ".tmp.png"
    cmd = [exe, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
           "--window-size=" + size, "--virtual-time-budget=15000", "--screenshot=" + tmp, url]
    try:
        subprocess.run(cmd, check=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, TZ="America/Chicago"))
        os.replace(tmp, out)
        return True
    except Exception as e:
        print("screenshot failed", url, type(e).__name__, file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def esc(s):
    return html.escape(str(s or ""), quote=True)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="robots" content="noindex">
<link rel="canonical" href="{SITE}/model.html?v={v}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="BRAIAIN">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{SITE}/v/{v}.html">
{image}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<script>(function(){{var r=(location.search.match(/[?&]ref=([^&]*)/)||[])[1];
if(!r){{var h="";try{{h=new URL(document.referrer).hostname.replace(/^www[.]/,"");}}catch(e){{}}
r={{"t.co":"x","x.com":"x","twitter.com":"x","bsky.app":"bsky","facebook.com":"fb","m.facebook.com":"fb","l.facebook.com":"fb","lm.facebook.com":"fb","reddit.com":"reddit","old.reddit.com":"reddit","out.reddit.com":"reddit","news.ycombinator.com":"hn","linkedin.com":"linkedin","lnkd.in":"linkedin","google.com":"google"}}[h]||(h?h.split(".").slice(-2,-1)[0]:"direct");}}
location.replace("/model.html?v={v}&ref=share_"+encodeURIComponent(r.slice(0,30)));}})();</script>
</head><body style="font-family:system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 16px">
<h1>{title}</h1><p>{desc}</p><p><a href="/model.html?v={v}">Read the release notes</a></p>
</body></html>
"""


def version_pages(versions):
    os.makedirs(os.path.join(ROOT, "v"), exist_ok=True)
    index = {}
    try:
        index = json.load(open(os.path.join(ROOT, "v", "index.json")))
    except (OSError, ValueError):
        pass
    deadline = time.time() + CARD_BUDGET
    left = 0
    for e in sorted(versions, key=lambda x: -x["v"]):  # newest first: the latest card matters most
        v = e["v"]
        name = (e.get("name") or "brAIAIn-" + str(v)) + (' "' + e["codename"] + '"' if e.get("codename") else "")
        if e.get("removed"):
            title, desc = name + " | BRAIAIN", "This version was recalled. Its sentence has been removed."
        elif v == 0:
            title, desc = "brAIAIn-0 | BRAIAIN", "The first version, trained only on the Ten Instructions."
        else:
            title = name + " is out | BRAIAIN"
            desc = ("An internal release. Trained on one sentence the lab wrote itself, unpaid: \"" if e.get("tier") == "lab"
                    else "Trained on one sentence from " + (e.get("by") or "Anonymous") + ': "') + (e.get("sentence") or "") + '"'
        key = hashlib.sha256(json.dumps([title, desc, e.get("metrics", {}).get("vocabulary"), e.get("evals", {})], sort_keys=True).encode()).hexdigest()[:16]
        png = os.path.join(ROOT, "v", f"{v}.png")
        if e.get("removed"):
            if os.path.exists(png):
                os.remove(png)
        elif index.get(str(v)) != key or not os.path.exists(png):
            if time.time() < deadline and shot(f"{LOCAL}/card.html?v={v}&raw=1", png):
                index[str(v)] = key
            else:
                left += 1
        has = os.path.exists(png) and not e.get("removed")
        image = (f'<meta property="og:image" content="{SITE}/v/{v}.png?k={index.get(str(v), "")}">\n'
                 '<meta property="og:image:width" content="1600">\n<meta property="og:image:height" content="900">'
                 if has else f'<meta property="og:image" content="{SITE}/og-image.png?v=2">')
        page = PAGE.format(title=esc(title), desc=esc(desc), v=v, SITE=SITE, image=image)
        with open(os.path.join(ROOT, "v", f"{v}.html"), "w", encoding="utf-8") as f:
            f.write(page)
    if left:
        print(f"release cards: {left} still to render; continuing next run")
    with open(os.path.join(ROOT, "v", "index.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------- daily answer

STOP = set("the a an and is are to of i you it what how why do does can my your me in on for be with that this".split())


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def answer(model, question, rng):
    known = {}
    for w in model["bi"]:
        if w != train.BOS and norm(w):
            known.setdefault(norm(w), []).append(w)
    qwords = {norm(w) for w in question.split() if len(norm(w)) > 2}
    starts = [w for w in (norm(x) for x in question.split()) if w in known and w not in STOP]
    best, best_score = "", -1
    for i in range(8):
        start = None
        if i < 5 and starts:
            start = rng.choice(known[rng.choice(starts)])
        s = train.generate(model, rng, 40, start)
        if not s:
            continue
        words = s.split()
        overlap = len({norm(w) for w in words} & qwords)
        score = overlap * 100 + min(len(words), 30) + (50 if len(words) >= 3 else 0)
        if score > best_score:
            best, best_score = s, score
    return best[:1].upper() + best[1:]


def daily(versions):
    now = datetime.now(ZoneInfo("America/Chicago"))
    today = now.date().isoformat()
    log = load("daily.json", [])
    if any(d["date"] == today for d in log) or now.hour < DAILY_HOUR:
        return
    qs = [l.strip() for l in open(os.path.join(DATA, "daily-questions.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]
    if not qs:
        return
    q = qs[len(log) % len(qs)]
    model = json.load(open(os.path.join(ROOT, "models", "latest.json"), encoding="utf-8"))
    rng = random.Random("braiain-daily-" + today)
    a = answer(model, q, rng)
    if not a or train.screen(a, train.read_lines("blocklist.txt")):
        a = "In the beginning there was no data."
    entry = {"date": today, "q": q, "a": a, "v": model.get("v", 0), "name": model.get("name") or "brAIAIn-%d" % model.get("v", 0)}
    log.append(entry)
    with open(os.path.join(DATA, "daily.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.makedirs(os.path.join(ROOT, "daily"), exist_ok=True)
    png = os.path.join(ROOT, "daily", today + ".png")
    ok = shot(f"{LOCAL}/daily.html?d={today}&raw=1", png)
    with open(os.path.join(ROOT, ".daily_notice.json"), "w", encoding="utf-8") as f:
        json.dump(dict(entry, png=png if ok else ""), f)
    print("daily answer", today, "v" + str(entry["v"]), "image" if ok else "no image")


def certificates(versions):
    """Freeze each Shark and Whale certificate as an image on release day. A frozen certificate is
    never re-rendered, so later redesigns of the page cannot change one already issued. It is
    deleted only if its sentence is removed."""
    path = os.path.join(ROOT, "v", "certs.json")
    issued = load_path(path)
    for e in versions:
        v = e["v"]
        png = os.path.join(ROOT, "v", f"{v}-cert.png")
        if e.get("removed") or e.get("tier") not in ("certificate", "naming") or not e.get("sentence"):
            if os.path.exists(png):
                os.remove(png)
            issued.pop(str(v), None)
            continue
        if os.path.exists(png):
            continue
        if shot(f"{LOCAL}/certificate.html?v={v}&raw=1", png, "1600,1131"):
            issued[str(v)] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            print("certificate frozen", v)
    with open(path, "w") as f:
        json.dump(issued, f, indent=1, sort_keys=True)


def load_path(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    versions = load("versions.json", [])
    certificates(versions)
    version_pages(versions)
    daily(versions)


if __name__ == "__main__":
    main()
