"""Title sponsorship: one slot, held until someone pays more.

Reads paid checkouts from the sponsorship Payment Link (sponsorLink in assets/config.js,
a "customers choose what to pay" link with custom fields "Sponsor name" and "Sponsor line").
A bid takes the slot if it beats the current holder by at least the increment. Bids that
are too low, or whose text fails the screen, are not published and the owner is emailed to
refund them. Flagged text waits for the owner, using the same approval links as sentences.
Public state is written to data/sponsor.json. Nothing private is stored there.
"""
import json, os, re, sys, urllib.error
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = T.ROOT
DATA = T.DATA
PATH = os.path.join(DATA, "sponsor.json")
OPEN_CENTS = 500
STEP_CENTS = 100
NAME_MAX, LINE_MAX = 40, 60


def sponsor_link():
    try:
        text = open(os.path.join(ROOT, "assets", "config.js"), encoding="utf-8").read()
    except OSError:
        return ""
    m = re.search(r'sponsorLink:\s*"([^"]*)"', text)
    return (m.group(1) if m else "").strip().rstrip("/")


def field(sess, words):
    for f in sess.get("custom_fields") or []:
        label = ((f.get("label") or {}).get("custom") or "").lower()
        k = (f.get("key") or "").lower()
        if any(w in k or w in label for w in words):
            return ((f.get("text") or {}).get("value") or "").strip()
    return ""


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def email(subject, lines):
    import smtplib
    from email.message import EmailMessage
    addr = os.environ.get("GMAIL_ADDRESS", "").strip()
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    if not addr or not pw:
        print("sponsor email not sent: Gmail secrets missing", file=sys.stderr)
        return False
    m = EmailMessage()
    m["Subject"], m["From"] = subject, addr
    m["To"] = os.environ.get("REVIEW_EMAIL_TO", "").strip() or addr
    m.set_content("\n".join(lines))
    try:
        s = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
        s.login(addr, pw)
        s.send_message(m)
        s.quit()
        return True
    except Exception as e:
        print("could not send sponsor email", type(e).__name__, file=sys.stderr)
        return False


def flags(name, line):
    # A sponsor name is supposed to be a name, so "looks like a full name" is not a reason to hold it.
    f = [x for x in T.review_flags(name) if x != "possible-full-name"]
    return f + [x for x in T.review_flags(line) if x not in f]


def main():
    state = T.load("sponsor.json", {})
    state.setdefault("current", None)
    state.setdefault("history", [])
    state.setdefault("seen", [])
    state.setdefault("notified", [])
    state["open_cents"], state["step_cents"] = OPEN_CENTS, STEP_CENTS
    before = json.dumps(state, sort_keys=True)
    url = sponsor_link()
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    mock = os.environ.get("SPONSOR_MOCK")
    if mock:
        incoming = json.load(open(mock, encoding="utf-8"))
    elif url and key:
        try:
            ids, missing = T.resolve_links(key, [url])
            if missing:
                print("sponsorLink not found in this Stripe account:", url, file=sys.stderr)
                return
            incoming = T.stripe_sessions(key, ids)
        except urllib.error.HTTPError as e:
            print("Stripe error reading sponsorship bids", e.code, file=sys.stderr)
            return
    else:
        incoming = []
    blocklist = T.read_lines("blocklist.txt")
    approved = {T.first_token(x)[-8:] for x in T.read_lines("approved.txt")}
    declined = {T.first_token(x)[-8:] for x in T.read_lines("declined.txt")}
    for folder, bucket in (("approvals", approved), ("declines", declined)):
        d = os.path.join(DATA, folder)
        if os.path.isdir(d):
            bucket |= {f.rsplit(".", 1)[0][-8:] for f in os.listdir(d) if not f.startswith(".") and f != "README.md"}
    seen, notified = set(state["seen"]), set(state["notified"])
    new = "https://github.com/dnilgis/braiain/new/main/data/"
    for s in incoming:
        h = T.session_hash(s["id"])
        if h in seen:
            continue
        if T.refunded(s):
            seen.add(h)
            continue
        pi = T.pi_id(s)
        code = pi[-8:] or s["id"][-8:]
        ids = {s["id"][-8:], pi[-8:]} - {""}
        cents = int(s.get("amount_total") or 0)
        name = T.clean_name(field(s, ("sponsor name", "name", "sponsor")))[:NAME_MAX] or "Anonymous"
        line = T.clean(field(s, ("line", "tagline", "slogan")))[:LINE_MAX]
        cur = state["current"]
        need = (cur["cents"] + STEP_CENTS) if cur else OPEN_CENTS
        pay = "  https://dashboard.stripe.com/payments/" + pi if pi else "  (find it in Stripe by code " + code + ")"
        reason = T.screen(name, blocklist) or (T.screen(line, blocklist) if line else None)
        if not reason and ids & declined:
            reason = "declined in review"
        if not reason and cents < need:
            reason = "bid of $%.2f was below the minimum of $%.2f when it arrived" % (cents / 100, need / 100)
        if reason:
            seen.add(h)
            if h + "-r" not in notified and email("brAIAIn: REFUND NEEDED, sponsorship bid not accepted (" + code + ")",
                                                 ["A sponsorship bid was not published: " + reason + ".", "Refund it in Stripe:", pay, "",
                                                  "Name: " + name, "Line: " + (line or "-"), "Amount: $%.2f" % (cents / 100)]):
                notified.add(h + "-r")
            print("sponsorship bid not accepted", code, reason.split(" when")[0])
            continue
        fl = flags(name, line)
        if fl and not ids & approved:
            if h not in notified and email("brAIAIn: sponsorship bid to review (" + code + ")",
                                           ["A title sponsorship bid is waiting. The filter flagged it: " + ", ".join(fl) + ".", "",
                                            "Name: " + name, "Line: " + (line or "-"), "Amount: $%.2f" % (cents / 100), "",
                                            "PUBLISH: open this link, then click Commit changes twice.",
                                            "  " + new + "approvals?filename=" + code + ".txt&value=approved", "",
                                            "DECLINE: open this link, click Commit changes twice, then refund it in Stripe:",
                                            "  " + new + "declines?filename=" + code + ".txt&value=declined", pay]):
                notified.add(h)
            print("sponsorship bid awaiting review", code, ",".join(fl))
            continue
        seen.add(h)
        at = datetime.fromtimestamp(s.get("created", 0), timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if cur:
            cur["until"] = at
            state["history"].append(cur)
        state["current"] = {"name": name, "line": line, "cents": cents, "since": at, "sh": h}
        print("new title sponsor", code, "$%.2f" % (cents / 100))
    state["seen"], state["notified"] = sorted(seen), sorted(notified)
    if json.dumps(state, sort_keys=True) != before or not os.path.exists(PATH):
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
