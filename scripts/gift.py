"""Gifts the recipient redeems: the giver pays, the recipient writes the sentence.

The giver buys from a gift Payment Link (giftLink, giftLinkCertificate, giftLinkNaming in
assets/config.js), with optional custom fields "Recipient's name" and "Your name". For each
paid gift this script creates a single-use Stripe promotion code, 100% off the matching tier's
regular Payment Link, and emails the giver a link to the printable ticket. The recipient
scans the ticket, writes a sentence, and checks out on the regular link for $0.00. train.py
then reads that checkout like any other, at the tier the giver paid for.

Codes are derived from the session id with the Stripe key, so a rerun recreates the same code
instead of a second one. They are never written to the repo: data/gifts.json holds only
hashes, tiers and Stripe promotion-code ids. Gifts are not refundable; if one is refunded or
disputed anyway, its code is switched off if it has not been used.

The restricted key needs Coupons: Write and Promotion Codes: Write on top of what train.py uses.
"""
import hashlib, hmac, json, os, re, sys, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import mailer

ROOT, DATA = T.ROOT, T.DATA
PATH = os.path.join(DATA, "gifts.json")
SITE = "https://www.braiain.com"
STRIPE_VERSION = "2024-06-20"  # pinned: newer API versions renamed the promotion-code fields
TIERS = (("giftLink", "paymentLink", "sentence"),
         ("giftLinkCertificate", "paymentLinkCertificate", "certificate"),
         ("giftLinkNaming", "paymentLinkNaming", "naming"),
         # Gift links at the old prices ($5, $15, $50), still read so a ticket bought from an old page is honored.
         ("giftLink5", "paymentLink", "sentence"),
         ("giftLinkCertificate15", "paymentLinkCertificate", "certificate"),
         ("giftLinkNaming50", "paymentLinkNaming", "naming"))
TIER_NAME = {"sentence": "The Minnow (one sentence)", "certificate": "The Shark (sentence and certificate)",
             "naming": "The Whale (sentence and codename)"}
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def config():
    try:
        text = open(os.path.join(ROOT, "assets", "config.js"), encoding="utf-8").read()
    except OSError:
        return {}
    return {k: v.strip().rstrip("/") for k, v in re.findall(r'(\w+):\s*"([^"]*)"', text)}


def stripe(key, path, params=None, method="GET"):
    data = urllib.parse.urlencode(params or {}, doseq=True)
    url = "https://api.stripe.com/v1/" + path + ("?" + data if method == "GET" and data else "")
    req = urllib.request.Request(url, data=data.encode() if method == "POST" else None, method=method,
                                 headers={"Authorization": "Bearer " + key, "Stripe-Version": STRIPE_VERSION})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def code_for(key, session_id):
    d = hmac.new(key.encode(), ("gift:" + session_id).encode(), hashlib.sha256).digest()
    s = "".join(ALPHABET[b % len(ALPHABET)] for b in d[:8])
    return "GIFT-" + s[:4] + "-" + s[4:]


def field(sess, words):
    for f in sess.get("custom_fields") or []:
        label = ((f.get("label") or {}).get("custom") or "").lower()
        k = (f.get("key") or "").lower()
        if any(w in k or w in label for w in words):
            return ((f.get("text") or {}).get("value") or "").strip()
    return ""


def links_for(code, tier, to, frm):
    q = {"code": code, "t": tier}
    if to:
        q["to"] = to
    if frm:
        q["from"] = frm
    qs = urllib.parse.urlencode(q)
    return SITE + "/gifts.html?" + qs + "#g-card", SITE + "/redeem.html?" + qs


def email(to, subject, lines):
    import smtplib
    from email.message import EmailMessage
    addr = os.environ.get("GMAIL_ADDRESS", "").strip()
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    if not addr or not pw or not to:
        print("gift email not sent: Gmail secrets or address missing", file=sys.stderr)
        return False
    m = EmailMessage()
    m["Subject"], m["From"], m["To"] = subject, "BRAIAIN Laboratories <" + addr + ">", to
    m["Reply-To"] = "hello@braiain.com"
    m.set_content("\n".join(lines))
    try:
        s = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
        s.login(addr, pw)
        s.send_message(m)
        s.quit()
        return True
    except Exception as e:
        print("could not send gift email", type(e).__name__, file=sys.stderr)
        return False


def lab_sender():
    """Buyers see the sending address, so only a lab Gmail may write to them (same rule as
    announce.py buyer emails): braiain in the address, and not the owner's own inbox."""
    sender = os.environ.get("GMAIL_ADDRESS", "").strip().lower()
    own = os.environ.get("REVIEW_EMAIL_TO", "").strip().lower()
    return bool(sender) and bool(own) and sender != own and "braiain" in sender.split("@")[0]


def owner():
    return os.environ.get("REVIEW_EMAIL_TO", "").strip() or os.environ.get("GMAIL_ADDRESS", "").strip()


def coupon(key, state, tier, plink):
    if state["coupons"].get(tier):
        return state["coupons"][tier]
    items = stripe(key, "payment_links/" + plink + "/line_items", {"limit": "1"})
    product = items["data"][0]["price"]["product"]
    c = stripe(key, "coupons", {"percent_off": "100", "duration": "once", "name": "brAIAIn gift: " + tier,
                                "applies_to[products][]": [product], "metadata[braiain]": "gift-" + tier}, "POST")
    state["coupons"][tier] = c["id"]
    print("created gift coupon for", tier)
    return c["id"]


def promo(key, cpn, code, h):
    try:
        return stripe(key, "promotion_codes", {"coupon": cpn, "code": code, "max_redemptions": "1",
                                              "metadata[gift]": h}, "POST")["id"]
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        found = stripe(key, "promotion_codes", {"code": code, "limit": "1"}).get("data") or []
        if found:  # created on an earlier run that did not get to save
            return found[0]["id"]
        raise


def main():
    cfg = config()
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    mock = os.environ.get("GIFT_MOCK")
    state = T.load("gifts.json", {})
    state.setdefault("coupons", {})
    state.setdefault("gifts", {})
    before = json.dumps(state, sort_keys=True)
    tiers = [(cfg.get(g, ""), cfg.get(p, ""), t) for g, p, t in TIERS if cfg.get(g)]
    if not tiers or not (key or mock):
        return
    if mock:  # tests: {"sessions": {tier: [session, ...]}}; Stripe calls are stubbed by the test
        by_tier = json.load(open(mock, encoding="utf-8"))["sessions"]
    else:
        by_tier = {}
        try:
            ids, missing = T.resolve_links(key, [g for g, _, _ in tiers] + [p for _, p, _ in tiers if p])
        except urllib.error.HTTPError as e:
            print("Stripe error listing Payment Links", e.code, file=sys.stderr)
            return
        if missing:
            print("gift or tier link not found in this Stripe account:", ", ".join(missing), file=sys.stderr)
        id_of = dict(zip([u for u in [g for g, _, _ in tiers] + [p for _, p, _ in tiers if p] if u not in missing], ids))
        for g, p, t in tiers:
            if g in id_of:
                by_tier.setdefault(t, []).extend(T.stripe_sessions(key, [id_of[g]]))
        state["_plinks"] = {t: id_of.get(p, "") for _, p, t in tiers}
    plinks = state.pop("_plinks", {})
    for tier, sessions in by_tier.items():
        for s in sessions:
            h = T.session_hash(s["id"])
            g = state["gifts"].get(h)
            if T.refunded(s):
                if g and g.get("active"):
                    try:
                        p = stripe(key, "promotion_codes/" + g["promo"])
                        if not p.get("times_redeemed"):
                            stripe(key, "promotion_codes/" + g["promo"], {"active": "false"}, "POST")
                            print("gift refunded or disputed: code switched off", h[:8])
                        g["active"] = False
                    except urllib.error.HTTPError as e:
                        print("could not switch off a refunded gift code", e.code, file=sys.stderr)
                continue
            to = T.clean_name(field(s, ("recipient",)))[:40]
            frm = T.clean_name(field(s, ("your name", "from", "giver")))[:40]
            code = code_for(key or "mock", s["id"])
            if not g:
                try:
                    cpn = coupon(key, state, tier, plinks.get(tier, "")) if not mock else "mock_" + tier
                    pid = promo(key, cpn, code, h) if not mock else "promo_mock_" + h[:6]
                except (urllib.error.HTTPError, KeyError, IndexError) as e:
                    print("could not create gift code", tier, getattr(e, "code", type(e).__name__), file=sys.stderr)
                    continue
                g = state["gifts"][h] = {"tier": tier, "promo": pid, "active": True, "emailed": False,
                                         "at": datetime.fromtimestamp(s.get("created", 0), timezone.utc).strftime("%Y-%m-%d")}
                print("gift code created", tier, h[:8])
            if not g.get("emailed"):
                ticket, redeem = links_for(code, tier, to, frm)
                buyer = ((s.get("customer_details") or {}).get("email") or "").strip()
                subject = "Your brAIAIn gift ticket" + (" for " + to if to else "")
                body = [
                    "Thank you for giving " + (to or "someone") + " a sentence in brAIAIn.", "",
                    "Gift: " + TIER_NAME[tier],
                    "Gift code: " + code + "  (works once)", "",
                    "Print the ticket (it has a QR code that opens the redeem page):",
                    "  " + ticket, "",
                    "Or send it: forward this link to " + (to or "them") + ":",
                    "  " + redeem, "",
                    "They write their own sentence and check out for $0.00. No card needed.",
                    "Their version usually goes live within minutes of that.", "",
                    "Keep this email. Anyone with the code can redeem it, once. Gifts are not refundable.", "",
                    "BRAIAIN Laboratories", SITE]
                if mailer.ready() and buyer:  # from hello@braiain.com, so no personal address reaches the buyer
                    ok = mailer.send(buyer, subject, "\n".join(body), mailer.gift_ticket(to, TIER_NAME[tier], code, ticket, redeem))
                    if not ok:  # Resend refused or was unreachable: the owner forwards it by hand rather than the buyer waiting
                        ok = email(owner(), "brAIAIn: FORWARD this gift ticket to " + buyer + " (sending from hello@ failed)",
                                   ["Resend did not send this gift ticket. Forward everything below the line to: " + buyer, "",
                                    "Subject: " + subject, "", "-" * 40] + body)
                    elif ok:
                        email(owner(), "brAIAIn: gift sold (" + TIER_NAME[tier] + ")",
                              ["A gift was bought and its ticket emailed to the buyer from hello@braiain.com.", "Tier: " + tier,
                               "Stripe promotion code id: " + g["promo"]])
                elif lab_sender():
                    ok = email(buyer, subject, body)
                    if ok:
                        email(owner(), "brAIAIn: gift sold (" + TIER_NAME[tier] + ")",
                              ["A gift was bought and its ticket emailed to the buyer.", "Tier: " + tier,
                               "Stripe promotion code id: " + g["promo"]])
                else:  # no lab sender: the owner forwards it, so the owner's address never reaches a buyer unasked
                    ok = email(owner(), "brAIAIn: FORWARD this gift ticket to " + (buyer or "the buyer (email not found)"),
                               ["A gift was bought. Forward everything below the line to: " + (buyer or "(no email on the checkout; find it in Stripe)"),
                                "from hello@braiain.com or another lab address, not your personal one.",
                                "Subject: " + subject, "", "-" * 40] + body)
                if ok:
                    g["emailed"] = True
    if json.dumps(state, sort_keys=True) != before or not os.path.exists(PATH):
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
