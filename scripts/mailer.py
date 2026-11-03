"""Email to buyers, sent from the lab's own address through Resend (resend.com).

Buyers must never see the owner's personal address, so anything sent to a buyer goes through
here, from hello@braiain.com, once the RESEND_API_KEY secret is set (a Resend account with
braiain.com verified as a sending domain). Without the key, ready() is False and callers keep
their old behaviour: the owner gets a copy to forward by hand.

Free plan: 100 emails a day, 3,000 a month. Standard library only.
"""
import json, os, sys, urllib.error, urllib.request

FROM = os.environ.get("MAIL_FROM", "").strip() or "BRAIAIN Laboratories <hello@braiain.com>"
REPLY_TO = "hello@braiain.com"


def ready():
    return bool(os.environ.get("RESEND_API_KEY", "").strip())


def send(to, subject, text):
    """Returns True if Resend accepted the email."""
    key = os.environ.get("RESEND_API_KEY", "").strip()
    if not key or not to:
        return False
    body = json.dumps({"from": FROM, "to": [to], "subject": subject, "text": text, "reply_to": REPLY_TO}).encode()
    req = urllib.request.Request("https://api.resend.com/emails", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": "braiain-mailer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.load(e).get("message", "")[:200]
        except Exception:
            pass
        print("Resend refused an email:", e.code, detail, file=sys.stderr)
    except Exception as e:
        print("could not reach Resend:", type(e).__name__, file=sys.stderr)
    return False
