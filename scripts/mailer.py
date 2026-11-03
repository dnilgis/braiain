"""Email to buyers, sent from the lab's own address through Resend (resend.com).

Buyers must never see the owner's personal address, so anything sent to a buyer goes through
here, from hello@braiain.com, once the RESEND_API_KEY secret is set (a Resend account with
braiain.com verified as a sending domain). Without the key, ready() is False and callers keep
their old behaviour: the owner gets a copy to forward by hand.

Free plan: 100 emails a day, 3,000 a month. Standard library only.
"""
import html as _h, json, os, sys, urllib.error, urllib.request

FROM = os.environ.get("MAIL_FROM", "").strip() or "BRAIAIN Laboratories <hello@braiain.com>"
REPLY_TO = "hello@braiain.com"


def ready():
    return bool(os.environ.get("RESEND_API_KEY", "").strip())


def send(to, subject, text, html=None, cc=None):
    """Returns True if Resend accepted the email. text is always sent too, for mail apps that block HTML."""
    key = os.environ.get("RESEND_API_KEY", "").strip()
    if not key or not to:
        return False
    msg = {"from": FROM, "to": [to], "subject": subject, "text": text, "reply_to": REPLY_TO}
    if html:
        msg["html"] = html
    if cc:
        msg["cc"] = [cc] if isinstance(cc, str) else list(cc)
    body = json.dumps(msg).encode()
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


# ---------------------------------------------------------------- branded HTML
# Table layout and inline styles: what mail apps (Gmail, Outlook, Apple Mail) render reliably.

SITE = "https://www.braiain.com"
INK, MUTED, LINE, BG = "#111111", "#5f5f5f", "#e6e6e6", "#f4f4f2"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Helvetica,Arial,sans-serif"


def esc(x):
    return _h.escape(str(x or ""), quote=True)


def button(label, url, primary=True):
    bg, fg, bd = (INK, "#ffffff", INK) if primary else ("#ffffff", INK, "#cfcfcf")
    return ('<table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 8px 10px 0;display:inline-table">'
            '<tr><td style="background:%s;border:1px solid %s;border-radius:999px">'
            '<a href="%s" style="display:inline-block;padding:12px 22px;font:600 15px %s;color:%s;text-decoration:none">%s</a>'
            '</td></tr></table>') % (bg, bd, esc(url), FONT, fg, esc(label))


def page(preheader, blocks):
    """blocks: HTML strings, each a section of the card."""
    inner = "".join('<tr><td style="padding:0 32px 22px">' + b + "</td></tr>" for b in blocks)
    return ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta name="color-scheme" content="light"><title>BRAIAIN</title></head>'
            '<body style="margin:0;padding:0;background:%(bg)s">'
            '<div style="display:none;max-height:0;overflow:hidden;opacity:0">%(pre)s</div>'
            '<table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="background:%(bg)s"><tr><td align="center" style="padding:28px 12px">'
            '<table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border:1px solid %(line)s;border-radius:16px">'
            '<tr><td style="padding:26px 32px 18px;font:700 17px %(font)s;letter-spacing:.04em;color:%(ink)s">BRAIAIN</td></tr>'
            '%(inner)s'
            '<tr><td style="padding:18px 32px 26px;border-top:1px solid %(line)s;font:400 12px/1.6 %(font)s;color:%(muted)s">'
            'BRAIAIN Laboratories is a parody AI lab. The model is real.<br>'
            'Questions? Reply to this email or write to <a href="mailto:hello@braiain.com" style="color:%(muted)s">hello@braiain.com</a>.<br>'
            '<a href="%(site)s" style="color:%(muted)s">braiain.com</a></td></tr>'
            '</table></td></tr></table></body></html>') % {
        "bg": BG, "line": LINE, "font": FONT, "ink": INK, "muted": MUTED, "site": SITE, "pre": esc(preheader), "inner": inner}


def h1(t):
    return '<h1 style="margin:0;font:700 26px/1.2 %s;color:%s">%s</h1>' % (FONT, INK, esc(t))


def p(t, muted=False, small=False):
    return '<p style="margin:0;font:400 %s/1.55 %s;color:%s">%s</p>' % ("13px" if small else "16px", FONT, MUTED if muted else INK, t)


def quote(t):
    return ('<div style="border-left:3px solid %s;padding:4px 0 4px 16px;font:600 19px/1.4 %s;color:%s">%s</div>'
            % (INK, FONT, INK, esc(t)))


def codebox(label, value, note):
    return ('<table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="background:%s;border-radius:12px">'
            '<tr><td style="padding:16px 18px;font:400 13px %s;color:%s">%s<br>'
            '<span style="font:700 22px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;color:%s">%s</span><br>%s</td></tr></table>'
            % (BG, FONT, MUTED, esc(label), INK, esc(value), esc(note)))


def gift_ticket(to, tier_name, code, ticket_url, redeem_url):
    who = to or "them"
    return page("Your gift ticket and code are inside. Print it, or forward the link.", [
        h1("Your gift ticket is ready."),
        p("Thank you for giving %s a sentence in brAIAIn. They write it, and a real language model is trained on it." % esc(to or "someone")),
        codebox("Gift: " + tier_name, code, "Works once. Keep this email."),
        p("<b>Print it.</b> The ticket has a QR code that opens the redeem page.") + '<div style="height:10px"></div>' + button("Print the ticket", ticket_url),
        p("<b>Or send it.</b> Forward this link to %s:" % esc(who)) + '<div style="height:10px"></div>' + button("Open the redeem page", redeem_url, False)
        + p('<a href="%s" style="color:%s;word-break:break-all">%s</a>' % (esc(redeem_url), MUTED, esc(redeem_url)), small=True),
        p("They write their sentence and check out for $0.00, with no card. Their version usually goes live within minutes.", muted=True, small=True),
        p("Gifts are not refundable. Anyone with the code can redeem it, once.", muted=True, small=True),
    ])


def version_live(name, sentence, vocab, notes_url, card_url, cert_url, share_url, x_url, bsky_url):
    blocks = [
        h1(name + " is live."),
        p("Your sentence is now in its training data, for good."),
        quote(sentence),
        p("It now knows %s words. Some of them are yours." % esc(vocab)),
        button("Read the release notes", notes_url) + button("Your release card", card_url, False)
        + (button("Your certificate", cert_url, False) if cert_url else ""),
        p("<b>Share it.</b> This link shows your release card in the preview:") + '<div style="height:10px"></div>'
        + button("Post on X", x_url, False) + button("Post on Bluesky", bsky_url, False)
        + p('<a href="%s" style="color:%s">%s</a>' % (esc(share_url), MUTED, esc(share_url)), small=True),
        p("If it was a gift, forward this email to them.", muted=True, small=True),
    ]
    return page(name + " was trained on your sentence.", blocks)
