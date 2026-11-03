/* The only file you need to edit after setup. */
window.BRAIAIN_CONFIG = {
  // The legal seller shown on the Terms and Privacy pages, e.g. "Example Ventures LLC".
  // Fill this in before taking payments.
  operator: "BRAIAIN, a sole proprietorship",
  // Stripe Payment Links, one per tier. Leave "" for a tier you do not offer.
  // Buying stays closed until opensAt, even if these are filled in early.
  paymentLink: "https://buy.stripe.com/3cI3cvcSY5IA95N5gu7N607",             // $2.49: one sentence
  // The old $5 sentence link, still read so a buyer holding the old URL is never missed. Not shown on the site.
  paymentLinkSentence5: "https://buy.stripe.com/28E6oHaKQdb2eq7eR47N600",
  paymentLinkCertificate: "https://buy.stripe.com/7sY3cv5qw1sk81JfV87N609",  // $9.99: one sentence plus a printable certificate
  paymentLinkNaming: "https://buy.stripe.com/dRm8wPdX2fja5TB9wK7N608",       // $24.99: one sentence plus the version's codename
  // The same three tiers with no boxes at checkout: the redeem and teach pages put the sentence in
  // the link itself (client_reference_id), so the buyer only confirms. Same products and prices.
  paymentLinkWrite: "https://buy.stripe.com/bJe9AT06cgne2Hp6ky7N60d",
  paymentLinkWriteCertificate: "https://buy.stripe.com/aFa7sLcSYc6YbdV5gu7N60e",
  paymentLinkWriteNaming: "https://buy.stripe.com/14A28r4msc6Y2Hp4cq7N60f",
  // Old $15 and $50 links, still read so nobody holding an old URL is missed. Not shown on the site.
  paymentLinkCertificate15: "https://buy.stripe.com/9B64gzdX29YQ3LtdN07N601",
  paymentLinkNaming50: "https://buy.stripe.com/6oUeVd3io2wogyf7oC7N602",
  // Title sponsorship: a Stripe Payment Link where the customer chooses the amount, with custom
  // fields "Sponsor name" and "Sponsor line (optional)". Leave "" until it exists.
  sponsorLink: "https://buy.stripe.com/8x200jf166MEdm36ky7N603",
  // Gifts the recipient redeems (scripts/gift.py): one Payment Link per tier, same prices, with
  // optional custom fields "Recipient's name" and "Your name". Leave "" until they exist.
  giftLink: "https://buy.stripe.com/dRm5kD5qw4Ew1Dl7oC7N60a",             // $2.49 gift: they choose one sentence
  giftLinkCertificate: "https://buy.stripe.com/6oUcN5dX24Ew2HpeR47N60c",  // $9.99 gift: sentence plus certificate
  giftLinkNaming: "https://buy.stripe.com/aFaaEX6uAc6YbdVgZc7N60b",       // $24.99 gift: sentence plus codename
  // Old gift links ($5, $15, $50), still read by scripts/gift.py. Not shown on the site.
  giftLink5: "https://buy.stripe.com/28E5kD1ag6MEdm3bES7N604",
  giftLinkCertificate15: "https://buy.stripe.com/28EfZh8CIb2U6XF8sG7N605",
  giftLinkNaming50: "https://buy.stripe.com/9B68wP6uAfjachZgZc7N606",
  price: "$2.49",
  priceCertificate: "$9.99",
  priceNaming: "$24.99",
  // When paid training opens. The countdown runs to this moment for every visitor.
  // 2026-11-01 02:00 UTC is Saturday, October 31 (Halloween), 9:00 pm Central.
  opensAt: "2026-10-31T00:00:00Z",
  // Waitlist form endpoint, e.g. "https://formspree.io/f/abcdwxyz". While this is "",
  // the waitlist box is hidden (no email is ever collected without somewhere to store it).
  waitlistEndpoint: "",
  // Your Cloudflare Worker address, e.g. "https://braiain-api.yourname.workers.dev". Leave "" until it's set up.
  apiBase: ""
};
