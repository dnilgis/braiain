/* The only file you need to edit after setup. */
window.BRAIAIN_CONFIG = {
  // Your Stripe Payment Link, e.g. "https://buy.stripe.com/abc123". While this is "",
  // the site runs in teaser mode: countdown and waitlist instead of buy buttons.
  paymentLink: "",
  // Shown on buttons.
  price: "$5",
  // When paid training opens. The countdown runs to this moment for every visitor.
  // 2026-11-01 02:00 UTC is Saturday, October 31 (Halloween), 9:00 pm Central.
  opensAt: "2026-11-01T02:00:00Z",
  // Waitlist form endpoint, e.g. "https://formspree.io/f/abcdwxyz". While this is "",
  // the waitlist box is hidden (no email is ever collected without somewhere to store it).
  waitlistEndpoint: "https://formspree.io/f/xzdlynrv",
  // Your Cloudflare Worker address, e.g. "https://braiain-api.yourname.workers.dev". Leave "" until it's set up.
  apiBase: ""
};
