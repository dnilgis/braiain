/* The only file you need to edit after setup. */
window.BRAIAIN_CONFIG = {
  // The legal seller shown on the Terms and Privacy pages, e.g. "Example Ventures LLC".
  // Fill this in before taking payments.
  operator: "BRAIAIN, a sole proprietorship",
  // Stripe Payment Links, one per tier. Leave "" for a tier you do not offer.
  // Buying stays closed until opensAt, even if these are filled in early.
  paymentLink: "https://buy.stripe.com/28E6oHaKQdb2eq7eR47N600",             // $5: one sentence
  paymentLinkCertificate: "https://buy.stripe.com/9B64gzdX29YQ3LtdN07N601",  // $15: one sentence plus a printable certificate
  paymentLinkNaming: "https://buy.stripe.com/6oUeVd3io2wogyf7oC7N602",       // $50: one sentence plus the version's codename
  price: "$5",
  priceCertificate: "$15",
  priceNaming: "$50",
  // When paid training opens. The countdown runs to this moment for every visitor.
  // 2026-11-01 02:00 UTC is Saturday, October 31 (Halloween), 9:00 pm Central.
  opensAt: "2026-10-31T00:00:00Z",
  // Waitlist form endpoint, e.g. "https://formspree.io/f/abcdwxyz". While this is "",
  // the waitlist box is hidden (no email is ever collected without somewhere to store it).
  waitlistEndpoint: "",
  // Your Cloudflare Worker address, e.g. "https://braiain-api.yourname.workers.dev". Leave "" until it's set up.
  apiBase: ""
};
