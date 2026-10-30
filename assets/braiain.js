/* brAIAIn: a trigram language model trained only on what it was given.
   This file rebuilds any version in the browser from the public training data
   and talks with it. No server, no API calls. */
(function () {
  "use strict";
  var BOS = "<s>", EOS = "</s>";
  var cache = {};

  function getJSON(path) {
    if (!cache[path]) {
      // One retry: the host occasionally answers 503 for a moment.
      var get = function () { return fetch(path, { cache: "no-cache" }).then(function (r) {
        if (!r.ok) throw new Error(path + " " + r.status);
        return r.json();
      }); };
      cache[path] = get().catch(function () {
        return new Promise(function (ok) { setTimeout(ok, 800); }).then(get);
      }).catch(function (e) { delete cache[path]; throw e; });
    }
    return cache[path];
  }

  function load() {
    return Promise.all([
      getJSON("/data/versions.json"),
      getJSON("/data/corpus.json"),
      getJSON("/data/training.json")
    ]).then(function (x) {
      return { versions: x[0], corpus: x[1], training: x[2] };
    });
  }

  // Tables have no prototype, so a word like "__proto__" or "constructor" is just a word.
  function dict() { return Object.create(null); }

  function train(passages) {
    var tri = dict(), bi = dict(), uni = dict();
    passages.forEach(function (p) {
      var t = [BOS, BOS].concat(p.split(/\s+/).filter(Boolean), [EOS]);
      for (var i = 2; i < t.length; i++) {
        var k = t[i - 2] + " " + t[i - 1], c = t[i], b = t[i - 1];
        (tri[k] = tri[k] || dict())[c] = (tri[k][c] || 0) + 1;
        (bi[b] = bi[b] || dict())[c] = (bi[b][c] || 0) + 1;
        if (c !== EOS) uni[c] = (uni[c] || 0) + 1;
      }
    });
    return { tri: tri, bi: bi, uni: uni };
  }

  // Same recipe as scripts/train.py: founding text + paid sentences up to v,
  // plus the last few passages earlier versions wrote, plus the newest sentence weighted.
  function passagesFor(d, v) {
    var human = d.training.genesis.slice();
    d.corpus.forEach(function (c) { if (c.text && c.v <= v) human.push(c.text); });
    var self = [];
    d.versions.forEach(function (e) { if (e.v >= 1 && e.v <= v && e.self_new) self = self.concat(e.self_new); });
    self = self.slice(-d.training.self_window);
    var newest = d.corpus.filter(function (c) { return c.text && c.v === v; }).map(function (c) { return c.text; });
    var extra = [];
    for (var i = 1; i < d.training.newest_weight; i++) extra = extra.concat(newest);
    return human.concat(self, extra);
  }

  var models = {};
  function model(d, v) {
    if (!models[v]) models[v] = train(passagesFor(d, v));
    return models[v];
  }

  function pick(table) {
    var keys = Object.keys(table).sort(), total = 0, i;
    for (i = 0; i < keys.length; i++) total += table[keys[i]];
    var r = Math.random() * total;
    for (i = 0; i < keys.length; i++) { r -= table[keys[i]]; if (r <= 0) return keys[i]; }
    return keys[keys.length - 1];
  }

  function norm(w) { return w.toLowerCase().replace(/[^a-z0-9']/g, ""); }

  // Choose a word from the question that the model knows, so it looks like it listened.
  function startWord(m, prompt) {
    if (!m.known) {
      m.known = dict();
      Object.keys(m.bi).forEach(function (w) { if (w !== BOS) { var n = norm(w); if (n) (m.known[n] = m.known[n] || []).push(w); } });
    }
    var known = m.known;
    var stop = { the: 1, a: 1, an: 1, and: 1, is: 1, are: 1, to: 1, of: 1, i: 1, you: 1, it: 1, what: 1, how: 1, why: 1, do: 1, does: 1, can: 1, my: 1, your: 1, me: 1, in: 1, on: 1, for: 1, be: 1, with: 1, that: 1, this: 1 };
    var words = prompt.split(/\s+/).map(norm).filter(function (w) { return w && known[w]; });
    var list = words.filter(function (w) { return !Object.prototype.hasOwnProperty.call(stop, w); });
    if (!list.length) return null;
    var w = list[Math.floor(Math.random() * list.length)];
    var forms = known[w];
    return forms[Math.floor(Math.random() * forms.length)];
  }

  // Decoding: draw eight answers and keep the one that shares the most words with the
  // question (ties go to the longer one). Short answers look broken, so three words minimum.
  function generate(m, prompt, maxWords) {
    var q = {};
    (prompt || "").split(/\s+/).map(norm).forEach(function (w) { if (w && w.length > 2) q[w] = 1; });
    var best = "", bestScore = -1;
    for (var i = 0; i < 8; i++) {
      var s = generateOnce(m, i < 5 ? prompt : "", maxWords), n = s.split(" ").length;
      if (!s) continue;
      var overlap = 0, seen = {};
      s.split(/\s+/).map(norm).forEach(function (w) { if (q[w] && !seen[w]) { overlap++; seen[w] = 1; } });
      var score = overlap * 100 + Math.min(n, 30) + (n >= 3 ? 50 : 0);
      if (score > bestScore) { best = s; bestScore = score; }
    }
    return best;
  }

  function generateOnce(m, prompt, maxWords) {
    maxWords = maxWords || 40;
    if (!Object.keys(m.uni).length) return "";
    var start = prompt ? startWord(m, prompt) : null;
    var a = BOS, b = BOS, out = [];
    if (start) { b = start; out.push(start); }
    while (out.length < maxWords) {
      var k = a + " " + b, c;
      if (m.tri[k]) c = pick(m.tri[k]);
      else if (m.bi[b]) c = pick(m.bi[b]);
      else break;
      if (c === EOS) break;
      out.push(c); a = b; b = c;
    }
    var s = out.join(" ");
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // Technical specifications. Every value is measured on the model in this browser;
  // only the vocabulary around it is borrowed from real labs.
  function specs(d, v) {
    var m = model(d, v), passages = passagesFor(d, v), e = d.versions.filter(function (x) { return x.v === v; })[0] || {};
    var params = 0, adds = 0, tokens = 0, k, j;
    for (k in m.tri) for (j in m.tri[k]) { params++; adds += m.tri[k][j]; }
    for (k in m.bi) for (j in m.bi[k]) { params++; adds += m.bi[k][j]; }
    for (k in m.uni) adds += m.uni[k];
    passages.forEach(function (p) { tokens += p.split(/\s+/).filter(Boolean).length; });
    var t0 = performance.now(), n = 0;
    while (performance.now() - t0 < 30) { generate(m, "", 40); n++; }
    var ms = (performance.now() - t0) / n;
    var cutoff = e.released ? new Date(e.released).toISOString().slice(0, 10) : "";
    var markers = 0;
    for (k in m.tri) for (j in m.tri[k]) if (k.indexOf(BOS) >= 0 || j === EOS) markers++;
    for (k in m.bi) for (j in m.bi[k]) if (k === BOS || j === EOS) markers++;
    return [
      ["Architecture", "Autoregressive count-based trigram language model with backoff", "Predicts the next word from the two before it. When it has never seen those two together, it falls back to one."],
      ["Context window", "2 tokens", "Industry term for how much it remembers. It remembers two words."],
      ["Parameters", params.toLocaleString(), "Distinct word pairs and triples it has counted, including " + markers.toLocaleString() + " that mark where a sentence starts or ends."],
      ["Embedding dimensions", "0", "It does not represent words as vectors. It represents them as words."],
      ["Layers / attention heads", "0 / 0", "Attention is implicit. It attends to the previous two words, fully."],
      ["Mixture of Experts", "1 expert", "Routing overhead: none."],
      ["Training tokens", tokens.toLocaleString(), "Every word in the training text, counted once per occurrence."],
      ["Training compute", adds.toLocaleString() + " count updates", "Training is counting. No multiplication, no gradients."],
      ["Gradient steps", "0", "The loss was never computed, so it never had to go down."],
      ["Precision", "Whole-number counts", "Stored exactly as counted."],
      ["Tokenizer", "Whitespace", "Splits on spaces, keeping case and punctuation. Vocabulary: " + Object.keys(m.uni).length.toLocaleString() + " distinct tokens."],
      ["Alignment", "RLHF", "Reinforcement Learning from Human Funding."],
      ["Constitution", "The Ten Instructions", "Twelve lines, written before the model existed. Binding."],
      ["Knowledge cutoff", cutoff, v === 0 ? "The day its founding text was written. No payments yet." : "The day it was trained on its most recent sentence."],
      ["Inference latency", took(ms / 1000) + " per answer", "Measured just now, on your device."],
      ["Emergent capabilities", "None observed", "Monitoring continues at current scale."]
    ];
  }
  function specTable(rows) {
    return '<div class="table-wrap"><table class="specs"><tbody>' + rows.map(function (r) {
      return '<tr><td><b>' + esc(r[0]) + '</b></td><td class="mono">' + esc(r[1]) + '</td><td class="muted small">' + esc(r[2]) + '</td></tr>';
    }).join("") + '</tbody></table></div>';
  }

  // One error state for every page: say it failed, offer a retry.
  function fail(el, what) {
    if (!el) return;
    el.innerHTML = '<div class="notice" role="alert">' + esc(what || "The model data could not be loaded.") +
      ' <a href="" onclick="location.reload();return false">Try again</a>.</div>';
  }
  function took(sec) {
    return sec < 0.001 ? Math.max(1, Math.round(sec * 1e6)) + " microseconds" : (sec * 1000).toFixed(1) + " milliseconds";
  }
  function live(cfg) {
    cfg = cfg || window.BRAIAIN_CONFIG || {};
    var t = Date.parse(cfg.opensAt || "");
    return !!cfg.paymentLink && (isNaN(t) || Date.now() >= t);
  }

  window.BRAIAIN = { load: load, model: model, generate: generate, passagesFor: passagesFor, getJSON: getJSON, specs: specs, specTable: specTable, fail: fail, took: took, live: live };

  // shared page bits
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".menu"), nav = document.querySelector(".nav");
    if (btn && nav) {
      nav.id = nav.id || "site-nav";
      btn.setAttribute("aria-controls", nav.id);
      var setOpen = function (open) { nav.classList.toggle("open", open); btn.setAttribute("aria-expanded", open ? "true" : "false"); };
      btn.addEventListener("click", function () { setOpen(!nav.classList.contains("open")); });
      document.addEventListener("keydown", function (e) { if (e.key === "Escape" && nav.classList.contains("open")) { setOpen(false); btn.focus(); } });
    }
    var cfg = window.BRAIAIN_CONFIG || {};
    applyMode(cfg);
    countdowns(cfg);
    waitlists(cfg);
  });
})();

// Buying is open only when a payment link exists AND the opening time has passed.
// Re-run when the countdown reaches zero, so the page flips at the same moment for everyone.
function applyMode(cfg) {
  var isLive = window.BRAIAIN.live(cfg);
  var links = { "": cfg.paymentLink, certificate: cfg.paymentLinkCertificate, naming: cfg.paymentLinkNaming };
  var prices = { "": cfg.price || "$5", certificate: cfg.priceCertificate || "$15", naming: cfg.priceNaming || "$50" };
  document.documentElement.classList.toggle("teaser", !isLive);
  document.documentElement.classList.toggle("is-live", isLive);
  document.querySelectorAll("[data-buy]").forEach(function (a) {
    var t = a.getAttribute("data-buy") || "", link = links[t];
    if (!a.hasAttribute("data-live-html")) a.setAttribute("data-live-html", a.innerHTML);
    if (isLive && link) { a.href = link; a.rel = "noopener"; a.innerHTML = a.getAttribute("data-live-html"); a.removeAttribute("aria-disabled"); a.classList.remove("off"); }
    else if (isLive && !link) { a.removeAttribute("href"); a.setAttribute("aria-disabled", "true"); a.classList.add("off"); }
    else { a.href = "/#waitlist"; if (a.hasAttribute("data-teaser-text")) a.textContent = a.getAttribute("data-teaser-text"); }
  });
  document.querySelectorAll("[data-price]").forEach(function (el) { el.textContent = prices[el.getAttribute("data-price") || ""] || prices[""]; });
}

// Countdown to cfg.opensAt, the same moment for every visitor.
function countdowns(cfg) {
  var els = document.querySelectorAll("[data-countdown]");
  if (!els.length) return;
  var target = Date.parse(cfg.opensAt || "");
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function plural(n, w) { return n + " " + w + (n === 1 ? "" : "s"); }
  function tick() {
    if (window.BRAIAIN.live(cfg)) return;
    var left = Math.max(0, Math.floor((target - Date.now()) / 1000));
    var h = Math.floor(left / 3600), m = Math.floor(left % 3600 / 60), sec = left % 60;
    var txt = isNaN(target) ? "soon" : (left > 0 ? pad(h) + ":" + pad(m) + ":" + pad(sec) : "opening shortly");
    els.forEach(function (el) { el.textContent = txt; });
    document.querySelectorAll("[data-countdown-words]").forEach(function (el) {
      el.textContent = isNaN(target) ? "" : left > 0 ? plural(h, "hour") + ", " + plural(m, "minute") + ", " + plural(sec, "second")
        : "The opening time has passed. Checkout opens as soon as it is switched on, and the waitlist hears first.";
    });
    if (left > 0) setTimeout(tick, 1000 - (Date.now() % 1000));
    else applyMode(cfg);
  }
  tick();
}

// Waitlist forms post to cfg.waitlistEndpoint (any form service that accepts a POST, e.g. Formspree).
function waitlists(cfg) {
  document.querySelectorAll("form[data-waitlist]").forEach(function (f) {
    if (!cfg.waitlistEndpoint) {
      f.style.display = "none";
      var off = document.createElement("p");
      off.className = "muted small";
      off.textContent = "Waitlist sign-ups open soon. There is a waitlist for the waitlist.";
      f.parentNode.insertBefore(off, f.nextSibling);
      return;
    }
    var msg = f.querySelector("[data-msg]");
    f.addEventListener("submit", function (e) {
      e.preventDefault();
      var email = f.querySelector("input[type=email]").value.trim();
      if (!email) return;
      var btn = f.querySelector("button"); btn.disabled = true;
      fetch(cfg.waitlistEndpoint, { method: "POST", headers: { "Accept": "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ email: email, list: "braiain-waitlist" }) })
        .then(function (r) {
          if (!r.ok) throw new Error(r.status);
          f.reset();
          var done = document.createElement("p");
          done.className = "small"; done.setAttribute("role", "status"); done.tabIndex = -1;
          done.textContent = "You are on the waitlist. We will email you once, when training opens, and then delete your address.";
          f.parentNode.insertBefore(done, f); f.style.display = "none";
          shareBox(done, cfg); done.focus();
        })
        .catch(function () { msg.textContent = "That did not go through. Please try again in a minute."; })
        .then(function () { btn.disabled = false; });
    });
  });
}

// After joining: let people share. There is no real position number, so none is shown.
function shareBox(f, cfg) {
  if (f.parentNode.querySelector(".share")) return;
  var url = "https://www.braiain.com/";
  var opens = Date.parse(cfg.opensAt || ""), mins = Math.round((opens - Date.now()) / 60000), hrs = Math.round(mins / 60);
  var when = mins <= 0 ? "" : mins < 60 ? " Paid training opens within the hour." : hrs === 1 ? " Paid training opens in about an hour." : " Paid training opens in " + hrs + " hours.";
  var text = "I joined the waitlist for brAIAIn, the AI model trained only on sentences people pay to add." + when;
  var box = document.createElement("div");
  box.className = "share";
  box.innerHTML = '<p class="small muted" style="margin:14px 0 8px">Tell someone. Every friend you bring is one more sentence the model does not know yet.</p>' +
    '<div class="share-row">' +
    (navigator.share ? '<button class="btn small" type="button" data-s="native">Share</button>' : "") +
    '<button class="btn ghost small" type="button" data-s="copy">Copy link</button>' +
    '<a class="btn ghost small" href="sms:?&body=' + encodeURIComponent(text + " " + url) + '">Text a friend</a>' +
    '<a class="btn ghost small" target="_blank" rel="noopener" href="https://twitter.com/intent/tweet?text=' + encodeURIComponent(text) + '&url=' + encodeURIComponent(url) + '">Post on X</a>' +
    '<a class="btn ghost small" target="_blank" rel="noopener" href="https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(url) + '">Facebook</a></div>';
  f.parentNode.insertBefore(box, f.nextSibling);
  var n = box.querySelector('[data-s="native"]');
  if (n) n.addEventListener("click", function () { navigator.share({ title: "brAIAIn", text: text, url: url }).catch(function () {}); });
  box.querySelector('[data-s="copy"]').addEventListener("click", function (e) {
    var b = e.currentTarget;
    (navigator.clipboard ? navigator.clipboard.writeText(text + " " + url) : Promise.reject()).then(function () { b.textContent = "Copied"; }, function () { b.textContent = url; });
  });
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function fmtDate(iso) {
  if (!iso) return "";
  var d = new Date(iso);
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}
function qp(name) { return new URLSearchParams(location.search).get(name); }
