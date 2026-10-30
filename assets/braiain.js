/* brAIAIn: a trigram language model trained only on what it was given.
   This file rebuilds any version in the browser from the public training data
   and talks with it. No server, no API calls. */
(function () {
  "use strict";
  var BOS = "<s>", EOS = "</s>";
  var cache = {};

  function getJSON(path) {
    if (!cache[path]) {
      cache[path] = fetch(path, { cache: "no-cache" }).then(function (r) {
        if (!r.ok) throw new Error(path + " " + r.status);
        return r.json();
      });
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

  // Short answers look broken, so try a few times for one with at least three words.
  function generate(m, prompt, maxWords) {
    var best = "";
    for (var i = 0; i < 6; i++) {
      var s = generateOnce(m, i < 4 ? prompt : "", maxWords);
      if (s.split(" ").length > best.split(" ").length) best = s;
      if (s.split(" ").length >= 3) return s;
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

  window.BRAIAIN = { load: load, model: model, generate: generate, passagesFor: passagesFor, getJSON: getJSON };

  // shared page bits
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".menu"), nav = document.querySelector(".nav");
    if (btn && nav) btn.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    var cfg = window.BRAIAIN_CONFIG || {}, link = cfg.paymentLink || "";
    document.documentElement.classList.toggle("teaser", !link);
    document.querySelectorAll("[data-buy]").forEach(function (a) {
      if (link) { a.href = link; a.rel = "noopener"; }
      else { a.href = "/#waitlist"; if (a.hasAttribute("data-teaser-text")) a.textContent = a.getAttribute("data-teaser-text"); }
    });
    countdowns(cfg);
    waitlists(cfg);
    document.querySelectorAll("[data-price]").forEach(function (el) { el.textContent = (window.BRAIAIN_CONFIG || {}).price || "$5"; });
  });
})();

// Countdown to cfg.opensAt, the same moment for every visitor.
function countdowns(cfg) {
  var els = document.querySelectorAll("[data-countdown]");
  if (!els.length) return;
  var target = Date.parse(cfg.opensAt || "");
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function plural(n, w) { return n + " " + w + (n === 1 ? "" : "s"); }
  function tick() {
    var left = Math.max(0, Math.floor((target - Date.now()) / 1000));
    var h = Math.floor(left / 3600), m = Math.floor(left % 3600 / 60), sec = left % 60;
    var txt = isNaN(target) ? "soon" : (left > 0 ? pad(h) + ":" + pad(m) + ":" + pad(sec) : "any minute now");
    els.forEach(function (el) { el.textContent = txt; });
    document.querySelectorAll("[data-countdown-words]").forEach(function (el) {
      el.textContent = isNaN(target) ? "" : left > 0 ? plural(h, "hour") + ", " + plural(m, "minute") + ", " + plural(sec, "second") : "Training was scheduled to open. Our engineers are aware.";
    });
    if (left > 0) setTimeout(tick, 1000 - (Date.now() % 1000));
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
        .then(function (r) { if (!r.ok) throw new Error(r.status); f.reset(); msg.textContent = "You are on the waitlist. We will email you once, when training opens. That is the only email."; shareBox(f, cfg); })
        .catch(function () { msg.textContent = "That did not go through. Please try again in a minute."; })
        .then(function () { btn.disabled = false; });
    });
  });
}

// After joining: let people share. There is no real position number, so none is shown.
function shareBox(f, cfg) {
  if (f.parentNode.querySelector(".share")) return;
  var url = "https://www.braiain.com/";
  var opens = Date.parse(cfg.opensAt || ""), hrs = Math.max(0, Math.round((opens - Date.now()) / 3600000));
  var text = "I joined the waitlist for brAIAIn, the AI model trained only on sentences people pay to add." +
    (hrs > 0 ? " Paid training opens in " + hrs + " hours." : "");
  var box = document.createElement("div");
  box.className = "share";
  box.innerHTML = '<p class="small muted" style="margin:14px 0 8px">Share your place in line. Every friend you bring is one more sentence the model does not know yet.</p>' +
    '<div class="share-row">' +
    (navigator.share ? '<button class="btn small" type="button" data-s="native">Share</button>' : "") +
    '<a class="btn ghost small" target="_blank" rel="noopener" href="https://twitter.com/intent/tweet?text=' + encodeURIComponent(text) + '&url=' + encodeURIComponent(url) + '">Post on X</a>' +
    '<a class="btn ghost small" target="_blank" rel="noopener" href="https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(url) + '">Facebook</a>' +
    '<a class="btn ghost small" href="sms:?&body=' + encodeURIComponent(text + " " + url) + '">Text a friend</a>' +
    '<button class="btn ghost small" type="button" data-s="copy">Copy link</button></div>';
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
