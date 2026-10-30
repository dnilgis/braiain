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
    var link = (window.BRAIAIN_CONFIG || {}).paymentLink || "";
    document.querySelectorAll("[data-buy]").forEach(function (a) {
      if (link) { a.href = link; a.rel = "noopener"; }
      else { a.href = "/feed.html"; }
    });
    document.querySelectorAll("[data-price]").forEach(function (el) { el.textContent = (window.BRAIAIN_CONFIG || {}).price || "$5"; });
  });
})();

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
