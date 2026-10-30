// brAIAIn API: a Cloudflare Worker.
// Paste this whole file into a new Worker in the Cloudflare dashboard. No settings needed.
// It reads the public training data from the site and runs the same model as the website.
const SITE = "https://www.braiain.com";
const BOS = "<s>", EOS = "</s>";
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

let cached = null, cachedAt = 0, models = new Map();
async function loadData() {
  if (cached && Date.now() - cachedAt < 5 * 60 * 1000) return cached;
  const get = async (p) => {
    const r = await fetch(SITE + p, { cf: { cacheTtl: 300 } });
    if (!r.ok) throw new Error(p + " " + r.status);
    return r.json();
  };
  try {
    const [versions, corpus, training] = await Promise.all([
      get("/data/versions.json"), get("/data/corpus.json"), get("/data/training.json"),
    ]);
    if (!cached || versions.length !== cached.versions.length || JSON.stringify(corpus) !== JSON.stringify(cached.corpus)) models = new Map();
    cached = { versions, corpus, training };
  } catch (e) {
    if (!cached) throw e; // keep serving the last good copy
  }
  cachedAt = Date.now();
  return cached;
}
// Keep the few most recently used versions in memory; rebuilding is the costly part.
function model(d, v) {
  if (models.has(v)) { const m = models.get(v); models.delete(v); models.set(v, m); return m; }
  const m = train(passagesFor(d, v));
  models.set(v, m);
  while (models.size > 4) models.delete(models.keys().next().value);
  return m;
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

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status, headers: { "Content-Type": "application/json", ...CORS },
  });
}
function error(message, status) {
  return json({ error: { message, type: "invalid_request_error" } }, status);
}

export default {
  async fetch(request) {
    try { return await handle(request); }
    catch (e) { return error("brAIAIn had a thought and it did not survive.", 500); }
  },
};

async function handle(request) {
  {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    let d;
    try { d = await loadData(); } catch (e) { return error("The model could not be loaded. It is probably thinking.", 503); }
    const latest = d.versions[d.versions.length - 1].v;

    if (url.pathname === "/v1/models" && request.method === "GET") {
      return json({
        object: "list",
        data: d.versions.map((e) => ({
          id: "brAIAIn-" + e.v, object: "model",
          created: Math.floor(new Date(e.released).getTime() / 1000), owned_by: "braiain-laboratories",
        })),
      });
    }

    if (url.pathname === "/v1/chat/completions" && request.method === "POST") {
      if (Number(request.headers.get("content-length") || 0) > 20000) return error("Request too large.", 413);
      let body;
      try { body = await request.json(); } catch (e) { return error("Request body must be JSON.", 400); }
      if (!body || typeof body !== "object") return error("Request body must be a JSON object.", 400);
      const msgs = Array.isArray(body.messages) ? body.messages : [];
      const last = [...msgs].reverse().find((m) => m && m.role === "user");
      if (!last) return error("Send at least one message with role user.", 400);
      const prompt = (typeof last.content === "string" ? last.content
        : Array.isArray(last.content) ? last.content.slice(0, 20).map((p) => String((p && p.text) || "").slice(0, 2000)).join(" ") : "").slice(0, 2000);
      let v = latest;
      const name = String(body.model || "braiain-latest");
      // Any model name that isn't a brAIAIn version gets the latest one, the way a real API would quietly upgrade you.
      const m = name.match(/^braiain-(\d+)$/i);
      if (m) v = parseInt(m[1], 10);
      if (!d.versions.some((e) => e.v === v)) return error("No such model: " + name, 404);
      // Only the newest few versions are served live; rebuilding old ones per request is too costly.
      if (v < latest - 2) return error("brAIAIn-" + v + " has been retired from the API. Its weights are on the site. Try brAIAIn-" + latest + ".", 410);
      const maxWords = Math.max(1, Math.min(80, parseInt(body.max_tokens, 10) || 40));
      if (body.stream) return error("Streaming is not supported. The whole answer arrives at once, like a thought.", 400);
      const text = generate(model(d, v), prompt.slice(0, 2000), maxWords) || "...";
      const pt = prompt.split(/\s+/).filter(Boolean).length, ct = text.split(/\s+/).filter(Boolean).length;
      return json({
        id: "chatcmpl-braiain-" + Date.now(), object: "chat.completion",
        created: Math.floor(Date.now() / 1000), model: "brAIAIn-" + v,
        choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: ct >= maxWords ? "length" : "stop" }],
        usage: { prompt_tokens: pt, completion_tokens: ct, total_tokens: pt + ct },
      });
    }

    if (url.pathname === "/" || url.pathname === "") {
      return json({ name: "brAIAIn API", docs: SITE + "/api.html", models: url.origin + "/v1/models", latest: "brAIAIn-" + latest });
    }
    return error("Not found. See " + SITE + "/api.html", 404);
  }
}
