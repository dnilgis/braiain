/**
 * BRAIAIN Standard — SEO Page Generator
 * Run: node generate-pages.js
 * Output: models/[id].html for each model in models.json
 *
 * Each page is fully static HTML with:
 * - Proper title/meta/OG tags targeting long-tail keywords
 * - JSON-LD structured data (Dataset + BreadcrumbList)
 * - Prerendered model data (no JS required for content)
 * - Link back to main leaderboard
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const data = JSON.parse(readFileSync('./models.json', 'utf8'));
const models = data.models;

mkdirSync('./models', { recursive: true });

const DIM_LABELS = {
  reasoning:'Reasoning',coding:'Coding',math:'Math',science:'Science',
  context:'Context',speed:'Speed',efficiency:'Efficiency',multimodal:'Multimodal'
};

function scoreDesc(s) {
  if (s >= 1350) return 'Exceptional — frontier performance across all benchmark categories.';
  if (s >= 1200) return 'Advanced — outperforms the majority of tested models.';
  if (s >= 1050) return 'Proficient — competitive across general production use cases.';
  return 'Capable — specialized strengths or efficiency-focused design.';
}

function generatePage(model, rank, allModels) {
  const total = model.analytical + model.technical;
  const desc = scoreDesc(total);
  const priceStr = model.price_input === 0 ? 'Free (open weights)' :
    `$${model.price_input.toFixed(3)}/1M input tokens`;

  const competitors = allModels
    .filter(m => m.id !== model.id)
    .sort((a, b) => (b.analytical + b.technical) - (a.analytical + a.technical))
    .slice(0, 3);

  const dimRows = Object.entries(model.dims).map(([k, v]) => `
    <tr>
      <td>${DIM_LABELS[k]}</td>
      <td style="font-family:monospace">${Math.round(v * 100)}/100</td>
      <td style="width:120px">
        <div style="height:4px;background:#e8e4e0;border-radius:0">
          <div style="height:100%;width:${Math.round(v * 100)}%;background:${model.color}"></div>
        </div>
      </td>
    </tr>`).join('');

  const competitorLinks = competitors.map(c => {
    const ct = c.analytical + c.technical;
    return `<a href="/models/${c.id}.html" style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border:1px solid #e8e4e0;margin-bottom:6px;text-decoration:none;color:#111;font-family:Space Grotesk,sans-serif;font-size:13px;transition:background .1s" onmouseover="this.style.background='#fafaf8'" onmouseout="this.style.background='transparent'">
      <span>${c.name} <span style="font-size:11px;color:#888;font-family:JetBrains Mono,monospace">${c.provider}</span></span>
      <span style="font-family:JetBrains Mono,monospace;font-weight:500">${ct}</span>
    </a>`;
  }).join('');

  const jsonLD = JSON.stringify({
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "BRAIAIN Standard", "item": "https://braiain.com/standard.html" },
          { "@type": "ListItem", "position": 2, "name": model.name, "item": `https://braiain.com/models/${model.id}.html` }
        ]
      },
      {
        "@type": "Dataset",
        "name": `${model.name} — BRAIAIN Standard Score`,
        "description": `${model.name} by ${model.provider} scored ${total} on the BRAIAIN Standard benchmark. ${desc}`,
        "url": `https://braiain.com/models/${model.id}.html`,
        "creator": { "@type": "Organization", "name": "BRAIAIN", "url": "https://braiain.com" },
        "variableMeasured": [
          { "@type": "PropertyValue", "name": "BRAIAIN Score", "value": total },
          { "@type": "PropertyValue", "name": "Analytical Score", "value": model.analytical },
          { "@type": "PropertyValue", "name": "Technical Score", "value": model.technical }
        ]
      }
    ]
  }, null, 2);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${model.name} BRAIAIN Score: ${total}/1600 — AI Intelligence Benchmark</title>
<meta name="description" content="${model.name} by ${model.provider} scores ${total}/1600 on the BRAIAIN Standard benchmark. Analytical: ${model.analytical}. Technical: ${model.technical}. ${desc}">
<meta name="keywords" content="${model.name} benchmark, ${model.name} score, ${model.provider} AI model, ${model.name} vs GPT-4o, ${model.name} performance 2026, AI model comparison">
<link rel="canonical" href="https://braiain.com/models/${model.id}.html">
<meta property="og:title" content="${model.name} — BRAIAIN Score ${total}">
<meta property="og:description" content="${model.name} scores ${total}/1600. Analytical: ${model.analytical} | Technical: ${model.technical}. ${desc}">
<meta property="og:url" content="https://braiain.com/models/${model.id}.html">
<meta property="og:image" content="https://braiain.com/og-image.png">
<script type="application/ld+json">${jsonLD}</script>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-3B3ZCE62NW"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag("js",new Date());gtag("config","G-3B3ZCE62NW");</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--red:#e8342a;--tx:#111;--mt:#888880;--dm:#ccc;--bg:#fff;--sf:#fafaf8;--bd:#e8e4e0;--bd2:#d4d0cc;--accent:${model.color}}
html{background:var(--bg);color:var(--tx);font-family:"Space Grotesk",sans-serif;-webkit-font-smoothing:antialiased}
body{max-width:800px;margin:0 auto;padding:0 24px 60px}
nav{display:flex;align-items:center;justify-content:space-between;height:52px;border-bottom:1px solid var(--bd);margin-bottom:0}
.nav-logo{font-size:14px;font-weight:600;color:var(--tx);text-decoration:none}
.nav-logo span{font-weight:300;color:var(--mt);font-size:12px;margin-left:4px}
.nav-back{font-size:12px;color:var(--mt);text-decoration:none;border:1px solid var(--bd2);padding:5px 12px;border-radius:2px;transition:all .15s}
.nav-back:hover{color:var(--red);border-color:var(--red)}
.breadcrumb{font-size:11px;font-family:"JetBrains Mono",monospace;color:var(--dm);padding:12px 0;letter-spacing:.04em}
.breadcrumb a{color:var(--mt);text-decoration:none}.breadcrumb a:hover{color:var(--red)}
.hero{padding:32px 0 24px;border-bottom:1px solid var(--bd)}
.hero-rank{font-size:10px;font-family:"JetBrains Mono",monospace;color:var(--red);letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px}
.hero h1{font-size:clamp(24px,5vw,40px);font-weight:300;letter-spacing:-.04em;color:var(--tx);margin-bottom:4px}
.hero-provider{font-size:13px;color:var(--mt);font-family:"JetBrains Mono",monospace;letter-spacing:.06em;margin-bottom:20px}
.score-display{display:flex;gap:24px;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px}
.score-main{font-size:72px;font-family:"JetBrains Mono",monospace;font-weight:300;letter-spacing:-.04em;color:var(--accent);line-height:1}
.score-subs{display:flex;flex-direction:column;gap:8px;padding-bottom:6px}
.score-sub{font-size:13px;color:var(--mt)}
.score-sub strong{color:var(--tx);font-family:"JetBrains Mono",monospace;font-weight:500}
.hero-desc{font-size:14px;color:var(--mt);line-height:1.75;max-width:520px}
.section{padding:28px 0;border-bottom:1px solid var(--bd)}
.section-h{font-size:10px;font-family:"JetBrains Mono",monospace;color:var(--red);letter-spacing:.14em;text-transform:uppercase;margin-bottom:16px}
.dim-table{width:100%;border-collapse:collapse;font-size:13px}
.dim-table th{text-align:left;padding:8px 0;font-size:9px;font-family:"JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--dm);border-bottom:1px solid var(--bd)}
.dim-table td{padding:10px 0;border-bottom:1px solid var(--bd);color:var(--mt);vertical-align:middle}
.dim-table td:first-child{font-family:"JetBrains Mono",monospace;color:var(--tx);width:130px}
.dim-table td:nth-child(2){width:60px}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.meta-item{background:var(--sf);padding:14px;border:1px solid var(--bd)}
.meta-label{font-size:9px;font-family:"JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--dm);margin-bottom:4px}
.meta-val{font-size:15px;font-family:"JetBrains Mono",monospace;color:var(--tx)}
.print-cta{background:var(--sf);border:1px solid var(--bd);padding:20px;text-align:center}
.print-cta h3{font-size:16px;font-weight:400;color:var(--tx);margin-bottom:6px;letter-spacing:-.2px}
.print-cta p{font-size:12px;color:var(--mt);margin-bottom:16px;line-height:1.6}
.print-btns{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.pb{font-size:12px;padding:9px 20px;border-radius:2px;cursor:pointer;font-family:"Space Grotesk",sans-serif;font-weight:500;letter-spacing:.02em;transition:all .12s;text-decoration:none;display:inline-block}
.pb.primary{background:var(--red);color:#fff;border:none}.pb.primary:hover{background:#c82b1e}
.pb.sec{border:1px solid var(--bd2);color:var(--mt);background:transparent}.pb.sec:hover{border-color:var(--tx);color:var(--tx)}
</style>
</head>
<body>

<nav>
  <a href="/standard.html" class="nav-logo">Braiain <span>standard</span></a>
  <a href="/standard.html" class="nav-back">← All models</a>
</nav>

<div class="breadcrumb">
  <a href="/standard.html">BRAIAIN Standard</a> / ${model.name}
</div>

<div class="hero">
  <div class="hero-rank">Rank #${rank} of ${allModels.length} models tested</div>
  <h1>${model.name}</h1>
  <div class="hero-provider">${model.provider} &nbsp;·&nbsp; Released ${model.released} &nbsp;·&nbsp; ${model.context_window} context</div>
  <div class="score-display">
    <div class="score-main" aria-label="BRAIAIN score ${total}">${total}</div>
    <div class="score-subs">
      <div class="score-sub">Analytical <strong>${model.analytical}</strong> / 800</div>
      <div class="score-sub">Technical <strong>${model.technical}</strong> / 800</div>
      <div class="score-sub" style="font-size:11px;font-family:JetBrains Mono,monospace;color:var(--dm)">${priceStr}</div>
    </div>
  </div>
  <p class="hero-desc">${desc}</p>
</div>

<div class="section">
  <div class="section-h">Dimension breakdown</div>
  <table class="dim-table" aria-label="Model dimension scores">
    <thead><tr><th scope="col">Dimension</th><th scope="col">Score</th><th scope="col">Bar</th></tr></thead>
    <tbody>${dimRows}</tbody>
  </table>
</div>

<div class="section">
  <div class="section-h">Model details</div>
  <div class="meta-grid">
    <div class="meta-item"><div class="meta-label">BRAIAIN score</div><div class="meta-val" style="color:${model.color}">${total}</div></div>
    <div class="meta-item"><div class="meta-label">Provider</div><div class="meta-val">${model.provider}</div></div>
    <div class="meta-item"><div class="meta-label">Context window</div><div class="meta-val">${model.context_window}</div></div>
    <div class="meta-item"><div class="meta-label">Pricing</div><div class="meta-val" style="font-size:12px">${priceStr}</div></div>
  </div>
</div>

<div class="section">
  <div class="section-h">Compare nearby models</div>
  ${competitorLinks}
  <a href="/standard.html" style="display:block;text-align:center;font-size:11px;font-family:JetBrains Mono,monospace;color:var(--mt);margin-top:12px;letter-spacing:.04em;text-decoration:none">View full leaderboard ↗</a>
</div>

<div class="section">
  <div class="print-cta">
    <h3>Own this model's DNA fingerprint</h3>
    <p>The Lissajous figure above is mathematically derived from ${model.name}'s exact benchmark scores. No two models produce the same figure. Archival giclée print — the shape is the score.</p>
    <div class="print-btns">
      <a href="https://braiain.com/shop/${model.id}" class="pb primary">Order print from $39</a>
      <a href="/standard.html#prints" class="pb sec">See all prints</a>
    </div>
  </div>
</div>

</body>
</html>`;
}

// Sort by score for rank calculation
const sorted = [...models].sort((a, b) => (b.analytical + b.technical) - (a.analytical + a.technical));

sorted.forEach((model, i) => {
  const page = generatePage(model, i + 1, models);
  const path = join('./models', `${model.id}.html`);
  writeFileSync(path, page);
  console.log(`✓ models/${model.id}.html (rank #${i + 1}, score ${model.analytical + model.technical})`);
});

console.log(`\nGenerated ${models.length} model pages in ./models/`);
console.log('Add these to your sitemap.xml and push.');
