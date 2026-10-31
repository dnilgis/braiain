const { chromium } = require('playwright');
const B = 'http://localhost:8000';
(async () => {
  const b = await chromium.launch();
  const shot = async (w, h, url, out, fn) => {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2, colorScheme: 'light' });
    await p.goto(B + url); await p.waitForTimeout(3000);
    await p.evaluate(()=>{const a=document.querySelector('.announce'); if(a) a.remove();});
    if (!fn) await p.screenshot({ path: out });
    else { const r = await p.evaluate(fn); await p.screenshot({ path: out, clip: r, fullPage: true }); }
    await p.close(); console.log('ok', out);
  };
  await shot(1280, 800, '/', 'press/braiain-home-desktop.png');
  await shot(390, 844, '/', 'press/braiain-home-phone.png');
  await shot(1280, 900, '/', 'press/braiain-specs.png', () => { const s=[...document.querySelectorAll('h2')].find(h=>/Under the hood/.test(h.textContent)).closest('section'); const r=s.getBoundingClientRect(); return {x:0,y:r.top+scrollY,width:innerWidth,height:r.height}; });
  await shot(1100, 900, '/benchmarks.html', 'press/braiain-leaderboard.png', () => { const t=document.getElementById('lb'); const h=t.closest('section').querySelector('h2'); const top=h.getBoundingClientRect().top-30, bot=t.getBoundingClientRect().bottom+60; return {x:0,y:top+scrollY,width:innerWidth,height:bot-top}; });
  await shot(1100, 900, '/company.html', 'press/braiain-board.png', () => { const h=[...document.querySelectorAll('h2')].find(h=>/Board of Directors/.test(h.textContent)); const bd=document.querySelector('.board'); const t=h.getBoundingClientRect().top-30,e=bd.getBoundingClientRect().bottom+30; return {x:0,y:t+scrollY,width:innerWidth,height:e-t}; });
  await b.close();
})();
