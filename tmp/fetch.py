import json, urllib.request, urllib.parse, os, io
from PIL import Image, ImageOps
UA = {"User-Agent": "braiain-board-photos/1.0 (hello@braiain.com)"}
SUBJECTS = {
  "note": ["yellow sticky note", "single post-it note", "sticky note on desk", "post-it"],
  "fern": ["boston fern", "fern houseplant", "potted fern", "fern in a pot", "Polypodiopsida pot"],
  "ball": ["8-ball billiard", "billiard ball 8", "magic 8 ball toy", "eight ball"],
}
OK = ("cc0", "public domain", "pd")
def api(q):
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","format":"json","generator":"search","gsrnamespace":"6","gsrsearch":q+" filetype:bitmap","gsrlimit":"40","prop":"imageinfo","iiprop":"url|extmetadata|size|mime","iiurlwidth":"900","iiextmetadatafilter":"LicenseShortName|Artist|ImageDescription"})
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30))
os.makedirs("tmp/cand", exist_ok=True)
credits = {}
for key, qs in SUBJECTS.items():
    got = 0
    seen = set()
    for q in qs:
        try: pages = api(q).get("query", {}).get("pages", {})
        except Exception as e: print("ERR", key, q, e); continue
        for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            ii = (p.get("imageinfo") or [{}])[0]
            lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "")
            if p["title"] in seen or ii.get("mime") not in ("image/jpeg", "image/png"): continue
            seen.add(p["title"])
            if not lic.lower().startswith(OK): continue
            if ii.get("width", 0) < 600: continue
            try:
                data = urllib.request.urlopen(urllib.request.Request(ii["thumburl"], headers=UA), timeout=30).read()
                im = Image.open(io.BytesIO(data)).convert("L")
                im = ImageOps.fit(im, (500, 600), centering=(0.5, 0.45))
                name = f"{key}-r{got}.jpg"
                im.save("tmp/cand/" + name, quality=85)
                credits[name] = {"title": p["title"], "license": lic, "page": ii.get("descriptionurl"), "artist": ((ii.get("extmetadata") or {}).get("Artist") or {}).get("value","")[:200]}
                got += 1
                print(key, got, p["title"], lic)
            except Exception as e:
                print("dl fail", p["title"], e)
            if got >= 8: break
        if got >= 8: break
old = json.load(open("tmp/cand/credits.json")) if os.path.exists("tmp/cand/credits.json") else {}
old.update(credits); json.dump(old, open("tmp/cand/credits.json", "w"), indent=1)
