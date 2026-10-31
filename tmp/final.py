import json, urllib.request, urllib.parse, io, os
from PIL import Image, ImageOps, ImageEnhance
UA = {"User-Agent": "braiain-board-photos/1.0 (hello@braiain.com)"}
PICKS = {  # key: (file title, crop box as fractions of the original, or None)
  "goose":   ("File:Anser cygnoides (white Chinese).jpg", None),
  "note":    ("File:Post-It-Note-Bureaucrat.jpg", (0.52, 0.42, 1.0, 1.0)),
  "fern":    ("File:Helecho de Boston (Nephrolepis exaltata).jpg", None),
  "ball":    ("File:Magic eight ball.png", None),
  "toaster": ("File:General Electric Model D-12 toaster, 1910s.jpg", None),
  "vacuum":  ("File:XiaomiMiRobotVacuumCleanerRobot.jpg", None),
  "mug":     ("File:PlainMug.jpg", None),
  "laptop":  ("File:LCD Crack of Laptop Computer.jpg", None),
}
os.makedirs("tmp/final", exist_ok=True)
credits = {}
for key, (title, box) in PICKS.items():
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","format":"json","titles":title,"prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":"1600","iiextmetadatafilter":"LicenseShortName|Artist"})
    p = list(json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA))) ["query"]["pages"].values())[0]
    ii = p["imageinfo"][0]
    im = Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(ii.get("thumburl") or ii["url"], headers=UA)).read()))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
    im = im.convert("L")
    if box:
        w, h = im.size; im = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
    im = ImageOps.fit(im, (360, 432), centering=(0.5, 0.45))
    im = ImageEnhance.Contrast(ImageOps.autocontrast(im, cutoff=1)).enhance(1.08)
    im.save(f"tmp/final/{key}.jpg", quality=86, optimize=True, progressive=True)
    md = ii.get("extmetadata", {})
    credits[key] = {"title": title[5:], "license": md.get("LicenseShortName", {}).get("value", ""), "page": ii.get("descriptionurl")}
    print(key, credits[key]["license"])
json.dump(credits, open("tmp/final/credits.json", "w"), indent=1)
