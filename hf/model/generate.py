"""Talk to brAIAIn from the command line. Standard library only.

    python generate.py "What is the meaning of life?"

Uses brAIAIn-latest.json (the newest version, updated after every release) if present,
otherwise brAIAIn-0.json.
"""
import json, random, re, sys

import os
_here = os.path.dirname(os.path.abspath(__file__))
_file = "brAIAIn-latest.json" if os.path.exists(os.path.join(_here, "brAIAIn-latest.json")) else "brAIAIn-0.json"
m = json.load(open(os.path.join(_here, _file), encoding="utf-8"))
BOS, EOS = "<s>", "</s>"


def pick(table):
    keys = sorted(table)
    r = random.random() * sum(table.values())
    for k in keys:
        r -= table[k]
        if r <= 0:
            return k
    return keys[-1]


def answer(prompt="", max_words=40):
    known = {re.sub(r"[^a-z0-9']", "", w.lower()): w for w in m["bi"] if w != BOS}
    words = [re.sub(r"[^a-z0-9']", "", w.lower()) for w in prompt.split()]
    starts = [known[w] for w in words if w in known and len(w) > 3]
    a, b, out = BOS, BOS, []
    if starts:
        b = random.choice(starts)
        out.append(b)
    while len(out) < max_words:
        k = a + " " + b
        nxt = m["tri"].get(k) or m["bi"].get(b)
        if not nxt:
            break
        c = pick(nxt)
        if c == EOS:
            break
        out.append(c)
        a, b = b, c
    s = " ".join(out)
    return s[:1].upper() + s[1:]


print(answer(" ".join(sys.argv[1:])))
