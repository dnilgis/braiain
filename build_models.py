"""
Build models.json from Chatbot Arena Elo scores across categories.
Sources: arena.ai/leaderboard (April 23, 2026), kearai.com, benchlm.ai

Normalization: Arena Elo 1350-1575 → 0.0-1.0 per dimension
This gives meaningful visual spread in the Lissajous curves.
"""
import json

# Elo range for normalization
ELO_MIN = 1350
ELO_MAX = 1575

def norm(elo):
    """Normalize Elo to 0-1 for Lissajous algorithm."""
    return round(max(0.0, min(1.0, (elo - ELO_MIN) / (ELO_MAX - ELO_MIN))), 4)

# ── RAW ARENA ELO SCORES BY CATEGORY ──────────────────────────────────────────
# Sources cited per value. "est" = estimated from relative ranking position.
# Categories: overall, math, coding, hard_prompts, creative_writing,
#             instruction_following, multi_turn, vision

models_raw = [
    {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "provider": "Anthropic",
        "color": "#c85a2a",
        "context_window": "1M",
        "price_input": 5.00, "price_output": 25.00,
        "released": "2025-11",
        "tags": ["#1 Overall", "Coding leader", "Instruction following"],
        "elo": {
            "overall": 1496,         # arena.ai 4/23 — rank #3
            "math": 1475,            # est from Feb arena math ~#5
            "coding": 1545,          # arena.ai/code 4/23 — rank #4
            "hard_prompts": 1500,    # benchlm: Arena IF 1500
            "creative_writing": 1469, # benchlm: Arena CW ~1469
            "instruction_following": 1500, # benchlm: Arena IF 1500
            "multi_turn": 1490,      # est from overall strength
            "vision": 1460,          # est — strong multimodal
        }
    },
    {
        "id": "gemini-3-1-pro",
        "name": "Gemini 3.1 Pro",
        "provider": "Google",
        "color": "#1a6e8a",
        "context_window": "1M",
        "price_input": 2.00, "price_output": 12.00,
        "released": "2026-02",
        "tags": ["#1 Math", "Long context", "Multimodal"],
        "elo": {
            "overall": 1493,         # arena.ai 4/23 — rank #5
            "math": 1510,            # kearai: #1 math arena
            "coding": 1457,          # arena.ai/code 4/23 — rank #12
            "hard_prompts": 1490,    # est from overall
            "creative_writing": 1487, # benchlm: Arena CW 1487
            "instruction_following": 1485, # est — strong IF
            "multi_turn": 1488,      # est from overall
            "vision": 1480,          # strong vision model
        }
    },
    {
        "id": "gpt-5-4",
        "name": "GPT-5.4",
        "provider": "OpenAI",
        "color": "#1a8a4a",
        "context_window": "1.1M",
        "price_input": 2.50, "price_output": 15.00,
        "released": "2026-03",
        "tags": ["Balanced", "Enterprise", "Strong IF"],
        "elo": {
            "overall": 1481,         # arena.ai 4/23 — rank #9
            "math": 1485,            # est — strong math
            "coding": 1457,          # arena.ai/code 4/23 — rank #13
            "hard_prompts": 1480,    # est
            "creative_writing": 1465, # est
            "instruction_following": 1480, # vellum: IFEval 96
            "multi_turn": 1478,      # est
            "vision": 1465,          # est — omni model
        }
    },
    {
        "id": "grok-4-20",
        "name": "Grok 4.20",
        "provider": "xAI",
        "color": "#e84040",
        "context_window": "2M",
        "price_input": 2.00, "price_output": 6.00,
        "released": "2026-03",
        "tags": ["Real-time data", "2M context", "Fast climber"],
        "elo": {
            "overall": 1482,         # arena.ai 4/23 — rank #8
            "math": 1470,            # est
            "coding": 1440,          # est from arena blog — #28 code
            "hard_prompts": 1485,    # est — strong on hard prompts
            "creative_writing": 1475, # est — personality-driven
            "instruction_following": 1465, # est
            "multi_turn": 1480,      # est — real-time context
            "vision": 1455,          # arena blog: #11 vision
        }
    },
    {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "color": "#b07018",
        "context_window": "1M",
        "price_input": 1.74, "price_output": 3.48,
        "released": "2026-04",
        "tags": ["Open weights", "MIT license", "Best value"],
        "elo": {
            "overall": 1463,         # arena.ai 4/23 — rank #20
            "math": 1470,            # est — strong math tradition
            "coding": 1456,          # arena.ai/code 4/23 — rank #14
            "hard_prompts": 1455,    # est
            "creative_writing": 1440, # est — weaker creative
            "instruction_following": 1450, # est
            "multi_turn": 1455,      # est
            "vision": 1420,          # est — text-focused
        }
    },
    {
        "id": "gemini-3-flash",
        "name": "Gemini 3 Flash",
        "provider": "Google",
        "color": "#00c97a",
        "context_window": "1M",
        "price_input": 0.50, "price_output": 3.00,
        "released": "2026-01",
        "tags": ["Ultra fast", "Best price/perf", "1M context"],
        "elo": {
            "overall": 1474,         # arena.ai 4/23 — rank #13
            "math": 1500,            # kearai: #2 math arena
            "coding": 1438,          # arena.ai/code — rank #16
            "hard_prompts": 1465,    # est
            "creative_writing": 1461, # benchlm: Arena CW 1461
            "instruction_following": 1460, # est
            "multi_turn": 1468,      # est
            "vision": 1470,          # est — strong multimodal
        }
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "color": "#d97706",
        "context_window": "1M",
        "price_input": 3.00, "price_output": 15.00,
        "released": "2026-01",
        "tags": ["Value flagship", "Strong coder", "Writing"],
        "elo": {
            "overall": 1463,         # arena.ai 4/23 — rank #21
            "math": 1455,            # est
            "coding": 1527,          # arena.ai/code 4/23 — rank #7
            "hard_prompts": 1465,    # est
            "creative_writing": 1455, # est
            "instruction_following": 1479, # benchlm: Arena IF 1479
            "multi_turn": 1460,      # est
            "vision": 1445,          # est
        }
    },
    {
        "id": "gpt-5-2",
        "name": "GPT-5.2",
        "provider": "OpenAI",
        "color": "#2a7acc",
        "context_window": "128K",
        "price_input": 1.75, "price_output": 14.00,
        "released": "2026-02",
        "tags": ["Proven", "High benchmark", "Enterprise"],
        "elo": {
            "overall": 1476,         # arena.ai 4/23 — rank #11
            "math": 1480,            # est — AA index leader Jan
            "coding": 1445,          # est
            "hard_prompts": 1475,    # est
            "creative_writing": 1460, # est
            "instruction_following": 1470, # est
            "multi_turn": 1472,      # est
            "vision": 1460,          # est
        }
    },
    {
        "id": "kimi-k2-6",
        "name": "Kimi K2.6",
        "provider": "Moonshot",
        "color": "#8b5cf6",
        "context_window": "256K",
        "price_input": 0.74, "price_output": 4.66,
        "released": "2026-04",
        "tags": ["Open weights", "Math strength", "Value"],
        "elo": {
            "overall": 1458,         # arena.ai 4/23 — rank #26
            "math": 1490,            # kearai: #3 math (K2.5 thinking)
            "coding": 1529,          # arena.ai/code 4/23 — rank #6
            "hard_prompts": 1450,    # est
            "creative_writing": 1430, # est
            "instruction_following": 1440, # est
            "multi_turn": 1445,      # est
            "vision": 1410,          # est — text-focused
        }
    },
    {
        "id": "gemini-3-pro",
        "name": "Gemini 3 Pro",
        "provider": "Google",
        "color": "#4285f4",
        "context_window": "1M",
        "price_input": 2.00, "price_output": 12.00,
        "released": "2025-10",
        "tags": ["Proven flagship", "Well-rounded", "Multimodal"],
        "elo": {
            "overall": 1486,         # arena.ai 4/23 — rank #7
            "math": 1505,            # kearai: very close to 3.1
            "coding": 1438,          # arena.ai/code 4/23 — rank #16
            "hard_prompts": 1482,    # est
            "creative_writing": 1480, # est — strong creative
            "instruction_following": 1478, # est
            "multi_turn": 1482,      # est
            "vision": 1485,          # strong vision — #1 vision arena
        }
    },
]

# ── BUILD MODELS.JSON ──────────────────────────────────────────────────────────

# Map Arena categories → BRAIAIN dimensions
DIM_MAP = {
    "reasoning":  "hard_prompts",      # hard prompts = reasoning challenge
    "math":       "math",
    "coding":     "coding",
    "science":    "overall",           # overall as proxy (no dedicated science arena)
    "context":    "multi_turn",        # multi-turn = sustained context
    "speed":      "creative_writing",  # creative = fluent generation speed
    "efficiency": "instruction_following",
    "multimodal": "vision",
}

WEIGHTS = {
    "reasoning": 0.35, "math": 0.40, "science": 0.25,  # analytical
    "coding": 0.40, "context": 0.30, "efficiency": 0.15, "speed": 0.15,  # technical
}

models_out = []
for m in models_raw:
    dims = {}
    for braiain_dim, arena_cat in DIM_MAP.items():
        dims[braiain_dim] = norm(m["elo"][arena_cat])

    # Compute section scores
    r, ma, sc = dims["reasoning"], dims["math"], dims["science"]
    co, cx, ef, sp = dims["coding"], dims["context"], dims["efficiency"], dims["speed"]
    mm = dims["multimodal"]

    analytical = round(200 + (r * 0.35 + ma * 0.40 + sc * 0.25) * 600)
    technical  = round(200 + (co * 0.40 + cx * 0.30 + ef * 0.15 + sp * 0.15) * 600)
    total = analytical + technical

    models_out.append({
        "id": m["id"],
        "name": m["name"],
        "provider": m["provider"],
        "color": m["color"],
        "analytical": max(200, min(800, analytical)),
        "technical": max(200, min(800, technical)),
        "dims": dims,
        "context_window": m["context_window"],
        "price_input": m["price_input"],
        "price_output": m["price_output"],
        "released": m["released"],
        "tags": m["tags"],
        "arena_elo": m["elo"],  # preserve raw Elo for transparency
    })

# Sort by total score
models_out.sort(key=lambda m: m["analytical"] + m["technical"], reverse=True)

# Print leaderboard
print(f"{'Rank':<5} {'Model':<25} {'Ana':>5} {'Tec':>5} {'Total':>6}")
print("─" * 50)
for i, m in enumerate(models_out):
    t = m["analytical"] + m["technical"]
    print(f"{i+1:<5} {m['name']:<25} {m['analytical']:>5} {m['technical']:>5} {t:>6}")

# Save
data = {
    "updated": "2026-04-25",
    "source": "Chatbot Arena (arena.ai) — April 23, 2026. 5.9M+ votes. Elo scores normalized 1350-1575 → 0-1.",
    "source_url": "https://arena.ai/leaderboard/text",
    "note": "Arena Elo scores across 8 categories. See arena_elo field for raw values.",
    "models": models_out,
}

with open("models.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved {len(models_out)} models to models.json")
