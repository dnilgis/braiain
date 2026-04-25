"""
Build historical timeline data for BRAIAIN "The Scale" visualization.
Picasso's Bulls concept: each generation's DNA figure grows more complex.

Source: BenchLM Arena Leaderboard History (benchlm.ai/llm-leaderboard-history)
Arena Elo milestones + crown changes + category leaders.
"""
import json

ELO_MIN = 1000
ELO_MAX = 1575

def norm(elo):
    return round(max(0.0, min(1.0, (elo - ELO_MIN) / (ELO_MAX - ELO_MIN))), 4)

# Each milestone model with estimated 8-dimension Elo scores.
# Early models: most dimensions weak, one or two strengths.
# Later models: all dimensions strengthen, profiles differentiate.

milestones = [
    {
        "id": "vicuna-13b",
        "name": "Vicuna 13B",
        "provider": "LMSYS",
        "date": "2023-05",
        "event": "The beginning. First Arena model tracked.",
        "overall_elo": 1094,
        "dims_elo": {
            "overall": 1094, "math": 1020, "coding": 1030,
            "hard_prompts": 1050, "creative_writing": 1080,
            "instruction_following": 1040, "multi_turn": 1060, "vision": 1000,
        }
    },
    {
        "id": "gpt-4-0314",
        "name": "GPT-4",
        "provider": "OpenAI",
        "date": "2023-12",
        "event": "The frontier opens. First model to cross 1100 Elo.",
        "overall_elo": 1150,
        "dims_elo": {
            "overall": 1150, "math": 1120, "coding": 1140,
            "hard_prompts": 1130, "creative_writing": 1100,
            "instruction_following": 1110, "multi_turn": 1090, "vision": 1000,
        }
    },
    {
        "id": "claude-3-opus",
        "name": "Claude 3 Opus",
        "provider": "Anthropic",
        "date": "2024-04",
        "event": "Anthropic takes the crown for the first time.",
        "overall_elo": 1256,
        "dims_elo": {
            "overall": 1256, "math": 1200, "coding": 1240,
            "hard_prompts": 1250, "creative_writing": 1230,
            "instruction_following": 1220, "multi_turn": 1210, "vision": 1050,
        }
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "OpenAI",
        "date": "2024-05",
        "event": "Multimodal arrives. Vision becomes a real dimension.",
        "overall_elo": 1285,
        "dims_elo": {
            "overall": 1285, "math": 1220, "coding": 1260,
            "hard_prompts": 1270, "creative_writing": 1250,
            "instruction_following": 1240, "multi_turn": 1230, "vision": 1180,
        }
    },
    {
        "id": "llama-3-1-405b",
        "name": "Llama 3.1 405B",
        "provider": "Meta",
        "date": "2024-08",
        "event": "Open source peaks. The gap narrows to 30 points.",
        "overall_elo": 1262,
        "dims_elo": {
            "overall": 1262, "math": 1210, "coding": 1250,
            "hard_prompts": 1230, "creative_writing": 1200,
            "instruction_following": 1190, "multi_turn": 1220, "vision": 1000,
        }
    },
    {
        "id": "o1-2024-12-17",
        "name": "o1",
        "provider": "OpenAI",
        "date": "2025-01",
        "event": "The reasoning revolution. Chain-of-thought changes everything.",
        "overall_elo": 1350,
        "dims_elo": {
            "overall": 1350, "math": 1380, "coding": 1340,
            "hard_prompts": 1370, "creative_writing": 1280,
            "instruction_following": 1300, "multi_turn": 1290, "vision": 1150,
        }
    },
    {
        "id": "deepseek-r1",
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "date": "2025-02",
        "event": "Open source nearly catches the frontier. Gap: 4 points.",
        "overall_elo": 1361,
        "dims_elo": {
            "overall": 1361, "math": 1390, "coding": 1350,
            "hard_prompts": 1360, "creative_writing": 1290,
            "instruction_following": 1310, "multi_turn": 1300, "vision": 1100,
        }
    },
    {
        "id": "gemini-2-5-pro",
        "name": "Gemini 2.5 Pro",
        "provider": "Google",
        "date": "2025-07",
        "event": "The longest reign begins. 5 months at #1. 1M context.",
        "overall_elo": 1450,
        "dims_elo": {
            "overall": 1450, "math": 1470, "coding": 1420,
            "hard_prompts": 1440, "creative_writing": 1430,
            "instruction_following": 1410, "multi_turn": 1440, "vision": 1280,
        }
    },
    {
        "id": "claude-opus-4-5",
        "name": "Claude Opus 4.5",
        "provider": "Anthropic",
        "date": "2025-11",
        "event": "The thinking models arrive. Extended reasoning becomes standard.",
        "overall_elo": 1469,
        "dims_elo": {
            "overall": 1469, "math": 1450, "coding": 1468,
            "hard_prompts": 1475, "creative_writing": 1440,
            "instruction_following": 1460, "multi_turn": 1455, "vision": 1290,
        }
    },
    {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "provider": "Anthropic",
        "date": "2026-02",
        "event": "First model to break 1500 Elo. The new benchmark.",
        "overall_elo": 1500,
        "dims_elo": {
            "overall": 1496, "math": 1475, "coding": 1545,
            "hard_prompts": 1500, "creative_writing": 1469,
            "instruction_following": 1500, "multi_turn": 1490, "vision": 1460,
        }
    },
]

# Map to BRAIAIN dimensions
DIM_MAP = {
    "reasoning": "hard_prompts",
    "math": "math",
    "coding": "coding",
    "science": "overall",
    "context": "multi_turn",
    "speed": "creative_writing",
    "efficiency": "instruction_following",
    "multimodal": "vision",
}

WEIGHTS = {
    "reasoning": 0.35, "math": 0.40, "science": 0.25,
    "coding": 0.40, "context": 0.30, "efficiency": 0.15, "speed": 0.15,
}

timeline = []
for m in milestones:
    dims = {}
    for bd, ac in DIM_MAP.items():
        dims[bd] = norm(m["dims_elo"][ac])

    r, ma, sc = dims["reasoning"], dims["math"], dims["science"]
    co, cx, ef, sp = dims["coding"], dims["context"], dims["efficiency"], dims["speed"]

    analytical = round(200 + (r*0.35 + ma*0.40 + sc*0.25) * 600)
    technical  = round(200 + (co*0.40 + cx*0.30 + ef*0.15 + sp*0.15) * 600)
    total = analytical + technical

    timeline.append({
        "id": m["id"],
        "name": m["name"],
        "provider": m["provider"],
        "date": m["date"],
        "event": m["event"],
        "arena_elo": m["overall_elo"],
        "analytical": max(200, min(800, analytical)),
        "technical": max(200, min(800, technical)),
        "total": max(400, min(1600, total)),
        "dims": dims,
    })

print("THE SCALE — From Silence to Perfection")
print("=" * 60)
for i, m in enumerate(timeline):
    print(f"  {m['date']}  {m['name']:<22} Elo {m['arena_elo']:>4} → BRAIAIN {m['total']:>4}")
    print(f"           {m['event']}")
    if i < len(timeline) - 1:
        print(f"           │")

with open("timeline.json", "w") as f:
    json.dump({
        "title": "The Scale",
        "subtitle": "From silence to perfection. 37 months of AI evolution.",
        "source": "Chatbot Arena (arena.ai) — Elo milestone data via BenchLM.ai",
        "source_url": "https://benchlm.ai/llm-leaderboard-history",
        "milestones": timeline,
    }, f, indent=2)
print(f"\nSaved to timeline.json")
