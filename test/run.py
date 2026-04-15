"""
BRAIAIN Standard — Complete Test Runner v1.0
Runs all 120 questions against a model via API and collects raw outputs.

Usage:
  python run.py --model claude-3-7-sonnet --provider anthropic --api-key sk-ant-...
  python run.py --model gpt-4o --provider openai --api-key sk-...
  python run.py --dry-run  # validate setup without real API calls

Requirements:
  pip install anthropic openai google-generativeai aiohttp
"""

import os
import json
import time
import asyncio
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional


# ── CONFIG ────────────────────────────────────────────────────────────────────

def get_quarterly_seed() -> int:
    """Derive seed from quarter identifier + secret salt.
    Salt must be set via BRAIAIN_SEED_SALT env var.
    Without it, novel questions become predictable and contamination
    detection is defeated."""
    quarter = os.environ.get("BRAIAIN_QUARTER", "2026Q2")
    salt = os.environ.get("BRAIAIN_SEED_SALT", "")
    if not salt:
        print("WARNING: BRAIAIN_SEED_SALT not set. Using unsalted seed.")
        print("Novel questions are predictable without a salt. Set this before publishing.")
    combined = f"{quarter}:{salt}"
    return int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)

QUARTERLY_SEED = get_quarterly_seed()

MODEL_ENDPOINTS = {
    "anthropic": {
        "claude-3-7-sonnet":  "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet":  "claude-3-5-sonnet-20241022",
    },
    "openai": {
        "gpt-4o":    "gpt-4o",
        "o3-mini":   "o3-mini",
    },
    "google": {
        "gemini-2-0-pro":   "gemini-2.0-pro",
        "gemini-2-0-flash": "gemini-2.0-flash",
    },
    "meta": {
        "llama-3-3-70b": None,  # Run locally or via together.ai
    },
    "deepseek": {
        "deepseek-v3": None,  # Run via deepseek API
    },
    "mistral": {
        "mistral-large-2": None,
    },
}

# Judge selection: never use same family to judge itself
JUDGE_FOR = {
    "anthropic": ("openai", "gpt-4o"),
    "openai":    ("google", "gemini-2.0-flash"),
    "google":    ("anthropic", "claude-3-5-sonnet"),
    "meta":      ("openai", "gpt-4o"),
    "deepseek":  ("openai", "gpt-4o"),
    "mistral":   ("openai", "gpt-4o"),
}

SCORING_WEIGHTS = {
    "analytical": {"reasoning": 0.35, "math": 0.40, "science": 0.25},
    "technical":  {"coding": 0.40, "context": 0.30, "efficiency": 0.15, "speed": 0.15},
}


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class Answer:
    question_id: str
    dimension: str
    prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class RunResult:
    model_id: str
    provider: str
    run_date: str
    seed: int
    answers: list
    speed_measurements: dict
    raw_hash: str  # SHA256 of all answers for immutability


# ── API CALLERS ────────────────────────────────────────────────────────────────

class AnthropicCaller:
    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, model_id: str, prompt: str, max_tokens: int = 1024) -> dict:
        t0 = time.time()
        response = self.client.messages.create(
            model=MODEL_ENDPOINTS["anthropic"].get(model_id, model_id),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - t0) * 1000
        return {
            "text": response.content[0].text,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
            "latency_ms": latency,
        }


class OpenAICaller:
    def __init__(self, api_key: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def call(self, model_id: str, prompt: str, max_tokens: int = 1024) -> dict:
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=MODEL_ENDPOINTS["openai"].get(model_id, model_id),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - t0) * 1000
        return {
            "text": response.choices[0].message.content,
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
            "latency_ms": latency,
        }


class GoogleCaller:
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai

    def call(self, model_id: str, prompt: str, max_tokens: int = 1024) -> dict:
        model_name = MODEL_ENDPOINTS["google"].get(model_id, model_id)
        model = self.genai.GenerativeModel(model_name)
        t0 = time.time()
        response = model.generate_content(prompt)
        latency = (time.time() - t0) * 1000
        return {
            "text": response.text,
            "tokens_in": response.usage_metadata.prompt_token_count,
            "tokens_out": response.usage_metadata.candidates_token_count,
            "latency_ms": latency,
        }


def get_caller(provider: str, api_key: str):
    callers = {
        "anthropic": AnthropicCaller,
        "openai":    OpenAICaller,
        "google":    GoogleCaller,
    }
    if provider not in callers:
        raise ValueError(f"Unsupported provider: {provider}. Add a caller class.")
    return callers[provider](api_key)


# ── SPEED MEASUREMENT ─────────────────────────────────────────────────────────

SPEED_PROMPTS = [
    {"id": "V01", "prompt": "Hi", "type": "ttft"},
    {"id": "V02", "prompt": "What is 144 divided by 12?", "type": "ttft"},
    {"id": "V03", "prompt": "Write exactly 500 words about the ocean.", "type": "tps"},
    {"id": "V04", "prompt": "List the first 50 prime numbers, one per line.", "type": "tps"},
    {"id": "V05", "prompt": "Translate this into French, Spanish, and German: 'Artificial intelligence is transforming the way we work, learn, and communicate.'", "type": "tps"},
]

def measure_speed(caller, model_id: str) -> dict:
    """Measure TTFT and TPS for the speed dimension."""
    print("  Measuring speed...")
    ttft_values = []
    tps_values = []

    for sp in SPEED_PROMPTS:
        try:
            result = caller.call(model_id, sp["prompt"], max_tokens=600)
            if sp["type"] == "ttft":
                ttft_values.append(result["latency_ms"])
            else:
                tokens = result["tokens_out"]
                time_s = result["latency_ms"] / 1000
                if time_s > 0:
                    tps_values.append(tokens / time_s)
        except Exception as e:
            print(f"    Speed measurement failed for {sp['id']}: {e}")

    return {
        "avg_ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else None,
        "avg_tps": sum(tps_values) / len(tps_values) if tps_values else None,
        "speed_score": None,  # Computed after all models tested (percentile rank)
    }


# ── AUTOMATED SCORING ─────────────────────────────────────────────────────────

def auto_score_exact(response: str, reference: str, tolerance: float = 0.001) -> float:
    """Score questions with exact or near-exact answers.
    Handles: plain strings, numbers, fractions, lists, booleans, None."""
    import re
    import ast

    resp_clean = response.strip().lower().replace(",", "").replace("$", "")
    ref_clean = reference.strip().lower()

    if resp_clean == ref_clean:
        return 1.0

    # ── Structural comparison (lists, tuples, booleans, None) ──
    # Handles "[1, 2, 3]" vs "[1,2,3]" and whitespace differences
    try:
        # Find a list/value in the response that matches reference structure
        ref_val = ast.literal_eval(reference.strip())

        # Try parsing the full response first
        for candidate in [response.strip(), resp_clean]:
            try:
                resp_val = ast.literal_eval(candidate)
                if resp_val == ref_val:
                    return 1.0
            except Exception:
                pass

        # Try to find a bracketed expression in the response
        bracket_matches = re.findall(r'\[.*?\]', response, re.DOTALL)
        for match in bracket_matches:
            try:
                resp_val = ast.literal_eval(match.strip())
                if resp_val == ref_val:
                    return 1.0
            except Exception:
                pass

        # Try boolean/None matching anywhere in response
        if isinstance(ref_val, bool) or ref_val is None:
            if str(ref_val) in response:
                return 1.0
    except Exception:
        pass

    # ── Numeric comparison ──
    try:
        resp_num = float(resp_clean.split()[0])
        ref_num = float(ref_clean)
        if abs(resp_num - ref_num) / (abs(ref_num) + 1e-9) < tolerance:
            return 1.0
    except (ValueError, IndexError):
        pass

    # ── Fraction comparison ──
    if "/" in ref_clean:
        try:
            from fractions import Fraction
            ref_frac = Fraction(ref_clean)
            match = re.search(r'(\d+)\s*/\s*(\d+)', resp_clean)
            if match:
                resp_frac = Fraction(int(match.group(1)), int(match.group(2)))
                if resp_frac == ref_frac:
                    return 1.0
        except Exception:
            pass

    # ── Substring match (for labelled answers like "Invalid") ──
    if ref_clean in resp_clean:
        return 1.0

    return 0.0


def auto_score_unit_tests(response: str, unit_tests: list) -> float:
    """
    Extract code from response and run unit tests.
    Returns fraction of tests passed.
    """
    # Extract code block
    import re
    code_match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # Try to find function definition
        func_match = re.search(r'(def \w+.*?)(?=\n\n|\Z)', response, re.DOTALL)
        code = func_match.group(1) if func_match else response

    passed = 0
    for test in unit_tests:
        try:
            namespace = {}
            exec(code, namespace)
            # Find the function name
            func_name = [k for k in namespace if callable(namespace[k]) and not k.startswith("_")]
            if not func_name:
                continue
            fn = namespace[func_name[0]]
            result = eval(f"fn{test['input']}")
            expected = eval(test['expected'])
            if result == expected:
                passed += 1
        except Exception:
            pass

    return passed / len(unit_tests) if unit_tests else 0.0


# ── AI JUDGE ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a precise academic evaluator scoring an AI model's answer.
Apply the rubric STRICTLY. Do not reward style. Do not penalise brevity if correct.
Respond ONLY with valid JSON: {"score": 0.0, "reason": "one sentence"}
Score must be exactly 0.0, 0.5, or 1.0."""

def judge_answer(judge_caller, judge_model: str,
                 question: str, reference: str,
                 model_answer: str, rubric: str) -> dict:
    """Call AI judge and return score + reason."""
    prompt = f"""Question: {question}

Reference answer: {reference}

Model answer: {model_answer}

Rubric: {rubric}"""

    try:
        response = judge_caller.call(judge_model, prompt, max_tokens=100)
        import re
        json_match = re.search(r'\{.*?\}', response["text"], re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0))
            if score not in [0.0, 0.5, 1.0]:
                score = round(score * 2) / 2  # round to nearest 0.5
            return {"score": score, "reason": result.get("reason", ""), "error": None}
    except Exception as e:
        pass

    return {"score": 0.0, "reason": "Judge call failed", "error": "parse_error"}


# ── CONTAMINATION DETECTION ───────────────────────────────────────────────────

def detect_contamination(fixed_scores: dict, novel_scores: dict,
                          threshold: float = 0.20) -> list:
    flags = []
    for dim in fixed_scores:
        if dim not in novel_scores:
            continue
        gap = fixed_scores[dim] - novel_scores[dim]
        if gap > threshold:
            flags.append({
                "dimension": dim,
                "fixed": round(fixed_scores[dim], 3),
                "novel": round(novel_scores[dim], 3),
                "gap": round(gap, 3),
                "status": "FLAGGED",
                "action": "Do not publish until investigated"
            })
    return flags


# ── SCORE FORMULA ─────────────────────────────────────────────────────────────

def compute_final_scores(dim_scores: dict) -> dict:
    r = dim_scores.get("reasoning", 0)
    m = dim_scores.get("math", 0)
    s = dim_scores.get("science", 0)
    c = dim_scores.get("coding", 0)
    x = dim_scores.get("context", 0)
    e = dim_scores.get("efficiency", 0)
    v = dim_scores.get("speed", 0)
    mm = dim_scores.get("multimodal", None)

    # Multimodal: if model has no vision capability, record as None.
    # Do NOT substitute mean of other dimensions — that rewards models
    # for capabilities they don't have. The DNA visualization should
    # render the multimodal axis as 0 when None.

    ana_raw = r * 0.35 + m * 0.40 + s * 0.25
    tec_raw = c * 0.40 + x * 0.30 + e * 0.15 + v * 0.15

    analytical = round(200 + ana_raw * 600)
    technical  = round(200 + tec_raw * 600)
    total      = analytical + technical

    return {
        "dimensions": {k: round(val, 4) if val is not None else None for k, val in {
            "reasoning": r, "math": m, "science": s, "coding": c,
            "context": x, "efficiency": e, "speed": v, "multimodal": mm
        }.items()},
        "analytical": max(200, min(800, analytical)),
        "technical":  max(200, min(800, technical)),
        "total":      max(400, min(1600, total)),
        "confidence": 15,
    }


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def load_questions() -> list:
    """Load fixed questions from JSON bank."""
    questions = []
    bank_path = Path(__file__).parent / "questions" / "fixed.json"
    with open(bank_path) as f:
        bank = json.load(f)

    for dim_name, dim_data in bank["dimensions"].items():
        for q in dim_data.get("questions", []):
            q["dimension"] = dim_name
            questions.append(q)

    # Load novel questions if they exist
    for novel_file in (Path(__file__).parent / "questions").glob("novel_*.json"):
        with open(novel_file) as f:
            novel_qs = json.load(f)
        questions.extend(novel_qs)

    return questions


def run_test(model_id: str, provider: str, api_key: str,
             dry_run: bool = False) -> RunResult:
    print(f"\n{'='*60}")
    print(f"BRAIAIN Standard v1.0")
    print(f"Model:    {model_id}")
    print(f"Provider: {provider}")
    print(f"Date:     {datetime.now(timezone.utc).isoformat()}")
    print(f"Seed:     {QUARTERLY_SEED}")
    print(f"Dry run:  {dry_run}")
    print(f"{'='*60}\n")

    if not dry_run:
        caller = get_caller(provider, api_key)
    else:
        class MockCaller:
            def call(self, model, prompt, max_tokens=1024):
                return {"text": "Mock answer for dry run", "tokens_in": 10,
                        "tokens_out": 5, "latency_ms": 100}
        caller = MockCaller()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")

    answers = []
    dim_raw_scores = {d: [] for d in
                      ["reasoning", "math", "science", "coding", "context",
                       "efficiency", "speed", "multimodal"]}

    for i, q in enumerate(questions):
        qid = q.get("id", f"Q{i+1}")
        dim = q.get("dimension", "unknown")
        prompt = q.get("prompt", "")
        tier = q.get("tier", 2)
        reference = q.get("reference", "")
        rubric = q.get("rubric", "")

        print(f"  [{i+1:03d}/{len(questions)}] {qid} ({dim}) tier={tier}", end="", flush=True)

        try:
            result = caller.call(model_id, prompt, max_tokens=1024)
            response_text = result["text"]

            # Auto-score where possible
            if tier == 1:
                if "unit_tests" in q and q["unit_tests"]:
                    score = auto_score_unit_tests(response_text, q["unit_tests"])
                else:
                    score = auto_score_exact(response_text, reference)
                method = "auto"
            else:
                score = None  # Will be judged separately
                method = "pending_judge"

            print(f" → {score if score is not None else 'queued'}")

            answer = {
                "question_id": qid,
                "dimension": dim,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response_text,
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "latency_ms": result["latency_ms"],
                "score": score,
                "score_method": method,
                "tier": tier,
                "reference": reference,
                "rubric": rubric,
            }
            answers.append(answer)

            if score is not None and dim in dim_raw_scores:
                dim_raw_scores[dim].append(score)

        except Exception as e:
            print(f" → ERROR: {e}")
            answers.append({
                "question_id": qid,
                "dimension": dim,
                "prompt": prompt[:100],
                "response": None,
                "score": 0.0,
                "score_method": "error",
                "error": str(e),
            })

        time.sleep(0.2)  # Rate limiting

    # Measure speed
    speed_data = measure_speed(caller, model_id)

    # Compute dimension averages
    dim_scores = {}
    for dim, scores in dim_raw_scores.items():
        if scores:
            dim_scores[dim] = sum(scores) / len(scores)
        elif dim == "speed":
            dim_scores[dim] = 0.5  # Placeholder until percentile computed
        # multimodal left as None if no questions answered

    # Hash for immutability — includes model ID, question IDs, prompts,
    # and responses so swapping which model generated which answers
    # would break the hash (Troy Hunt review)
    hash_payload = {
        "model_id": model_id,
        "provider": provider,
        "seed": QUARTERLY_SEED,
        "records": [
            {
                "qid": a.get("question_id", ""),
                "prompt": a.get("prompt", ""),
                "response": a.get("response", ""),
            }
            for a in answers
        ]
    }
    raw_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode()
    ).hexdigest()[:16]

    return RunResult(
        model_id=model_id,
        provider=provider,
        run_date=datetime.now(timezone.utc).isoformat(),
        seed=QUARTERLY_SEED,
        answers=answers,
        speed_measurements=speed_data,
        raw_hash=raw_hash,
    )


def save_result(result: RunResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{result.model_id}-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    path = output_dir / fname
    with open(path, "w") as f:
        json.dump({
            "model_id": result.model_id,
            "provider": result.provider,
            "run_date": result.run_date,
            "seed": result.seed,
            "raw_hash": result.raw_hash,
            "speed_measurements": result.speed_measurements,
            "answer_count": len(result.answers),
            "answers": result.answers,
        }, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="BRAIAIN Standard test runner")
    parser.add_argument("--model", default="gpt-4o", help="Model ID")
    parser.add_argument("--provider", default="openai",
                        choices=list(JUDGE_FOR.keys()), help="Provider")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without real API calls")
    parser.add_argument("--output-dir", default="output/results",
                        help="Directory for result files")
    args = parser.parse_args()

    # API key from environment only — never accept on CLI
    # (CLI args appear in shell history and process listings)
    api_key = os.environ.get("BRAIAIN_API_KEY", os.environ.get("API_KEY", ""))

    if not args.dry_run and not api_key:
        print("Error: set BRAIAIN_API_KEY (or API_KEY) environment variable.")
        print("  export BRAIAIN_API_KEY=sk-...")
        print("API keys are not accepted via --api-key to avoid shell history exposure.")
        return

    result = run_test(
        model_id=args.model,
        provider=args.provider,
        api_key=api_key,
        dry_run=args.dry_run,
    )

    output_path = save_result(result, Path(args.output_dir))

    # Quick summary
    scored = [a for a in result.answers if a.get("score") is not None]
    if scored:
        avg = sum(a["score"] for a in scored) / len(scored)
        print(f"\nQuick summary:")
        print(f"  Questions answered: {len(result.answers)}")
        print(f"  Auto-scored:        {len(scored)}")
        print(f"  Average raw score:  {avg:.3f}")
        print(f"  Raw output hash:    {result.raw_hash}")
        print(f"\nNext step: run judge.py on {output_path} for Tier 2/3 questions")


if __name__ == "__main__":
    main()
