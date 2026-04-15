"""
BRAIAIN Standard — AI Judge Runner
Processes Tier 2 answers through the AI judge panel.
Run after run.py completes.

Usage:
  python judge.py --result output/results/gpt-4o-20260401-1200.json \
                  --judge-provider google --judge-api-key YOUR_KEY
"""

import json
import re
import time
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone


JUDGE_SYSTEM = """You are a precise academic evaluator scoring an AI model's answer against a reference answer and rubric.

Apply the rubric STRICTLY.
Do not reward style over substance.
Do not penalise concise answers if they are correct.
Do not give partial credit unless the rubric explicitly allows 0.5.

Respond ONLY with valid JSON in this exact format:
{"score": 0.0, "reason": "one sentence, max 20 words"}

score must be exactly 0.0, 0.5, or 1.0 — no other values."""


def call_anthropic_judge(api_key: str, model: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=100,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"text": response.content[0].text}


def call_openai_judge(api_key: str, model: str, prompt: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=100,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt}
        ]
    )
    return {"text": response.choices[0].message.content}


def call_google_judge(api_key: str, model: str, prompt: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model,
        system_instruction=JUDGE_SYSTEM
    )
    response = m.generate_content(prompt)
    return {"text": response.text}


JUDGE_CALLERS = {
    "anthropic": call_anthropic_judge,
    "openai":    call_openai_judge,
    "google":    call_google_judge,
}

# Which judge model to use per provider
JUDGE_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
}


def build_judge_prompt(question: str, reference: str,
                       model_answer: str, rubric: str) -> str:
    return f"""Question: {question}

Reference answer: {reference}

Model answer: {model_answer}

Rubric: {rubric}"""


def parse_judge_response(text: str) -> dict:
    """Extract score and reason from judge response."""
    # Try direct JSON parse
    try:
        result = json.loads(text.strip())
        score = float(result.get("score", -1))
        if score in [0.0, 0.5, 1.0]:
            return {"score": score, "reason": result.get("reason", ""), "error": None}
    except Exception:
        pass

    # Try to find JSON in the response
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            score = float(result.get("score", -1))
            if score in [0.0, 0.5, 1.0]:
                return {"score": score, "reason": result.get("reason", ""), "error": None}
        except Exception:
            pass

    # Last resort: look for score value
    score_match = re.search(r'"score"\s*:\s*(0\.0|0\.5|1\.0)', text)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "reason": "extracted from malformed response",
            "error": "parse_warning"
        }

    return {"score": 0.0, "reason": "failed to parse judge response", "error": "parse_error"}


def judge_file(result_path: Path, judge_provider: str, judge_api_key: str,
               dry_run: bool = False) -> dict:
    """Load a result file and run AI judge on all Tier 2/3 answers."""

    with open(result_path) as f:
        result = json.load(f)

    model_id = result["model_id"]
    provider = result["provider"]
    judge_model = JUDGE_MODELS[judge_provider]

    print(f"\n{'='*60}")
    print(f"BRAIAIN Judge Runner")
    print(f"Model being judged: {model_id} ({provider})")
    print(f"Judge: {judge_provider} / {judge_model}")
    print(f"{'='*60}\n")

    # ── INDEPENDENCE POLICY: hard stop if judge is same family as model ──
    # Map providers to their corporate family for broader matching
    PROVIDER_FAMILIES = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "google",
        "meta": "meta",
        "deepseek": "deepseek",
        "mistral": "mistral",
    }
    model_family = PROVIDER_FAMILIES.get(provider, provider)
    judge_family = PROVIDER_FAMILIES.get(judge_provider, judge_provider)

    if judge_family == model_family:
        print(f"\n ERROR: Judge provider '{judge_provider}' is in the same family "
              f"as model provider '{provider}'.")
        print("This violates BRAIAIN independence policy.")
        print("Use a different --judge-provider. Aborting.")
        import sys
        sys.exit(1)

    caller = JUDGE_CALLERS[judge_provider]
    pending = [a for a in result["answers"] if a.get("score_method") == "pending_judge"]
    print(f"Tier 2/3 answers to judge: {len(pending)}\n")

    judged_count = 0
    failed_count = 0

    for answer in result["answers"]:
        if answer.get("score_method") != "pending_judge":
            continue
        if answer.get("response") is None:
            answer["score"] = 0.0
            answer["score_method"] = "error_no_response"
            continue

        prompt = build_judge_prompt(
            question=answer.get("prompt", ""),
            reference=answer.get("reference", ""),
            model_answer=answer["response"],
            rubric=answer.get("rubric", "No rubric provided")
        )

        qid = answer["question_id"]
        dim = answer["dimension"]
        print(f"  Judging {qid} ({dim})...", end="", flush=True)

        if dry_run:
            answer["score"] = 0.75
            answer["score_method"] = "dry_run_judge"
            answer["judge_reason"] = "dry run"
            print(" [dry run] 0.75")
            continue

        try:
            response = caller(judge_api_key, judge_model, prompt)
            parsed = parse_judge_response(response["text"])

            answer["score"] = parsed["score"]
            answer["judge_reason"] = parsed["reason"]
            answer["score_method"] = f"judge_{judge_provider}"
            if parsed["error"]:
                answer["judge_error"] = parsed["error"]

            print(f" → {parsed['score']} ({parsed['reason'][:40]})")
            judged_count += 1

        except Exception as e:
            print(f" → FAILED: {e}")
            answer["score"] = 0.0
            answer["score_method"] = "judge_failed"
            answer["judge_error"] = str(e)
            failed_count += 1

        time.sleep(0.3)  # Rate limiting

    print(f"\nJudged: {judged_count} | Failed: {failed_count}")

    # Recompute dimension scores with newly judged answers
    dim_raw_scores = {}
    for answer in result["answers"]:
        dim = answer.get("dimension")
        score = answer.get("score")
        if dim and score is not None:
            dim_raw_scores.setdefault(dim, []).append(score)

    dim_averages = {
        dim: sum(scores) / len(scores)
        for dim, scores in dim_raw_scores.items()
        if scores
    }

    # Recompute final scores
    r = dim_averages.get("reasoning", 0)
    m = dim_averages.get("math", 0)
    s = dim_averages.get("science", 0)
    c = dim_averages.get("coding", 0)
    x = dim_averages.get("context", 0)
    e = dim_averages.get("efficiency", 0)
    v = dim_averages.get("speed", 0.5)
    mm = dim_averages.get("multimodal", None)
    # Do NOT substitute mean for non-vision models (see panel review)

    analytical = round(200 + (r * 0.35 + m * 0.40 + s * 0.25) * 600)
    technical  = round(200 + (c * 0.40 + x * 0.30 + e * 0.15 + v * 0.15) * 600)
    total      = analytical + technical

    result["scores_after_judging"] = {
        "dimensions": {k: round(val, 4) if val is not None else None for k, val in {
            "reasoning": r, "math": m, "science": s, "coding": c,
            "context": x, "efficiency": e, "speed": v, "multimodal": mm
        }.items()},
        "analytical": max(200, min(800, analytical)),
        "technical":  max(200, min(800, technical)),
        "total":      max(400, min(1600, total)),
        "confidence": 15,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n── PRELIMINARY SCORES (before human Tier 3 review) ──")
    print(f"  Reasoning:   {r:.3f}")
    print(f"  Math:        {m:.3f}")
    print(f"  Science:     {s:.3f}")
    print(f"  Coding:      {c:.3f}")
    print(f"  Context:     {x:.3f}")
    print(f"  Efficiency:  {e:.3f}")
    print(f"  Speed:       {v:.3f} (percentile TBD)")
    print(f"  Multimodal:  {mm:.3f}")
    print(f"  ────────────────────")
    print(f"  Analytical:  {analytical}")
    print(f"  Technical:   {technical}")
    print(f"  TOTAL:       {total}")
    print(f"\n  NOTE: Tier 3 (human expert) review still required before publishing.")

    # Save updated result
    judged_path = result_path.parent / result_path.name.replace(".json", "-judged.json")
    with open(judged_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nUpdated result saved to {judged_path}")
    print(f"Next step: human expert reviews Tier 3 questions, then run update_models.py")

    return result


def main():
    parser = argparse.ArgumentParser(description="BRAIAIN AI judge runner")
    parser.add_argument("--result", required=True, help="Path to result JSON from run.py")
    parser.add_argument("--judge-provider", default="openai",
                        choices=list(JUDGE_CALLERS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # API key from environment only — never accept on CLI
    judge_api_key = os.environ.get("BRAIAIN_JUDGE_KEY", os.environ.get("JUDGE_API_KEY", ""))

    if not args.dry_run and not judge_api_key:
        print("Error: set BRAIAIN_JUDGE_KEY (or JUDGE_API_KEY) environment variable.")
        print("  export BRAIAIN_JUDGE_KEY=sk-...")
        return

    judge_file(
        result_path=Path(args.result),
        judge_provider=args.judge_provider,
        judge_api_key=judge_api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
