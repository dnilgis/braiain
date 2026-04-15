"""
BRAIAIN Standard — Models.json Updater
Reads judged result files and updates models.json with final scores.
Run after human expert has reviewed Tier 3 questions.

Usage:
  python update_models.py --result output/results/gpt-4o-20260401-judged.json \
                          --human-review output/human/gpt-4o-20260401-tier3.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone


def load_human_review(path: Path) -> dict:
    """
    Load human expert Tier 3 review.
    Format: {"question_id": {"score": 0.0|0.5|1.0, "note": "..."}, ...}
    """
    if not path.exists():
        print(f"WARNING: No human review file found at {path}")
        print("Proceeding with AI judge scores only. All Tier 3 questions will score 0.5.")
        return {}
    with open(path) as f:
        return json.load(f)


def check_contamination(result: dict, n_bootstrap: int = 10000) -> list:
    """Run contamination detection on fixed vs novel question splits.

    Uses bootstrap confidence interval on the gap (fixed_avg - novel_avg).
    Flags if the 95% CI lower bound > 0 AND the gap > 0.10.
    This avoids false positives from small sample variance (Chollet review).
    """
    import random

    fixed_by_dim = {}
    novel_by_dim = {}

    for answer in result.get("answers", []):
        dim = answer.get("dimension")
        score = answer.get("score")
        qid = answer.get("question_id", "")
        if dim is None or score is None:
            continue

        if qid.startswith("NOVEL_"):
            novel_by_dim.setdefault(dim, []).append(score)
        else:
            fixed_by_dim.setdefault(dim, []).append(score)

    flags = []
    rng = random.Random(42)

    for dim in fixed_by_dim:
        if dim not in novel_by_dim or not novel_by_dim[dim]:
            continue

        fixed_scores = fixed_by_dim[dim]
        novel_scores = novel_by_dim[dim]
        fixed_avg = sum(fixed_scores) / len(fixed_scores)
        novel_avg = sum(novel_scores) / len(novel_scores)
        observed_gap = fixed_avg - novel_avg

        # Bootstrap: resample each group with replacement, compute gap
        bootstrap_gaps = []
        for _ in range(n_bootstrap):
            f_sample = [rng.choice(fixed_scores) for _ in range(len(fixed_scores))]
            n_sample = [rng.choice(novel_scores) for _ in range(len(novel_scores))]
            f_mean = sum(f_sample) / len(f_sample)
            n_mean = sum(n_sample) / len(n_sample)
            bootstrap_gaps.append(f_mean - n_mean)

        bootstrap_gaps.sort()
        ci_lower = bootstrap_gaps[int(n_bootstrap * 0.025)]
        ci_upper = bootstrap_gaps[int(n_bootstrap * 0.975)]

        # Flag if: 95% CI lower bound > 0 (statistically significant gap)
        # AND observed gap > 0.10 (meaningful effect size)
        if ci_lower > 0 and observed_gap > 0.10:
            flags.append({
                "dimension": dim,
                "fixed_avg": round(fixed_avg, 3),
                "novel_avg": round(novel_avg, 3),
                "gap": round(observed_gap, 3),
                "ci_95": [round(ci_lower, 3), round(ci_upper, 3)],
                "n_fixed": len(fixed_scores),
                "n_novel": len(novel_scores),
                "status": "FLAGGED — statistically significant gap, investigate before publishing"
            })
        elif observed_gap > 0.20:
            # Large raw gap but CI includes zero — warn but don't hard-flag
            flags.append({
                "dimension": dim,
                "fixed_avg": round(fixed_avg, 3),
                "novel_avg": round(novel_avg, 3),
                "gap": round(observed_gap, 3),
                "ci_95": [round(ci_lower, 3), round(ci_upper, 3)],
                "n_fixed": len(fixed_scores),
                "n_novel": len(novel_scores),
                "status": "WARNING — large gap but not statistically significant (small sample). Monitor."
            })

    return flags


def compute_final_scores(result: dict, human_review: dict) -> dict:
    """
    Apply human review scores to Tier 3 questions, then compute final scores.
    """
    # Apply human scores
    tier3_applied = 0
    for answer in result["answers"]:
        if answer.get("tier") == 3:
            qid = answer["question_id"]
            if qid in human_review:
                answer["score"] = human_review[qid]["score"]
                answer["score_method"] = "human_expert"
                answer["human_note"] = human_review[qid].get("note", "")
                tier3_applied += 1
            elif answer.get("score") is None:
                answer["score"] = 0.5  # Default if no human review
                answer["score_method"] = "default_no_review"

    print(f"Human expert scores applied: {tier3_applied}")

    # Aggregate by dimension
    dim_scores = {}
    dim_counts = {}
    for answer in result["answers"]:
        dim = answer.get("dimension")
        score = answer.get("score")
        if dim and score is not None:
            dim_scores[dim] = dim_scores.get(dim, 0) + score
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

    dim_averages = {
        dim: dim_scores[dim] / dim_counts[dim]
        for dim in dim_scores
    }

    # Speed percentile: placeholder until all models in current cycle tested
    speed = result.get("speed_measurements", {})
    speed_score = dim_averages.get("speed", 0.5)

    r = dim_averages.get("reasoning", 0)
    m = dim_averages.get("math", 0)
    s = dim_averages.get("science", 0)
    c = dim_averages.get("coding", 0)
    x = dim_averages.get("context", 0)
    e = dim_averages.get("efficiency", 0)
    v = speed_score
    mm = dim_averages.get("multimodal", None)
    # Do NOT substitute mean for non-vision models (see panel review)

    analytical = round(200 + (r * 0.35 + m * 0.40 + s * 0.25) * 600)
    technical  = round(200 + (c * 0.40 + x * 0.30 + e * 0.15 + v * 0.15) * 600)
    total      = analytical + technical

    return {
        "reasoning": round(r, 4),
        "coding": round(c, 4),
        "math": round(m, 4),
        "science": round(s, 4),
        "context": round(x, 4),
        "speed": round(v, 4),
        "efficiency": round(e, 4),
        "multimodal": round(mm, 4) if mm is not None else None,
        "analytical": max(200, min(800, analytical)),
        "technical": max(200, min(800, technical)),
        "total": max(400, min(1600, total)),
    }


def update_models_json(model_id: str, provider: str, scores: dict,
                       contamination_flags: list, models_json_path: Path,
                       dry_run: bool = False):
    """Update models.json with new scores."""
    with open(models_json_path) as f:
        data = json.load(f)

    now = datetime.now(timezone.utc).isoformat()

    # Find existing model or create new entry
    existing = next((m for m in data["models"] if m["id"] == model_id), None)

    if existing:
        old_total = existing.get("analytical", 200) + existing.get("technical", 200)
        new_total = scores["total"]
        delta = new_total - old_total

        # Record history entry
        if "score_history" not in existing:
            existing["score_history"] = []
        existing["score_history"].append({
            "date": now,
            "analytical": existing.get("analytical"),
            "technical": existing.get("technical"),
            "total": old_total,
        })

        # Update scores
        existing["analytical"] = scores["analytical"]
        existing["technical"] = scores["technical"]
        existing["dims"] = {
            "reasoning": scores["reasoning"],
            "coding": scores["coding"],
            "math": scores["math"],
            "science": scores["science"],
            "context": scores["context"],
            "speed": scores["speed"],
            "efficiency": scores["efficiency"],
            "multimodal": scores["multimodal"],
        }
        existing["last_tested"] = now[:10]

        if delta < -5:
            existing["regression_note"] = f"Score decreased {abs(delta)} points from {old_total} to {new_total} on {now[:10]}"
            print(f"\n⚠️  REGRESSION DETECTED: {model_id}")
            print(f"   Old score: {old_total}")
            print(f"   New score: {new_total}")
            print(f"   Delta:     {delta}")
            print(f"   Regression note will be published with the score update.")
        elif delta > 5:
            print(f"\n✅ Score improved: {model_id} {old_total} → {new_total} (+{delta})")
        else:
            print(f"\n→ Score unchanged: {model_id} {old_total} → {new_total}")

        if contamination_flags:
            existing["contamination_flags"] = contamination_flags
            print(f"\n⚠️  CONTAMINATION FLAGS: {len(contamination_flags)} dimensions")
            for flag in contamination_flags:
                print(f"   {flag['dimension']}: fixed={flag['fixed_avg']}, novel={flag['novel_avg']}, gap={flag['gap']}")

    else:
        print(f"\n+ New model added: {model_id}")
        # TODO: create full model entry — add name, color, tags, context_window, pricing, etc.
        data["models"].append({
            "id": model_id,
            "name": model_id,  # Update manually
            "provider": provider,
            "color": "#888880",  # Update manually
            "analytical": scores["analytical"],
            "technical": scores["technical"],
            "dims": {
                "reasoning": scores["reasoning"],
                "coding": scores["coding"],
                "math": scores["math"],
                "science": scores["science"],
                "context": scores["context"],
                "speed": scores["speed"],
                "efficiency": scores["efficiency"],
                "multimodal": scores["multimodal"],
            },
            "context_window": "Unknown",  # Update manually
            "price_input": 0,
            "price_output": 0,
            "released": now[:7],
            "tags": [],
            "last_tested": now[:10],
        })

    data["updated"] = now

    if dry_run:
        print("\n[DRY RUN] models.json would be updated as follows:")
        target = next((m for m in data["models"] if m["id"] == model_id), None)
        if target:
            print(json.dumps({k: v for k, v in target.items() if k != "score_history"}, indent=2))
        return

    with open(models_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nmodels.json updated: {models_json_path}")
    print(f"BRAIAIN score for {model_id}: {scores['total']}")
    print(f"\nNext: push to GitHub, DNA figures regenerate automatically on site.")


def main():
    parser = argparse.ArgumentParser(description="Update models.json with new scores")
    parser.add_argument("--result", required=True, help="Judged result JSON")
    parser.add_argument("--human-review", default=None,
                        help="Human expert Tier 3 review JSON")
    parser.add_argument("--models-json", default="../models.json",
                        help="Path to models.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_path = Path(args.result)
    with open(result_path) as f:
        result = json.load(f)

    model_id = result["model_id"]
    provider = result["provider"]

    # Load human review if provided
    human_review = {}
    if args.human_review:
        human_review = load_human_review(Path(args.human_review))

    # Check contamination
    contamination_flags = check_contamination(result)

    if contamination_flags and not args.dry_run:
        print("\n⚠️  CONTAMINATION FLAGS DETECTED")
        print("Scores should not be published until investigated.")
        print("Use --dry-run to preview scores, then investigate flags.")
        for flag in contamination_flags:
            print(f"  {flag}")
        confirm = input("\nProceed anyway? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    # Compute final scores
    scores = compute_final_scores(result, human_review)

    print(f"\n── FINAL SCORES: {model_id} ──")
    print(f"  Reasoning:   {scores['reasoning']:.4f}")
    print(f"  Math:        {scores['math']:.4f}")
    print(f"  Science:     {scores['science']:.4f}")
    print(f"  Coding:      {scores['coding']:.4f}")
    print(f"  Context:     {scores['context']:.4f}")
    print(f"  Efficiency:  {scores['efficiency']:.4f}")
    print(f"  Speed:       {scores['speed']:.4f}")
    print(f"  Multimodal:  {scores['multimodal']:.4f}")
    print(f"  ────────────")
    print(f"  Analytical:  {scores['analytical']}")
    print(f"  Technical:   {scores['technical']}")
    print(f"  TOTAL:       {scores['total']}")

    models_json_path = Path(args.models_json)
    if not models_json_path.exists():
        print(f"\nError: models.json not found at {models_json_path}")
        return

    update_models_json(
        model_id=model_id,
        provider=provider,
        scores=scores,
        contamination_flags=contamination_flags,
        models_json_path=models_json_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
