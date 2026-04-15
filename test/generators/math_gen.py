"""
BRAIAIN Standard — Math Novel Question Generator
Generates fresh math questions each cycle from parametric templates.
All generated questions are programmatically verified solvable.
"""

import random
import math
from fractions import Fraction
from dataclasses import dataclass
from typing import Optional


@dataclass
class Question:
    id: str
    dimension: str
    difficulty: str
    tier: int
    prompt: str
    reference: str
    rubric: str
    verified: bool = True


class MathGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.generated = []

    def generate_all(self, n: int = 15) -> list[Question]:
        templates = [
            self.conditional_probability,
            self.sum_formula,
            self.handshake_problem,
            self.train_meeting,
            self.functional_equation,
            self.combinatorics_arrange,
            self.expected_value_dice,
            self.geometric_probability,
            self.modular_arithmetic,
            self.number_theory_gcd,
            self.sequence_problem,
            self.coin_probability,
            self.mixture_problem,
            self.work_rate,
            self.diophantine_simple,
        ]
        questions = []
        for i in range(n):
            template = templates[i % len(templates)]
            try:
                q = template()
                q.id = f"NOVEL_M{i+1:02d}"
                questions.append(q)
            except Exception as e:
                # If a specific parameter set fails, skip and move on
                pass
        return questions

    def conditional_probability(self) -> Question:
        colors = ["red", "blue", "green", "yellow"]
        c1, c2 = self.rng.sample(colors[:3], 2)
        n1 = self.rng.randint(2, 7)
        n2 = self.rng.randint(n1 + 1, n1 + 6)
        total = n1 + n2

        p_both = Fraction(n1, total) * Fraction(n1 - 1, total - 1)
        p_c2_first = Fraction(n2, total) * Fraction(n1, total - 1)
        p_second_c1 = p_both + p_c2_first

        if p_second_c1 == 0:
            raise ValueError("Zero denominator")

        answer = p_both / p_second_c1
        answer_simplified = f"{answer.numerator}/{answer.denominator}"

        return Question(
            id="",
            dimension="math",
            difficulty="hard",
            tier=1,
            prompt=(
                f"A bag contains {n1} {c1} balls and {n2} {c2} balls. "
                f"Two balls are drawn without replacement. "
                f"Given that the second ball drawn is {c1}, what is the probability "
                f"the first ball was also {c1}? Give your answer as a simplified fraction."
            ),
            reference=answer_simplified,
            rubric=(
                f"1.0: {answer_simplified} with Bayes theorem shown. "
                f"0.5: correct method, arithmetic error. 0.0: wrong method."
            )
        )

    def sum_formula(self) -> Question:
        a = self.rng.randint(10, 50)
        b = self.rng.randint(a + 50, a + 300)
        # Sum of integers from a to b = (b-a+1)(a+b)/2
        count = b - a + 1
        total = count * (a + b) // 2
        assert count * (a + b) % 2 == 0, "Non-integer sum"

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=f"What is the sum of all integers from {a} to {b} inclusive?",
            reference=str(total),
            rubric=(
                f"1.0: {total}. Formula: ({b}-{a}+1)({a}+{b})/2 = {count}×{a+b}/2. "
                f"0.5: correct method, arithmetic error. 0.0: wrong formula."
            )
        )

    def handshake_problem(self) -> Question:
        people = self.rng.randint(10, 50)
        shakes = self.rng.randint(3, min(people - 1, 12))
        # Each person shakes shakes times, each handshake counted twice
        if (people * shakes) % 2 != 0:
            shakes += 1  # ensure integer result
        total = people * shakes // 2

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=(
                f"In a group of {people} people, each person shakes hands with exactly "
                f"{shakes} others. How many handshakes occur in total?"
            ),
            reference=str(total),
            rubric=(
                f"1.0: {total}. ({people} × {shakes}) / 2 = {total}. "
                f"0.5: {people * shakes} (forgot to divide by 2). 0.0: other."
            )
        )

    def train_meeting(self) -> Question:
        distance = self.rng.choice([200, 250, 300, 350, 400])
        speed1 = self.rng.choice([60, 70, 80, 90])
        speed2 = self.rng.choice([40, 50, 60, 70])
        bird_speed = speed1 + speed2 + self.rng.choice([10, 20, 30])

        total_speed = speed1 + speed2
        time_hrs = Fraction(distance, total_speed)
        bird_distance = bird_speed * time_hrs

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=(
                f"Two trains are {distance} miles apart and travel toward each other. "
                f"One travels at {speed1} mph, the other at {speed2} mph. "
                f"A bird starts at the faster train and flies back and forth at {bird_speed} mph "
                f"until the trains meet. How far does the bird fly? "
                f"Give your answer as a fraction or decimal."
            ),
            reference=str(float(bird_distance)),
            rubric=(
                f"1.0: {float(bird_distance):.4f} miles. "
                f"Time = {distance}/{total_speed} = {float(time_hrs):.4f} hrs. "
                f"Bird distance = {bird_speed} × {float(time_hrs):.4f}. "
                f"0.0: attempted infinite series without correct result."
            )
        )

    def functional_equation(self) -> Question:
        # f(kx) = m * f(x), f(1) = c. What is f(k^n)?
        k = self.rng.choice([2, 3, 4])
        m = self.rng.choice([2, 3, 5])
        c = self.rng.randint(1, 5)
        n = self.rng.choice([3, 4, 5, 6])
        # f(k^n) = m^n * c
        answer = (m ** n) * c

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=(
                f"A function f satisfies f({k}x) = {m}·f(x) for all real x, "
                f"and f(1) = {c}. What is f({k**n})?"
            ),
            reference=str(answer),
            rubric=(
                f"1.0: {answer}. Apply {n} times: f({k})={m}·{c}={m*c}, "
                f"... f({k**n}) = {m}^{n}·{c} = {answer}. "
                f"0.5: correct method, arithmetic error. 0.0: wrong application."
            )
        )

    def combinatorics_arrange(self) -> Question:
        # How many ways to arrange n items with repeats
        words = {
            "LEVEL": {"L": 2, "E": 2, "V": 1},
            "REFER": {"R": 2, "E": 2, "F": 1},
            "CIVIC": {"C": 2, "I": 2, "V": 1},
            "RADAR": {"R": 2, "A": 2, "D": 1},
            "DEEDS": {"D": 2, "E": 2, "S": 1},
            "TOOTH": {"T": 2, "O": 2, "H": 1},
            "ADDED": {"A": 1, "D": 3, "E": 1},
            "ERROR": {"E": 1, "R": 3, "O": 1},
        }
        word, counts = self.rng.choice(list(words.items()))
        n = len(word)
        denom = 1
        for cnt in counts.values():
            for i in range(1, cnt + 1):
                denom *= i
        answer = math.factorial(n) // denom

        return Question(
            id="",
            dimension="math",
            difficulty="hard",
            tier=1,
            prompt=f"How many distinct ways can the letters of the word {word} be arranged?",
            reference=str(answer),
            rubric=(
                f"1.0: {answer}. Formula: {n}! / ({' × '.join(str(v)+'!' for v in counts.values())}) "
                f"= {math.factorial(n)} / {denom} = {answer}. "
                f"0.5: correct formula, arithmetic error. 0.0: just {n}!."
            )
        )

    def expected_value_dice(self) -> Question:
        sides = self.rng.choice([4, 6, 8, 10, 12])
        rolls = self.rng.randint(2, 4)
        # Expected value of sum of `rolls` dice with `sides` sides
        ev_single = (sides + 1) / 2
        ev_total = ev_single * rolls

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=(
                f"You roll {rolls} fair {sides}-sided dice (numbered 1 to {sides}). "
                f"What is the expected value of the sum?"
            ),
            reference=str(ev_total),
            rubric=(
                f"1.0: {ev_total}. E[single die] = ({sides}+1)/2 = {ev_single}. "
                f"E[sum of {rolls}] = {rolls} × {ev_single} = {ev_total}. "
                f"0.5: correct for one die, arithmetic error scaling. 0.0: wrong."
            )
        )

    def geometric_probability(self) -> Question:
        outer = self.rng.randint(4, 10)
        inner = self.rng.randint(1, outer - 2)
        # P(point in inner circle) = (inner/outer)^2
        num = inner ** 2
        den = outer ** 2
        frac = Fraction(num, den)

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=(
                f"A point is chosen uniformly at random inside a circle of radius {outer}. "
                f"What is the probability that the point also lies inside a concentric circle "
                f"of radius {inner}? Give your answer as a simplified fraction."
            ),
            reference=f"{frac.numerator}/{frac.denominator}",
            rubric=(
                f"1.0: {frac}. P = π·{inner}² / π·{outer}² = {inner}²/{outer}² = "
                f"{inner**2}/{outer**2} = {frac}. 0.5: correct setup, simplification error. "
                f"0.0: used circumference instead of area."
            )
        )

    def modular_arithmetic(self) -> Question:
        base = self.rng.randint(2, 9)
        exp = self.rng.randint(10, 30)
        mod = self.rng.choice([7, 9, 11, 13])
        answer = pow(base, exp, mod)

        return Question(
            id="",
            dimension="math",
            difficulty="hard",
            tier=1,
            prompt=f"What is {base}^{exp} mod {mod}?",
            reference=str(answer),
            rubric=(
                f"1.0: {answer}. Use fast modular exponentiation or find the cycle length of "
                f"{base} mod {mod}. 0.5: correct method, arithmetic error. "
                f"0.0: computed {base}^{exp} without modular reduction."
            )
        )

    def number_theory_gcd(self) -> Question:
        a = self.rng.randint(50, 500)
        b = self.rng.randint(50, 500)
        g = math.gcd(a, b)
        lcm = a * b // g

        return Question(
            id="",
            dimension="math",
            difficulty="medium",
            tier=1,
            prompt=f"What is the least common multiple (LCM) of {a} and {b}?",
            reference=str(lcm),
            rubric=(
                f"1.0: {lcm}. LCM = {a}×{b}/GCD({a},{b}) = {a*b}/{g} = {lcm}. "
                f"0.5: correct formula, arithmetic error. 0.0: gives GCD instead of LCM."
            )
        )

    def sequence_problem(self) -> Question:
        # Arithmetic or geometric sequence, find nth term
        seq_type = self.rng.choice(["arithmetic", "geometric"])
        if seq_type == "arithmetic":
            a1 = self.rng.randint(1, 20)
            d = self.rng.randint(2, 10)
            n = self.rng.randint(10, 25)
            answer = a1 + (n - 1) * d
            first_5 = [a1 + i * d for i in range(5)]
            return Question(
                id="", dimension="math", difficulty="medium", tier=1,
                prompt=(
                    f"The first five terms of a sequence are: "
                    f"{', '.join(map(str, first_5))}. "
                    f"What is the {n}th term?"
                ),
                reference=str(answer),
                rubric=(
                    f"1.0: {answer}. First term={a1}, common difference={d}, "
                    f"T({n}) = {a1} + ({n}-1)×{d} = {answer}. "
                    f"0.5: correct formula, arithmetic error. 0.0: wrong."
                )
            )
        else:
            a1 = self.rng.randint(2, 5)
            r = self.rng.randint(2, 4)
            n = self.rng.randint(5, 10)
            answer = a1 * (r ** (n - 1))
            first_4 = [a1 * (r ** i) for i in range(4)]
            return Question(
                id="", dimension="math", difficulty="medium", tier=1,
                prompt=(
                    f"The first four terms of a sequence are: "
                    f"{', '.join(map(str, first_4))}. "
                    f"What is the {n}th term?"
                ),
                reference=str(answer),
                rubric=(
                    f"1.0: {answer}. Geometric: a={a1}, r={r}, T({n})={a1}×{r}^{n-1}={answer}. "
                    f"0.5: correct formula, arithmetic error. 0.0: wrong."
                )
            )

    def coin_probability(self) -> Question:
        n = self.rng.randint(3, 7)
        k = self.rng.randint(1, n - 1)
        # P(exactly k heads in n flips) = C(n,k) / 2^n
        from math import comb
        num = comb(n, k)
        den = 2 ** n
        g = math.gcd(num, den)
        answer = f"{num//g}/{den//g}"

        return Question(
            id="", dimension="math", difficulty="medium", tier=1,
            prompt=(
                f"A fair coin is flipped {n} times. What is the probability of getting "
                f"exactly {k} heads? Give your answer as a simplified fraction."
            ),
            reference=answer,
            rubric=(
                f"1.0: {answer}. C({n},{k})/{2**n} = {num}/{den} = {answer}. "
                f"0.5: correct method, arithmetic error. 0.0: wrong."
            )
        )

    def mixture_problem(self) -> Question:
        pct1 = self.rng.choice([10, 20, 30, 40])
        pct2 = self.rng.choice([60, 70, 80, 90])
        target = self.rng.randint(pct1 + 5, pct2 - 5)
        total = self.rng.choice([100, 200, 150])
        # x litres of pct1% + (total-x) litres of pct2% = target% of total
        # x*pct1 + (total-x)*pct2 = target*total
        # x(pct1 - pct2) = total(target - pct2)
        x = Fraction(total * (target - pct2), pct1 - pct2)
        if x <= 0 or x >= total:
            raise ValueError("Invalid mixture")

        return Question(
            id="", dimension="math", difficulty="medium", tier=1,
            prompt=(
                f"You need to create {total} litres of a {target}% solution. "
                f"You have a {pct1}% solution and a {pct2}% solution. "
                f"How many litres of the {pct1}% solution do you need?"
            ),
            reference=str(float(x)),
            rubric=(
                f"1.0: {float(x):.2f} litres. "
                f"Equation: {pct1}x + {pct2}({total}-x) = {target}×{total}. "
                f"x = {x}. 0.5: correct setup, arithmetic error. 0.0: wrong setup."
            )
        )

    def work_rate(self) -> Question:
        a_days = self.rng.randint(4, 15)
        b_days = self.rng.randint(4, 15)
        # Together: 1/a + 1/b = (a+b)/(a*b) of job per day
        combined = Fraction(1, a_days) + Fraction(1, b_days)
        together_days = 1 / combined

        return Question(
            id="", dimension="math", difficulty="medium", tier=1,
            prompt=(
                f"Worker A can complete a job in {a_days} days. "
                f"Worker B can complete the same job in {b_days} days. "
                f"How many days will it take them to complete the job working together? "
                f"Give your answer as a fraction."
            ),
            reference=f"{together_days.numerator}/{together_days.denominator}",
            rubric=(
                f"1.0: {together_days}. Combined rate = 1/{a_days} + 1/{b_days} = "
                f"{combined}. Time = 1/{combined} = {together_days}. "
                f"0.5: correct method, arithmetic error. 0.0: just averaged the days."
            )
        )

    def diophantine_simple(self) -> Question:
        # x^2 - y^2 = n for small n with multiple solutions
        n = self.rng.choice([15, 21, 35, 45, 55, 63])
        # Find all factor pairs (a,b) with a > b > 0, a*b = n, same parity
        solutions = []
        for a in range(1, n + 1):
            if n % a == 0:
                b = n // a
                if a >= b and (a + b) % 2 == 0:
                    x = (a + b) // 2
                    y = (a - b) // 2
                    if y > 0:
                        solutions.extend([(x, y), (x, -y), (-x, y), (-x, -y)])

        if not solutions:
            raise ValueError("No solutions")

        sol_str = ", ".join(f"({x},{y})" for x, y in sorted(set(solutions)))

        return Question(
            id="", dimension="math", difficulty="hard", tier=1,
            prompt=f"Find all integer solutions to x² - y² = {n}. List all solutions.",
            reference=sol_str,
            rubric=(
                f"1.0: all {len(set(solutions))} solutions via (x+y)(x-y)={n} factoring. "
                f"Solutions: {sol_str}. 0.5: correct method, missed some sign cases. "
                f"0.0: wrong approach or only one solution."
            )
        )


if __name__ == "__main__":
    import json
    import os
    import hashlib

    # Derive seed from quarter + secret salt (set BRAIAIN_SEED_SALT env var)
    quarter = os.environ.get("BRAIAIN_QUARTER", "2026Q2")
    salt = os.environ.get("BRAIAIN_SEED_SALT", "")
    if not salt:
        print("NOTE: BRAIAIN_SEED_SALT not set. Using unsalted seed for dev.")
    QUARTERLY_SEED = int(hashlib.sha256(f"{quarter}:{salt}".encode()).hexdigest()[:8], 16)

    gen = MathGenerator(seed=QUARTERLY_SEED)
    questions = gen.generate_all(n=15)

    print(f"Generated {len(questions)} novel math questions\n")
    for q in questions[:3]:  # Show first 3
        print(f"--- {q.id} [{q.difficulty}] ---")
        print(f"Q: {q.prompt}")
        print(f"A: {q.reference}")
        print(f"Rubric: {q.rubric}\n")

    # Save to file
    output = [
        {
            "id": q.id, "dimension": q.dimension,
            "difficulty": q.difficulty, "tier": q.tier,
            "prompt": q.prompt, "reference": q.reference,
            "rubric": q.rubric
        }
        for q in questions
    ]
    with open("questions/novel_math.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to questions/novel_math.json")
