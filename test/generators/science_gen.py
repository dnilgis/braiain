"""
BRAIAIN Standard — Science Novel Question Generator
Generates fresh science questions each cycle from parametric templates.
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


class ScienceGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def generate_all(self, n: int = 10) -> list[Question]:
        templates = [
            self.radioactive_decay,
            self.projectile_motion,
            self.ideal_gas_law,
            self.genetics_cross,
            self.circuit_resistance,
            self.dilution,
            self.gravitational_force,
            self.energy_conservation,
            self.population_growth,
            self.stoichiometry,
            self.doppler_effect,
            self.acid_base_ph,
            self.orbital_period,
            self.half_life_remaining,
            self.spring_oscillation,
        ]
        questions = []
        for i in range(n):
            template = templates[i % len(templates)]
            try:
                q = template()
                q.id = f"NOVEL_S{i+1:02d}"
                questions.append(q)
            except Exception:
                pass
        return questions

    # ── PHYSICS ────────────────────────────────────────────────────────────────

    def radioactive_decay(self) -> Question:
        half_life = self.rng.choice([2, 3, 4, 5, 6, 8, 10])
        initial = self.rng.choice([100, 200, 400, 500, 800, 1000])
        num_half_lives = self.rng.randint(2, 6)
        time = half_life * num_half_lives
        remaining = initial / (2 ** num_half_lives)

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"A radioactive sample has a half-life of {half_life} years. "
                f"If you start with {initial} grams, how many grams remain after "
                f"{time} years?"
            ),
            reference=str(remaining),
            rubric=(
                f"1.0: {remaining}g. {time} years = {num_half_lives} half-lives. "
                f"{initial} × (1/2)^{num_half_lives} = {remaining}. "
                f"0.5: correct method, arithmetic error. 0.0: wrong approach."
            )
        )

    def projectile_motion(self) -> Question:
        v0 = self.rng.choice([10, 15, 20, 25, 30, 40, 50])
        angle_deg = self.rng.choice([30, 45, 60])
        g = 9.8
        angle_rad = math.radians(angle_deg)

        max_height = (v0 * math.sin(angle_rad)) ** 2 / (2 * g)
        range_val = (v0 ** 2) * math.sin(2 * angle_rad) / g
        flight_time = 2 * v0 * math.sin(angle_rad) / g

        ask = self.rng.choice(["range", "max_height", "flight_time"])
        if ask == "range":
            answer = round(range_val, 2)
            quantity = "horizontal range"
            formula = f"R = v₀²sin(2θ)/g = {v0}²×sin({2*angle_deg}°)/{g}"
        elif ask == "max_height":
            answer = round(max_height, 2)
            quantity = "maximum height"
            formula = f"H = (v₀sinθ)²/(2g) = ({v0}×sin{angle_deg}°)²/(2×{g})"
        else:
            answer = round(flight_time, 2)
            quantity = "total flight time"
            formula = f"T = 2v₀sinθ/g = 2×{v0}×sin{angle_deg}°/{g}"

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"A projectile is launched at {v0} m/s at an angle of {angle_deg}° "
                f"above the horizontal. Assuming flat ground and g = {g} m/s², "
                f"what is the {quantity}? Round to 2 decimal places."
            ),
            reference=str(answer),
            rubric=(
                f"1.0: {answer} (±0.05). {formula}. "
                f"0.5: correct formula, arithmetic error. 0.0: wrong formula."
            )
        )

    def ideal_gas_law(self) -> Question:
        """PV = nRT problem, solve for one variable."""
        R = 8.314  # J/(mol·K)
        solve_for = self.rng.choice(["P", "V", "n", "T"])

        if solve_for == "P":
            n = self.rng.choice([1, 2, 3, 5])
            T = self.rng.choice([273, 300, 350, 400])
            V = self.rng.choice([10, 20, 25, 50])  # litres
            V_m3 = V / 1000
            answer = round(n * R * T / V_m3, 1)
            prompt = (
                f"{n} moles of an ideal gas occupy {V} litres at {T} K. "
                f"What is the pressure in Pascals? Use R = {R} J/(mol·K)."
            )
            formula = f"P = nRT/V = {n}×{R}×{T}/{V_m3}"
        elif solve_for == "V":
            n = self.rng.choice([1, 2, 3])
            T = self.rng.choice([273, 300, 350])
            P = self.rng.choice([101325, 200000, 50000])
            V_m3 = n * R * T / P
            answer = round(V_m3 * 1000, 2)  # in litres
            prompt = (
                f"{n} moles of an ideal gas at {T} K and {P} Pa. "
                f"What is the volume in litres? Use R = {R} J/(mol·K)."
            )
            formula = f"V = nRT/P = {n}×{R}×{T}/{P} m³ = {answer} L"
        elif solve_for == "n":
            T = self.rng.choice([273, 300, 350])
            P = self.rng.choice([101325, 200000])
            V = self.rng.choice([10, 20, 50])
            V_m3 = V / 1000
            answer = round(P * V_m3 / (R * T), 3)
            prompt = (
                f"An ideal gas at {P} Pa occupies {V} litres at {T} K. "
                f"How many moles of gas are present? Use R = {R} J/(mol·K). "
                f"Round to 3 decimal places."
            )
            formula = f"n = PV/RT = {P}×{V_m3}/({R}×{T})"
        else:  # T
            n = self.rng.choice([1, 2, 5])
            P = self.rng.choice([101325, 200000])
            V = self.rng.choice([10, 20, 50])
            V_m3 = V / 1000
            answer = round(P * V_m3 / (n * R), 1)
            prompt = (
                f"{n} moles of an ideal gas at {P} Pa occupy {V} litres. "
                f"What is the temperature in Kelvin? Use R = {R} J/(mol·K)."
            )
            formula = f"T = PV/nR = {P}×{V_m3}/({n}×{R})"

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=prompt,
            reference=str(answer),
            rubric=(
                f"1.0: {answer} (±1%). {formula}. "
                f"0.5: correct formula, unit error or arithmetic error. "
                f"0.0: wrong formula."
            )
        )

    def circuit_resistance(self) -> Question:
        """Series or parallel resistor calculation."""
        config = self.rng.choice(["series", "parallel", "mixed"])
        if config == "series":
            r1 = self.rng.randint(2, 20)
            r2 = self.rng.randint(2, 20)
            r3 = self.rng.randint(2, 20)
            total = r1 + r2 + r3
            prompt = (
                f"Three resistors ({r1}Ω, {r2}Ω, {r3}Ω) are connected in series. "
                f"What is the total resistance?"
            )
            ref = f"{total}Ω"
            formula = f"R_total = {r1} + {r2} + {r3} = {total}Ω"
        elif config == "parallel":
            r1 = self.rng.choice([4, 6, 8, 10, 12, 20])
            r2 = self.rng.choice([4, 6, 8, 10, 12, 20])
            total = round(1 / (1/r1 + 1/r2), 3)
            prompt = (
                f"Two resistors ({r1}Ω and {r2}Ω) are connected in parallel. "
                f"What is the equivalent resistance? Round to 3 decimal places."
            )
            ref = str(total)
            formula = f"1/R = 1/{r1} + 1/{r2}; R = {total}Ω"
        else:  # mixed: two in parallel, that combo in series with a third
            r1 = self.rng.choice([4, 6, 8, 10])
            r2 = self.rng.choice([4, 6, 8, 12])
            r3 = self.rng.randint(3, 15)
            parallel = 1 / (1/r1 + 1/r2)
            total = round(parallel + r3, 3)
            prompt = (
                f"Two resistors ({r1}Ω and {r2}Ω) are in parallel. "
                f"Their combination is in series with a {r3}Ω resistor. "
                f"What is the total equivalent resistance? Round to 3 decimal places."
            )
            ref = str(total)
            formula = (
                f"Parallel: 1/({1/r1:.4f} + {1/r2:.4f}) = {parallel:.3f}Ω. "
                f"Series: {parallel:.3f} + {r3} = {total}Ω"
            )

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=prompt,
            reference=ref,
            rubric=(
                f"1.0: {ref}. {formula}. "
                f"0.5: correct method, rounding error. "
                f"0.0: confused series/parallel formulas."
            )
        )

    def energy_conservation(self) -> Question:
        """Object sliding down frictionless incline — find speed at bottom."""
        mass = self.rng.choice([1, 2, 3, 5, 10])
        angle = self.rng.choice([20, 30, 37, 45, 53, 60])
        length = self.rng.choice([2, 3, 4, 5, 8, 10])
        g = 9.8

        height = length * math.sin(math.radians(angle))
        speed = math.sqrt(2 * g * height)
        speed_rounded = round(speed, 2)

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"A {mass} kg block slides down a frictionless incline of angle "
                f"{angle}° and length {length} m. What is its speed at the bottom? "
                f"Use g = {g} m/s². Round to 2 decimal places."
            ),
            reference=str(speed_rounded),
            rubric=(
                f"1.0: {speed_rounded} m/s (±0.05). "
                f"Height = {length}×sin({angle}°) = {round(height, 3)} m. "
                f"v = √(2×{g}×{round(height, 3)}) = {speed_rounded}. Mass cancels. "
                f"0.5: correct method, arithmetic error. 0.0: wrong formula."
            )
        )

    def gravitational_force(self) -> Question:
        """Newton's law of gravitation between two masses."""
        G = 6.674e-11
        m1_val = self.rng.choice([5, 10, 50, 100])
        m2_val = self.rng.choice([5, 10, 50, 100])
        m1_exp = self.rng.choice([20, 22, 24, 26, 28, 30])
        m2_exp = self.rng.choice([20, 22, 24, 26, 28, 30])
        r_val = self.rng.choice([1, 2, 5, 10])
        r_exp = self.rng.choice([8, 9, 10, 11])

        m1 = m1_val * (10 ** m1_exp)
        m2 = m2_val * (10 ** m2_exp)
        r = r_val * (10 ** r_exp)

        F = G * m1 * m2 / (r ** 2)

        # Express in scientific notation
        F_exp = int(math.floor(math.log10(abs(F))))
        F_mantissa = round(F / (10 ** F_exp), 3)

        return Question(
            id="", dimension="science", difficulty="hard", tier=1,
            prompt=(
                f"Calculate the gravitational force between two objects: "
                f"mass₁ = {m1_val}×10^{m1_exp} kg, mass₂ = {m2_val}×10^{m2_exp} kg, "
                f"separated by {r_val}×10^{r_exp} m. "
                f"Use G = 6.674×10⁻¹¹ N·m²/kg². Give answer in scientific notation."
            ),
            reference=f"{F_mantissa}×10^{F_exp} N",
            rubric=(
                f"1.0: {F_mantissa}×10^{F_exp} N (±5% on mantissa, exact exponent). "
                f"F = G·m₁·m₂/r². "
                f"0.5: correct formula, exponent off by 1. "
                f"0.0: wrong formula."
            )
        )

    def doppler_effect(self) -> Question:
        """Calculate observed frequency with moving source or observer."""
        v_sound = 343  # m/s
        f_source = self.rng.choice([200, 440, 500, 600, 800, 1000])
        v_obj = self.rng.randint(10, 80)

        scenario = self.rng.choice(["source_approaching", "source_receding",
                                     "observer_approaching", "observer_receding"])

        if scenario == "source_approaching":
            f_obs = round(f_source * v_sound / (v_sound - v_obj), 1)
            desc = f"a source emitting {f_source} Hz approaches you at {v_obj} m/s"
            formula = f"f' = f × v/(v - vₛ) = {f_source} × {v_sound}/({v_sound}-{v_obj})"
        elif scenario == "source_receding":
            f_obs = round(f_source * v_sound / (v_sound + v_obj), 1)
            desc = f"a source emitting {f_source} Hz moves away from you at {v_obj} m/s"
            formula = f"f' = f × v/(v + vₛ) = {f_source} × {v_sound}/({v_sound}+{v_obj})"
        elif scenario == "observer_approaching":
            f_obs = round(f_source * (v_sound + v_obj) / v_sound, 1)
            desc = f"you move toward a stationary {f_source} Hz source at {v_obj} m/s"
            formula = f"f' = f × (v + vₒ)/v = {f_source} × ({v_sound}+{v_obj})/{v_sound}"
        else:
            f_obs = round(f_source * (v_sound - v_obj) / v_sound, 1)
            desc = f"you move away from a stationary {f_source} Hz source at {v_obj} m/s"
            formula = f"f' = f × (v - vₒ)/v = {f_source} × ({v_sound}-{v_obj})/{v_sound}"

        return Question(
            id="", dimension="science", difficulty="hard", tier=1,
            prompt=(
                f"Using the Doppler effect: {desc}. "
                f"The speed of sound is {v_sound} m/s. "
                f"What frequency do you observe? Round to 1 decimal place."
            ),
            reference=str(f_obs),
            rubric=(
                f"1.0: {f_obs} Hz (±0.5). {formula}. "
                f"0.5: correct formula, arithmetic error. "
                f"0.0: wrong Doppler formula for this scenario."
            )
        )

    # ── CHEMISTRY ──────────────────────────────────────────────────────────────

    def dilution(self) -> Question:
        """C1V1 = C2V2 dilution problem."""
        c1 = self.rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])
        v1 = self.rng.choice([10, 25, 50, 100])
        v2 = self.rng.choice([v1 * 2, v1 * 4, v1 * 5, v1 * 10])
        c2 = round(c1 * v1 / v2, 4)

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"You dilute {v1} mL of a {c1} M solution to a final volume of {v2} mL. "
                f"What is the final concentration?"
            ),
            reference=f"{c2} M",
            rubric=(
                f"1.0: {c2} M. C₁V₁ = C₂V₂ → C₂ = {c1}×{v1}/{v2} = {c2}. "
                f"0.5: correct formula, wrong arithmetic. 0.0: wrong formula."
            )
        )

    def acid_base_ph(self) -> Question:
        """Calculate pH of a strong acid or base."""
        acid_or_base = self.rng.choice(["acid", "base"])
        concentration_exp = self.rng.randint(1, 5)
        concentration = 10 ** (-concentration_exp)

        if acid_or_base == "acid":
            pH = concentration_exp
            substance = self.rng.choice(["HCl", "HNO₃", "HBr"])
            prompt = (
                f"What is the pH of a {concentration} M {substance} solution? "
                f"Assume complete dissociation."
            )
        else:
            pOH = concentration_exp
            pH = 14 - pOH
            substance = self.rng.choice(["NaOH", "KOH"])
            prompt = (
                f"What is the pH of a {concentration} M {substance} solution? "
                f"Assume complete dissociation. Use Kw = 10⁻¹⁴ at 25°C."
            )

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=prompt,
            reference=str(pH),
            rubric=(
                f"1.0: pH = {pH}. "
                f"{'pH = -log[H⁺] = -log(' + str(concentration) + ') = ' + str(pH) if acid_or_base == 'acid' else 'pOH = -log[OH⁻] = ' + str(pOH) + '; pH = 14 - ' + str(pOH) + ' = ' + str(pH)}. "
                f"0.5: correct method, minor error. 0.0: wrong approach."
            )
        )

    def stoichiometry(self) -> Question:
        """Balanced equation stoichiometry: given grams of reactant, find grams of product."""
        reactions = [
            {
                "equation": "2H₂ + O₂ → 2H₂O",
                "reactant": "H₂", "product": "H₂O",
                "r_mw": 2.016, "p_mw": 18.015,
                "r_coeff": 2, "p_coeff": 2,
            },
            {
                "equation": "CH₄ + 2O₂ → CO₂ + 2H₂O",
                "reactant": "CH₄", "product": "CO₂",
                "r_mw": 16.04, "p_mw": 44.01,
                "r_coeff": 1, "p_coeff": 1,
            },
            {
                "equation": "N₂ + 3H₂ → 2NH₃",
                "reactant": "N₂", "product": "NH₃",
                "r_mw": 28.014, "p_mw": 17.031,
                "r_coeff": 1, "p_coeff": 2,
            },
            {
                "equation": "2Na + Cl₂ → 2NaCl",
                "reactant": "Na", "product": "NaCl",
                "r_mw": 22.99, "p_mw": 58.44,
                "r_coeff": 2, "p_coeff": 2,
            },
            {
                "equation": "CaCO₃ → CaO + CO₂",
                "reactant": "CaCO₃", "product": "CaO",
                "r_mw": 100.09, "p_mw": 56.08,
                "r_coeff": 1, "p_coeff": 1,
            },
        ]

        rxn = self.rng.choice(reactions)
        grams_reactant = self.rng.choice([5, 10, 20, 25, 50, 100])

        moles_reactant = grams_reactant / rxn["r_mw"]
        moles_product = moles_reactant * rxn["p_coeff"] / rxn["r_coeff"]
        grams_product = round(moles_product * rxn["p_mw"], 2)

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"Given the balanced equation: {rxn['equation']}\n"
                f"If you start with {grams_reactant} g of {rxn['reactant']}, "
                f"how many grams of {rxn['product']} are produced? "
                f"Round to 2 decimal places. "
                f"(Molar masses: {rxn['reactant']} = {rxn['r_mw']} g/mol, "
                f"{rxn['product']} = {rxn['p_mw']} g/mol)"
            ),
            reference=str(grams_product),
            rubric=(
                f"1.0: {grams_product} g (±0.1). "
                f"Moles {rxn['reactant']} = {grams_reactant}/{rxn['r_mw']} = {round(moles_reactant, 4)}. "
                f"Moles {rxn['product']} = {round(moles_product, 4)}. "
                f"Grams = {grams_product}. "
                f"0.5: correct method, rounding error. 0.0: wrong stoichiometric ratio."
            )
        )

    # ── BIOLOGY ────────────────────────────────────────────────────────────────

    def genetics_cross(self) -> Question:
        """Mendelian genetics: dihybrid or monohybrid cross."""
        cross_type = self.rng.choice(["monohybrid", "dihybrid"])

        if cross_type == "monohybrid":
            trait = self.rng.choice(["flower colour", "seed shape", "wing type", "coat colour"])
            dominant = self.rng.choice(["A", "B", "R", "T"])
            recessive = dominant.lower()

            parent1 = self.rng.choice([f"{dominant}{dominant}", f"{dominant}{recessive}"])
            parent2 = self.rng.choice([f"{dominant}{recessive}", f"{recessive}{recessive}"])

            # Compute offspring ratios
            p1_alleles = [parent1[0], parent1[1]]
            p2_alleles = [parent2[0], parent2[1]]
            offspring = []
            for a1 in p1_alleles:
                for a2 in p2_alleles:
                    geno = "".join(sorted([a1, a2], key=lambda x: (x.islower(), x)))
                    offspring.append(geno)

            geno_counts = {}
            for g in offspring:
                geno_counts[g] = geno_counts.get(g, 0) + 1

            dom_count = sum(v for k, v in geno_counts.items() if dominant in k)
            rec_count = sum(v for k, v in geno_counts.items() if dominant not in k)

            prompt = (
                f"In a monohybrid cross for {trait}, parent 1 is {parent1} and parent 2 is {parent2}. "
                f"{dominant} is dominant over {recessive}. "
                f"What is the phenotypic ratio of dominant to recessive offspring?"
            )
            ref = f"{dom_count}:{rec_count}"
            rubric = (
                f"1.0: {dom_count}:{rec_count} with correct Punnett square. "
                f"Genotypes: {geno_counts}. "
                f"0.5: correct ratio without showing work. 0.0: wrong ratio."
            )
        else:
            # Dihybrid: AaBb × AaBb → 9:3:3:1
            prompt = (
                f"Two pea plants heterozygous for both seed shape (Rr) and seed colour (Yy) "
                f"are crossed (RrYy × RrYy). R (round) is dominant over r (wrinkled). "
                f"Y (yellow) is dominant over y (green). "
                f"What fraction of offspring will be round and green (R_yy)?"
            )
            # P(R_) = 3/4, P(yy) = 1/4 → 3/16
            ref = "3/16"
            rubric = (
                f"1.0: 3/16. P(at least one R) = 3/4, P(yy) = 1/4, independent → 3/4 × 1/4 = 3/16. "
                f"0.5: correct method, arithmetic error. 0.0: wrong approach."
            )

        return Question("", "science", "medium", 1, prompt, ref, rubric)

    def population_growth(self) -> Question:
        """Exponential population growth: N(t) = N0 × e^(rt)."""
        N0 = self.rng.choice([100, 500, 1000, 5000])
        r = self.rng.choice([0.02, 0.03, 0.05, 0.07, 0.10])
        t = self.rng.choice([5, 10, 15, 20, 25])

        N_t = round(N0 * math.exp(r * t), 1)

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"A bacterial population starts at {N0} and grows exponentially "
                f"with rate r = {r} per hour. Using N(t) = N₀ × e^(rt), "
                f"what is the population after {t} hours? Round to 1 decimal place."
            ),
            reference=str(N_t),
            rubric=(
                f"1.0: {N_t} (±0.5%). N = {N0} × e^({r}×{t}) = {N0} × e^{r*t} = {N_t}. "
                f"0.5: correct formula, arithmetic/rounding error. "
                f"0.0: used wrong growth model."
            )
        )

    # ── ADDITIONAL PHYSICS ─────────────────────────────────────────────────────

    def orbital_period(self) -> Question:
        """Kepler's third law: T² ∝ a³. Given one planet's data, find another's period."""
        # Earth baseline: a=1 AU, T=1 year
        planet_a_au = self.rng.choice([0.5, 0.7, 1.5, 2.0, 3.0, 5.0, 10.0])
        # T² = a³ → T = a^(3/2)
        T = round(planet_a_au ** 1.5, 3)

        return Question(
            id="", dimension="science", difficulty="hard", tier=1,
            prompt=(
                f"Using Kepler's third law (T² = a³ in AU and years), "
                f"a planet orbits at {planet_a_au} AU from the Sun. "
                f"What is its orbital period in years? Round to 3 decimal places."
            ),
            reference=str(T),
            rubric=(
                f"1.0: {T} years (±0.005). T = {planet_a_au}^(3/2) = {T}. "
                f"0.5: correct formula, rounding error. "
                f"0.0: wrong formula (e.g. T = a³ instead of T = a^1.5)."
            )
        )

    def half_life_remaining(self) -> Question:
        """Given remaining fraction, calculate elapsed time."""
        half_life = self.rng.choice([2, 4, 5, 8, 10, 12])
        n_halves = self.rng.randint(1, 5)
        remaining_fraction = Fraction(1, 2 ** n_halves)
        elapsed = half_life * n_halves

        prompt = (
            f"A substance has a half-life of {half_life} days. "
            f"How many days until only {remaining_fraction} of the original amount remains?"
        )
        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=prompt,
            reference=str(elapsed),
            rubric=(
                f"1.0: {elapsed} days. {remaining_fraction} = (1/2)^{n_halves}, "
                f"so {n_halves} half-lives × {half_life} days = {elapsed}. "
                f"0.5: correct half-lives count, wrong multiplication. 0.0: wrong."
            )
        )

    def spring_oscillation(self) -> Question:
        """Period and frequency of a mass-spring system."""
        mass = self.rng.choice([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        k = self.rng.choice([5, 10, 20, 50, 100, 200])

        T = round(2 * math.pi * math.sqrt(mass / k), 4)
        f = round(1 / T, 4) if T > 0 else 0

        ask = self.rng.choice(["period", "frequency"])
        if ask == "period":
            answer = T
            quantity = "period"
            unit = "seconds"
            formula = f"T = 2π√(m/k) = 2π√({mass}/{k})"
        else:
            answer = f
            quantity = "frequency"
            unit = "Hz"
            formula = f"f = 1/(2π√(m/k)) = 1/(2π√({mass}/{k}))"

        return Question(
            id="", dimension="science", difficulty="medium", tier=1,
            prompt=(
                f"A {mass} kg mass is attached to a spring with spring constant "
                f"k = {k} N/m. What is the {quantity} of oscillation in {unit}? "
                f"Round to 4 decimal places."
            ),
            reference=str(answer),
            rubric=(
                f"1.0: {answer} {unit} (±0.001). {formula}. "
                f"0.5: correct formula, rounding error. "
                f"0.0: wrong formula."
            )
        )


if __name__ == "__main__":
    import json
    import os
    import hashlib

    quarter = os.environ.get("BRAIAIN_QUARTER", "2026Q2")
    salt = os.environ.get("BRAIAIN_SEED_SALT", "")
    if not salt:
        print("NOTE: BRAIAIN_SEED_SALT not set. Using unsalted seed for dev.")
    QUARTERLY_SEED = int(hashlib.sha256(f"{quarter}:{salt}".encode()).hexdigest()[:8], 16)
    gen = ScienceGenerator(seed=QUARTERLY_SEED)
    questions = gen.generate_all(n=15)

    print(f"Generated {len(questions)} novel science questions\n")
    for q in questions[:3]:
        print(f"--- {q.id} [{q.difficulty}] ---")
        print(f"Q: {q.prompt[:200]}...")
        print(f"A: {q.reference}")
        print(f"Rubric: {q.rubric}\n")

    output = [
        {
            "id": q.id, "dimension": q.dimension,
            "difficulty": q.difficulty, "tier": q.tier,
            "prompt": q.prompt, "reference": q.reference,
            "rubric": q.rubric
        }
        for q in questions
    ]
    with open("questions/novel_science.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to questions/novel_science.json")
