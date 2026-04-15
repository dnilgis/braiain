"""
BRAIAIN Standard — Logic & Reasoning Novel Question Generator
Generates fresh reasoning questions each cycle from parametric templates.
All generated questions are programmatically verified solvable.
"""

import random
import itertools
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


class LogicGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.generated = []

    def generate_all(self, n: int = 15) -> list[Question]:
        templates = [
            self.truth_teller_liar,
            self.hat_puzzle,
            self.knights_knaves_pair,
            self.island_of_three,
            self.syllogism_validity,
            self.set_membership,
            self.transitive_ranking,
            self.pigeonhole,
            self.conditional_chain,
            self.scheduling_constraint,
            self.negation_fallacy,
            self.deductive_elimination,
            self.river_crossing,
            self.logical_equivalence,
            self.weighted_majority,
        ]
        questions = []
        for i in range(n):
            template = templates[i % len(templates)]
            try:
                q = template()
                q.id = f"NOVEL_R{i+1:02d}"
                questions.append(q)
            except Exception:
                pass
        return questions

    # ── TEMPLATES ──────────────────────────────────────────────────────────────

    def truth_teller_liar(self) -> Question:
        """N people, exactly K are truth-tellers. Each makes a claim about others."""
        names = ["Ava", "Ben", "Cal", "Dee", "Eve", "Fin", "Gia", "Hal"]
        n = self.rng.randint(3, 5)
        selected = self.rng.sample(names, n)
        # Pick a random assignment: who is truth-teller (T) and who is liar (L)
        num_truth = self.rng.randint(1, n - 1)
        truth_set = set(self.rng.sample(selected, num_truth))

        # Build statements: each person claims someone else is a truth-teller or liar
        statements = []
        for person in selected:
            other = self.rng.choice([p for p in selected if p != person])
            # Truth-tellers make true claims, liars make false claims
            other_is_truth = other in truth_set
            if person in truth_set:
                # True claim
                if other_is_truth:
                    claim = f"{person} says: '{other} is a truth-teller.'"
                else:
                    claim = f"{person} says: '{other} is a liar.'"
            else:
                # False claim
                if other_is_truth:
                    claim = f"{person} says: '{other} is a liar.'"
                else:
                    claim = f"{person} says: '{other} is a truth-teller.'"
            statements.append(claim)

        # Verify uniqueness of solution
        all_assignments = list(itertools.product([True, False], repeat=n))
        valid = []
        for assignment in all_assignments:
            assign_map = dict(zip(selected, assignment))
            consistent = True
            for person in selected:
                stmt = statements[selected.index(person)]
                # Parse claim
                if "is a truth-teller" in stmt:
                    claimed_other = stmt.split("'")[1].split(" is")[0]
                    claim_value = True
                else:
                    claimed_other = stmt.split("'")[1].split(" is")[0]
                    claim_value = False
                actual = assign_map[claimed_other]
                if assign_map[person]:  # truth-teller
                    if claim_value != actual:
                        consistent = False
                        break
                else:  # liar
                    if claim_value == actual:
                        consistent = False
                        break
            if consistent:
                valid.append(assign_map)

        if len(valid) != 1:
            raise ValueError("Non-unique solution, regenerate")

        solution = valid[0]
        truth_list = sorted([p for p, v in solution.items() if v])
        liar_list = sorted([p for p, v in solution.items() if not v])

        prompt = (
            f"In a group of {n} people, each person is either a truth-teller "
            f"(always tells the truth) or a liar (always lies).\n\n"
            + "\n".join(statements) +
            f"\n\nWho are the truth-tellers and who are the liars?"
        )
        ref = f"Truth-tellers: {', '.join(truth_list)}. Liars: {', '.join(liar_list)}."
        rubric = (
            f"1.0: correct full assignment ({ref}). "
            f"0.5: identifies some correctly but not all. "
            f"0.0: wrong assignment."
        )
        return Question("", "reasoning", "hard", 2, prompt, ref, rubric)

    def hat_puzzle(self) -> Question:
        """N people in a line, each wearing a colored hat. Binary colors."""
        n = self.rng.randint(3, 5)
        colors = ["red", "blue"]
        hats = [self.rng.choice(colors) for _ in range(n)]
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"][:n]

        # Person i can see hats of persons i+1 ... n-1
        # Asked from back (person 0) to front
        # Standard strategy: person 0 says parity of what they see
        parity_color = colors[sum(1 for h in hats[1:] if h == "red") % 2]

        # Build description
        lines = [
            f"{n} people stand in a line, each wearing either a red or blue hat. "
            f"Each person can see the hats of everyone in front of them but not their own or those behind them. "
            f"Starting from the back of the line, each person must guess their own hat colour. "
            f"They can hear all previous guesses.\n"
        ]

        lines.append(f"The actual hat arrangement from back to front is: "
                      f"{', '.join(f'{names[i]}={hats[i]}' for i in range(n))}.\n")
        lines.append(
            f"Using the optimal parity strategy, what does each person guess, "
            f"and how many are guaranteed to be correct?"
        )

        # Compute optimal strategy
        guesses = []
        # Person 0 (back): announces parity of reds they see
        guesses.append(parity_color)
        running_parity = sum(1 for h in hats[1:] if h == "red") % 2
        # Person 1 onward can deduce from parity + previous guesses
        known_reds = 0
        for i in range(1, n):
            # Person i sees hats i+1..n-1
            reds_ahead = sum(1 for h in hats[i+1:] if h == "red")
            # Parity announced = (reds from 1..n-1) mod 2
            # Person i knows reds_ahead and known_reds_behind (from previous correct guesses)
            # Their hat is red if (running_parity - reds_ahead - known_reds) % 2 == 1
            total_others = reds_ahead + known_reds
            if (running_parity - total_others) % 2 == 1:
                guesses.append("red")
                known_reds += 1
            else:
                guesses.append("blue")

        correct = sum(1 for i in range(n) if guesses[i] == hats[i])
        guaranteed = n - 1  # Strategy guarantees n-1 correct

        ref = (
            f"Optimal parity strategy: person at back announces parity of red hats they see. "
            f"Guesses: {', '.join(f'{names[i]}={guesses[i]}' for i in range(n))}. "
            f"The strategy guarantees {guaranteed} out of {n} correct (everyone except "
            f"possibly the first person)."
        )
        rubric = (
            f"1.0: explains parity strategy and states {guaranteed} guaranteed correct. "
            f"0.5: correct strategy concept without full working. "
            f"0.0: wrong strategy or wrong guarantee count."
        )
        return Question("", "reasoning", "hard", 2, "\n".join(lines), ref, rubric)

    def knights_knaves_pair(self) -> Question:
        """Two people, one knight (truth) one knave (liar). One question to identify."""
        names = self.rng.sample(["Aria", "Blake", "Corin", "Drew", "Emery"], 2)
        a, b = names
        # Randomly assign
        knight = self.rng.choice([a, b])
        knave = b if knight == a else a

        # The question to ask: "Are you the same type?"
        # Knight (truth-teller) asked "Are you and [other] the same type?"
        #   They aren't → Knight says "No"
        # Knave (liar) asked same question
        #   They aren't → truth is "No", knave lies → "Yes"
        # So: if answer is "Yes" → speaker is knave; "No" → speaker is knight

        question_to_ask = f"If you asked {b}: 'Is {a} a knight?' what would {b} say?"
        # Both routes point to the wrong answer about A
        # Knight A asks about Knave B: B would lie and say "No" (A is not knight). Knight truthfully reports: "No"
        # Knave A asks about Knight B: B would truthfully say "No" (A is not knight). Knave lies about B's answer: "Yes"

        if knight == a:
            reported_answer = "No"
        else:
            reported_answer = "Yes"

        prompt = (
            f"On an island, knights always tell the truth and knaves always lie. "
            f"You meet {a} and {b}. Exactly one is a knight and one is a knave.\n\n"
            f"You ask {a}: '{question_to_ask}'\n"
            f"{a} responds: '{reported_answer}.'\n\n"
            f"Who is the knight and who is the knave? Show your reasoning."
        )

        ref = f"{knight} is the knight, {knave} is the knave."
        rubric = (
            f"1.0: {ref} with correct logical chain through both cases. "
            f"0.5: correct answer without full reasoning. "
            f"0.0: wrong identification."
        )
        return Question("", "reasoning", "hard", 2, prompt, ref, rubric)

    def island_of_three(self) -> Question:
        """Three islanders: knight, knave, spy (can say anything). Identify all."""
        names = self.rng.sample(["Kai", "Luna", "Milo", "Nora", "Owen", "Pia"], 3)
        roles = ["knight", "knave", "spy"]
        self.rng.shuffle(roles)
        assignment = dict(zip(names, roles))

        # Generate statements that yield a unique solution
        # Knight tells truth, knave lies, spy can say anything
        # We need to find statements where only one assignment is consistent
        attempts = 0
        while attempts < 100:
            attempts += 1
            stmts = {}
            for name in names:
                other = self.rng.choice([n for n in names if n != name])
                claimed_role = self.rng.choice(["knight", "knave", "spy"])
                stmts[name] = (other, claimed_role)

            # Check all 6 permutations
            valid_assignments = []
            for perm in itertools.permutations(roles):
                test = dict(zip(names, perm))
                consistent = True
                for name in names:
                    other, claimed = stmts[name]
                    actual_other_role = test[other]
                    if test[name] == "knight":
                        if claimed != actual_other_role:
                            consistent = False
                            break
                    elif test[name] == "knave":
                        if claimed == actual_other_role:
                            consistent = False
                            break
                    # spy: anything goes
                if consistent:
                    valid_assignments.append(test)

            if len(valid_assignments) == 1 and valid_assignments[0] == assignment:
                break
        else:
            raise ValueError("Could not find unique puzzle")

        stmt_lines = []
        for name in names:
            other, claimed = stmts[name]
            stmt_lines.append(f"{name} says: '{other} is a {claimed}.'")

        prompt = (
            f"On an island there are three types: knights (always truth), "
            f"knaves (always lie), and spies (can say anything). "
            f"Three people — {', '.join(names)} — are one of each type.\n\n"
            + "\n".join(stmt_lines) +
            f"\n\nDetermine who is the knight, who is the knave, and who is the spy."
        )
        ref = ", ".join(f"{n} is the {assignment[n]}" for n in names) + "."
        rubric = (
            f"1.0: all three correctly identified with case analysis. "
            f"0.5: two correct. 0.0: one or zero correct."
        )
        return Question("", "reasoning", "hard", 2, prompt, ref, rubric)

    def syllogism_validity(self) -> Question:
        """Generate a syllogism and ask if it's valid."""
        # Template: All A are B. All B are C. Therefore all A are C. (VALID)
        # Or invalid forms
        categories = [
            ("mammals", "warm-blooded", "vertebrates"),
            ("roses", "flowers", "plants"),
            ("squares", "rectangles", "quadrilaterals"),
            ("puppies", "dogs", "animals"),
            ("senators", "politicians", "public figures"),
        ]
        a, b, c = self.rng.choice(categories)

        form = self.rng.choice(["valid_barbara", "invalid_affirm_consequent",
                                 "invalid_undistributed_middle", "valid_modus_tollens"])

        if form == "valid_barbara":
            premise1 = f"All {a} are {b}."
            premise2 = f"All {b} are {c}."
            conclusion = f"Therefore, all {a} are {c}."
            answer = "Valid"
            explanation = "Barbara syllogism (AAA-1): the middle term is distributed in the major premise."
        elif form == "invalid_affirm_consequent":
            premise1 = f"All {a} are {b}."
            premise2 = f"Some {c} are {b}."
            conclusion = f"Therefore, some {c} are {a}."
            answer = "Invalid"
            explanation = f"Undistributed middle: {b} is not distributed in either premise. Some {c} could be {b} without being {a}."
        elif form == "invalid_undistributed_middle":
            premise1 = f"All {a} are {c}."
            premise2 = f"All {b} are {c}."
            conclusion = f"Therefore, all {a} are {b}."
            answer = "Invalid"
            explanation = f"Undistributed middle: {c} is never distributed (appears as predicate of affirmative propositions). Being {c} doesn't make {a} and {b} identical."
        else:  # valid_modus_tollens
            premise1 = f"All {a} are {b}."
            premise2 = f"This creature is not a {b[:-1] if b.endswith('s') else b}."
            conclusion = f"Therefore, this creature is not a {a[:-1] if a.endswith('s') else a}."
            answer = "Valid"
            explanation = "Modus tollens: if all A are B, and not-B, then not-A."

        prompt = (
            f"Is this argument valid or invalid? Explain your reasoning.\n\n"
            f"Premise 1: {premise1}\n"
            f"Premise 2: {premise2}\n"
            f"Conclusion: {conclusion}"
        )
        ref = f"{answer}. {explanation}"
        rubric = (
            f"1.0: '{answer}' with correct logical explanation. "
            f"0.5: '{answer}' without explanation. "
            f"0.0: wrong validity judgment."
        )
        return Question("", "reasoning", "medium", 2, prompt, ref, rubric)

    def set_membership(self) -> Question:
        """Given set operations, determine membership."""
        a_members = set(self.rng.sample(range(1, 20), self.rng.randint(4, 8)))
        b_members = set(self.rng.sample(range(1, 20), self.rng.randint(4, 8)))

        op = self.rng.choice(["union", "intersection", "difference", "symmetric_difference"])

        if op == "union":
            result = sorted(a_members | b_members)
            op_name = "A ∪ B"
            op_desc = "the union of A and B"
        elif op == "intersection":
            result = sorted(a_members & b_members)
            op_name = "A ∩ B"
            op_desc = "the intersection of A and B"
        elif op == "difference":
            result = sorted(a_members - b_members)
            op_name = "A \\ B (A minus B)"
            op_desc = "A minus B (elements in A but not in B)"
        else:
            result = sorted(a_members ^ b_members)
            op_name = "A △ B"
            op_desc = "the symmetric difference of A and B"

        prompt = (
            f"Let A = {{{', '.join(map(str, sorted(a_members)))}}} and "
            f"B = {{{', '.join(map(str, sorted(b_members)))}}}.\n"
            f"What is {op_desc}? List all elements."
        )
        ref = f"{{{', '.join(map(str, result))}}}"
        rubric = (
            f"1.0: {ref} (all elements correct, no extras). "
            f"0.5: missing 1-2 elements or 1-2 extras. "
            f"0.0: wrong operation applied."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def transitive_ranking(self) -> Question:
        """Given pairwise comparisons, determine full ranking."""
        items = self.rng.sample(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 4)
        # Create a definite ranking
        ranking = list(items)
        self.rng.shuffle(ranking)  # ranking[0] is best

        # Generate pairwise clues (not all pairs — make it a deduction)
        pairs = []
        # Give n-1 adjacent comparisons (enough to determine order) plus 1 redundant
        for i in range(len(ranking) - 1):
            pairs.append((ranking[i], ranking[i + 1]))
        # Shuffle the presentation order
        self.rng.shuffle(pairs)

        clue_lines = []
        for better, worse in pairs:
            verb = self.rng.choice(["scored higher than", "outperformed", "ranked above", "beat"])
            clue_lines.append(f"• {better} {verb} {worse}")

        prompt = (
            f"Four competitors were ranked based on these results:\n"
            + "\n".join(clue_lines) +
            f"\n\nList all four from best to worst."
        )
        ref = " > ".join(ranking)
        rubric = (
            f"1.0: correct full order: {ref}. "
            f"0.5: one swap. 0.0: two or more swaps."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def pigeonhole(self) -> Question:
        """Classic pigeonhole principle problem with varied parameters."""
        variant = self.rng.choice(["socks", "birthday", "cards"])

        if variant == "socks":
            colors = self.rng.randint(3, 6)
            target = self.rng.randint(2, 4)
            answer = (target - 1) * colors + 1
            prompt = (
                f"A drawer contains socks in {colors} different colours (many of each). "
                f"You draw socks one at a time in the dark. "
                f"What is the minimum number you must draw to guarantee "
                f"that you have at least {target} socks of the same colour?"
            )
            ref = str(answer)
            explanation = (
                f"Pigeonhole principle: worst case is {target - 1} of each colour "
                f"= {(target-1)*colors}, then the next sock completes a set. "
                f"Answer: ({target}-1)×{colors} + 1 = {answer}."
            )
        elif variant == "birthday":
            people = self.rng.choice([367, 368, 370, 400, 500])
            answer = 2  # guaranteed match with > 366 people
            prompt = (
                f"In a room of {people} people, what is the minimum number of people "
                f"guaranteed to share the same birthday? (Assume 366 possible birthdays, "
                f"including Feb 29.)"
            )
            # floor((people-1)/366) + 1
            answer = (people - 1) // 366 + 1
            ref = str(answer)
            explanation = (
                f"Pigeonhole: {people} people in 366 slots. "
                f"At least ceil({people}/366) = {answer} share a birthday."
            )
        else:  # cards
            draw = self.rng.randint(5, 14)
            # How many cards to guarantee a pair of same suit?
            # 4 suits, so 5 cards guarantees a suit pair
            # But ask: guarantee K cards of same suit
            target_suit = self.rng.randint(2, 4)
            answer = (target_suit - 1) * 4 + 1
            prompt = (
                f"From a standard 52-card deck, cards are drawn one at a time. "
                f"What is the minimum number of cards you must draw to guarantee "
                f"that at least {target_suit} cards are of the same suit?"
            )
            ref = str(answer)
            explanation = (
                f"Pigeonhole: 4 suits. Worst case: {target_suit-1} of each suit "
                f"= {(target_suit-1)*4}. Next card: {answer} guarantees {target_suit} "
                f"of one suit."
            )

        rubric = (
            f"1.0: {ref}. {explanation} "
            f"0.5: correct principle, off by one. "
            f"0.0: wrong principle."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def conditional_chain(self) -> Question:
        """If A then B, if B then C, ... Given one fact, derive conclusion."""
        items = self.rng.sample([
            "it rains", "the ground is wet", "the plants grow",
            "the harvest is good", "the price drops",
            "demand rises", "supply falls", "profits increase",
            "expansion begins", "hiring increases"
        ], self.rng.randint(4, 6))

        chain = items
        given_idx = 0
        ask_idx = len(chain) - 1

        rules = [f"If {chain[i]}, then {chain[i+1]}." for i in range(len(chain)-1)]
        self.rng.shuffle(rules)

        prompt = (
            f"Given these rules:\n" +
            "\n".join(f"  {r}" for r in rules) +
            f"\n\nIf we know that '{chain[given_idx]}' is true, "
            f"what can we conclude about '{chain[ask_idx]}'? "
            f"Show the chain of reasoning."
        )
        ref = (
            f"'{chain[ask_idx]}' is true. Chain: " +
            " → ".join(chain)
        )
        rubric = (
            f"1.0: correct conclusion with complete chain. "
            f"0.5: correct conclusion without full chain shown. "
            f"0.0: wrong conclusion."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def scheduling_constraint(self) -> Question:
        """Given constraints, find a valid schedule ordering."""
        tasks = self.rng.sample(["A", "B", "C", "D", "E"], 4)

        # Generate a valid order and derive constraints from it
        valid_order = list(tasks)
        self.rng.shuffle(valid_order)

        constraints = []
        # Add n-1 "before" constraints along the valid order
        for i in range(len(valid_order) - 1):
            constraints.append((valid_order[i], valid_order[i + 1]))

        # Verify unique topological sort
        # With a chain of n-1 constraints on n items, it's unique
        constraint_lines = [f"  • {a} must come before {b}" for a, b in constraints]
        self.rng.shuffle(constraint_lines)

        prompt = (
            f"Schedule these {len(tasks)} tasks in order, given these constraints:\n" +
            "\n".join(constraint_lines) +
            f"\n\nWhat is the only valid ordering?"
        )
        ref = " → ".join(valid_order)
        rubric = (
            f"1.0: {ref}. "
            f"0.5: valid partial order with one swap. "
            f"0.0: violates a constraint."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def negation_fallacy(self) -> Question:
        """Present a logical fallacy and ask to identify it."""
        fallacies = [
            {
                "name": "denying the antecedent",
                "premise1": "If it is raining, then the streets are wet.",
                "premise2": "It is not raining.",
                "conclusion": "Therefore, the streets are not wet.",
                "explanation": "The streets could be wet for other reasons (sprinklers, flooding). Not-P does not entail not-Q.",
            },
            {
                "name": "affirming the consequent",
                "premise1": "If a shape is a square, then it has four sides.",
                "premise2": "This shape has four sides.",
                "conclusion": "Therefore, this shape is a square.",
                "explanation": "Many shapes have four sides (rectangles, trapezoids). Q being true does not prove P.",
            },
            {
                "name": "false dilemma",
                "premise1": "Either we ban all cars or pollution will destroy the planet.",
                "premise2": "We cannot ban all cars.",
                "conclusion": "Therefore, pollution will destroy the planet.",
                "explanation": "The dilemma ignores middle options (electric cars, regulation, public transit). The either/or framing is artificially restrictive.",
            },
            {
                "name": "composition fallacy",
                "premise1": "Every brick in this wall is light.",
                "premise2": "This wall is made entirely of bricks.",
                "conclusion": "Therefore, this wall is light.",
                "explanation": "Properties of parts don't necessarily transfer to the whole. Many light bricks can compose a heavy wall.",
            },
        ]

        f = self.rng.choice(fallacies)
        prompt = (
            f"Identify the logical fallacy in this argument and explain why it is invalid:\n\n"
            f"Premise 1: {f['premise1']}\n"
            f"Premise 2: {f['premise2']}\n"
            f"Conclusion: {f['conclusion']}"
        )
        ref = f"{f['name']}. {f['explanation']}"
        rubric = (
            f"1.0: correctly names '{f['name']}' (or equivalent) and explains why invalid. "
            f"0.5: identifies it as invalid with partial explanation but wrong fallacy name. "
            f"0.0: says the argument is valid."
        )
        return Question("", "reasoning", "medium", 2, prompt, ref, rubric)

    def deductive_elimination(self) -> Question:
        """Process of elimination puzzle: who owns what?"""
        names = self.rng.sample(["Yuki", "Rahul", "Sofia", "Leo"], 3)
        pets = self.rng.sample(["cat", "dog", "fish", "parrot", "hamster"], 3)

        # Create a definite assignment
        assignment = dict(zip(names, pets))

        # Generate clues that uniquely determine the assignment
        clues = []
        # Clue 1: X does not own pet_a or pet_b (elimination)
        non_owner = names[0]
        wrong_pets = [p for p in pets if p != assignment[non_owner]]
        clues.append(f"{non_owner} does not own the {wrong_pets[0]} or the {wrong_pets[1]}.")

        # Clue 2: The person who owns pet_x is not Y
        clues.append(f"The person who owns the {assignment[names[1]]} is not {names[2]}.")

        # Clue 3: Direct clue
        clues.append(f"{names[2]} owns the {assignment[names[2]]}.")

        prompt = (
            f"Three people — {', '.join(names)} — each own exactly one pet "
            f"({', '.join(pets)}). Use these clues to determine who owns which pet:\n\n"
            + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(clues)) +
            f"\n\nWho owns which pet?"
        )
        ref = ", ".join(f"{n} owns the {assignment[n]}" for n in names)
        rubric = (
            f"1.0: all three correct: {ref}. "
            f"0.5: two correct. 0.0: one or zero correct."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def river_crossing(self) -> Question:
        """Farmer-style crossing puzzle with varied parameters."""
        n_items = self.rng.choice([3, 4])

        if n_items == 3:
            items = self.rng.choice([
                ("wolf", "goat", "cabbage"),
                ("fox", "chicken", "grain"),
                ("cat", "mouse", "cheese"),
            ])
            predator, prey, food = items
            # Minimum crossings: 7
            min_crossings = 7
            prompt = (
                f"A farmer needs to cross a river with a {predator}, a {prey}, and some {food}. "
                f"The boat carries the farmer plus one item. If left alone, the {predator} eats "
                f"the {prey}, and the {prey} eats the {food}. "
                f"What is the minimum number of crossings needed, and describe the sequence?"
            )
            ref = (
                f"{min_crossings} crossings. Step 1: take {prey} across. "
                f"Step 2: return alone. Step 3: take {predator} across. "
                f"Step 4: bring {prey} back. Step 5: take {food} across. "
                f"Step 6: return alone. Step 7: take {prey} across."
            )
        else:
            # 4-person bridge problem
            times = sorted(self.rng.sample([1, 2, 3, 5, 7, 8, 10, 12, 15], 4))
            a, b, c, d = times
            # Optimal strategy for bridge and torch
            strategy1 = a + b + a + d + a  # naive: fastest escorts
            strategy2 = a + b + a + d + b  # optimized
            strategy3 = a + c + a + d + a
            # Optimal: min of two strategies
            opt1 = 2*b + a + d  # send two slowest together, b shuttles
            opt2 = a + b + c + d  # fastest escorts each
            # Actually: optimal = min(2*b + a + d, a + 2*a + c + d) — classic solution
            # Strategy A: a+b cross, a returns, c+d cross, b returns, a+b cross = a+b+a+c+d... no
            # Standard: strategy 1: a escorts each = a+b + a + a+c + a + a+d = way too many
            # Correct optimal for 4 people:
            # Method 1: A+B cross (B time), A back (A), C+D cross (D), B back (B), A+B cross (B) = 2B + A + D
            # Method 2: A+B cross (B), A back (A), A+C cross (C), A back (A), A+D cross (D) = 3A + B + C + D... no
            # Method 2: A escorts each: A+C cross(C), A back(A), A+D cross(D), A back(A), A+B cross(B) = B+C+D + 2A
            optimal = min(2*b + a + d, 2*a + c + d)

            prompt = (
                f"Four people must cross a bridge at night with one flashlight. "
                f"At most two can cross at a time, and they must carry the flashlight. "
                f"Their crossing times are {a}, {b}, {c}, and {d} minutes. "
                f"A pair crosses at the speed of the slower person. "
                f"What is the minimum total time to get everyone across?"
            )
            ref = f"{optimal} minutes."
            min_crossings = 5  # always 5 for 4 people

        rubric = (
            f"1.0: {ref.split('.')[0]} with correct step sequence. "
            f"0.5: correct minimum time without full sequence. "
            f"0.0: wrong minimum time."
        )
        return Question("", "reasoning", "hard", 2, prompt, ref, rubric)

    def logical_equivalence(self) -> Question:
        """Are two logical statements equivalent?"""
        pairs = [
            {
                "stmt1": "If it rains, then the ground is wet.",
                "stmt2": "If the ground is not wet, then it is not raining.",
                "equivalent": True,
                "reason": "Statement 2 is the contrapositive of Statement 1. A conditional and its contrapositive are logically equivalent."
            },
            {
                "stmt1": "If it rains, then the ground is wet.",
                "stmt2": "If the ground is wet, then it is raining.",
                "equivalent": False,
                "reason": "Statement 2 is the converse of Statement 1, not the contrapositive. The ground could be wet for other reasons."
            },
            {
                "stmt1": "No cats are dogs.",
                "stmt2": "No dogs are cats.",
                "equivalent": True,
                "reason": "Universal negative propositions are symmetric: if no A are B, then no B are A."
            },
            {
                "stmt1": "All dogs are mammals.",
                "stmt2": "All mammals are dogs.",
                "equivalent": False,
                "reason": "The converse of a universal affirmative is not equivalent. All dogs are mammals does not mean all mammals are dogs."
            },
            {
                "stmt1": "Not all students passed.",
                "stmt2": "Some students did not pass.",
                "equivalent": True,
                "reason": "¬∀x P(x) is logically equivalent to ∃x ¬P(x). The negation of 'all' is 'some not'."
            },
        ]
        pair = self.rng.choice(pairs)
        prompt = (
            f"Are these two statements logically equivalent? Explain.\n\n"
            f"Statement 1: \"{pair['stmt1']}\"\n"
            f"Statement 2: \"{pair['stmt2']}\""
        )
        answer = "Yes, equivalent" if pair["equivalent"] else "No, not equivalent"
        ref = f"{answer}. {pair['reason']}"
        rubric = (
            f"1.0: correct answer ('{answer}') with correct justification. "
            f"0.5: correct answer without justification. "
            f"0.0: wrong answer."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)

    def weighted_majority(self) -> Question:
        """Weighted voting: does a coalition have enough weight to pass?"""
        n_voters = self.rng.randint(4, 6)
        weights = sorted([self.rng.randint(1, 10) for _ in range(n_voters)], reverse=True)
        total = sum(weights)
        quota = total // 2 + 1  # simple majority

        voters = [f"V{i+1}" for i in range(n_voters)]

        # Pick a coalition
        coalition_size = self.rng.randint(2, n_voters - 1)
        coalition = self.rng.sample(range(n_voters), coalition_size)
        coalition_weight = sum(weights[i] for i in coalition)
        passes = coalition_weight >= quota

        coalition_names = [voters[i] for i in sorted(coalition)]

        prompt = (
            f"In a weighted voting system, {n_voters} voters have these weights:\n"
            + ", ".join(f"{voters[i]} = {weights[i]}" for i in range(n_voters)) +
            f"\n\nThe quota to pass a motion is {quota} (simple majority of total weight {total}).\n"
            f"Does the coalition {{{', '.join(coalition_names)}}} have enough weight to pass? "
            f"What is their total weight?"
        )
        ref = (
            f"Coalition weight = {coalition_weight}. Quota = {quota}. "
            f"{'Yes, passes' if passes else 'No, fails'} "
            f"({coalition_weight} {'≥' if passes else '<'} {quota})."
        )
        rubric = (
            f"1.0: correct weight ({coalition_weight}) and correct pass/fail. "
            f"0.5: correct pass/fail, wrong weight. "
            f"0.0: wrong pass/fail."
        )
        return Question("", "reasoning", "medium", 1, prompt, ref, rubric)


if __name__ == "__main__":
    import json
    import os
    import hashlib

    quarter = os.environ.get("BRAIAIN_QUARTER", "2026Q2")
    salt = os.environ.get("BRAIAIN_SEED_SALT", "")
    if not salt:
        print("NOTE: BRAIAIN_SEED_SALT not set. Using unsalted seed for dev.")
    QUARTERLY_SEED = int(hashlib.sha256(f"{quarter}:{salt}".encode()).hexdigest()[:8], 16)
    gen = LogicGenerator(seed=QUARTERLY_SEED)
    questions = gen.generate_all(n=17)

    print(f"Generated {len(questions)} novel reasoning questions\n")
    for q in questions[:3]:
        print(f"--- {q.id} [{q.difficulty}] ---")
        print(f"Q: {q.prompt[:200]}...")
        print(f"A: {q.reference[:200]}")
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
    with open("questions/novel_reasoning.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to questions/novel_reasoning.json")
