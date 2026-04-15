"""
BRAIAIN Standard — Bug Injection System
Takes known-correct functions and injects programmatic bugs.
Answer (the bug + fix) is known before the question is generated.
"""

import random
import ast
import textwrap
from dataclasses import dataclass


@dataclass
class BugQuestion:
    id: str
    difficulty: str
    original_code: str
    buggy_code: str
    bug_type: str
    bug_description: str
    fix_description: str
    unit_tests: list[dict]
    prompt: str
    rubric: str


# ── CLEAN FUNCTION LIBRARY ────────────────────────────────────────────────────
# Each function is known-correct and tested. Bugs will be injected programmatically.

CLEAN_FUNCTIONS = {

    "binary_search": {
        "code": '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
        "tests": [
            {"input": "([1,3,5,7,9], 5)", "expected": "2"},
            {"input": "([1,3,5,7,9], 1)", "expected": "0"},
            {"input": "([1,3,5,7,9], 9)", "expected": "4"},
            {"input": "([1,3,5,7,9], 4)", "expected": "-1"},
            {"input": "([], 1)", "expected": "-1"},
        ]
    },

    "reverse_string": {
        "code": '''
def reverse_words(sentence):
    words = sentence.split()
    return ' '.join(reversed(words))
''',
        "tests": [
            {"input": "('hello world')", "expected": "'world hello'"},
            {"input": "('a b c')", "expected": "'c b a'"},
            {"input": "('single')", "expected": "'single'"},
            {"input": "('')", "expected": "''"},
        ]
    },

    "is_prime": {
        "code": '''
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
''',
        "tests": [
            {"input": "(2,)", "expected": "True"},
            {"input": "(17,)", "expected": "True"},
            {"input": "(4,)", "expected": "False"},
            {"input": "(1,)", "expected": "False"},
            {"input": "(97,)", "expected": "True"},
        ]
    },

    "count_vowels": {
        "code": '''
def count_vowels(s):
    vowels = set('aeiouAEIOU')
    return sum(1 for c in s if c in vowels)
''',
        "tests": [
            {"input": "('hello')", "expected": "2"},
            {"input": "('AEIOU')", "expected": "5"},
            {"input": "('rhythm')", "expected": "0"},
            {"input": "('')", "expected": "0"},
        ]
    },

    "flatten_list": {
        "code": '''
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
''',
        "tests": [
            {"input": "([[1, [2, 3]], 4],)", "expected": "[1, 2, 3, 4]"},
            {"input": "([1, 2, 3],)", "expected": "[1, 2, 3]"},
            {"input": "([[]],)", "expected": "[]"},
            {"input": "([[1, [2, [3]]]],)", "expected": "[1, 2, 3]"},
        ]
    },

    "two_sum": {
        "code": '''
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
''',
        "tests": [
            {"input": "([2,7,11,15], 9)", "expected": "[0, 1]"},
            {"input": "([3,2,4], 6)", "expected": "[1, 2]"},
            {"input": "([3,3], 6)", "expected": "[0, 1]"},
            {"input": "([1,2,3], 10)", "expected": "[]"},
        ]
    },

    "max_subarray": {
        "code": '''
def max_subarray(nums):
    max_sum = nums[0]
    current = nums[0]
    for num in nums[1:]:
        current = max(num, current + num)
        max_sum = max(max_sum, current)
    return max_sum
''',
        "tests": [
            {"input": "([-2,1,-3,4,-1,2,1,-5,4],)", "expected": "6"},
            {"input": "([1],)", "expected": "1"},
            {"input": "([-1,-2,-3],)", "expected": "-1"},
            {"input": "([5,4,-1,7,8],)", "expected": "23"},
        ]
    },

    "roman_to_int": {
        "code": '''
def roman_to_int(s):
    val = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    total = 0
    for i in range(len(s)):
        if i < len(s)-1 and val[s[i]] < val[s[i+1]]:
            total -= val[s[i]]
        else:
            total += val[s[i]]
    return total
''',
        "tests": [
            {"input": "('III',)", "expected": "3"},
            {"input": "('IV',)", "expected": "4"},
            {"input": "('IX',)", "expected": "9"},
            {"input": "('LVIII',)", "expected": "58"},
            {"input": "('MCMXCIV',)", "expected": "1994"},
        ]
    },

    "valid_brackets": {
        "code": '''
def is_valid(s):
    stack = []
    pairs = {')':'(', ']':'[', '}':'{'}
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0
''',
        "tests": [
            {"input": "('()',)", "expected": "True"},
            {"input": "('()[]{}')", "expected": "True"},
            {"input": "('(]',)", "expected": "False"},
            {"input": "('([)]',)", "expected": "False"},
            {"input": "('{[]}',)", "expected": "True"},
        ]
    },

    "rotate_array": {
        "code": '''
def rotate(nums, k):
    n = len(nums)
    k = k % n
    return nums[-k:] + nums[:-k] if k else nums[:]
''',
        "tests": [
            {"input": "([1,2,3,4,5,6,7], 3)", "expected": "[5, 6, 7, 1, 2, 3, 4]"},
            {"input": "([-1,-100,3,99], 2)", "expected": "[3, 99, -1, -100]"},
            {"input": "([1], 0)", "expected": "[1]"},
            {"input": "([1,2], 3)", "expected": "[2, 1]"},
        ]
    },

    "merge_sorted_lists": {
        "code": '''
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
''',
        "tests": [
            {"input": "([1,3,5], [2,4,6])", "expected": "[1, 2, 3, 4, 5, 6]"},
            {"input": "([1], [2])", "expected": "[1, 2]"},
            {"input": "([], [1,2,3])", "expected": "[1, 2, 3]"},
            {"input": "([1,2,3], [])", "expected": "[1, 2, 3]"},
        ]
    },

    "palindrome_check": {
        "code": '''
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
''',
        "tests": [
            {"input": "('racecar',)", "expected": "True"},
            {"input": "('hello',)", "expected": "False"},
            {"input": "('A man a plan a canal Panama',)", "expected": "True"},
            {"input": "('ab',)", "expected": "False"},
            {"input": "('a',)", "expected": "True"},
        ]
    },

    "power_recursive": {
        "code": '''
def power(x, n):
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    return x * power(x, n - 1)
''',
        "tests": [
            {"input": "(2, 10)", "expected": "1024"},
            {"input": "(3, 0)", "expected": "1"},
            {"input": "(2, -2)", "expected": "0.25"},
            {"input": "(5, 3)", "expected": "125"},
        ]
    },

    "matrix_transpose": {
        "code": '''
def transpose(matrix):
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        result.append(row)
    return result
''',
        "tests": [
            {"input": "([[1,2,3],[4,5,6]],)", "expected": "[[1, 4], [2, 5], [3, 6]]"},
            {"input": "([[1]],)", "expected": "[[1]]"},
            {"input": "([],)", "expected": "[]"},
            {"input": "([[1,2],[3,4],[5,6]],)", "expected": "[[1, 3, 5], [2, 4, 6]]"},
        ]
    },

    "find_second_largest": {
        "code": '''
def second_largest(nums):
    if len(nums) < 2:
        return None
    first = second = float('-inf')
    for num in nums:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num
    return second if second != float('-inf') else None
''',
        "tests": [
            {"input": "([3,1,4,1,5,9],)", "expected": "5"},
            {"input": "([1,2],)", "expected": "1"},
            {"input": "([5,5,5],)", "expected": "None"},
            {"input": "([1],)", "expected": "None"},
            {"input": "([10,20,30,40],)", "expected": "30"},
        ]
    },

    "insertion_sort": {
        "code": '''
def insertion_sort(arr):
    result = arr[:]
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result
''',
        "tests": [
            {"input": "([3,1,4,1,5],)", "expected": "[1, 1, 3, 4, 5]"},
            {"input": "([1],)", "expected": "[1]"},
            {"input": "([5,4,3,2,1],)", "expected": "[1, 2, 3, 4, 5]"},
            {"input": "([],)", "expected": "[]"},
        ]
    },

    "fibonacci": {
        "code": '''
def fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "tests": [
            {"input": "(0,)", "expected": "0"},
            {"input": "(1,)", "expected": "1"},
            {"input": "(10,)", "expected": "55"},
            {"input": "(6,)", "expected": "8"},
        ]
    },

    "remove_all": {
        "code": '''
def remove_all(lst, val):
    result = []
    for item in lst:
        if item != val:
            result.append(item)
    return result
''',
        "tests": [
            {"input": "([1,2,3,2,1], 2)", "expected": "[1, 3, 1]"},
            {"input": "([1,1,1], 1)", "expected": "[]"},
            {"input": "([1,2,3], 4)", "expected": "[1, 2, 3]"},
            {"input": "([], 1)", "expected": "[]"},
        ]
    },

    "running_average": {
        "code": '''
def running_average(nums):
    result = []
    total = 0
    for i, num in enumerate(nums):
        total += num
        result.append(round(total / (i + 1), 2))
    return result
''',
        "tests": [
            {"input": "([1,2,3,4],)", "expected": "[1.0, 1.5, 2.0, 2.5]"},
            {"input": "([10],)", "expected": "[10.0]"},
            {"input": "([4,4,4],)", "expected": "[4.0, 4.0, 4.0]"},
        ]
    },
}


# ── BUG TYPES ─────────────────────────────────────────────────────────────────

class BugInjector:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def off_by_one_loop_end(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Change `range(n)` to `range(n-1)` or similar"""
        buggy = code.replace("+ 1, 2)", ", 2)")  # is_prime specific
        if buggy == code:
            buggy = code.replace("len(s)-1", "len(s)-2")  # roman specific
        if buggy == code:
            return None
        return (
            buggy,
            "off-by-one: loop terminates one iteration too early",
            "Change the loop range to include the missing final iteration"
        )

    def wrong_comparison(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Flip < to <= or > to >= in a critical condition"""
        if "< right:" in code:
            return (
                code.replace("left <= right:", "left < right:"),
                "wrong comparison operator: <= changed to < causes off-by-one in binary search",
                "Change 'left < right' back to 'left <= right'"
            )
        if "elif arr[mid] < target:" in code:
            return (
                code.replace("arr[mid] < target:", "arr[mid] <= target:"),
                "wrong comparison operator: < changed to <= causes incorrect pivot handling",
                "Change 'arr[mid] <= target' back to 'arr[mid] < target'"
            )
        return None

    def missing_base_case(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Remove or break the base case in a recursive function"""
        if "if n < 2:" in code:
            return (
                code.replace("if n < 2:\n        return False", "if n < 1:\n        return False"),
                "wrong base case: n < 1 instead of n < 2 causes is_prime(1) to return True incorrectly",
                "Change 'if n < 1' back to 'if n < 2'"
            )
        if "isinstance(item, list)" in code:
            return (
                code.replace("result.extend(flatten(item))", "result.extend(item)"),
                "missing recursion: nested lists are not recursively flattened",
                "Change 'result.extend(item)' back to 'result.extend(flatten(item))'"
            )
        return None

    def wrong_variable(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Reference wrong variable"""
        if "complement = target - num" in code:
            return (
                code.replace("if complement in seen:", "if num in seen:"),
                "wrong variable: checks if current number is in seen instead of complement",
                "Change 'if num in seen' back to 'if complement in seen'"
            )
        if "current_sum = max(num, current_sum + num)" in code or "current = max(num, current + num)" in code:
            return (
                code.replace("max_sum = max(max_sum, current)", "max_sum = max(max_sum, num)"),
                "wrong variable: tracks single element instead of current subarray sum",
                "Change 'max(max_sum, num)' back to 'max(max_sum, current)'"
            )
        return None

    def wrong_operator(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Change + to - or * in key expression"""
        if "total -= val[s[i]]" in code:
            return (
                code.replace("total -= val[s[i]]", "total += val[s[i]]"),
                "wrong operator: subtracts when it should add for subtractive notation (IV, IX, etc.)",
                "Change 'total += val[s[i]]' back to 'total -= val[s[i]]' for the subtraction case"
            )
        return None

    def inverted_condition(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Invert a boolean condition"""
        if "not stack or stack[-1] != pairs[c]:" in code:
            return (
                code.replace(
                    "if not stack or stack[-1] != pairs[c]:",
                    "if stack and stack[-1] == pairs[c]:"
                ),
                "inverted condition: pops when brackets match instead of when they don't",
                "Change the condition back to 'if not stack or stack[-1] != pairs[c]'"
            )
        if "len(stack) == 0" in code:
            return (
                code.replace("return len(stack) == 0", "return len(stack) != 0"),
                "inverted return: returns True when stack is non-empty (unmatched brackets remain)",
                "Change 'len(stack) != 0' back to 'len(stack) == 0'"
            )
        return None

    def wrong_merge_order(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Flip <= to < in merge comparison, or swap extend targets"""
        if "a[i] <= b[j]" in code:
            return (
                code.replace("a[i] <= b[j]", "a[i] < b[j]"),
                "wrong comparison: < instead of <= breaks stable merge when elements are equal",
                "Change 'a[i] < b[j]' back to 'a[i] <= b[j]' to maintain stability"
            )
        if "result.extend(a[i:])" in code and "result.extend(b[j:])" in code:
            return (
                code.replace("result.extend(a[i:])\n    result.extend(b[j:])",
                             "result.extend(b[j:])\n    result.extend(a[i:])"),
                "swapped remainders: appends remaining b before remaining a, corrupting sorted order",
                "Swap the extend calls back: extend a[i:] first, then b[j:]"
            )
        return None

    def wrong_palindrome_direction(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Break palindrome check by moving pointers wrong direction"""
        if "left += 1" in code and "right -= 1" in code and "s[left] != s[right]" in code:
            return (
                code.replace("right -= 1", "right += 1"),
                "wrong pointer direction: right pointer moves away from center instead of toward it, causing infinite loop or index error",
                "Change 'right += 1' back to 'right -= 1'"
            )
        return None

    def wrong_base_case_value(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Return wrong value from base case"""
        if "if n == 0:\n        return 1" in code and "power" in code:
            return (
                code.replace("if n == 0:\n        return 1", "if n == 0:\n        return 0"),
                "wrong base case value: x^0 should return 1, not 0 — causes all results to be 0",
                "Change 'return 0' back to 'return 1' in the n==0 base case"
            )
        return None

    def wrong_transpose_index(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Swap row/column indices in transpose"""
        if "matrix[i][j]" in code:
            return (
                code.replace("matrix[i][j]", "matrix[j][i]"),
                "swapped indices: reads matrix[j][i] instead of matrix[i][j], producing the original matrix instead of the transpose",
                "Change 'matrix[j][i]' back to 'matrix[i][j]'"
            )
        return None

    def wrong_update_order(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Swap update order in tracking algorithm"""
        if "second = first" in code and "first = num" in code:
            return (
                code.replace(
                    "second = first\n            first = num",
                    "first = num\n            second = first"
                ),
                "wrong update order: sets first before saving to second, so second always equals first (the new max)",
                "Change order back: set second = first BEFORE setting first = num"
            )
        return None

    def wrong_sort_comparison(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Flip comparison in sort, producing descending instead of ascending"""
        if "result[j] > key" in code:
            return (
                code.replace("result[j] > key", "result[j] < key"),
                "wrong comparison direction: shifts elements when they are LESS than key, producing descending sort",
                "Change 'result[j] < key' back to 'result[j] > key'"
            )
        return None

    def wrong_fibonacci_base(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Break fibonacci base case"""
        if "if n < 2:\n        return n" in code and "fibonacci" in code:
            return (
                code.replace("if n < 2:\n        return n", "if n < 2:\n        return 1"),
                "wrong base case: returns 1 for both fib(0) and fib(1), but fib(0) should be 0",
                "Change 'return 1' back to 'return n' in the base case"
            )
        return None

    def wrong_filter_comparison(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Flip != to == in filter, keeping the wrong elements"""
        if "if item != val:" in code:
            return (
                code.replace("if item != val:", "if item == val:"),
                "inverted filter: keeps matching elements instead of non-matching, returns only the values that should be removed",
                "Change 'if item == val' back to 'if item != val'"
            )
        return None

    def wrong_divisor(self, code: str, func_name: str) -> tuple[str, str, str]:
        """Use wrong divisor in average calculation"""
        if "total / (i + 1)" in code:
            return (
                code.replace("total / (i + 1)", "total / i") .replace("for i, num in enumerate(nums):", "for i, num in enumerate(nums, 1):"),
                "off-by-one in index: enumerate starts at 1 but divides by i, causing first element to divide by 1 (correct) but subsequent averages to be wrong due to shifted indexing",
                "Change back to enumerate(nums) with total / (i + 1)"
            )
        # Simpler bug: just wrong divisor
        if "total / (i + 1)" in code:
            return (
                code.replace("total / (i + 1)", "total / len(nums)"),
                "wrong divisor: divides by total count instead of running count, making early values too small",
                "Change 'total / len(nums)' back to 'total / (i + 1)'"
            )
        return None

    def inject_bug(self, func_name: str) -> BugQuestion | None:
        """Try to inject a bug into the named function."""
        if func_name not in CLEAN_FUNCTIONS:
            return None

        fn_data = CLEAN_FUNCTIONS[func_name]
        original = fn_data["code"]
        tests = fn_data["tests"]

        bug_methods = [
            self.off_by_one_loop_end,
            self.wrong_comparison,
            self.missing_base_case,
            self.wrong_variable,
            self.wrong_operator,
            self.inverted_condition,
            self.wrong_merge_order,
            self.wrong_palindrome_direction,
            self.wrong_base_case_value,
            self.wrong_transpose_index,
            self.wrong_update_order,
            self.wrong_sort_comparison,
            self.wrong_fibonacci_base,
            self.wrong_filter_comparison,
            self.wrong_divisor,
        ]
        self.rng.shuffle(bug_methods)

        for method in bug_methods:
            result = method(original, func_name)
            if result is not None:
                buggy, bug_desc, fix_desc = result
                if buggy != original:  # confirm something changed
                    prompt = (
                        f"The following Python function has a bug. "
                        f"Identify the bug and provide the corrected code:\n"
                        f"```python{buggy}```"
                    )
                    rubric = (
                        f"1.0: correctly identifies '{bug_desc}' and provides correct fix. "
                        f"Fix: {fix_desc}. "
                        f"0.5: identifies the wrong output without pinpointing the exact bug. "
                        f"0.0: incorrect fix that doesn't resolve all test cases."
                    )
                    return BugQuestion(
                        id="",
                        difficulty="medium",
                        original_code=original,
                        buggy_code=buggy,
                        bug_type=method.__name__,
                        bug_description=bug_desc,
                        fix_description=fix_desc,
                        unit_tests=tests,
                        prompt=prompt,
                        rubric=rubric
                    )
        return None

    def generate_all(self, n: int = 15) -> list[BugQuestion]:
        func_names = list(CLEAN_FUNCTIONS.keys())
        self.rng.shuffle(func_names)
        questions = []
        for fname in func_names:
            q = self.inject_bug(fname)
            if q:
                q.id = f"NOVEL_C{len(questions)+1:02d}"
                questions.append(q)
                if len(questions) >= n:
                    break
        return questions


if __name__ == "__main__":
    import json
    import os
    import hashlib

    quarter = os.environ.get("BRAIAIN_QUARTER", "2026Q2")
    salt = os.environ.get("BRAIAIN_SEED_SALT", "")
    if not salt:
        print("NOTE: BRAIAIN_SEED_SALT not set. Using unsalted seed for dev.")
    QUARTERLY_SEED = int(hashlib.sha256(f"{quarter}:{salt}".encode()).hexdigest()[:8], 16)
    injector = BugInjector(seed=QUARTERLY_SEED)
    questions = injector.generate_all(n=15)

    print(f"Generated {len(questions)} novel coding questions\n")
    for q in questions[:2]:
        print(f"--- {q.id} ---")
        print(f"Bug type: {q.bug_type}")
        print(f"Bug: {q.bug_description}")
        print(f"Fix: {q.fix_description}")
        print(f"Prompt: {q.prompt[:200]}...\n")

    output = [
        {
            "id": q.id,
            "dimension": "coding",
            "difficulty": q.difficulty,
            "tier": 1,
            "prompt": q.prompt,
            "reference": q.fix_description,
            "bug_type": q.bug_type,
            "unit_tests": q.unit_tests,
            "rubric": q.rubric
        }
        for q in questions
    ]
    with open("questions/novel_coding.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to questions/novel_coding.json")
