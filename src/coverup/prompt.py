import abc
from pathlib import Path
from .utils import lines_branches_do
from .segment import CodeSegment
import typing as T


def get_module_name(src_file: Path, src_dir: Path) -> str:
    # assumes both src_file and src_dir Path.resolve()'d
    try:
        relative = src_file.relative_to(src_dir)
        return ".".join((src_dir.stem,) + relative.parts[:-1] + (relative.stem,))
    except ValueError:
        return None  # not relative to source


class Prompter(abc.ABC):
    """Interface for a CoverUp prompter."""

    def __init__(self, args, segment: CodeSegment):
        self.args = args
        self.segment = segment


    @abc.abstractmethod
    def initial_prompt(self) -> T.List[dict]:
        """Returns initial prompt(s) for a code segment."""


    @abc.abstractmethod
    def error_prompt(self, error: str) -> T.List[dict]:
        """Returns prompts(s) in response to an error."""


    @abc.abstractmethod
    def missing_coverage_prompt(self) -> T.List[dict]:
        """Returns prompts(s) in response to the suggested test lacking coverage."""


def message(content: str, *, role="user") -> dict:
    return {
        'role': role,
        'content': content
    }


class Gpt4Prompter(Prompter):
    """Prompter for GPT-4."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)


    def initial_prompt(self) -> T.List[dict]:
        args = self.args
        seg = self.segment
        module_name = get_module_name(seg.path, args.source_dir)

        return [
            message(f"""
You are an expert Python test-driven developer.
The code below, extracted from {seg.filename},{' module ' + module_name + ',' if module_name else ''} does not achieve full coverage:
when tested, {seg.lines_branches_missing_do()} not execute.
Create a new pytest test function that executes these missing lines/branches, always making
sure that the new test is correct and indeed improves coverage.
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as not to affect other tests;
use 'pytest-mock' if appropriate.
Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{seg.get_excerpt()}
```
""")
        ]


    def error_prompt(self, error: str) -> T.List[dict]:
        return [message(f"""\
Executing the test yields an error, shown below.
Modify the test to correct it; respond only with the complete Python code in backticks.

{error}""")
        ]


    def missing_coverage_prompt(self, now_missing_lines: set, now_missing_branches: set) -> T.List[dict]:
        return [message(f"""\
This test still lacks coverage: {lines_branches_do(now_missing_lines, set(), now_missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
""")
        ]


class ClaudePrompter(Prompter):
    """Prompter for Claude."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)


    def initial_prompt(self) -> T.List[str]:
        args = self.args
        seg = self.segment
        module_name = get_module_name(seg.path, args.source_dir)

        return [
            message("You are an expert Python test-driven developer who creates pytest test functions that achieve high coverage.",
                    role="system"),
            message(f"""
<file path="{seg.filename}" module_name="{module_name}">
{seg.get_excerpt()}
</file>

<instructions>

The code above does not achieve full coverage:
when tested, {seg.lines_branches_missing_do()} not execute.

1. Create a new pytest test function that executes these missing lines/branches, always making
sure that the new test is correct and indeed improves coverage.

2. Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.

3. Be sure to include assertions in the test that verify any applicable postconditions.

4. Please also make VERY SURE to clean up after the test, so as not to affect other tests;
use 'pytest-mock' if appropriate.

5. Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.

6.  Respond with the Python code enclosed in backticks. Before answering the question, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]


    def error_prompt(self, error: str) -> T.List[dict]:
        return [message(f"""\
<error>{error}</error>
Executing the test yields an error, shown above.
<instructions>
1. Modify the test to correct it.
2. Respond with the complete Python code in backticks.
3. Before answering the question, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]


    def missing_coverage_prompt(self, now_missing_lines: set, now_missing_branches: set) -> T.List[dict]:
        return [message(f"""\
This test still lacks coverage: {lines_branches_do(now_missing_lines, set(), now_missing_branches)} not execute.
<instructions>
1. Modify it to execute those lines.
2. Respond with the complete Python code in backticks.
3. Before responding, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]


# prompter registry
prompters = {
    "gpt": Gpt4Prompter,
    "claude": ClaudePrompter
}
