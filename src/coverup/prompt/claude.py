import typing as T
from .prompter import Prompter, CodeSegment, mk_message, get_module_name
from ..utils import lines_branches_do


class ClaudePrompter(Prompter):
    """Prompter for Claude."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        module_name = get_module_name(segment.path, self.args.src_base_dir)
        filename = segment.path.relative_to(self.args.src_base_dir)

        return [
            mk_message("You are an expert Python test-driven developer who creates pytest test functions that achieve high coverage.",
                    role="system"),
            mk_message(f"""
<file path="{filename}" module_name="{module_name}">
{segment.get_excerpt(tag_lines=bool(segment.executed_lines))}
</file>

<instructions>

The code above does not achieve full coverage:
when tested, {'it does' if not segment.executed_lines else segment.lines_branches_missing_do()} not execute.

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


    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict]:
        return [mk_message(f"""\
<error>{error}</error>
Executing the test yields an error, shown above.
<instructions>
1. Modify the test to correct it.
2. Respond with the complete Python code in backticks.
3. Before answering the question, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict]:
        return [mk_message(f"""\
This test still lacks coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
<instructions>
1. Modify it to execute those lines.
2. Respond with the complete Python code in backticks.
3. Before responding, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]
