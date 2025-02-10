import typing as T
from .prompter import Prompter, CodeSegment, mk_message, get_module_name
from ..utils import lines_branches_do


class GptV1Prompter(Prompter):
    """Prompter for GPT-4."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        module_name = get_module_name(segment.path, self.args.src_base_dir)
        filename = segment.path.relative_to(self.args.src_base_dir)

        return [
            mk_message(f"""
You are an expert Python test-driven developer.
The code below, extracted from {filename},{' module ' + module_name + ',' if module_name else ''} does not achieve full coverage:
when tested, {'it does' if not segment.executed_lines else segment.lines_branches_missing_do()} not execute.
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
{segment.get_excerpt(tag_lines=bool(segment.executed_lines))}
```
""")
        ]

    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict]:
        return [mk_message(f"""\
Executing the test yields an error, shown below.
Modify the test to correct it; respond only with the complete Python code in backticks.

{error}""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict]:
        return [mk_message(f"""\
This test still lacks coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
""")
        ]
