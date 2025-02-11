import typing as T
from .gpt_v2 import *


class GptV2AblatedPrompter(GptV2Prompter):
    """Prompter for GPT 4."""

    def __init__(
        self,
        cmd_args,
        *,
        with_coverage: bool = True,
        with_get_info: bool = True,
        with_imports: bool = True,
        with_error_fixing: bool = True
    ):
        super().__init__(cmd_args)
        self.with_coverage = with_coverage
        self.with_get_info = with_get_info
        self.with_imports = with_imports
        self.with_error_fixing = with_error_fixing


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        filename = segment.path.relative_to(self.args.src_base_dir)

        return [
            mk_message(f"""
You are an expert Python test-driven developer.
""" + (f"""\
The code below, extracted from {filename}, does not achieve full coverage:
when tested, {segment.lines_branches_missing_do()} not execute.
Create new pytest test functions that execute all missing lines and branches, always making
""" if self.with_coverage else f"""\
The code below, extracted from {filename}, does not achieve full coverage.
Create new pytest test functions that execute all lines and branches, always making
""") + f"""\
sure that each test is correct and indeed improves coverage.
""" + ("Use the get_info tool function as necessary.\n" if self.with_get_info else "") + f"""\
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as to avoid state pollution;
use 'monkeypatch' or 'pytest-mock' if appropriate.
Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{segment.get_excerpt(tag_lines=self.with_coverage, add_imports=self.with_imports)}
```
""")
        ]


    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict] | None:
        if not self.with_error_fixing: return None
        return [mk_message(f"""\
Executing the test yields an error, shown below.
Modify or rewrite the test to correct it; respond only with the complete Python code in backticks.
""" + ("Use the get_info tool function as necessary.\n" if self.with_get_info else "") + f"""\

{error}""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        if not self.with_coverage: return None
        return [mk_message(f"""\
The tests still lack coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
""" + ("Use the get_info tool function as necessary.\n" if self.with_get_info else "")
)
        ]


    def get_info(self, ctx: CodeSegment, name: str) -> str:
        """
        {
            "name": "get_info",
            "description": "Returns information about a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "class, function or method name, as in 'f' for function f or 'C.foo' for method foo in class C."
                    }
                },
                "required": ["name"]
            }
        }
        """

        if info := codeinfo.get_info(
            codeinfo.parse_file(ctx.path),
            name,
            line=ctx.begin,
            generate_imports=self.with_imports
        ):
            return "\"...\" below indicates omitted code.\n\n" + info

        return f"Unable to obtain information on {name}."


    def get_functions(self) -> T.List[T.Callable]:
        if not self.with_get_info: return []
        return [self.get_info]
