import typing as T
from .prompter import *
import coverup.codeinfo as codeinfo


class ClaudePrompter(Prompter):
    """Prompter tuned for Claude 3 Sonnet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def system_prompt(self) -> T.List[dict]:
        """Optional: prepend this message when using Sonnet for better reliability."""
        return [mk_message(
            "You are a code generator that writes complete Python test files. "
            "You must respond only with valid Python code enclosed in triple backticks, and nothing else. "
            "Do not include explanations or commentary."
        )]

    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        filename = segment.path.relative_to(self.args.src_base_dir)

        return [
            *self.system_prompt(),
            mk_message(f"""
You are an expert Python test-driven developer.

The following code, extracted from {filename}, does not achieve full coverage:
{segment.lines_branches_missing_do()} do not execute.

Your task:
- Write new **pytest test functions** that cause all missing lines and branches to execute.
- Tests must be correct and include assertions that verify postconditions.
- If necessary, use the `get_info` tool function to learn more about symbols.
- Ensure each test leaves no state behind; use `monkeypatch` or `pytest-mock` if helpful.
- Do NOT include any top-level code that calls `pytest.main` or the test itself.

Respond with **only the full Python test file**, enclosed in triple backticks.

Here is the code to test:

```python
{segment.get_excerpt()}
```
""")
        ]


    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict] | None:
        return [
            *self.system_prompt(),
            mk_message(f"""\
The test produced an error:

{error}

Please revise the test to correct the error.

Respond with only the complete revised Python test file, enclosed in triple backticks.
You may use the `get_info` tool function if needed.
""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        return [
            *self.system_prompt(),
            mk_message(f"""\
The test still lacks coverage: {lines_branches_do(missing_lines, set(), missing_branches)} do not execute.

Revise the test to ensure full coverage.

Respond with only the complete revised Python test file, enclosed in triple backticks.
You may use the `get_info` tool function if helpful.
""")
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

        if info := codeinfo.get_info(codeinfo.parse_file(ctx.path), name, line=ctx.begin):
            return "\"...\" below indicates omitted code.\n\n" + info

        return f"Unable to obtain information on {name}."


    def get_functions(self) -> T.List[T.Callable]:
        return [self.get_info]
