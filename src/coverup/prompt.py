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

    def __init__(self, args):
        self.args = args


    @abc.abstractmethod
    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        """Returns initial prompt(s) for a code segment."""


    @abc.abstractmethod
    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict] | None:
        """Returns prompts(s) in response to an error."""


    @abc.abstractmethod
    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        """Returns prompts(s) in response to the suggested test(s) lacking coverage."""


    def get_functions(self) -> T.List[T.Callable]:
        """Returns a list of functions to be made available to the LLM.
           Each function's docstring must consist of its schema in JSON format."""
        return []


def _message(content: str, *, role="user") -> dict:
    return {
        'role': role,
        'content': content
    }


class Gpt4PrompterV1(Prompter):
    """Prompter for GPT-4 used in paper submission."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        args = self.args
        module_name = get_module_name(segment.path, args.package_dir)
        filename = segment.path.relative_to(args.package_dir.parent)

        return [
            _message(f"""
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
        return [_message(f"""\
Executing the test yields an error, shown below.
Modify the test to correct it; respond only with the complete Python code in backticks.

{error}""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict]:
        return [_message(f"""\
This test still lacks coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
""")
        ]


class Gpt4PrompterV2(Prompter):
    """Prompter for GPT."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        args = self.args
        module_name = get_module_name(segment.path, args.package_dir)
        filename = segment.path.relative_to(args.package_dir.parent)

        return [
            _message(f"""
You are an expert Python test-driven developer.
The code below, extracted from {filename}, does not achieve full coverage:
when tested, {segment.lines_branches_missing_do()} not execute.
Create new pytest test functions that execute all missing lines and branches, always making
sure that each test is correct and indeed improves coverage.
Use the get_info tool function as necessary.
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as to avoid state pollution;
use 'monkeypatch' or 'pytest-mock' if appropriate.
Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{segment.get_excerpt()}
```
""")
        ]


    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict]:
        return [_message(f"""\
Executing the test yields an error, shown below.
Modify or rewrite the test to correct it; respond only with the complete Python code in backticks.
Use the get_info tool function as necessary.

{error}""")
        ]


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict]:
        return [_message(f"""\
The tests still lack coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
Use the get_info tool function as necessary.
""")
        ]


    @staticmethod
    def get_info(ctx: CodeSegment, name: str) -> str:
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

        from .codeinfo import get_info, parse_file

        if info := get_info(parse_file(ctx.path), name, line=ctx.begin):
            return "\"...\" below indicates omitted code.\n\n" + info

        return f"Unable to obtain information on {name}."


    def get_functions(self) -> T.List[T.Callable]:
        return [__class__.get_info]


class Gpt4PrompterV2Ablated(Prompter):
    """Fully ablated GPT prompter."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)

    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        args = self.args
        module_name = get_module_name(segment.path, args.package_dir)
        filename = segment.path.relative_to(args.package_dir.parent)

        return [
            _message(f"""
You are an expert Python test-driven developer.
The code below, extracted from {filename}, does not achieve full coverage.
Create new pytest test functions that execute all lines and branches, always making
sure that each test is correct and indeed improves coverage.
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as to avoid state pollution;
use 'monkeypatch' or 'pytest-mock' if appropriate.
Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{segment.get_excerpt(tag_lines=False, include_imports=False)}
```
""")
        ]


    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict] | None:
        return None


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        return None


class Gpt4PrompterV2CoverageAblated(Prompter):
    """Partially ablated GPT prompter that lacks coverage information."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)

    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        args = self.args
        module_name = get_module_name(segment.path, args.package_dir)
        filename = segment.path.relative_to(args.package_dir.parent)

        return [
            _message(f"""
You are an expert Python test-driven developer.
The code below, extracted from {filename}, does not achieve full coverage.
Create new pytest test functions that execute all lines and branches, always making
sure that each test is correct and indeed improves coverage.
Use the get_info tool function as necessary.
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as to avoid state pollution;
use 'monkeypatch' or 'pytest-mock' if appropriate.
Write as little top-level code as possible, and in particular do not include any top-level code
calling into pytest.main or the test itself.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{segment.get_excerpt(tag_lines=False, include_imports=True)}
```
""")
        ]

    def error_prompt(self, segment: CodeSegment, error: str) -> T.List[dict] | None:
        return [_message(f"""\
Executing the test yields an error, shown below.
Modify the test to correct it; respond only with the complete Python code in backticks.
Use the get_info tool function as necessary.

{error}""")
        ]

    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        return None

    get_info = Gpt4PrompterV2.get_info

    def get_functions(self) -> T.List[T.Callable]:
        return [__class__.get_info]


class ClaudePrompter(Prompter):
    """Prompter for Claude."""

    def __init__(self, *args, **kwargs):
        Prompter.__init__(self, *args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        args = self.args
        module_name = get_module_name(segment.path, args.package_dir)
        filename = segment.path.relative_to(args.package_dir.parent)

        return [
            _message("You are an expert Python test-driven developer who creates pytest test functions that achieve high coverage.",
                    role="system"),
            _message(f"""
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
        return [_message(f"""\
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
        return [_message(f"""\
This test still lacks coverage: {lines_branches_do(missing_lines, set(), missing_branches)} not execute.
<instructions>
1. Modify it to execute those lines.
2. Respond with the complete Python code in backticks.
3. Before responding, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.
</instructions>
""")
        ]


# prompter registry
prompters = {
    "gpt": Gpt4PrompterV2,
    "gpt-v1": Gpt4PrompterV1,
    "gpt-v2": Gpt4PrompterV2,
    "gpt-v2-fully-ablated": Gpt4PrompterV2Ablated,
    "gpt-v2-ablated": Gpt4PrompterV2Ablated,
    "gpt-v2-coverage-ablated": Gpt4PrompterV2CoverageAblated,
    "claude": ClaudePrompter
}
