import typing as T
from .gpt_v2 import *


class GptV2NoCoveragePrompter(GptV2Prompter):
    """Partially ablated GPT prompter that lacks coverage information."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def initial_prompt(self, segment: CodeSegment) -> T.List[dict]:
        module_name = get_module_name(segment.path, self.args.package_dir)
        filename = segment.path.relative_to(self.args.package_dir.parent)

        return [
            mk_message(f"""
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


    def missing_coverage_prompt(self, segment: CodeSegment,
                                missing_lines: set, missing_branches: set) -> T.List[dict] | None:
        return None


