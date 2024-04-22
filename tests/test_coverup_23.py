# file src/coverup/prompt.py:155-190
# lines [156, 157, 158, 160, 161, 162, 163, 164, 165, 171]
# branches []

import pytest
from coverup.prompt import ClaudePrompter, _message
from unittest.mock import MagicMock

# Mocking the necessary parts to test ClaudePrompter.initial_prompt
class MockSegment:
    def __init__(self, path, filename, lines_branches_missing_do):
        self.path = path
        self.filename = filename
        self.lines_branches_missing_do = lines_branches_missing_do

    def get_excerpt(self):
        return "Code excerpt"

def get_module_name_mock(path, source_dir):
    return "mock_module"

@pytest.fixture
def mock_segment():
    return MockSegment(path="mock_path", filename="mock_file.py", lines_branches_missing_do=lambda: "lines 156-171")

@pytest.fixture
def mock_args():
    return MagicMock(source_dir="mock_source_dir")

@pytest.fixture
def claude_prompter(mock_args, mock_segment):
    prompter = ClaudePrompter(args=mock_args, segment=mock_segment)
    return prompter

def test_initial_prompt_executes_missing_lines_branches(monkeypatch, claude_prompter):
    # Patch the get_module_name function to return a mock module name
    monkeypatch.setattr('coverup.prompt.get_module_name', get_module_name_mock)

    expected_messages = [
        _message("You are an expert Python test-driven developer who creates pytest test functions that achieve high coverage.",
                 role="system"),
        _message(f"""
<file path="{claude_prompter.segment.filename}" module_name="mock_module">
Code excerpt
</file>

<instructions>

The code above does not achieve full coverage:
when tested, lines 156-171 not execute.

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

    result = claude_prompter.initial_prompt()
    assert result == expected_messages
