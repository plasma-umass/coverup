# file src/coverup/prompt.py:206-215
# lines [207, 208]
# branches []

import pytest
from coverup.prompt import ClaudePrompter

# Assuming the existence of a function `lines_branches_do` that is used in the ClaudePrompter class.
# Since it's not provided, we'll create a mock for it.
def lines_branches_do(missing_lines, _, missing_branches):
    return f"lines {missing_lines} and branches {missing_branches}"

# Mocking the function in the ClaudePrompter class
@pytest.fixture
def mock_lines_branches_do(mocker):
    mocker.patch('coverup.prompt.lines_branches_do', side_effect=lines_branches_do)

# Test function to cover lines 207-208
def test_missing_coverage_prompt(mock_lines_branches_do, mocker):
    # Mocking the Prompter __init__ to not require arguments
    mocker.patch('coverup.prompt.Prompter.__init__', return_value=None)
    
    prompter = ClaudePrompter()
    missing_lines = {1, 2}
    missing_branches = {3, 4}
    result = prompter.missing_coverage_prompt(missing_lines, missing_branches)
    expected_message = {
        'content': "This test still lacks coverage: lines {1, 2} and branches {3, 4} not execute.\n<instructions>\n1. Modify it to execute those lines.\n2. Respond with the complete Python code in backticks.\n3. Before responding, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.\n</instructions>\n",
        'role': 'user'
    }
    assert len(result) == 1
    assert result[0] == expected_message
