# file src/coverup/prompt.py:193-203
# lines [194, 195]
# branches []

import pytest
from coverup.prompt import ClaudePrompter

@pytest.fixture
def mock_prompter_args(mocker):
    mocker.patch('coverup.prompt.Prompter.__init__', return_value=None)

def test_error_prompt(mock_prompter_args):
    prompter = ClaudePrompter()
    error_message = "Test Error"
    expected_output = [{'content': f"<error>{error_message}</error>\nExecuting the test yields an error, shown above.\n<instructions>\n1. Modify the test to correct it.\n2. Respond with the complete Python code in backticks.\n3. Before answering the question, please think about it step-by-step within <thinking></thinking> tags. Then, provide your final answer within <answer></answer> tags.\n</instructions>\n", 'role': 'user'}]
    
    result = prompter.error_prompt(error_message)
    
    assert result == expected_output, "The error_prompt method should return the correct error message format."
