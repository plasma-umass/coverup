# file src/coverup/prompt.py:151-152
# lines [152]
# branches []

import pytest
from coverup.prompt import ClaudePrompter, Prompter

# Assuming that Prompter is a class that can be initialized without any special requirements
# and that it does not have any side effects that need to be cleaned up.

class TestClaudePrompter:
    def test_claude_prompter_initialization(self, monkeypatch):
        # Mock the __init__ method of the Prompter class to track if it's called
        init_called = False
        def mock_init(self, *args, **kwargs):
            nonlocal init_called
            init_called = True

        monkeypatch.setattr(Prompter, "__init__", mock_init)

        # Create an instance of ClaudePrompter, which should call the mocked __init__ method
        claude_prompter = ClaudePrompter()

        # Assert that the Prompter.__init__ method was called
        assert init_called, "Prompter.__init__ was not called during ClaudePrompter initialization"
