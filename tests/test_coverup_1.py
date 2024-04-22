# file src/coverup/prompt.py:40-44
# lines [41, 42, 43]
# branches []

import pytest
from coverup.prompt import _message

def test_message_default_role():
    content = "Test content"
    expected_result = {'role': 'user', 'content': content}
    result = _message(content)
    assert result == expected_result

def test_message_custom_role():
    content = "Test content"
    role = "admin"
    expected_result = {'role': role, 'content': content}
    result = _message(content, role=role)
    assert result == expected_result
