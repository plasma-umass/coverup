# file src/coverup/coverup.py:203-222
# lines [204, 206, 207, 208, 209, 211, 213, 214, 215, 216, 217, 219, 220, 222]
# branches ['213->214', '213->222', '214->215', '214->219', '215->213', '215->216', '216->215', '216->217', '219->213', '219->220']

import ast
import pytest
from coverup.coverup import find_imports

@pytest.fixture
def python_code_with_syntax_error():
    return "invalid python code"

@pytest.fixture
def python_code_with_imports():
    return """
import os
import sys
from pathlib import Path
import re
"""

def test_find_imports_with_syntax_error(python_code_with_syntax_error):
    result = find_imports(python_code_with_syntax_error)
    assert result == []

def test_find_imports_with_valid_code(python_code_with_imports):
    result = find_imports(python_code_with_imports)
    assert set(result) == {"os", "sys", "pathlib", "re"}
