# file src/coverup/utils.py:49-51
# lines [50, 51]
# branches ['50->exit', '50->51']

import pytest
from coverup.utils import format_branches

def test_format_branches_with_exit_branch():
    branches = [(1, 2), (3, 0)]
    expected_output = ['1->2', '3->exit']
    assert list(format_branches(branches)) == expected_output
