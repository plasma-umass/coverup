# file src/coverup/utils.py:49-51
# lines [50, 51]
# branches ['50->exit', '50->51']

import pytest
from coverup.utils import format_branches

@pytest.fixture
def mock_branches():
    return [(1, 2), (3, 0), (5, 6)]

def test_format_branches(mock_branches):
    formatted_branches = list(format_branches(mock_branches))
    assert formatted_branches == ['1->2', '3->exit', '5->6']
