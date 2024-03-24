# file src/coverup/coverup.py:311-320
# lines [313, 315, 316, 317, 318, 319, 320]
# branches []

import pytest
from unittest.mock import MagicMock
from src.coverup.coverup import State

@pytest.fixture
def initial_coverage():
    return {'file1.py': {(1, 2), (3, 4)}}

def test_state_initialization(initial_coverage, monkeypatch):
    mock_defaultdict = MagicMock()
    monkeypatch.setattr('collections.defaultdict', mock_defaultdict)
    state = State(initial_coverage)
    mock_defaultdict.assert_called_once_with(set)
    assert state.coverage == initial_coverage
    assert state.usage == {'prompt_tokens': 0, 'completion_tokens': 0}
    assert all(counter == 0 for counter in state.counters.values())
    assert state.final_coverage is None
    assert state.bar is None
