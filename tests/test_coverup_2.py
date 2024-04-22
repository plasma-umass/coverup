# file src/coverup/coverup.py:345-354
# lines [347, 349, 350, 351, 352, 353, 354]
# branches []

import pytest
from coverup.coverup import State

PROGRESS_COUNTERS = ['counter1', 'counter2', 'counter3']

@pytest.fixture
def initial_coverage():
    return {'file1.py': {(1, 2), (3, 4)}, 'file2.py': {(5, 6)}}

@pytest.fixture
def state_cleanup(monkeypatch):
    # Cleanup code to reset any global state if necessary
    monkeypatch.setattr('coverup.coverup.PROGRESS_COUNTERS', PROGRESS_COUNTERS)

def test_state_initialization(initial_coverage, state_cleanup):
    state = State(initial_coverage)
    assert state.done == {}
    assert state.coverage == initial_coverage
    assert state.usage == {'prompt_tokens': 0, 'completion_tokens': 0}
    assert state.counters == {k: 0 for k in PROGRESS_COUNTERS}
    assert state.final_coverage is None
    assert state.bar is None
