# file src/coverup/coverup.py:363-365
# lines [365]
# branches []

import pytest
from src.coverup.coverup import State
from collections import namedtuple

# Assuming CodeSegment is a namedtuple or similar simple class
# If it's not, you would need to adjust the definition accordingly
CodeSegment = namedtuple('CodeSegment', ['filename', 'begin', 'end'])

@pytest.fixture
def state():
    initial_coverage = {}  # Assuming initial_coverage is a dictionary
    s = State(initial_coverage)
    s.done = {}
    yield s
    # No cleanup needed as state is re-created for each test

def test_state_is_done_executes_line_365(state):
    # Setup
    filename = 'test_file.py'
    begin = 10
    end = 20
    seg = CodeSegment(filename, begin, end)
    state.done[filename] = {(begin, end)}

    # Exercise & Verify
    assert state.is_done(seg) == True

    # Cleanup is handled by the fixture
