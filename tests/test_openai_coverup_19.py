# file src/coverup/coverup.py:390-405
# lines [392, 393, 394, 395, 396, 397, 401, 402, 404, 405]
# branches ['401->402', '401->404']

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from coverup.coverup import State

@pytest.fixture
def mock_state(tmp_path):
    initial_coverage = {'module1': 70.0, 'module2': 0.0}
    state = State(initial_coverage)
    state.done = {'module1': {'test1', 'test2'}, 'module2': set()}
    state.usage = {'module1': 2, 'module2': 0}
    state.counters = {'counter1': 10}
    state.coverage = {'module1': 75.0}
    state.final_coverage = {'module1': 80.0}
    return state

@pytest.fixture
def ckpt_file(tmp_path):
    return tmp_path / "checkpoint.json"

def test_save_checkpoint_with_final_coverage(mock_state, ckpt_file):
    mock_state.save_checkpoint(ckpt_file)
    assert ckpt_file.exists()
    with ckpt_file.open() as f:
        ckpt_data = json.load(f)
    assert ckpt_data['version'] == 1
    # Convert lists to sets for comparison to ignore order
    assert set(ckpt_data['done']['module1']) == {'test1', 'test2'}
    assert ckpt_data['usage'] == {'module1': 2, 'module2': 0}
    assert ckpt_data['counters'] == {'counter1': 10}
    assert ckpt_data['coverage'] == {'module1': 75.0}
    assert ckpt_data['final_coverage'] == {'module1': 80.0}
