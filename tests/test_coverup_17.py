# file src/coverup/coverup.py:424-439
# lines [426, 427, 428, 429, 430, 431, 435, 436, 438, 439]
# branches ['435->436', '435->438']

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Assuming the State class is defined in coverup/coverup.py
from coverup.coverup import State

# Test to cover lines 426-431 and 438-439
def test_save_checkpoint_basic(tmp_path):
    initial_coverage = MagicMock()
    state = State(initial_coverage)
    state.done = {'key': [1, 2, 3]}
    state.usage = 10
    state.counters = {'counter': 1}
    state.coverage = {'cov': 'data'}
    state.final_coverage = None

    ckpt_file = tmp_path / "checkpoint.json"
    state.save_checkpoint(ckpt_file)

    with ckpt_file.open("r") as f:
        data = json.load(f)

    assert data['version'] == 1
    assert data['done'] == {'key': [1, 2, 3]}
    assert data['usage'] == 10
    assert data['counters'] == {'counter': 1}
    assert data['coverage'] == {'cov': 'data'}
    assert 'final_coverage' not in data

# Test to cover lines 435-436
def test_save_checkpoint_with_final_coverage(tmp_path):
    initial_coverage = MagicMock()
    state = State(initial_coverage)
    state.done = {'key': [1, 2, 3]}
    state.usage = 10
    state.counters = {'counter': 1}
    state.coverage = {'cov': 'data'}
    state.final_coverage = {'final': 'coverage_data'}

    ckpt_file = tmp_path / "checkpoint_final.json"
    state.save_checkpoint(ckpt_file)

    with ckpt_file.open("r") as f:
        data = json.load(f)

    assert data['version'] == 1
    assert data['done'] == {'key': [1, 2, 3]}
    assert data['usage'] == 10
    assert data['counters'] == {'counter': 1}
    assert data['coverage'] == {'cov': 'data'}
    assert data['final_coverage'] == {'final': 'coverage_data'}
