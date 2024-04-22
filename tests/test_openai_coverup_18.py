# file src/coverup/coverup.py:368-387
# lines [371, 372, 373, 374, 375, 376, 378, 379, 380, 381, 382, 383, 384, 386, 387]
# branches ['379->380', '379->381', '382->383', '382->384']

import json
import pytest
from pathlib import Path
from src.coverup.coverup import State

@pytest.fixture
def mock_checkpoint_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "checkpoint.json"
    return p

def test_load_checkpoint_success(mock_checkpoint_file):
    # Prepare the checkpoint data
    checkpoint_data = {
        'version': 1,
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
        'coverage': {},
        'done': {}
    }
    # Write the checkpoint data to the mock file
    mock_checkpoint_file.write_text(json.dumps(checkpoint_data))
    
    # Call the method under test
    state = State.load_checkpoint(mock_checkpoint_file)
    
    # Assertions to verify postconditions
    assert state is not None
    assert state.coverage == checkpoint_data['coverage']
    assert state.usage == checkpoint_data['usage']
    assert state.done == {filename: set(tuple(d) for d in done_list) for filename, done_list in checkpoint_data['done'].items()}

def test_load_checkpoint_file_not_found(mock_checkpoint_file):
    # Ensure the file does not exist
    if mock_checkpoint_file.exists():
        mock_checkpoint_file.unlink()
    
    # Call the method under test
    state = State.load_checkpoint(mock_checkpoint_file)
    
    # Assertions to verify postconditions
    assert state is None
