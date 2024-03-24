# file src/coverup/delta.py:5-32
# lines [16]
# branches ['13->16']

import pytest
from pathlib import Path
from src.coverup.delta import _compact

def test_compact_with_path_without_number(tmp_path):
    # Create a temporary file without a number in its name
    temp_file = tmp_path / "test_file.py"
    temp_file.touch()

    # Call the _compact function with a set containing the Path
    result = _compact({temp_file})

    # Assert that the result contains the name of the temporary file
    assert temp_file.name in result

    # Clean up the temporary file
    temp_file.unlink()
