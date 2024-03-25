# file src/coverup/coverup.py:27-109
# lines []
# branches ['37->37', '103->103']

import pytest
from pathlib import Path
from unittest.mock import patch
from coverup.coverup import parse_args

def test_parse_args_with_invalid_dir_and_negative_int(tmp_path):
    invalid_dir = tmp_path / "invalid_dir"
    invalid_dir_file = tmp_path / "invalid_file.txt"
    invalid_dir_file.touch()

    with pytest.raises(SystemExit):
        parse_args(['--tests-dir', str(invalid_dir), '--source-dir', str(tmp_path)])

    with pytest.raises(SystemExit):
        parse_args(['--tests-dir', str(tmp_path), '--source-dir', str(invalid_dir_file)])

    with pytest.raises(SystemExit):
        parse_args(['--max-concurrency', '-1'])

    # Cleanup is not necessary as tmp_path is a fixture that automatically handles isolation
