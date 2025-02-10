import pytest
from pathlib import Path
from unittest.mock import MagicMock


class MockArgs:
    def __init__(self, tests_dir, prefix):
        self.tests_dir = Path(tests_dir)
        self.prefix = prefix


def test_file_path():
    from coverup.coverup import test_file_path
    mock_args = MockArgs('/tmp', 'mock_prefix')

    test_seq = 1
    actual_path = test_file_path(mock_args, test_seq)
    
    assert actual_path == Path('/tmp') / f"test_mock_prefix_{test_seq}.py"
