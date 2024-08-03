# file src/coverup/coverup.py:32-139
# lines [33, 35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 48, 49, 51, 52, 53, 54, 56, 57, 59, 60, 61, 62, 64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80, 81, 83, 84, 85, 87, 88, 90, 91, 93, 94, 95, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 117, 119, 120, 121, 123, 124, 125, 126, 128, 129, 131, 133, 134, 136, 137, 139]
# branches ['42->42', '42->43', '125->125', '125->126', '133->134', '133->136', '136->137', '136->139']

import pytest
from pathlib import Path
from coverup.coverup import parse_args

def test_parse_args_with_source_files(tmp_path):
    source_file = tmp_path / "source.py"
    source_file.touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "m.py").touch()
    args = parse_args([str(source_file), '--tests-dir', str(tests_dir), '--source-dir', str(source_dir), '--model', 'gpt-4'])
    assert args.source_files[0] == source_file.resolve()
