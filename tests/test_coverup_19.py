# file src/coverup/testrunner.py:72-87
# lines []
# branches ['82->77']

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from coverup.testrunner import BadTestsFinder

# Assuming the existence of a DeltaDebugger class in coverup.testrunner
# If not, this would need to be adjusted accordingly.

def test_find_tests_includes_correct_files(tmp_path, monkeypatch):
    # Create a directory structure with files to be discovered
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    included_file = tests_dir / "test_included.py"
    included_file.touch()
    excluded_file = tests_dir / "not_a_test.txt"
    excluded_file.touch()
    nested_dir = tests_dir / "nested"
    nested_dir.mkdir()
    nested_test_file = nested_dir / "nested_test.py"
    nested_test_file.touch()

    # Mock the progress callable to avoid side effects
    mock_progress = MagicMock()

    # Instantiate the BadTestsFinder
    finder = BadTestsFinder(tests_dir=tests_dir, progress=mock_progress)

    # Check that the correct files are included
    assert included_file in finder.all_tests
    assert nested_test_file in finder.all_tests
    assert excluded_file not in finder.all_tests

    # Check that the progress callable was not called, indicating no side effects
    mock_progress.assert_not_called()

def test_find_tests_excludes_incorrect_files(tmp_path, monkeypatch):
    # Create a directory structure with files to be discovered
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    excluded_file_1 = tests_dir / "no_test_prefix_suffix.py"
    excluded_file_1.touch()
    excluded_file_2 = tests_dir / "test_.txt"
    excluded_file_2.touch()
    excluded_file_3 = tests_dir / "test_no_py_suffix.txt"
    excluded_file_3.touch()

    # Mock the progress callable to avoid side effects
    mock_progress = MagicMock()

    # Instantiate the BadTestsFinder
    finder = BadTestsFinder(tests_dir=tests_dir, progress=mock_progress)

    # Check that the incorrect files are excluded
    assert excluded_file_1 not in finder.all_tests
    assert excluded_file_2 not in finder.all_tests
    assert excluded_file_3 not in finder.all_tests

    # Check that the progress callable was not called, indicating no side effects
    mock_progress.assert_not_called()
