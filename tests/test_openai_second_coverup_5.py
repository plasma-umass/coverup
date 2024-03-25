# file src/coverup/testrunner.py:86-100
# lines []
# branches ['96->91']

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from coverup.testrunner import BadTestsFinder

def test_missing_branch(tmp_path):
    # Create a directory structure with files that should and should not be found
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_included.py").touch()
    (tests_dir / "included_test.py").touch()
    (tests_dir / "not_a_test.txt").touch()
    (tests_dir / "test_not_included.txt").touch()
    sub_dir = tests_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "test_included.py").touch()
    (sub_dir / "included_test.py").touch()
    (sub_dir / "not_a_test.txt").touch()

    # Mock the DeltaDebugger to avoid side effects
    with pytest.MonkeyPatch.context() as m:
        m.setattr('coverup.testrunner.DeltaDebugger', MagicMock())

        # Instantiate BadTestsFinder
        finder = BadTestsFinder(tests_dir=tests_dir)

        # Assert that only the correct test files are found
        expected_files = {
            tests_dir / "test_included.py",
            tests_dir / "included_test.py",
            sub_dir / "test_included.py",
            sub_dir / "included_test.py",
        }
        assert finder.all_tests == expected_files
