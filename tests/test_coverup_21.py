# file src/coverup/testrunner.py:40-63
# lines [62, 63]
# branches ['53->53']

import json
import os
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Test to cover lines 62-63
def test_measure_suite_coverage_file_not_found(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_example.py"
    test_file.write_text("def test_pass(): assert True")

    def mock_run(*args, **kwargs):
        # Simulate the behavior of subprocess.run
        class MockCompletedProcess:
            def __init__(self):
                self.returncode = 0
                self.stdout = b""

            def check_returncode(self):
                pass

        return MockCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)

    def mock_unlink(path):
        # Simulate the behavior of os.unlink when file is not found
        raise FileNotFoundError

    monkeypatch.setattr(os, "unlink", mock_unlink)

    def mock_json_load(file):
        # Simulate json.load returning an empty dictionary
        return {}

    monkeypatch.setattr(json, "load", mock_json_load)

    from coverup.testrunner import measure_suite_coverage

    # Run the test suite coverage measurement
    coverage_data = measure_suite_coverage(tests_dir=tests_dir, source_dir=source_dir)
    assert isinstance(coverage_data, dict)  # Postcondition: coverage_data should be a dict

# Test to cover branch 53->53
def test_measure_suite_coverage_non_zero_returncode(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_example.py"
    test_file.write_text("def test_fail(): assert False")

    def mock_run(*args, **kwargs):
        # Simulate the behavior of subprocess.run with non-zero returncode
        class MockCompletedProcess:
            def __init__(self):
                self.returncode = 1  # Non-zero returncode
                self.stdout = b""

            def check_returncode(self):
                raise subprocess.CalledProcessError(self.returncode, args)

        return MockCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)

    from coverup.testrunner import measure_suite_coverage

    # Run the test suite coverage measurement and expect a CalledProcessError
    with pytest.raises(subprocess.CalledProcessError):
        measure_suite_coverage(tests_dir=tests_dir, source_dir=source_dir)

    # No postcondition to assert since the test is expected to raise an exception
