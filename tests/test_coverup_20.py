# file src/coverup/delta.py:5-32
# lines [16]
# branches ['13->16']

import pytest
from pathlib import Path
from coverup.delta import _compact

def test_compact_with_path_numbers():
    test_set = {Path("test_1.py"), Path("test_2.py"), Path("test_3.py")}
    result = _compact(test_set)
    assert result == "{1-3}"

def test_compact_with_path_names():
    test_set = {Path("test_a.py"), Path("test_b.py")}
    result = _compact(test_set)
    assert result == "{test_a.py, test_b.py}"

def test_compact_with_mixed_paths():
    test_set = {Path("test_1.py"), Path("test_2.py"), Path("test_a.py")}
    result = _compact(test_set)
    assert result == "{1-2, test_a.py}"

def test_compact_with_non_path_objects():
    test_set = {Path("test_1.py"), "some_test"}
    result = _compact(test_set)
    assert result == "{1, some_test}"

def test_compact_with_path_missing_number():
    test_set = {Path("test_a.py"), Path("test.py")}
    result = _compact(test_set)
    assert result == "{test.py, test_a.py}"

def test_compact_with_empty_set():
    test_set = set()
    result = _compact(test_set)
    assert result == "{}"

@pytest.fixture
def cleanup_test_files():
    # Setup: create test files
    test_files = [Path(f"test_{i}.py") for i in range(1, 4)]
    test_files += [Path("test_a.py"), Path("test.py")]
    for test_file in test_files:
        test_file.touch()

    yield test_files

    # Teardown: remove test files
    for test_file in test_files:
        test_file.unlink()

def test_compact_with_real_files(cleanup_test_files):
    test_set = set(cleanup_test_files)
    result = _compact(test_set)
    expected_names = {"test.py", "test_a.py"}
    expected_result = "{" + ", ".join(["1-3"] + sorted(expected_names)) + "}"
    assert result == expected_result
