# file src/coverup/coverup.py:27-109
# lines [28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 43, 44, 46, 47, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 80, 81, 83, 84, 85, 87, 88, 90, 91, 93, 94, 95, 97, 98, 99, 101, 102, 103, 104, 106, 107, 109]
# branches ['37->37', '37->38', '103->103', '103->104']

import pytest
from pathlib import Path
from coverup.coverup import parse_args

def test_parse_args_full_coverage(tmp_path):
    # Create a dummy source file and test directory
    source_file = tmp_path / "source.py"
    source_file.touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    # Create a dummy checkpoint file
    checkpoint_file = tmp_path / "checkpoint"
    checkpoint_file.touch()

    # Create a dummy log file
    log_file = tmp_path / "log"

    # Create a dummy requirements file
    requirements_file = tmp_path / "requirements.txt"

    # Arguments to cover all lines
    args = [
        str(source_file),
        '--tests-dir', str(tests_dir),
        '--source-dir', str(source_dir),
        '--checkpoint', str(checkpoint_file),
        '--no-checkpoint',
        '--model', 'gpt-3',
        '--model-temperature', '0.7',
        '--line-limit', '60',
        '--rate-limit', '1000',
        '--max-attempts', '5',
        '--max-backoff', '30',
        '--dry-run',
        '--show-details',
        '--log-file', str(log_file),
        '--pytest-args', 'some args',
        '--install-missing-modules',
        '--write-requirements-to', str(requirements_file),
        '--failing-test-action', 'find-culprit',
        '--only-disable-interfering-tests',
        '--debug',
        '--max-concurrency', '10'
    ]

    # Parse the arguments
    parsed_args = parse_args(args)

    # Assertions to verify postconditions
    assert parsed_args.source_files == [source_file]
    assert parsed_args.tests_dir == tests_dir
    assert parsed_args.source_dir == source_dir
    assert parsed_args.checkpoint is None  # Because --no-checkpoint was used
    assert parsed_args.model == 'gpt-3'
    assert parsed_args.model_temperature == '0.7'
    assert parsed_args.line_limit == 60
    assert parsed_args.rate_limit == 1000
    assert parsed_args.max_attempts == 5
    assert parsed_args.max_backoff == 30
    assert parsed_args.dry_run is True
    assert parsed_args.show_details is True
    assert parsed_args.log_file == str(log_file)
    assert parsed_args.pytest_args == 'some args'
    assert parsed_args.install_missing_modules is True
    assert parsed_args.write_requirements_to == requirements_file
    assert parsed_args.failing_test_action == 'find-culprit'
    assert parsed_args.only_disable_interfering_tests is True
    assert parsed_args.debug is True
    assert parsed_args.max_concurrency == 10
