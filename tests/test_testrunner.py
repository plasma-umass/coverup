import pytest
import subprocess
import coverup.testrunner as tr
from pathlib import Path
import tempfile


@pytest.mark.asyncio
async def test_measure_test_coverage_exit_1(tmpdir):
    with pytest.raises(subprocess.CalledProcessError) as einfo:
        await tr.measure_test_coverage(test="import os;\ndef test_foo(): os.exit(1)\n", tests_dir=Path(tmpdir))


@pytest.mark.parametrize("absolute", [True, False])
def test_measure_suite_coverage_empty_dir(absolute):
    with tempfile.TemporaryDirectory(dir=Path('.')) as tests_dir:
        tests_dir = Path(tests_dir)
        if absolute:
            tests_dir = tests_dir.resolve()

        coverage = tr.measure_suite_coverage(tests_dir=tests_dir, source_dir=tests_dir, trace=print)   # shouldn't throw
        assert {} == coverage['files']


def seq2p(tests_dir, seq):
    return tests_dir / f"test_coverup_{seq}.py"


N_TESTS=10
def make_failing_suite(tests_dir: Path, fail_collect: bool):
    """In a suite with 10 tests, test 6 fails; test 3 doesn't fail, but causes 6 to fail."""

    for seq in range(N_TESTS):
        seq2p(tests_dir, seq).write_text('def test_foo(): pass')

    culprit = seq2p(tests_dir, 3)
    culprit.write_text("import sys\n" + "sys.foobar = True\n" + "def test_foo(): pass")

    failing = seq2p(tests_dir, 6)
    if fail_collect:
        failing.write_text("import sys\n" + "assert not getattr(sys, 'foobar', False)\n" + "def test_foo(): pass")
    else:
        failing.write_text("import sys\n" + "def test_foo(): assert not getattr(sys, 'foobar', False)")

    return failing, culprit


@pytest.mark.parametrize("fail_collect", [True, False])
@pytest.mark.parametrize("absolute", [True, False])
def test_measure_suite_coverage_test_fails(absolute, fail_collect):
    with tempfile.TemporaryDirectory(dir=Path('.')) as tests_dir:
        tests_dir = Path(tests_dir)
        if absolute:
            tests_dir = tests_dir.resolve()

        failing, culprit = make_failing_suite(tests_dir, fail_collect)

        with pytest.raises(subprocess.CalledProcessError) as einfo:
            tr.measure_suite_coverage(tests_dir=tests_dir, source_dir=Path('src'), isolate_tests=False)


@pytest.mark.parametrize("fail_collect", [True, False])
@pytest.mark.parametrize("absolute", [True, False])
def test_measure_suite_coverage_isolated(absolute, fail_collect):
    with tempfile.TemporaryDirectory(dir=Path('.')) as tests_dir:
        tests_dir = Path(tests_dir)
        if absolute:
            tests_dir = tests_dir.resolve()

        failing, culprit = make_failing_suite(tests_dir, fail_collect)

        tr.measure_suite_coverage(tests_dir=tests_dir, source_dir=Path('src'), isolate_tests=True)

        # FIXME check coverage
