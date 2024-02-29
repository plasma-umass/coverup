import pytest
import subprocess
import coverup.testrunner as tr
from pathlib import Path


@pytest.mark.asyncio
async def test_measure_test_coverage_exit_1(tmpdir):
    with pytest.raises(subprocess.CalledProcessError) as einfo:
        await tr.measure_test_coverage(test="import os;\ndef test_foo(): os.exit(1)\n", tests_dir=Path(tmpdir))

def test_measure_suite_coverage_empty_dir(tmpdir):
    coverage = tr.measure_suite_coverage(tests_dir=Path(tmpdir), source_dir=Path('src'))   # shouldn't throw
    assert coverage['summary']['covered_lines'] == 0


def seq2p(tests_dir, seq):
    return tests_dir / f"test_coverup_{seq}.py"


def make_failing_suite(tests_dir: Path, fail_collect: bool):
    """In a suite with 10 tests, test 6 fails; test 3 doesn't fail, but causes 6 to fail."""

    for seq in range(10):
        seq2p(tests_dir, seq).write_text('def test_foo(): pass')

    culprit = seq2p(tests_dir, 3)
    culprit.write_text("import sys\n" + "sys.foobar = True")

    failing = seq2p(tests_dir, 6)
    if fail_collect:
        failing.write_text("import sys\n" + "assert not getattr(sys, 'foobar', False)\n" + "def test_foo(): pass")
    else:
        failing.write_text("import sys\n" + "def test_foo(): assert not getattr(sys, 'foobar', False)")

    return failing, culprit


@pytest.mark.parametrize("fail_collect", [True, False])
def test_measure_suite_coverage_test_fails(tmpdir, fail_collect):

    tests_dir = Path(tmpdir)

    failing, culprit = make_failing_suite(tests_dir, fail_collect)

    with pytest.raises(subprocess.CalledProcessError) as einfo:
        tr.measure_suite_coverage(tests_dir=tests_dir, source_dir=Path('src'))

    assert [failing] == tr.parse_failed_tests(tests_dir, einfo.value)


def test_finds_tests_in_subdir(tmpdir):
    tests_dir = Path(tmpdir)

    (tests_dir / "test_foo.py").write_text("def test(): pass")
    subdir = tests_dir / "subdir"
    subdir.mkdir()
    test_in_subdir = (subdir / "test_bar.py")
    test_in_subdir.write_text("def test(): pass")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    assert test_in_subdir in btf.all_tests


@pytest.mark.parametrize("fail_collect", [True, False])
def test_run_tests(tmpdir, fail_collect):
    tests_dir = Path(tmpdir)

    failing, _ = make_failing_suite(tests_dir, fail_collect)

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    assert {failing} == btf.run_tests()


def test_run_tests_multiple_failures(tmpdir):
    tests_dir = Path(tmpdir)

    for seq in range(10):
        seq2p(tests_dir, seq).write_text("import sys\n" + "def test_foo(): assert not getattr(sys, 'foobar', False)")

    culprit = seq2p(tests_dir, 3)
    culprit.write_text("import sys\n" + "sys.foobar = True")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    failing = btf.run_tests()

    for seq in range(10):
        if seq != 3: assert seq2p(tests_dir, seq) in failing

def test_run_tests_no_tests(tmpdir):
    tests_dir = Path(tmpdir)

    (tests_dir / "test_foo.py").write_text("# no tests here")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    failed = btf.run_tests()
    assert failed == set()


@pytest.mark.parametrize("fail_collect", [True, False])
def test_find_culprit(tmpdir, fail_collect):
    tests_dir = Path(tmpdir)

    failing, culprit = make_failing_suite(tests_dir, fail_collect)

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)

    assert not btf.run_tests({culprit})
    assert {culprit} == btf.find_culprit(failing)


def test_find_culprit_multiple_failures(tmpdir):
    tests_dir = Path(tmpdir)

    for seq in range(10):
        seq2p(tests_dir, seq).write_text("import sys\n" + "def test_foo(): assert not getattr(sys, 'foobar', False)")

    culprit = seq2p(tests_dir, 3)
    culprit.write_text("import sys\n" + "sys.foobar = True")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    assert {culprit} == btf.find_culprit(seq2p(tests_dir, 6))


@pytest.mark.skip(reason="no good handling for this yet, it takes a long time")
def test_find_culprit_hanging_collect(tmpdir):
    tests_dir = Path(tmpdir)

    all_tests = {seq2p(tests_dir, seq) for seq in range(10)}
    for t in all_tests:
        t.write_text('def test_foo(): pass')

    culprit = seq2p(tests_dir, 3)
    culprit.write_text("""\
import pytest

def test_foo(): pass

pytest.main(["--verbose"])
pytest.main(["--verbose"])
pytest.main(["--verbose"])
""")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    btf.run_tests()
