import pytest
import subprocess
import coverup.testrunner as tr
from pathlib import Path


def test_measure_suite_coverage_empty_dir(tmpdir):
    coverage = tr.measure_suite_coverage(tests_dir=Path(tmpdir), source_dir=Path('src'))   # shouldn't throw
    assert coverage['summary']['covered_lines'] == 0


def test_finds_tests_in_subdir(tmpdir):
    tests_dir = Path(tmpdir)

    (tests_dir / "test_foo.py").write_text("def test(): pass")
    subdir = tests_dir / "subdir"
    subdir.mkdir()
    test_in_subdir = (subdir / "test_bar.py")
    test_in_subdir.write_text("def test(): pass")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    assert test_in_subdir in btf.all_tests


@pytest.mark.parametrize("fail_load", [True, False])
def test_run_tests(tmpdir, fail_load):
    tests_dir = Path(tmpdir)

    def seq2p(seq):
        return tests_dir / f"test_coverup_{seq}.py"

    all_tests = {seq2p(seq) for seq in range(10)}
    for t in all_tests:
        t.write_text('def test_foo(): pass')

    culprit = seq2p(3)
    culprit.write_text("import sys\n" + "sys.hexversion=0")

    failing = seq2p(6)
    if fail_load:
        failing.write_text("import sys\n" + "assert sys.hexversion != 0\n" + "def test_foo(): pass")
    else:
        failing.write_text("import sys\n" + "def test_foo(): assert sys.hexversion != 0")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    failed = btf.run_tests()

    assert failed == failing


def test_run_tests_no_tests(tmpdir):
    tests_dir = Path(tmpdir)

    (tests_dir / "test_foo.py").write_text("# no tests here")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)
    failed = btf.run_tests()
    assert failed is None


@pytest.mark.parametrize("fail_load", [True, False])
def test_find_culprit(tmpdir, fail_load):
    tests_dir = Path(tmpdir)

    def seq2p(seq):
        return tests_dir / f"test_coverup_{seq}.py"

    all_tests = {seq2p(seq) for seq in range(10)}
    for t in all_tests:
        t.write_text('def test_foo(): pass')

    culprit = seq2p(3)
    culprit.write_text("import sys\n" + "sys.hexversion=0")

    failing = seq2p(6)
    if fail_load:
        failing.write_text("import sys\n" + "assert sys.hexversion != 0\n" + "def test_foo(): pass")
    else:
        failing.write_text("import sys\n" + "def test_foo(): assert sys.hexversion != 0")

    btf = tr.BadTestsFinder(tests_dir=tests_dir, trace=print)

    assert not btf.test({failing})

    assert {seq2p(3)} == btf.find_culprit(failing)
