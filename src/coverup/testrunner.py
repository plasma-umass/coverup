from pathlib import Path
import tempfile
import subprocess
import pytest
import typing as T
import sys
import json
import re
from .delta import DeltaDebugger


def measure_coverage(*, test: str, tests_dir: Path, pytest_args='', log_write=None):
    """Runs a given test and returns the coverage obtained."""
    with tempfile.NamedTemporaryFile(prefix="tmp_test_", suffix='.py',
                                     dir=str(tests_dir), mode="w") as t:
        t.write(test)
        t.flush()

        with tempfile.NamedTemporaryFile() as j:
            # -qq to cut down on tokens
            p = subprocess.run((f"{sys.executable} -m slipcover --branch --json --out {j.name} " +
                                f"-m pytest {pytest_args} -qq --disable-warnings {t.name}").split(),
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            if log_write:
                log_write(str(p.stdout, 'UTF-8'))

            cov = json.load(j)

    return cov["files"]


def measure_suite_coverage(*, tests_dir: Path, source_dir: Path, pytest_args=''):
    """Runs an entire test suite and returns the coverage obtained."""
    with tempfile.NamedTemporaryFile() as j:
        p = subprocess.run((f"{sys.executable} -m slipcover --source {source_dir} --branch --json --out {j.name} " +
                            f"-m pytest {pytest_args} -qq -x --disable-warnings {tests_dir}").split(),
                           check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if p.returncode not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            p.check_returncode()

#        log_write(seg, str(p.stdout, 'UTF-8'))
        return json.load(j)



class ParseError(Exception):
    pass

class EarlierFailureException(Exception):
    def __init__(self, failed, test_set):
        super().__init__(f'Earlier test failed: {failed}')
        self.failed = failed
        self.test_set = test_set


def parse_failed_test(tests_dir: Path, p: (subprocess.CompletedProcess, subprocess.CalledProcessError)) -> Path:
    output = str(p.stdout, 'UTF-8')
    if (m := re.search("^===+ short test summary info ===+\n" +\
                       "^(?:ERROR|FAILED) ([^\\s:]+)", output, re.MULTILINE)):

        failed = Path(m.group(1))
        if tests_dir.is_absolute():
            # pytest sometimes makes absolute paths into relative ones by adding ../../.. to root...
            failed = failed.resolve()

        return failed

    raise ParseError(f"Unable to parse failing test out of pytest output. RC={p.returncode}; output:\n{output}")


class BadTestsFinder(DeltaDebugger):
    """Finds tests that cause other tests to fail."""
    def __init__(self, *, tests_dir: Path, pytest_args: str = '', trace = None):
        super(BadTestsFinder, self).__init__(trace=trace)
        self.tests_dir = tests_dir

        def find_tests(p):
            for f in p.iterdir():
                if f.is_dir():
                    yield from find_tests(f)
                else:
                    # TODO filter according to pytest customization rules
                    if f.is_file() and (f.stem.startswith('test_') or f.stem.endswith('_test')) and f.suffix == '.py':
                        yield f

        self.all_tests = set(find_tests(self.tests_dir))
        self.pytest_args = pytest_args


    def run_tests(self, tests_to_run: set = None) -> Path:
        """Runs the tests, by default all, returning the first one that fails, or None.
           Throws RuntimeError if unable to parse pytest's output.
        """

        test_set = tests_to_run if tests_to_run else self.all_tests

        def link_tree(src, dst):
            dst.mkdir(parents=True, exist_ok=True)

            for src_file in src.iterdir():
                dst_file = dst / src_file.name

                if src_file.is_dir():
                    link_tree(src_file, dst_file)
                else:
                    if src_file not in self.all_tests or src_file in test_set:
                        dst_file.hardlink_to(src_file)

        assert self.tests_dir.parent != self.tests_dir # we need a parent directory

        if self.trace: self.trace(f"running {len(test_set)} test(s).")
        with tempfile.TemporaryDirectory(dir=self.tests_dir.parent) as tmpdir:
            tmpdir = Path(tmpdir)
            link_tree(self.tests_dir, tmpdir)

            p = subprocess.run((f"{sys.executable} -m pytest {self.pytest_args} -x -qq --disable-warnings " +\
                                f"--rootdir {tmpdir} {tmpdir}").split(),
                               check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2*60*60)

            if p.returncode in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
                if self.trace: self.trace(f"tests passed")
                return None

            first_failing = parse_failed_test(tmpdir, p)

            # bring it back to its normal path
            first_failing = self.tests_dir / first_failing.relative_to(tmpdir)

            if self.trace: self.trace(f"tests rc={p.returncode} first_failing={first_failing}")

            return first_failing


    def test(self, test_set: set, **kwargs) -> bool:
        if not (first_failing := self.run_tests(test_set)):
            return False

        # If given, check that it's target_test that failed: a different test may have failed.
        # If a test that comes after target_test fails, then this is really a success; but if it's a test
        # that comes before target_test, this is "inconsistent" (in delta debugging terms). We side
        # step the issue by re-starting the process looking for what made first_failing fail.
        if target_test := kwargs.get('target_test'):
            # FIXME compare in pytest order (with subdirectories last)
            if first_failing < target_test:
                raise EarlierFailureException(first_failing, test_set)

            if first_failing != target_test:
                return False

        return True # "interesting"/"reproduced"


    def find_culprit(self, failing_test: Path) -> T.Set[Path]:
        """Returns the set of tests causing 'failing_test' to fail."""
        assert failing_test in self.all_tests

        # TODO first test collection using --collect-only, with short timeout
        # TODO reduce timeout for actually running tests

# we unfortunately can't do this... the code that causes test(s) to fail may execute during pytest collection.
#        sorted_tests = sorted(self.all_tests)
#        test_set = set(sorted_tests[:sorted_tests.index(failing_test)])
#        assert self.test(test_set), "Test set doesn't fail!"

        return self.debug(changes=self.all_tests - {failing_test}, rest={failing_test}, target_test=failing_test)
