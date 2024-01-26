import abc
import typing as T
from pathlib import Path

def compact(test_set):
    """Generates a (more) compact test name representation for debugging messages."""
    import re

    numbers = set()
    names = set()
    for t in test_set:
        if isinstance(t, Path):
            if (m := re.search("_(\\d+).py$", t.name)):
                numbers.add(int(m.group(1)))
            else:
                names.add(t.name)
        else:
            names.add(str(t))

    def get_ranges():
        it = iter(sorted(numbers))

        a = next(it, None)
        while a is not None:
            b = a
            while (n := next(it, None)) == b+1:
                b = n

            yield str(a) if a == b else f"{a}-{b}"
            a = n

    return ", ".join(list(get_ranges()) + sorted(names))


class DeltaDebugger(abc.ABC):
    """Implements delta debugging ("dd" algorithm), as per 'Yesterday, my program worked. Today, it does not. Why?' "
       https://dl.acm.org/doi/10.1145/318774.318946
    """
    def __init__(self, *, trace=None):
        self.trace = trace


    @abc.abstractmethod
    def test(self, test_set: set, **kwargs) -> bool:
        """Invoked to test whether the case is 'interesting'"""


    def debug(self, changes: set, rest: set = set(), **kwargs) -> set:
        if self.trace: self.trace(f"debug(changes={compact(changes)}; rest={compact(rest)})")

        len_changes = len(changes)
        if len_changes == 1: return changes # got it

        change_list = list(changes)

        c1 = set(change_list[:len_changes//2])
        if self.test(c1.union(rest), **kwargs):
            return self.debug(c1, rest)    # in 1st half

        c2 = set(change_list[len_changes//2:])
        if self.test(c2.union(rest), **kwargs):
            return self.debug(c2, rest)    # in 2nd half

        # "interference"
        return self.debug(c1, c2.union(rest), **kwargs).union(
               self.debug(c2, c1.union(rest), **kwargs))

class RuntimeException(Exception):
    pass

class BadTestsFinder(DeltaDebugger):
    """Finds tests that cause other tests to fail."""

    def __init__(self, test_dir: Path, *, pytest_args: str = '', trace = None):
        super(BadTestsFinder, self).__init__(trace=trace)
        self.test_dir = test_dir

        def find_tests(p):
            for f in p.iterdir():
                if f.is_dir():
                    yield from find_tests(f)
                else:
                    # TODO filter according to pytest customization rules
                    if f.is_file() and (f.stem.startswith('test_') or f.stem.endswith('_test')) and f.suffix == '.py':
                        yield f

        self.all_tests = set(find_tests(self.test_dir))
        self.pytest_args = pytest_args


    def run_tests(self, tests_to_run: set = None) -> Path:
        """Runs the tests, by default all, returning the first one that fails, or None.
           Throws RuntimeError if unable to parse pytest's output.
        """
        import tempfile
        import subprocess
        import sys
        import pytest

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

        if self.trace: self.trace(f"running {len(test_set)} test(s).")
        with tempfile.TemporaryDirectory(dir=self.test_dir.parent) as tmpdir:
            tmpdir = Path(tmpdir)
            link_tree(self.test_dir, tmpdir)

            p = subprocess.run((f"{sys.executable} -m pytest {self.pytest_args} -x -qq --disable-warnings " +\
                                f"--rootdir {tmpdir} {tmpdir}").split(),
                               check=False, capture_output=True, timeout=60*60)

            if p.returncode in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
                if self.trace: self.trace(f"tests passed")
                return None

            if not (first_failing := self.find_failed_test(str(p.stdout, 'UTF-8'))):
                raise RuntimeException(f"Unable to parse failing test out of pytest output. RC={p.returncode}; output:\n" +\
                                       str(p.stdout, 'UTF-8') + "\n----\n")

            if tmpdir.is_absolute():
                # pytest sometimes makes absolute paths into relative ones by adding ../../.. to root...
                first_failing = first_failing.resolve()

            # bring it back to its normal path
            first_failing = self.test_dir / first_failing.relative_to(tmpdir)

            if self.trace: self.trace(f"tests rc={p.returncode} first_failing={first_failing}")

            return first_failing


    def test(self, test_set: set, **kwargs) -> bool:
        if first_failing := self.run_tests(test_set):
            # If given, check that it's target_test that failed: a different test may have failed.
            # If a test that comes after target_test fails, then this is really a success; but if it's a test
            # that comes before target_test, this may be "inconsistent" (in delta debugging terms).
            if not (target_test := kwargs.get('target_test')) or first_failing == target_test:
                return True # "interesting"/"reproduced"

        return False


    def find_culprit(self, failing_test: Path) -> T.Set[Path]:
        """Returns the set of tests causing 'failing_test' to fail."""
        assert failing_test in self.all_tests

# we unfortunately can't do this... the code that causes test(s) to fail may execute during pytest collection.
#        sorted_tests = sorted(self.all_tests)
#        test_set = set(sorted_tests[:sorted_tests.index(failing_test)])
#        assert self.test(test_set), "Test set doesn't fail!"

        return self.debug(changes=self.all_tests - {failing_test}, rest={failing_test}, target_test=failing_test)


    @staticmethod
    def find_failed_test(output: str) -> Path:
        import re
        if (m := re.search("^===+ short test summary info ===+\n" +\
                           "^(?:ERROR|FAILED) ([^\\s:]+)", output, re.MULTILINE)):
            return Path(m.group(1))

        return None


