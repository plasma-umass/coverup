from pathlib import Path
from coverup.segment import *
import json


class mockfs:
    """Mocks the built-in open() function"""

    def __init__(self, files: dict):
        self.files = files

    def __enter__(self):
        import unittest.mock as mock

        def _open(filename, mode="r"):
            if filename not in self.files: raise FileNotFoundError(filename)
            return mock.mock_open(read_data=self.files[filename]).return_value

        self.mock = mock.patch('builtins.open', new=_open)
        self.mock.__enter__()
        return self

    def __exit__(self, *args):
        self.mock.__exit__(*args)


somecode_py = (Path("tests") / "somecode.py").read_text()
somecode_json = """\
{
    "files": {
        "tests/somecode.py": {
            "executed_lines": [
                3, 4, 6, 9, 20, 21, 25, 27, 29, 38, 39, 40
            ],
            "missing_lines": [
                7, 10, 12, 13, 15, 16, 18, 23, 32, 34, 36
            ],
            "executed_branches": [
                [38, 39]
            ],
            "missing_branches": [
                [12, 13], [12, 15], [38, 0]
            ]
        }
    }
}
"""


def test_basic():
    coverage = json.loads(somecode_json)

    with mockfs({"tests/somecode.py": somecode_py}):
        segs = get_missing_coverage(coverage, line_limit=2)

        assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
        seg_names = [seg.name for seg in segs]
        assert ['__init__', 'foo', 'bar', 'globalDef2'] == seg_names

        bar = segs[seg_names.index('bar')]
        assert bar.begin == 20 # decorator line
        assert '@staticmethod' in bar.get_excerpt(), "Decorator missing"

        for seg in segs:
            for l in seg.missing_lines:
                assert seg.begin <= l <= seg.end


def test_coarse():
    coverage = json.loads(somecode_json)

    with mockfs({"tests/somecode.py": somecode_py}):
        segs = get_missing_coverage(coverage, line_limit=100)

        assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
        seg_names = [seg.name for seg in segs]
        assert ['SomeCode', 'globalDef2'] == seg_names

        assert segs[seg_names.index('SomeCode')].begin == 3 # entire class?
        assert segs[seg_names.index('SomeCode')].end == 24  # entire class?

        for seg in segs:
            for l in seg.missing_lines:
                assert seg.begin <= l <= seg.end


def test_no_branch_coverage():
    somecode_json_no_branch = """\
{
    "files": {
        "tests/somecode.py": {
            "executed_lines": [
                3, 4, 6, 9, 20, 21, 25, 27, 29, 38, 39, 40
            ],
            "missing_lines": [
                7, 10, 12, 13, 15, 16, 18, 23, 32, 34, 36
            ]
        }
    }
}
"""
    coverage = json.loads(somecode_json_no_branch)

    with mockfs({"tests/somecode.py": somecode_py}):
        segs = get_missing_coverage(coverage, line_limit=2)

        assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
        assert ['__init__', 'foo', 'bar', 'globalDef2'] == [seg.name for seg in segs]


def test_all_missing():
    somecode_json = """\
{
    "files": {
        "tests/somecode.py": {
            "executed_lines": [
            ],
            "missing_lines": [
                3, 4, 6, 9, 20, 21, 25, 27, 29, 38, 39, 40,
                7, 10, 12, 13, 15, 16, 18, 23, 32, 34, 36
            ]
        }
    }
}
"""

    coverage = json.loads(somecode_json)
    with mockfs({"tests/somecode.py": somecode_py}):
        segs = get_missing_coverage(coverage, line_limit=3)

        print("\n".join(f"{s} {s.lines_of_interest=}" for s in segs))

        assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
        assert ['__init__', 'foo', 'bar', 'globalDef', 'globalDef2'] == [seg.name for seg in segs]

        for i in range(1, len(segs)):
            assert segs[i-1].end <= segs[i].begin     # no overlaps

        # FIXME global statements missing... how to best capture them?


def test_class_excludes_decorator_of_function_if_at_limit():
    code_py = """\
class Foo:
    x = 0

    @staticmethod
    def foo():
        pass
"""

    code_json = """\
{
    "files": {
        "code.py": {
            "executed_lines": [
            ],
            "missing_lines": [
                1, 2, 4, 5, 6
            ]
        }
    }
}
"""
    coverage = json.loads(code_json)
    with mockfs({"code.py": code_py}):
        segs = get_missing_coverage(coverage, line_limit=4)

        print("\n".join(f"{s} {s.lines_of_interest=}" for s in segs))

        assert ['Foo', 'foo'] == [seg.name for seg in segs]
        assert segs[0].begin == 1
        assert segs[0].end <= 4 # shouldn't include "@staticmethod"
        assert segs[0].missing_lines == {1,2}


def test_class_statements_after_methods():
    code_py = """\
class Foo:
    @staticmethod
    def foo():
        pass

    x = 0
    y = 1

    def bar():
        pass
"""

    code_json = """\
{
    "files": {
        "code.py": {
            "executed_lines": [
            ],
            "missing_lines": [
                1, 2, 3, 4, 6, 7, 9, 10
            ]
        }
    }
}
"""
    coverage = json.loads(code_json)
    with mockfs({"code.py": code_py}):
        segs = get_missing_coverage(coverage, line_limit=4)

        print("\n".join(str(s) for s in segs))

        for seg in segs:
            for l in seg.missing_lines:
                assert seg.begin <= l <= seg.end


def test_class_within_class():
    code_py = """\
class Foo:
    foo_ = 0
    class Bar:
        bar_ = 0
        def __init__(self):
            self.x = 0
            self.y = 0

"""

    code_json = """\
{
    "files": {
        "code.py": {
            "executed_lines": [
            ],
            "missing_lines": [
                1, 2, 3, 4, 5, 6, 7
            ]
        }
    }
}
"""
    coverage = json.loads(code_json)
    with mockfs({"code.py": code_py}):
        segs = get_missing_coverage(coverage, line_limit=2)

        print("\n".join(str(s) for s in segs))

        for seg in segs:
            for l in seg.missing_lines:
                assert seg.begin <= l <= seg.end

        seg_names = [seg.name for seg in segs]
        assert seg_names == ['Foo', 'Bar', '__init__']

        init = segs[seg_names.index('__init__')]
        assert init.context == [(1,2), (3,4)]


def test_only_coverage_missing():
    code_py = """\
class Foo:
    class Bar:
        def __init__(self, x):
            self.x = None
            if x:
                self.x = x
            self.y = 0

"""

    code_json = """\
{
    "files": {
        "code.py": {
            "executed_lines": [
                1, 2, 3, 4, 5, 6, 7
            ],
            "missing_lines": [
            ],
            "executed_branches": [
                [5, 6]
            ],
            "missing_branches": [
                [5, 7]
            ]
        }
    }
}
"""
    coverage = json.loads(code_json)
    with mockfs({"code.py": code_py}):
        segs = get_missing_coverage(coverage, line_limit=4)

        print("\n".join(f"{s} {s.lines_of_interest=}" for s in segs))

        for seg in segs:
            for l in seg.missing_lines:
                assert seg.begin <= l <= seg.end

        seg_names = [seg.name for seg in segs]
        assert seg_names == ['__init__']
        assert segs[0].context == [(1,2), (2,3)]
        assert segs[0].missing_lines == set()
        assert segs[0].missing_branches == {(5,7)}


def test_class_long_init():
    code_py = """\
class SomeClass(object):

    def __init__(self, module):
        ...
        ...
        ...
        ...
        ...
"""

    code_cov = {
        'files': {
            'code.py': {
                'executed_lines': [],
                'missing_lines': [1, *range(3, 9)]
            }
        }
    }

    with mockfs({"code.py": code_py}):
        segs = get_missing_coverage(code_cov, line_limit=5)

    print("\n".join(f"{s} {s.lines_of_interest=}" for s in segs))
    assert segs[0].begin == 3
