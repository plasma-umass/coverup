from pathlib import Path
import coverup.segment as segment
import textwrap
import json


somecode_py = textwrap.dedent("""\
    # Sample Python code used to create some tests.
    import os

    class Foo:
        '''docstring...'''

        @staticmethod
        def foo():
            return os.path.exists('here')

        def __init__(self):
            '''initializes...'''
            self._foo = 0

        x = 0
        if x != 0:
            y = 2

        class Bar:
            z = 10

        def bar():
            '''docstring'''
            class Baz:
                assert False

            def baz():
                pass

    GLOBAL = 42

    def func(x):
        '''docstring'''
        if x > 0:
            return 42**42

    if __name__ == "__main__":
        ...
""")

somecode_json = """\
{
    "files": {
        "tests/somecode.py": {
            "executed_lines": [
                2, 4, 5, 7, 8, 11, 15, 16, 19,
                20, 22, 30, 32, 37, 38
            ],
            "missing_lines": [
                9, 13, 17, 24, 25, 27, 28, 34, 35
            ],
            "executed_branches": [
                [ 16, 19 ], [ 37, 38 ]
            ],
            "missing_branches": [
                [ 16, 17 ],
                [ 34, 0 ],
                [ 34, 35 ],
                [ 37, 0 ]
            ]
        }
    }
}
"""


def test_large_limit_whole_class(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "somecode.py").write_text(somecode_py)

    coverage = json.loads(somecode_json)
    segs = segment.get_missing_coverage(coverage, line_limit=100)

    assert ['Foo', 'func'] == [seg.name for seg in segs]
    assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
    assert all(seg.begin < seg.end for seg in segs)

    assert textwrap.dedent(segs[0].get_excerpt(tag_lines=False)) == textwrap.dedent("""\
        import os
        class Foo:
            '''docstring...'''

            @staticmethod
            def foo():
                return os.path.exists('here')

            def __init__(self):
                '''initializes...'''
                self._foo = 0

            x = 0
            if x != 0:
                y = 2

            class Bar:
                z = 10

            def bar():
                '''docstring'''
                class Baz:
                    assert False

                def baz():
                    pass
        """)

    assert textwrap.dedent(segs[1].get_excerpt(tag_lines=False)) == textwrap.dedent("""\
        def func(x):
            '''docstring'''
            if x > 0:
                return 42**42
        """)

    # FIXME check executed_lines, missing_lines, ..., interesting_lines


def test_small_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "somecode.py").write_text(somecode_py)

    coverage = json.loads(somecode_json)
    segs = segment.get_missing_coverage(coverage, line_limit=3)

    # "Bar" omitted because executed
    assert ['foo', '__init__', 'bar', 'func'] == [seg.name for seg in segs]
    assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
    assert all(seg.begin < seg.end for seg in segs)


    assert textwrap.dedent(segs[0].get_excerpt(tag_lines=False)) == textwrap.dedent("""\
        import os
        class Foo:
            @staticmethod
            def foo():
                return os.path.exists('here')
        """)

    assert textwrap.dedent(segs[2].get_excerpt(tag_lines=False)) == textwrap.dedent("""\
        class Foo:
            def bar():
                '''docstring'''
                class Baz:
                    assert False

                def baz():
                    pass
        """)

    # FIXME check executed_lines, missing_lines, ..., interesting_lines


def test_all_missing_not_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "somecode.py").write_text(somecode_py)

    neverloaded_json = """\
        {
            "files": {
                "tests/somecode.py": {
                    "executed_lines": [],
                    "missing_lines": [
                        2, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17,
                        19, 20, 22, 24, 25, 27, 28, 30, 32, 34,
                        35, 37, 38 ],
                    "executed_branches": [],
                    "missing_branches": [
                        [ 16, 17 ], [ 16, 19 ], [ 34, 0 ], [ 34, 35 ],
                        [ 37, 0 ], [ 37, 38 ]
                    ]
                }
            }
        }"""

    coverage = json.loads(neverloaded_json)
    segs = segment.get_missing_coverage(coverage, line_limit=2)

    assert ['foo', '__init__', 'Bar', 'bar', 'func'] == [seg.name for seg in segs]
    assert all([Path(seg.filename).name == 'somecode.py' for seg in segs])
    assert all(seg.begin < seg.end for seg in segs)

    assert textwrap.dedent(segs[3].get_excerpt(tag_lines=False)) == textwrap.dedent("""\
        class Foo:
            def bar():
                '''docstring'''
                class Baz:
                    assert False

                def baz():
                    pass
        """)

    # FIXME check executed_lines, missing_lines, ..., interesting_lines


def test_only_branch_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "code.py").write_text(textwrap.dedent("""\
        class Foo:
            class Bar:
                def __init__(self, x):
                    self.x = None
                    if x:
                        self.x = x
                    self.y = 0

        """))

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
        }"""

    coverage = json.loads(code_json)
    segs = segment.get_missing_coverage(coverage, line_limit=4)

    assert len(segs) == 1
    assert "__init__" == segs[0].name

    assert textwrap.dedent(segs[0].get_excerpt()) == textwrap.dedent("""\
           class Foo:
               class Bar:
                   def __init__(self, x):
                       self.x = None
        5:             if x:
                           self.x = x
        7:             self.y = 0
        """)

