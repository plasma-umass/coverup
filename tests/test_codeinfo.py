import textwrap
import ast
from pathlib import Path
import typing as T
import pytest
import coverup.codeinfo as codeinfo


@pytest.fixture
def import_fixture(monkeypatch):
    import importlib
    import tempfile
    import sys

    previously_loaded = {m for m in sys.modules}

    # Avoid using the tmp_path fixture because it retains the directories
    # while other tests execute... if importlib then still has something
    # cached, we confusingly get other tests' contents, rather than clean
    # "file not found" errors.
    with tempfile.TemporaryDirectory(dir=Path('.')) as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(tmp_path)
        yield tmp_path

    importlib.invalidate_caches()

    # It's not enough to call 'invalidate_caches'... :/
    for m in list(sys.modules):
        if m not in previously_loaded:
            del sys.modules[m]


def test_resolve_import(import_fixture):
    tmp_path = import_fixture

    def I(imp_code):
        return ast.parse(imp_code).body[0]

    (tmp_path / "code.py").touch()
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").touch()
    (tmp_path / "foo" / "bar").mkdir()
    (tmp_path / "foo" / "bar" / "__init__.py").touch()
    (tmp_path / "foo" / "bar" / "none.py").touch()
    (tmp_path / "foo" / "baz.py").touch()
    (tmp_path / "bar.py").touch()

    file = tmp_path / "code.py"
    foo = tmp_path / "foo"
    assert "foo" == codeinfo._resolve_import(file, I("import foo"))
    assert "foo.bar" == codeinfo._resolve_import(file, I("import foo.bar"))
    assert "bar"  == codeinfo._resolve_import(file, I("import bar"))

    file = foo / "__init__.py"
    assert "bar" == codeinfo._resolve_import(file, I("import bar"))
    assert "foo.bar" == codeinfo._resolve_import(file, I("import foo.bar"))
    assert "foo.bar" == codeinfo._resolve_import(file, I("from . import bar"))

    file = foo / "bar" / "__init__.py"
    assert "foo.bar.none" == codeinfo._resolve_import(file, I("import foo.bar.none"))
    assert "foo.bar.none" == codeinfo._resolve_import(file, I("from . import none"))
    assert "foo.baz" == codeinfo._resolve_import(file, I("from .. import baz"))
    assert "foo.bar.none" == codeinfo._resolve_import(file, I("from .none import *"))
    assert "foo.bar.none" == codeinfo._resolve_import(file, I("from ..bar.none import *"))


def test_find_name_path_import():
    code = textwrap.dedent("""\
        import os
        from . import foo
        import bar

        class C:
            from bar import bar
            x = 10

            def __init__(self, x: int) -> C:
                self._foo = x

            class B:
                z = 42

            @deco
            def foo(self) -> None:
                print(self._foo)

        def func(x: C):
            import xyzzy
            x.foo()
        """
    )

    tree = ast.parse(code)

    def U(x):
        if x: return ast.unparse(x[-1])

    assert None == U(codeinfo._find_name_path(tree, ["sys"]))
    assert 'import os' == U(codeinfo._find_name_path(tree, ["os"]))
    assert 'from bar import bar' == U(codeinfo._find_name_path(tree, ["C", "bar"]))
    assert None == U(codeinfo._find_name_path(tree, ["xyzzy"]))

    assert 'import bar' == U(codeinfo._find_name_path(tree, ["bar", "baz"]))


def test_get_info_class():
    code = textwrap.dedent("""\
        class C:
            x = 10

            def __init__(self, x: int) -> C:
                self._foo = x

            class B:
                z = 42

                def foo():
                    '''far niente'''
                    pass

            @deco
            def foo(self) -> None:
                print(self._foo)

            def bar(self):
                pass

        def func(x: C):
            x.foo()
        """
    )

    tree = ast.parse(code)

    assert codeinfo.get_info(tree, 'C') == textwrap.dedent("""\
        class C:
            x = 10

            def __init__(self, x: int) -> C:
                self._foo = x

            class B:
                ...

            @deco
            def foo(self) -> None:
                ...

            def bar(self):
                ..."""
    )

    assert codeinfo.get_info(tree, 'C.foo') == textwrap.dedent("""\
        class C:
            ...

            @deco
            def foo(self) -> None:
                print(self._foo)"""
    )

    assert codeinfo.get_info(tree, 'C.B') == textwrap.dedent("""\
        class C:
            ...

            class B:
                z = 42

                def foo():
                    ..."""
    )

    assert codeinfo.get_info(tree, 'C.B.foo') == textwrap.dedent('''\
        class C:
            ...

            class B:
                ...

                def foo():
                    """far niente"""
                    pass'''
    )

    assert codeinfo.get_info(tree, 'C.B.bar') == None
    assert codeinfo.get_info(tree, 'foo') == None


def test_get_info_imported(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os

        import foo.bar
        from foo.foo import foofoo as foo2
        from foo import Bar as FooBar
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        from .bar import Bar

        class Foo:
            pass
        """
    ))

    (tmp_path / "foo" / "foo.py").write_text(textwrap.dedent("""\
        def foofoo():
            return 42
        """
    ))

    (tmp_path / "foo" / "bar").mkdir()
    (tmp_path / "foo" / "bar" / "__init__.py").write_text(textwrap.dedent("""\
        from ..baz import Baz as Bar
        """
    ))
    (tmp_path / "foo" / "baz.py").write_text(textwrap.dedent("""\
        class Baz:
            class Qux:
                answer = 42
        """
    ))


    tree = codeinfo._parse_file(code)

    assert codeinfo.get_info(tree, 'foo2') == textwrap.dedent('''\
        def foofoo():
            return 42'''
    )

    assert codeinfo.get_info(tree, 'FooBar') == textwrap.dedent('''\
        class Baz:

            class Qux:
                ...'''
    )

    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        class Foo:
            pass'''
    )


def test_get_info_import_and_class_in_block(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        try:
            import foo
        except ImportError:
            import bar as foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        if True:
            class Foo:
                pass
        """
    ))
    (tmp_path / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            answer = 42
        """
    ))

    tree = codeinfo._parse_file(code)

    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        class Foo:
            pass'''
    )
