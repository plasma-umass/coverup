import textwrap
import ast
from pathlib import Path
import typing as T
import pytest
import coverup.codeinfo as codeinfo
import sys


codeinfo._debug = print     # enables debugging


@pytest.fixture
def importlib_cleanup():
    import importlib

    previously_loaded = {m for m in sys.modules}

    yield

    importlib.invalidate_caches()

    # It's not enough to call 'invalidate_caches'... :/
    for m in list(sys.modules):
        if m not in previously_loaded:
            del sys.modules[m]


@pytest.fixture
def import_fixture(importlib_cleanup, monkeypatch):
    import tempfile
    import shutil

    # Avoid using the tmp_path fixture because it retains the directories
    # while other tests execute... if importlib then still has something
    # cached, we confusingly get other tests' contents, rather than clean
    # "file not found" errors.
    with tempfile.TemporaryDirectory(dir=Path('.')) as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(tmp_path)
        yield tmp_path

    shutil.rmtree(tmp_path, ignore_errors=True)


def get_fqn(p):
    fqn = codeinfo._get_fqn(p)
    return '.'.join(fqn) if fqn else fqn


def test_get_fqn(import_fixture):
    tmp_path = import_fixture

    (tmp_path / "foo").mkdir()

    assert "foo" == get_fqn(tmp_path / "foo" / "__init__.py")
    assert "foo.bar" == get_fqn(tmp_path / "foo" / "bar.py")
    assert "foo.bar.baz" == get_fqn(tmp_path / "foo" / "bar" / "baz.py")

    assert "foo" == get_fqn(Path("foo") / "__init__.py")
    assert "foo.bar" == get_fqn(Path("foo") / "bar.py")
    assert "foo.bar.baz" == get_fqn(Path("foo") / "bar" / "baz.py")


def test_get_fqn_relative_syspath(importlib_cleanup, monkeypatch):
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path('.')) as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        rel_path = tmp_path.relative_to(Path.cwd())

        monkeypatch.chdir(rel_path)
        monkeypatch.syspath_prepend('.')

        assert "foo" == get_fqn(tmp_path / "foo" / "__init__.py")
        assert "foo.bar" == get_fqn(tmp_path / "foo" / "bar.py")
        assert "foo.bar.baz" == get_fqn(tmp_path / "foo" / "bar" / "baz.py")

        assert "foo" == get_fqn(Path("foo") / "__init__.py")
        assert "foo.bar" == get_fqn(Path("foo") / "bar.py")
        assert "foo.bar.baz" == get_fqn(Path("foo") / "bar" / "baz.py")


def test_resolve_from_import(import_fixture):
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

    foo = tmp_path / "foo"
    file = foo / "__init__.py"
    assert "foo" == codeinfo._resolve_from_import(file, I("from . import bar"))
    assert "foo.bar" == codeinfo._resolve_from_import(file, I("from .bar import baz"))
    assert "foo.bar.none" == codeinfo._resolve_from_import(file, I("from .bar.none import baz"))

    file = foo / "bar" / "__init__.py"
    assert "foo.bar" == codeinfo._resolve_from_import(file, I("from . import none"))
    assert "foo.bar.none" == codeinfo._resolve_from_import(file, I("from .none import *"))
    assert "foo.baz" == codeinfo._resolve_from_import(file, I("from ..baz import *"))
    assert "foo.bar.none" == codeinfo._resolve_from_import(file, I("from ..bar.none import *"))


def test_get_info_class(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
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

        class D:
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)

            def d(self):
                pass
        """
    ))

    tree = codeinfo.parse_file(code)

    assert codeinfo.get_info(tree, 'C') == textwrap.dedent("""\
        ```python
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
                ...
        ```"""
    )

    assert codeinfo.get_info(tree, 'D') == textwrap.dedent('''\
        ```python
        class D:

            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)

            def d(self):
                ...
        ```'''
    )

    assert codeinfo.get_info(tree, 'C.foo') == textwrap.dedent("""\
        ```python
        class C:
            ...

            @deco
            def foo(self) -> None:
                print(self._foo)
        ```"""
    )

    assert codeinfo.get_info(tree, 'C.B') == textwrap.dedent("""\
        ```python
        class C:
            ...

            class B:
                z = 42

                def foo():
                    ...
        ```"""
    )

    assert codeinfo.get_info(tree, 'C.B.foo') == textwrap.dedent('''\
        ```python
        class C:
            ...

            class B:
                ...

                def foo():
                    """far niente"""
                    pass
        ```'''
    )

    assert codeinfo.get_info(tree, 'C.B.bar') == None
    assert codeinfo.get_info(tree, 'foo') == None


def test_get_info_method_from_parent(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        class A:
            def a(self):
                return 42

        class B(A):
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'B.a') == textwrap.dedent("""\
        ```python
        class A:
            ...

            def a(self):
                return 42
        ```"""
    )


def test_get_info_method_from_parent_is_attribute(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        class A:
            class B:
                def a(self):
                    return 42

        class C(A.B):
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'C.a') == textwrap.dedent("""\
        ```python
        class A:
            ...

            class B:
                ...

                def a(self):
                    return 42
        ```"""
    )


def test_get_info_method_from_parent_parent_missing(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        from doesnotexist import A

        class B(A):
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'B.a') == None


def test_get_info_method_from_parent_imported(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        from bar import A

        class B(A):
            pass
        """
    ))

    (tmp_path / "bar.py").write_text(textwrap.dedent("""\
        class A:
            def a(self):
                return 42
        """
    ))

    # TODO it would be better if it showed class B(A)
    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'B.a') == textwrap.dedent("""\
        in bar.py:
        ```python
        class A:
            ...

            def a(self):
                return 42
        ```"""
    )


def test_get_info_method_from_parent_imported_in_class(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        class A:
            from bar import B

            class C(B):
                pass

            class D(C):
                pass
        """
    ))

    (tmp_path / "bar.py").write_text(textwrap.dedent("""\
        class B:
            def b(self):
                return 42
        """
    ))

    # FIXME this would be better if it showed class C(B) and class D(C)
    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'A.D.b') == textwrap.dedent("""\
        in foo.py:
        ```python
        class A:
            ...
            from bar import B
        ```

        in bar.py:
        ```python
        class B:
            ...

            def b(self):
                return 42
        ```"""
    )


def test_get_info_assignment(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        PI = 3.1415

        x = 0
        if x == 0:
            TYPES = (
                int,
                str,
            )

        class C:
            x = 10

            def __init__(self, x: int) -> C:
                self._foo = x
        """
    ))

    tree = codeinfo.parse_file(code)

    assert codeinfo.get_info(tree, 'PI') == textwrap.dedent("""\
        ```python
        PI = 3.1415
        ```"""
    )

    # FIXME do proper code slicing
    assert codeinfo.get_info(tree, 'TYPES') == textwrap.dedent("""\
        ```python
        TYPES = (int, str)
        ```"""
    )

    assert codeinfo.get_info(tree, 'C.x') == textwrap.dedent("""\
        ```python
        class C:
            ...
            x = 10
        ```"""
    )


def test_get_info_import(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        import foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        in foo/__init__.py:
        ```python
        class Foo:
            pass
        ```'''
    )

    assert codeinfo.get_info(tree, 'foo.baz') == None # doesn't exist


def test_get_info_import_submodule(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        import foo.bar
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        in foo/__init__.py:
        ```python
        class Foo:
            pass
        ```'''
    )

    assert codeinfo.get_info(tree, 'foo.bar.Bar') == textwrap.dedent('''\
        in foo/bar.py:
        ```python
        class Bar:
            pass
        ```'''
    )


def test_get_info_import_as(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        import foo as baz
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'baz.Foo') == textwrap.dedent('''\
        in code.py:
        ```python
        import foo as baz
        ```

        in foo/__init__.py:
        ```python
        class Foo:
            pass
        ```'''
    )


def test_get_info_from_import_symbol_exists(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        from foo import bar
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass

        class bar:  # 'bar' is defined here -- takes precedence over bar.py
            class Bar:
                pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'bar.Bar') == textwrap.dedent('''\
        in foo/__init__.py:
        ```python
        class bar:
            ...

            class Bar:
                pass
        ```'''
    )


def test_get_info_from_import_as_symbol_exists(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        from foo import bar as baz
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass

        class bar:  # 'bar' is defined here -- takes precedence over bar.py
            class Bar:
                pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'baz.Bar') == textwrap.dedent('''\
        in code.py:
        ```python
        from foo import bar as baz
        ```

        in foo/__init__.py:
        ```python
        class bar:
            ...

            class Bar:
                pass
        ```'''
    )


def test_get_info_from_import_symbol_doesnt_exist(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        from foo import bar
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'bar.Bar') == textwrap.dedent('''\
        in foo/bar.py:
        ```python
        class Bar:
            pass
        ```'''
    )


def test_get_info_from_import_as_symbol_doesnt_exist(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        from foo import bar as baz
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'baz.Bar') == textwrap.dedent('''\
        in code.py:
        ```python
        from foo import bar as baz
        ```

        in foo/bar.py:
        ```python
        class Bar:
            pass
        ```'''
    )


def test_get_info_from_import_relative(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        import foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        from . import bar
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        from .baz import Baz as Bar
        """
    ))
    (tmp_path / "foo" / "baz.py").write_text(textwrap.dedent("""\
        class Baz:
            answer = 42
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'foo.bar.Bar') == textwrap.dedent('''\
        in code.py:
        ```python
        import foo
        ```

        in foo/__init__.py:
        ```python
        from . import bar
        ```

        in foo/bar.py:
        ```python
        from .baz import Baz as Bar
        ```

        in foo/baz.py:
        ```python
        class Baz:
            answer = 42
        ```'''
    )


def test_get_info_import_in_excerpt_function(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os

        def something():
            def something_else():
                from foo import Foo
                Foo()
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'Foo', line=3) == textwrap.dedent('''\
        in foo/__init__.py:
        ```python
        class Foo:
            pass
        ```'''
    )


def test_get_info_import_in_excerpt_class(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os

        class X:
            from foo import Foo

            def __init__(self, x: Foo):
                self.x = x
        """
    ))

    (tmp_path / "foo.py").write_text(textwrap.dedent("""\
        class Foo:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'Foo', line=3) == textwrap.dedent('''\
        in foo.py:
        ```python
        class Foo:
            pass
        ```'''
    )


def test_get_info_import_in_class(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        class C:
            import foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        from . import bar

        class Foo:
            pass
        """
    ))
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'C.foo.bar.Bar') == textwrap.dedent('''\
        in code.py:
        ```python
        class C:
            ...
            import foo
        ```

        in foo/__init__.py:
        ```python
        from . import bar
        ```

        in foo/bar.py:
        ```python
        class Bar:
            pass
        ```'''
    )


def test_get_info_imported_assignment(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import foo.constants as C
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text("")
    (tmp_path / "foo" / "constants.py").write_text(textwrap.dedent("""\
        PI = 3.1415926
        """
    ))

    tree = codeinfo.parse_file(code)

    assert codeinfo.get_info(tree, 'C.PI') == textwrap.dedent('''\
        in code.py:
        ```python
        import foo.constants as C
        ```

        in foo/constants.py:
        ```python
        PI = 3.1415926
        ```'''
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

    tree = codeinfo.parse_file(code)

    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        in foo/__init__.py:
        ```python
        class Foo:
            pass
        ```'''
    )


@pytest.mark.parametrize("from_module", [True, False])
def test_get_info_name_includes_module_fqn(import_fixture, from_module):
    tmp_path = import_fixture

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text("")
    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class C:
            pass
        """
    ))

    if from_module:
        tree = codeinfo.parse_file(tmp_path / "foo" / "bar.py")
        assert codeinfo.get_info(tree, 'foo.bar.C') == textwrap.dedent('''\
            ```python
            class C:
                pass
            ```'''
        )

    else:
        (tmp_path / "x.py").write_text(textwrap.dedent("""\
            from foo.bar import C
            """
        ))

        # class C really apears under the name "C", not "foo.bar.C", but gpt-4o sometimes
        # asks for names like it after seeing the equivalent of  "from foo.bar import C"
        tree = codeinfo.parse_file(tmp_path / "x.py")
        assert codeinfo.get_info(tree, 'foo.bar.C') == textwrap.dedent('''\
            ```python
            class C:
                pass
            ```'''
        )


def test_get_info_includes_imports(import_fixture):
    tmp_path = import_fixture
    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import os
        from foo import bar as R, baz as Z
        import sys, ast

        class C(Z):
            from foo import Foo
            x = len(sys.path)

            def __init__(self, x: int) -> C:
                self._foo = Z(x)

            class B:
                z = 42

        def func(x: C):
            x.foo(os.environ)
        """
    ))

    (tmp_path / "foo.py").write_text(textwrap.dedent("""\
        import sys
        from .bar import Bar

        class Foo(Bar):
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'C') == textwrap.dedent("""\
        ```python
        from foo import baz as Z
        import sys

        class C(Z):
            from foo import Foo
            x = len(sys.path)

            def __init__(self, x: int) -> C:
                self._foo = Z(x)

            class B:
                ...
        ```"""
    )

    assert codeinfo.get_info(tree, 'C', generate_imports=False) == textwrap.dedent("""\
        ```python
        class C(Z):
            from foo import Foo
            x = len(sys.path)

            def __init__(self, x: int) -> C:
                self._foo = Z(x)

            class B:
                ...
        ```"""
    )

    assert codeinfo.get_info(tree, 'C.Foo') == textwrap.dedent("""\
        in code.py:
        ```python
        from foo import baz as Z

        class C(Z):
            ...
            from foo import Foo
        ```

        in foo.py:
        ```python
        from .bar import Bar

        class Foo(Bar):
            pass
        ```"""
    )

    assert codeinfo.get_info(tree, 'C.Foo', generate_imports=False) == textwrap.dedent("""\
        in code.py:
        ```python
        class C(Z):
            ...
            from foo import Foo
        ```

        in foo.py:
        ```python
        class Foo(Bar):
            pass
        ```"""
    )

    assert codeinfo.get_info(tree, 'func') == textwrap.dedent("""\
        ```python
        import os

        def func(x: C):
            x.foo(os.environ)
        ```"""
    )

    assert codeinfo.get_info(tree, 'func', generate_imports=False) == textwrap.dedent("""\
        ```python
        def func(x: C):
            x.foo(os.environ)
        ```"""
    )


def test_get_info_included_imports_always_absolute(import_fixture):
    tmp_path = import_fixture
    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        import sys
        from .bar import Bar

        class Foo(Bar):
            pass
        """
    ))

    (tmp_path / "foo" / "bar.py").write_text(textwrap.dedent("""\
        class Bar:
            pass
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'foo.Foo') == textwrap.dedent("""\
        in foo/__init__.py:
        ```python
        from foo.bar import Bar

        class Foo(Bar):
            pass
        ```"""
    )


def test_get_global_imports(import_fixture):
    tmp_path = import_fixture
    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import a, b
        from c import d as e
        import os

        try:
            from hashlib import sha1
        except ImportError:
            from sha import sha as sha1
        
        class Foo:
            import abc as os

            def f():
                if os.path.exists("foo"):
                    sha1(a.x, b.x, e.x)
        """))

    print(code)

    def find_node(tree, name):
        for n in ast.walk(tree):
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return n

    tree = codeinfo.parse_file(code)
#    print(ast.dump(tree, indent=2))

    f = find_node(tree, 'f')
    assert [ast.unparse(x) for x in codeinfo.get_global_imports(tree, f)] == [
        'import a, b',
        'from c import d as e',
        'import os',
        'from hashlib import sha1'
    ]


def test_get_info_module(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        import bar
        """
    ))
    (tmp_path / "bar.py").write_text(textwrap.dedent("""\
        try:
            import os
        except ImportError:
            import os2

        class A:
            def __init__(self):
                pass

            class B:
                pass

        def foo(a, b):
            return a+b

        answer = 42

        if __name__ == '__main__':
            import sys
            sys.exit(0)
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'bar') == textwrap.dedent("""\
        in bar.py:
        ```python
        try:
            import os
        except ImportError:
            import os2

        class A:
            ...

        def foo(a, b):
            ...
        answer = 42
        if __name__ == '__main__':
            import sys
            sys.exit(0)
        ```"""
    )


@pytest.mark.skipif(sys.version_info[0:2] < (3,11), reason="not a frozen module then")
def test_get_info_frozen_module(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        import os
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'os.path.join') == None


def test_get_info_builtin_module(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text(textwrap.dedent("""\
        import sys
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'sys.path') == None


def test_get_info_non_py(import_fixture):
    """On Python 3.10 on Linux, this module maps to a .so, not a Python file"""
    tmp_path = import_fixture

    code = tmp_path / "foo.py"
    code.write_text("")

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'termios.tcgetattr') == None


def test_get_info_import_star(import_fixture):
    tmp_path = import_fixture

    code = tmp_path / "code.py"
    code.write_text(textwrap.dedent("""\
        import foo
        """
    ))

    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "__init__.py").write_text(textwrap.dedent("""\
        import sys

        if sys.platform == 'win32':
            from .win32 import *
        else:
            from .unix import *
        """
    ))
    (tmp_path / "foo" / "unix.py").write_text(textwrap.dedent("""\
        answer = 'eunuchs'
        """
    ))
    (tmp_path / "foo" / "win32.py").write_text(textwrap.dedent("""\
        answer = '$$$$'
        """
    ))

    tree = codeinfo.parse_file(code)
    assert codeinfo.get_info(tree, 'foo.answer') == textwrap.dedent("""\
        in foo/win32.py:
        ```python
        answer = '$$$$'
        ```"""
    )
