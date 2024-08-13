import textwrap
import ast
import typing as T
import copy
from pathlib import Path
import pytest
#import coverup.codeinfo as codeinfo


# TODO use 'ast' alternative that retains comments?
# TODO test with name being a local variable

# variable -> last assignment?
#   what about any other symbols? do slicing?
# function -> function definition
# class -> class definition, globals, __init__, ...
# method -> "class X:"


def resolve_import(file: Path, imp: ast.Import | ast.ImportFrom) -> Path | None:
    """Given a file containing an `import` and the `import` itself, determines
       what module to read for the import."""

#    print(f"file={file.relative_to(Path.cwd())} imp={ast.dump(imp)}")
    import importlib.util

    if isinstance(imp, ast.ImportFrom):
        if imp.level > 0:  # relative import
            path = file
            for _ in range(imp.level):
                path = path.parent

            # FIXME this doesn't handle namespace packages; would it be better to
            # go through sys.path, looking for the root from which 'file' was loaded?
            # Beware of possible relative paths in sys.path, though.
            for parent in path.parents:
                if not (parent / "__init__.py").exists():
                    break

            path = path.relative_to(parent)
            return f"{'.'.join(path.parts)}.{imp.module if imp.module else imp.names[0].name}"

        return imp.module # absolute from ... import 

    assert isinstance(imp, ast.Import)
    assert len(imp.names) == 1
    return imp.names[0].name


def find_name_path(node: ast.AST, name: T.List[str]) -> T.List[ast.AST]:
    """Looks for a class or function by name, prefixing it with the path of parent class objects, if any."""

    for c in ast.iter_child_nodes(node):
        if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if c.name == name[0]:
                if len(name) == 1:
                    return [c]

                if isinstance(c, ast.ClassDef) and len(name) > 1 and (path := find_name_path(c, name[1:])):
                    return [c, *path]

                return []

        else:
            if path := find_name_path(c, name):
                return path

        if isinstance(c, ast.ClassDef) and c.name == name[0]:
            if len(name) == 1: return [c]
            return [c, *path] if (path := find_name_path(c, name[1:])) else []

        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == name[0]:
            return [c] if len(name) == 1 else []

        if isinstance(c, (ast.Import, ast.ImportFrom)):
            # FIXME the first matching import needn't be the one that resolves the name:
            #   import foo.bar
            #   from qux import xyzzy as foo
            for alias in c.names:
                if (not alias.asname and alias.name.split('.')[0] == name[0]) or alias.asname == name[0]:
                    # don't need to check for len(name) == 1 here: 'import foo' is a solution for 'foo.bar'
                    imp = copy.copy(c)
                    imp.names = [alias]
                    return [imp]

        # FIXME variable assignments create aliases...

    return []


def summarize(path: T.List[ast.AST]) -> ast.AST:
    """Replaces portions of the element to be shown with "..." to make it shorter."""

    # first the object
    if isinstance(path[-1], ast.ClassDef):
        path[-1] = copy.deepcopy(path[-1])
        for c in ast.iter_child_nodes(path[-1]):
            if (isinstance(c, ast.ClassDef) or
                (isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name != "__init__")):
                # Leave "__init__" unmodified as it's likely to contain important member information
                c.body = [ast.Expr(ast.Constant(value=ast.literal_eval("...")))]

    # now the path of Class objects
    for i in reversed(range(len(path)-1)):
        assert isinstance(path[i], ast.ClassDef)
        path[i] = copy.copy(path[i])
        path[i].body = [
            ast.Expr(ast.Constant(value=ast.literal_eval("..."))),
            path[i+1]
        ]

    return path[0]


def parse_file(file: Path) -> ast.AST:
    with file.open("r") as f:
        tree = ast.parse(f.read())

    assert isinstance(tree, ast.Module)
    tree._attributes = (*tree._attributes, 'path')
    tree.path = file
    return tree


def get_info(tree: ast.Module, name: str, context: str = None) -> T.Optional[str]:
#    print(f"get_info {name=}")

    import importlib.util
    # FIXME see what 'name' is in 'context'

    key = name.split('.')

    while (path := find_name_path(tree, key)) and isinstance(path[-1], (ast.Import, ast.ImportFrom)):
        if not (module_name := resolve_import(tree.path, path[-1])):
            return None

        key = key[len(path)-1:]

        if isinstance(path[-1], ast.Import):
            # "import a.b.c" imports a, a.b and a.b.c...
            segments = module_name.split('.')
            len_equal_prefix = next((i for i, (x, y) in enumerate(zip(segments, key)) if x != y),
                                    min(len(segments), len(key)))

            module_name = '.'.join(segments[:len_equal_prefix])
            key = key[len_equal_prefix:]

        else:
            # replace with name out of 'from S import N as A'
            key[0] = path[-1].names[0].name

        if not (spec := importlib.util.find_spec(module_name)) or not spec.origin:
            return None

        import_file = Path(spec.origin)
        tree = parse_file(import_file)
        file = import_file

    if path:
        return ast.unparse(summarize(path))

    return None


@pytest.fixture
def import_fixture(monkeypatch):
    import importlib
    import tempfile
    import sys

    previously_loaded = {m for m in sys.modules}

    # Avoid the tmp_path fixture because it retains the directories
    # while other tests execute... if importlib then still has something
    # cached, we confusingly get other tests' contents, rather than a
    # clean "file not found" error.
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
    assert "foo" == resolve_import(file, I("import foo"))
    assert "foo.bar" == resolve_import(file, I("import foo.bar"))
    assert "bar"  == resolve_import(file, I("import bar"))

    file = foo / "__init__.py"
    assert "bar" == resolve_import(file, I("import bar"))
    assert "foo.bar" == resolve_import(file, I("import foo.bar"))
    assert "foo.bar" == resolve_import(file, I("from . import bar"))

    file = foo / "bar" / "__init__.py"
    assert "foo.bar.none" == resolve_import(file, I("import foo.bar.none"))
    assert "foo.bar.none" == resolve_import(file, I("from . import none"))
    assert "foo.baz" == resolve_import(file, I("from .. import baz"))
    assert "foo.bar.none" == resolve_import(file, I("from .none import *"))
    assert "foo.bar.none" == resolve_import(file, I("from ..bar.none import *"))


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

    assert None == U(find_name_path(tree, ["sys"]))
    assert 'import os' == U(find_name_path(tree, ["os"]))
    assert 'from bar import bar' == U(find_name_path(tree, ["C", "bar"]))
    assert None == U(find_name_path(tree, ["xyzzy"]))

    assert 'import bar' == U(find_name_path(tree, ["bar", "baz"]))


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

    assert get_info(tree, 'C') == textwrap.dedent("""\
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

    assert get_info(tree, 'C.foo') == textwrap.dedent("""\
        class C:
            ...

            @deco
            def foo(self) -> None:
                print(self._foo)"""
    )

    assert get_info(tree, 'C.B') == textwrap.dedent("""\
        class C:
            ...

            class B:
                z = 42

                def foo():
                    ..."""
    )

    assert get_info(tree, 'C.B.foo') == textwrap.dedent('''\
        class C:
            ...

            class B:
                ...

                def foo():
                    """far niente"""
                    pass'''
    )

    assert get_info(tree, 'C.B.bar') == None
    assert get_info(tree, 'foo') == None


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


    tree = parse_file(code)

    assert get_info(tree, 'foo2') == textwrap.dedent('''\
        def foofoo():
            return 42'''
    )

    assert get_info(tree, 'FooBar') == textwrap.dedent('''\
        class Baz:

            class Qux:
                ...'''
    )

    assert get_info(tree, 'foo.Foo') == textwrap.dedent('''\
        class Foo:
            pass'''
    )
