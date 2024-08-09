import textwrap
import ast
import typing as T
import copy
#import coverup.codeinfo as codeinfo


# variable -> last assignment?
#   what about any other symbols? do slicing?
# function -> function definition
# class -> class definition, globals, __init__, ...
# method -> "class X:"

def find_path(node: ast.AST, name: T.List[str]) -> T.List[ast.AST]:
    """Looks for a class or function by name, prefixing it with the path of parent class objects, if any."""
    for c in ast.iter_child_nodes(node):
        if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if c.name == name[0]:
                if len(name) == 1:
                    return [c]

                if isinstance(c, ast.ClassDef) and len(name) > 1 and (path := find_path(c, name[1:])):
                    return [c, *path]

                return []

        else:
            if path := find_path(c, name):
                return path

        if isinstance(c, ast.ClassDef) and c.name == name[0]:
            if len(name) == 1: return [c]
            return [c, *path] if (path := find_path(c, name[1:])) else []

        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == name[0]:
            return [c] if len(name) == 1 else []


    return []


def summarize(path: T.List[ast.AST]) -> ast.AST:
    """Replaces portions of the element to be shown with "..." to make it shorter."""

    # first the object
    if isinstance(path[-1], ast.ClassDef):
        path[-1] = copy.deepcopy(path[-1])
        for c in ast.iter_child_nodes(path[-1]):
            if (isinstance(c, ast.ClassDef) or
                (isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name != "__init__")):
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


def get_info(tree: ast.AST, name: str, context: str = None) -> T.Optional[str]:
    # FIXME handle imports
    # FIXME see what 'name' is in 'context'

    if path := find_path(tree, name.split('.')):
        return ast.unparse(summarize(path))

    return None


def test_get_info_class():
    code = textwrap.dedent("""\
        class C:
            x = 10

            def __init__(self, x: int) -> None:
                self._foo = x

            class B:
                z = 42

                def foo():
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

            def __init__(self, x: int) -> None:
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

    assert get_info(tree, 'C.B.foo') == textwrap.dedent("""\
        class C:
            ...

            class B:
                ...

                def foo():
                    pass"""
    )

    assert get_info(tree, 'C.B.bar') == None


# TODO test with name having been imported
# TODO test with name being a local variable
