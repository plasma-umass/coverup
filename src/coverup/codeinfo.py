import ast
import copy
from pathlib import Path
import typing as T
import importlib.util


# TODO use 'ast' alternative that retains comments?

def _get_fqn(file: Path) -> T.Optional[T.List[str]]:
    """Returns a source file's Python Fully Qualified Name, as a list name parts."""
    import sys

    if not file.is_absolute():
        file = file.resolve()

    parents = list(file.parents)

    for path in sys.path:
        path = Path(path)
        if not path.is_absolute():
            path = path.resolve()

        for p in parents:
            if p == path:
                relative = file.relative_to(p)
                relative = relative.parent if relative.name == '__init__.py' else relative.parent / relative.stem
                return relative.parts


def _resolve_import(file: Path, imp: ast.Import | ast.ImportFrom) -> str:
    """Given a file containing an `import` and the `import` itself, determines
       what module to read for the import."""

    if isinstance(imp, ast.ImportFrom):
        if imp.level > 0:  # relative import
            if not (fqn := _get_fqn(file)):
                return None

            if imp.level > len(fqn):
                return None # would go beyond top-level package

            if imp.level > 1:
                fqn = fqn[:-(imp.level-1)]

            return f"{'.'.join(fqn)}.{imp.module if imp.module else imp.names[0].name}"

        return imp.module # absolute from ... import 

    assert isinstance(imp, ast.Import)
    assert len(imp.names) == 1
    return imp.names[0].name


def _find_name_path(node: ast.AST, name: T.List[str]) -> T.List[ast.AST]:
    """Looks for a class or function by name, returning the "path" of ast.ClassDef modules crossed
       to find it.  If an `import` is found for the sought, it is returned instead.
    """

    for c in ast.iter_child_nodes(node):
        if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if c.name == name[0]:
                if len(name) == 1:
                    return [c]

                if isinstance(c, ast.ClassDef) and len(name) > 1 and (path := _find_name_path(c, name[1:])):
                    return [c, *path]

                return []

        else:
            if path := _find_name_path(c, name):
                return path

        if isinstance(c, ast.ClassDef) and c.name == name[0]:
            if len(name) == 1: return [c]
            return [c, *path] if (path := _find_name_path(c, name[1:])) else []

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

        # FIXME variable assignments create aliases... follow them

    return []


def _summarize(path: T.List[ast.AST]) -> ast.AST:
    """Replaces portions of Class elements with "..." to reduce noise and tokens."""

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
    """Reads a python source file, annotating it with its path/filename."""
    with file.open("r") as f:
        tree = ast.parse(f.read())

    assert isinstance(tree, ast.Module)
    tree._attributes = (*tree._attributes, 'path')
    tree.path = file
    return tree


def _common_prefix_len(a: T.List[str], b: T.List[str]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def get_info(module: ast.Module, name: str) -> T.Optional[str]:
    """Returns summarized information on a class or function, following imports if necessary."""

    key = name.split('.')

    if (len(key) > 1 and (module_fqn := _get_fqn(module.path)) and
        (common_prefix := _common_prefix_len(module_fqn, key))):
        key = key[common_prefix:]

    while (path := _find_name_path(module, key)) and isinstance(path[-1], (ast.Import, ast.ImportFrom)):
        if not (module_name := _resolve_import(module.path, path[-1])):
            return None

        key = key[len(path)-1:]

        if isinstance(path[-1], ast.Import):
            # "import a.b.c" actually imports a, a.b and a.b.c...
            segments = module_name.split('.')
            common_prefix = _common_prefix_len(segments, key)

            module_name = '.'.join(segments[:common_prefix])
            key = key[common_prefix:]

        else:
            # replace with name out of 'from S import N as A'
            key[0] = path[-1].names[0].name

        if not (spec := importlib.util.find_spec(module_name)) or not spec.origin:
            return None

        import_file = Path(spec.origin)
        module = parse_file(import_file)
        file = import_file

    # XXX add any relevant imports

    if path:
        return ast.unparse(_summarize(path))

    return None
