import ast
import copy
from pathlib import Path
import typing as T
import importlib.util

#_debug = lambda x: None
_debug = print


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

    _debug(f"looking up {name} in {ast.dump(node)}")

    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name == name[0]:
            if len(name) == 1:
                return [node]

            if isinstance(node, ast.ClassDef):
                for c in ast.iter_child_nodes(node):
                    if (path := _find_name_path(c, name[1:])):
                        return [node, *path]

        return []

    if (isinstance(node, ast.Assign) and
        any(isinstance(n, ast.Name) and n.id == name[0] for t in node.targets for n in ast.walk(t))):
        return [node] if len(name) == 1 else []

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # FIXME the first matching import needn't be the one that resolves the name:
        #   import foo.bar
        #   from qux import xyzzy as foo
        c_module = getattr(node, "module", None)
        for alias in node.names:
            a_name = [alias.asname] if alias.asname else alias.name.split('.')
            if (a_name[0] == name[0] or (c_module and c_module.split('.') + a_name == name)):
                # don't need to check for len(name) == 1 here: 'import foo' is a solution for 'foo.bar'
                imp = copy.copy(node)
                imp.names = [alias]
                return [imp]

        return []

    for c in ast.iter_child_nodes(node):
        if (path := _find_name_path(c, name)):
            return path

    return []


def _summarize(path: T.List[ast.AST]) -> ast.AST:
    """Replaces portions of Class elements with "..." to reduce noise and tokens."""

    # first the object
    if isinstance(path[-1], ast.ClassDef):
        path[-1] = copy.deepcopy(path[-1])
        for c in ast.iter_child_nodes(path[-1]):
            if (isinstance(c, ast.ClassDef) or
                (isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                 c.name not in ("__init__", "__new__"))):
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


def get_global_imports(module: ast.Module, node: ast.AST) -> T.List[ast.Import | ast.ImportFrom]:
    """Looks for module-level `import`s that (may) define the names seen in "node"."""

    def get_names(node: ast.AST):
        # TODO this ignores numerous ways in which a global import might not be visible,
        # such as when local symbols are created, etc.  In such cases, showing the
        # import in the excerpt is extraneous, but not incorrect.
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                yield n.id

    def get_imports(n: ast.AST):
        # TODO imports inside Class defines name in the class' namespace; they are uncommon
        # Imports within functions are included in the excerpt, so there's no need for us
        # to find them.
        if not isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.iter_child_nodes(n):
                yield from get_imports(child)

        if isinstance(n, (ast.Import, ast.ImportFrom)):
            yield n

    imports = []

    names = set(get_names(node))
    if names:
        for imp in get_imports(module):
            new_imp = None

            for alias in imp.names:
                alias_name = alias.asname if alias.asname else alias.name.split('.')[0]
                if alias_name in names:
                    if not new_imp:
                        new_imp = copy.copy(imp)
                        new_imp.names = []
                        imports.append(new_imp)
                    new_imp.names.append(alias)
                    names.remove(alias_name)

            if not names:
                break

    return imports


def get_info(module: ast.Module, name: str) -> T.Optional[str]:
    """Returns summarized information on a class or function, following imports if necessary."""

    key = name.split('.')

    if (len(key) > 1 and (module_fqn := _get_fqn(module.path)) and
        (common_prefix := _common_prefix_len(module_fqn, key))):
        key = key[common_prefix:]

    while (path := _find_name_path(module, key)) and isinstance(path[-1], (ast.Import, ast.ImportFrom)):
        imp = path[-1]
        if not (module_name := _resolve_import(module.path, imp)):
            return None

        key = key[len(path)-1:]    # currently only relevant for 'import' within class

        # import N
        # import N.x                imports N and N.x
        # import a.b as N           'a.b' is renamed 'N'
        # from a.b import N         either gets symbol N out of a.b, or imports a.b.N as N
        # from a.b import c as N

        if imp.names[0].asname:
            # replace with name out of 'from S import N as A'
            key[0:1] = imp.names[0].name.split('.')

        segments = module_name.split('.')

        if isinstance(imp, ast.ImportFrom):
            segments = segments[imp.level:]

        common_prefix = _common_prefix_len(segments, key)
        _debug(f"{module_name=} {common_prefix=} {key=}")
        key = key[common_prefix:]

        if not (spec := importlib.util.find_spec(module_name)) or not spec.origin:
            return None

        import_file = Path(spec.origin)
        module = parse_file(import_file)
        file = import_file

        # FIXME catch and deal with inclusion loops?

    if path:
        summary = _summarize(path)
        imports = get_global_imports(module, summary)
        return ast.unparse(ast.Module(body=[*imports, summary], type_ignores=[]))

    return None
