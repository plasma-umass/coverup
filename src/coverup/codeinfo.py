import ast
import copy
from pathlib import Path
import typing as T
import importlib.util

_debug = lambda x: None


# TODO use 'ast' alternative that retains comments?

def _package_path(file: Path) -> T.Optional[T.List[str]]:
    """Returns a Python source file's path relative to its package"""
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
                return file.relative_to(p)


def _get_fqn(file: Path) -> T.Optional[T.List[str]]:
    """Returns a source file's Python Fully Qualified Name, as a list name parts."""
    if not (path := _package_path(file)):
        return none

    path = path.parent if path.name == '__init__.py' else path.parent / path.stem
    return path.parts


def _resolve_from_import(file: Path, imp: ast.ImportFrom) -> str:
    """Resolves the module name in a `from X import Y` statement."""

    if imp.level > 0:  # relative import
        if not (pkg_path := _package_path(file)):
            return None

        pkg_path = pkg_path.parts
        if imp.level > len(pkg_path):
            return None # would go beyond top-level package

        return ".".join(pkg_path[:-imp.level]) + (f".{imp.module}" if imp.module else "")

    return imp.module # absolute from ... import 


def _load_module(module_name: str) -> ast.Module | None:
    try:
        if ((spec := importlib.util.find_spec(module_name))
            and spec.origin and spec.origin.endswith('.py')):
            _debug(f"Loading {spec.origin}")
            return parse_file(Path(spec.origin))

    except ModuleNotFoundError:
        pass

    return None


def _auto_stack(func):
    """Decorator that adds a stack of the first argument of the function being called."""
    def helper(*args):
        helper.stack.append(args[0])
        _debug(f"{'.'.join((n.name if getattr(n, 'name', None) else '<' + type(n).__name__ + '>') for n in helper.stack)}")
        retval = func(*args)
        helper.stack.pop()
        return retval
    helper.stack = []
    return helper


def _handle_import(module: ast.Module, node: ast.Import | ast.ImportFrom, name: T.List[str],
                   *, paths_seen: T.Set[Path] = None) -> T.Optional[T.List[ast.AST]]:

    def transition(node: ast.Import | ast.ImportFrom, alias: ast.alias, mod: ast.Module) -> T.List:
        imp = copy.copy(node)
        imp.names = [alias]
        return [imp, mod]

    if isinstance(node, ast.Import):
        # import N
        # import N.x                imports N and N.x
        # import a.b as N           'a.b' is renamed 'N'
        for alias in node.names:
            if alias.asname:
                if alias.asname == name[0]:
                    mod = _load_module(alias.name)
                    if path := _find_name_path(mod, name[1:], paths_seen=paths_seen):
                        return transition(node, alias, mod) + path

            elif (import_name := alias.name.split('.'))[0] == name[0]:
                common_prefix = _common_prefix_len(import_name, name)
                mod = _load_module('.'.join(import_name[:common_prefix]))
                if path := _find_name_path(mod, name[common_prefix:], paths_seen=paths_seen):
                    return transition(node, alias, mod) + path

    elif isinstance(node, ast.ImportFrom):
        # from a.b import N         either gets symbol N out of a.b, or imports a.b.N as N
        # from a.b import c as N

        for alias in node.names:
            if alias.name == '*':
                modname = _resolve_from_import(module.path, node)
                if (mod := _load_module(modname)) and \
                   (path := _find_name_path(mod, name, paths_seen=paths_seen)):
                    return transition(node, alias, mod) + path

            elif (alias.asname if alias.asname else alias.name) == name[0]:
                modname = _resolve_from_import(module.path, node)
                _debug(f"looking for symbol ({[alias.name, *name[1:]]} in {modname})")
                mod = _load_module(modname)
                if path := _find_name_path(mod, [alias.name, *name[1:]], paths_seen=paths_seen):
                    return transition(node, alias, mod) + path

                _debug(f"looking for module ({name[1:]} in {modname}.{alias.name})")
                if (mod := _load_module(f"{modname}.{alias.name}")) and \
                   (path := _find_name_path(mod, name[1:], paths_seen=paths_seen)):
                    return transition(node, alias, mod) + path


def _find_name_path(module: ast.Module, name: T.List[str], *, paths_seen: T.Set[Path] = None) -> T.List[ast.AST]:
    """Looks for a symbol's definition by its name, returning the "path" of ast.ClassDef, ast.Import, etc.,
       crossed to find it.
    """
    if not module: return None
    if not name: return [module]

    _debug(f"looking up {name} in {module.path}")

    if not paths_seen: paths_seen = set()
    if module.path in paths_seen: return None
    paths_seen.add(module.path)

    @_auto_stack
    def find_name(node: ast.AST, name: T.List[str]) -> T.List[ast.AST]:
        _debug(f"_find_name {name} in {ast.dump(node)}")
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name[0]:
                if len(name) == 1:
                    return [node]

                if isinstance(node, ast.ClassDef):
                    for c in node.body:
                        _debug(f"{node.name} checking {ast.dump(c)}")
                        if (path := find_name(c, name[1:])):
                            return [node, *path]

                    for base in node.bases:
                        base_name = ast.unparse(base).split('.')
                        if (len(find_name.stack) > 1 and
                            isinstance(context := find_name.stack[-2], ast.ClassDef)):
                            if (base_path := find_name(context, [context.name, *base_name, *name[1:]])):
                                return base_path[1:]

                        if (path := find_name(module, [*base_name, *name[1:]])):
                            return path

                elif isinstance(module, (ast.Function, ast.AsyncFunction)):
                    # searching within a function in the excerpt
                    for c in node.body:
                        _debug(f"{node.name} checking {ast.dump(c)}")
                        if (path := find_name(c, name[1:])):
                            return [node, *path]

            return []

        if (isinstance(node, ast.Assign) and
            any(isinstance(n, ast.Name) and n.id == name[0] for t in node.targets for n in ast.walk(t))):
            return [node] if len(name) == 1 else []

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (path := _handle_import(module, node, name, paths_seen=paths_seen)):
                return path

            return []

        # not isinstance(...) is just to trim down unnecessary work
        elif not isinstance(node, (ast.Expression, ast.Expr, ast.Name, ast.Attribute, ast.Compare)):
            for c in ast.iter_child_nodes(node):
                if (path := find_name(c, name)):
                    return path

        return []

    return find_name(module, name)


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

    elif isinstance(path[-1], ast.Module):
        path[-1] = copy.deepcopy(path[-1])
        for c in ast.iter_child_nodes(path[-1]):
            if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                c.body = [ast.Expr(ast.Constant(value=ast.literal_eval("...")))]

    # now Class objects
    for i in reversed(range(len(path)-1)):
        if isinstance(path[i], ast.ClassDef):
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
                        if isinstance(new_imp, ast.ImportFrom):
                            # make all imports absolute, as GPT, seeing relative imports in
                            # modules, may want to repeat them in tests as well
                            new_imp.module = _resolve_from_import(module.path, new_imp)
                            new_imp.level = 0

                        new_imp.names = []
                        imports.append(new_imp)
                    new_imp.names.append(alias)
                    names.remove(alias_name)

            if not names:
                break

    return imports


def _find_excerpt(module: ast.Module, line: int) -> ast.AST:
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            begin = min([node.lineno] + [d.lineno for d in node.decorator_list])
            if begin <= line <= node.end_lineno:
                return node


def get_info(module: ast.Module, name: str, *, line: int = 0) -> T.Optional[str]:
    """Returns summarized information on a class or function, following imports if necessary."""

    key = name.split('.')

    if (len(key) > 1 and (module_fqn := _get_fqn(module.path)) and
        (common_prefix := _common_prefix_len(module_fqn, key))):
        key = key[common_prefix:]

    path = None
    # first look in the excerpt node, such as the focal function, if specified
    if (excerpt_node := _find_excerpt(module, line)):
        for c in ast.walk(excerpt_node):
            if (path := _handle_import(module, c, key)):
                break

    if not path:
        # Try looking among globals and classes
        path = _find_name_path(module, key)

    if not path:
        # Couldn't find the name in the context of the given module;
        # try to interpret it as an absolute FQN (GPT asks for that sometimes)
        key = name.split('.')
        for i in range(len(key)-1, 0, -1):
            if (mod := _load_module('.'.join(key[:i]))) and (path := _find_name_path(mod, key[i:])):
                break


    def any_import_as_or_import_in_class() -> bool:
        return any(
            isinstance(n, (ast.Import, ast.ImportFrom)) and (
                n.names[0].asname or (
                    i > 0 and isinstance(path[i-1], ast.ClassDef)
                )
            )
            for i, n in enumerate(path)
        )

    if path:
        _summarize(path)

        for i in range(len(path)):
            _debug(f"path[{i}]={ast.dump(path[i])}")

        if any_import_as_or_import_in_class():
            # include the full path for best context 
            path = [module] + path
        else:
            # just include the last module
            modules = [i for i in range(len(path)) if isinstance(path[i], ast.Module)]
            if modules:
                path = path[modules[-1]:]

        if any(isinstance(n, ast.Module) for n in path):
            result = ""
            for i, mod in enumerate(path):
                if isinstance(mod, ast.Module):
                    content = path[i+1] if i < len(path)-1 else mod
                    # When a module itself is the content, all imports are retained,
                    # so there's no need to look for them.
                    imports = get_global_imports(mod, content) if mod != content else []
                    if result: result += "\n\n"
                    result += f"""\
in {_package_path(mod.path)}:
```python
{ast.unparse(ast.Module(body=[*imports, content], type_ignores=[]))}
```"""
            return result
        else:
            imports = get_global_imports(module, path[0])
            return f"""\
```python
{ast.unparse(ast.Module(body=[*imports, path[0]], type_ignores=[]))}
```"""

    return None
