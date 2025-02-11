import ast
import copy
from pathlib import Path
import typing as T
import importlib.util

_debug = lambda x: None


class Module(ast.Module):
    def __init__(self, original: ast.Module, path: Path):
        super().__init__(original.body, original.type_ignores)
        self.path = path

    def __reduce__(self):
        # for pickle/deepcopy
        return (self.__class__, (ast.Module(self.body, self.type_ignores), self.path))


# TODO use 'ast' alternative that retains comments?

def _package_path(file: Path) -> Path|None:
    """Returns a Python source file's path relative to its package"""
    import sys

    if not file.is_absolute():
        file = file.resolve()

    parents = list(file.parents)

    path: str|Path
    for path in sys.path:
        path = Path(path)
        if not path.is_absolute():
            path = path.resolve()

        for p in parents:
            if p == path:
                return file.relative_to(p)

    return None


def _get_fqn(file: Path) -> T.Sequence[str]|None:
    """Returns a source file's Python Fully Qualified Name, as a list name parts."""
    if not (path := _package_path(file)):
        return None

    path = path.parent if path.name == '__init__.py' else path.parent / path.stem
    return path.parts


def _resolve_from_import(file: Path, imp: ast.ImportFrom) -> str|None:
    """Resolves the module name in a `from X import Y` statement."""

    if imp.level > 0:  # relative import
        if not (pkg_path := _package_path(file)):
            return None

        pkg_path_parts = pkg_path.parts
        if imp.level > len(pkg_path_parts):
            return None # would go beyond top-level package

        return ".".join(pkg_path_parts[:-imp.level]) + (f".{imp.module}" if imp.module else "")

    return imp.module # absolute from ... import 


def _load_module(module_name: str) -> Module | None:
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


def _handle_import(
    module: Module,
    node: ast.AST,
    name: list[str],
    *,
    paths_seen: set[Path]|None = None
) -> list[Module|ast.stmt] | None:

    def transition(
        node: ast.Import | ast.ImportFrom,
        alias: ast.alias,
        mod: Module
    ) -> list[Module|ast.stmt]:
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
                        # _find_name_path returns None if mod is None
                        return transition(node, alias, T.cast(Module, mod)) + path

            elif (import_name := alias.name.split('.'))[0] == name[0]:
                common_prefix = _common_prefix_len(import_name, name)
                mod = _load_module('.'.join(import_name[:common_prefix]))
                if path := _find_name_path(mod, name[common_prefix:], paths_seen=paths_seen):
                    return transition(node, alias, T.cast(Module, mod)) + path

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
                    return transition(node, alias, T.cast(Module, mod)) + path

                _debug(f"looking for module ({name[1:]} in {modname}.{alias.name})")
                if (mod := _load_module(f"{modname}.{alias.name}")) and \
                   (path := _find_name_path(mod, name[1:], paths_seen=paths_seen)):
                    return transition(node, alias, T.cast(Module, mod)) + path

    return None


def _find_name_path(
    module: Module|None,
    name: list[str],
    *,
    paths_seen: set[Path]|None = None
) -> list[Module|ast.stmt]|None:
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
    def find_name(node: ast.AST, name: list[str]) -> list[Module|ast.stmt]:
        _debug(f"_find_name {name} in {ast.dump(node)}")
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name[0]:
                if len(name) == 1:
                    return [node]

                if isinstance(node, ast.ClassDef):
                    for stmt in node.body:
                        _debug(f"{node.name} checking {ast.dump(stmt)}")
                        if (path := find_name(stmt, name[1:])):
                            return [node, *path]

                    for base in node.bases:
                        base_name = ast.unparse(base).split('.')
                        if (len(find_name.stack) > 1 and
                            isinstance(context := find_name.stack[-2], ast.ClassDef)):
                            if (base_path := find_name(context, [context.name, *base_name, *name[1:]])):
                                return base_path[1:]

                        if (path := find_name(module, [*base_name, *name[1:]])):
                            return path

            return []

        if (isinstance(node, ast.Assign) and
            any(isinstance(n, ast.Name) and n.id == name[0] for t in node.targets for n in ast.walk(t))):
            return [node] if len(name) == 1 else []

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (path := _handle_import(T.cast(Module, module), node, name, paths_seen=paths_seen)):
                return path

            return []

        # not isinstance(...) is just to trim down unnecessary work
        elif not isinstance(node, (ast.Expression, ast.Expr, ast.Name, ast.Attribute, ast.Compare)):
            for c in ast.iter_child_nodes(node):
                if (path := find_name(c, name)):
                    return path

        return []

    return find_name(module, name)


def _summarize(path: list[Module|ast.stmt]) -> Module|ast.stmt:
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

    elif isinstance(path[-1], Module):
        path[-1] = copy.deepcopy(path[-1])
        for c in ast.iter_child_nodes(path[-1]):
            if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                c.body = [ast.Expr(ast.Constant(value=ast.literal_eval("...")))]

    # now Class objects
    for i in reversed(range(len(path)-1)):
        if isinstance(path[i], ast.ClassDef):
            path[i] = copy.copy(path[i])
            path_i = T.cast(ast.ClassDef, path[i])
            path_i.body = [
                ast.Expr(ast.Constant(value=ast.literal_eval("..."))),
                T.cast(ast.stmt, path[i+1])
            ]

    return path[0]


def parse_file(file: Path) -> Module:
    """Reads a python source file, annotating it with its path/filename."""
    with file.open("r") as f:
        tree = ast.parse(f.read())

    return Module(tree, file)


def _common_prefix_len(a: T.Sequence[str], b: T.Sequence[str]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def get_global_imports(module: Module, node: ast.AST) -> T.Sequence[ast.Import | ast.ImportFrom]:
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


def _find_excerpt(node: ast.AST, line: int) -> ast.AST|None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        begin = min([node.lineno] + [d.lineno for d in node.decorator_list])
        if begin <= line:
            return node

    for c in ast.iter_child_nodes(node):
        if (excerpt := _find_excerpt(c, line)):
            return excerpt

    return None


def get_info(module: Module, name: str, *, line: int = 0, generate_imports: bool = True) -> T.Optional[str]:
    """Returns summarized information on a class or function, following imports if necessary."""

    key = name.split('.')

    if (len(key) > 1 and (module_fqn := _get_fqn(module.path)) and
        (common_prefix := _common_prefix_len(module_fqn, key))):
        key = key[common_prefix:]

    path: list[Module|ast.stmt]|None = None
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


    def any_import_as_or_import_in_class(path: T.Sequence[Module|ast.stmt]) -> bool:
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

        if any_import_as_or_import_in_class(path):
            # include the full path for best context 
            path.insert(0, module)
        else:
            # just include the last module
            modules = [i for i in range(len(path)) if isinstance(path[i], ast.Module)]
            if modules:
                path = path[modules[-1]:]

        if not any(isinstance(n, ast.Module) for n in path):
            # no module in the path: we stay within the same file
            content = T.cast(ast.stmt, path[0])
            imports = get_global_imports(module, content) if generate_imports else []
            return f"""\
```python
{ast.unparse(ast.Module(body=[*imports, content], type_ignores=[]))}
```"""

        # there's a module in the path: build 'result' with all file transitions
        result = ""
        for i, item in enumerate(path):
            if isinstance(item, ast.Module):
                mod = item
                if i >= len(path)-1: # ends in a module
                    # When a module itself is the content, all imports are retained,
                    # so there's no need to look for them.
                    if result: result += "\n\n"
                    result += f"""\
in {_package_path(mod.path)}:
```python
{ast.unparse(mod)}
```"""
                else:
                    # continues to some content within a module
                    content = T.cast(ast.stmt, path[i+1])
                    imports = get_global_imports(mod, content) if generate_imports else []
                    if result: result += "\n\n"
                    result += f"""\
in {_package_path(mod.path)}:
```python
{ast.unparse(ast.Module(body=[*imports, content], type_ignores=[]))}
```"""
        return result

    return None
