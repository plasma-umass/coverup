from pathlib import Path
import typing as T


class TemporaryOverwrite:
    """Context handler that overwrites a file, and restores it upon exit."""
    def __init__(self, file: Path, new_content: str):
        self.file = file
        self.new_content = new_content
        self.backup = file.parent / (file.name + ".bak") if file.exists() else None

    def __enter__(self):
        if self.file.exists():
            self.file.replace(self.backup)

        self.file.write_text(self.new_content)
        self.file.touch()

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.unlink()
        if self.backup:
            self.backup.replace(self.file)


def format_ranges(lines: T.Set[int], negative: T.Set[int]) -> str:
    """Formats sets of line numbers as comma-separated lists, collapsing neighboring lines into ranges
       for brevity."""

    def get_range(lines):
        it = iter(sorted(lines))

        a = next(it, None)
        while a is not None:
            b = a
            while (n := next(it, None)) is not None and not (set(range(b+1,n+1)) & negative):
                b = n

            if a == b:
                yield str(a)
            else:
                yield f"{a}-{b}"

            a = n

    return ", ".join(get_range(lines))


def format_branches(branches):
    for br in sorted(branches):
        yield f"{br[0]}->exit" if br[1] == 0 else f"{br[0]}->{br[1]}"


def lines_branches_do(lines: T.Set[int], neg_lines: T.Set[int], branches: T.Set[T.Tuple[int, int]]) -> str:
    relevant_branches = {b for b in branches if b[0] not in lines and b[1] not in lines} if branches else set()

    s = ''
    if lines:
        s += f"line{'s' if len(lines)>1 else ''} {format_ranges(lines, neg_lines)}"

        if relevant_branches:
            s += " and "

    if relevant_branches:
        s += f"branch{'es' if len(relevant_branches)>1 else ''} "
        s += ", ".join(format_branches(relevant_branches))

    s += " does" if len(lines)+len(relevant_branches) == 1 else " do"
    return s
