from pathlib import Path
import typing as T
import subprocess


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


async def subprocess_run(args: T.Sequence[str], check: bool = False, timeout: T.Optional[int] = None) -> subprocess.CompletedProcess:
    """Provides an asynchronous version of subprocess.run"""
    import asyncio

    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.STDOUT)

    try:
        if timeout is not None:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        else:
            output, _ = await process.communicate()

    except (asyncio.TimeoutError, asyncio.exceptions.TimeoutError):
        try:
            process.terminate()
            await process.wait()
        except ProcessLookupError:
            pass

        if timeout:
            timeout_f = float(timeout)
        else:
            timeout_f = 0.0
        raise subprocess.TimeoutExpired(args, timeout_f) from None

    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args, output=output)

    # process.returncode is None iff the process hasn't terminated yet
    return subprocess.CompletedProcess(args=args, returncode=T.cast(int, process.returncode), stdout=output)


def summary_coverage(cov: dict, sources: T.List[Path]) -> str:
    """Returns the summary coverage, limiting it to the source files if any are given."""

    if sources:
        import copy
        from slipcover.slipcover import add_summaries

        filtered = {
            'meta': cov['meta'],
            'files': {}
        }

        for f in cov['files']:
            resolved = Path(f).resolve()
            if resolved in sources:
                filtered['files'][str(resolved)] = copy.deepcopy(cov['files'][f])

        add_summaries(filtered)
        cov = filtered

    return f"{cov['summary']['percent_covered']:.1f}%"
