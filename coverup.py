import asyncio
import openai
import json
import subprocess
import os
from collections import defaultdict
from pathlib import Path
import typing as T
import re
import llm_utils


PREFIX = 'coverup'
CKPT_FILE = PREFIX + "-ckpt.json"
DEFAULT_MODEL='gpt-4-1106-preview'

# Tier 5 rate limits for models; tuples indicate limit and interval in seconds
# Extracted from https://platform.openai.com/account/limits on 11/22/23
MODEL_RATE_LIMITS = {
    'gpt-3.5-turbo': {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-0301': {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-0613': {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-1106': {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-16k':  {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-16k-0613': {
        'token': (1_000_000, 60), 'request': (10_000, 60)
    },
    'gpt-3.5-turbo-instruct': {
        'token': (250_000, 60), 'request': (3_000, 60)
    },
    'gpt-3.5-turbo-instruct-0914': {
        'token': (250_000, 60), 'request': (3_000, 60)
    },
    'gpt-4': {
        'token': (300_000, 60), 'request': (10_000, 60)
    },
    'gpt-4-0314': {
        'token': (300_000, 60), 'request': (10_000, 60)
    },
    'gpt-4-0613': {
        'token': (300_000, 60), 'request': (10_000, 60)
    },
    'gpt-4-1106-preview': {
        'token': (300_000, 60), 'request': (5_000, 60)
    }
}

def token_rate_limit_for_model(model_name: str) -> T.Tuple[int, int]:
    if (model_limits := MODEL_RATE_LIMITS.get(model_name)):
        return model_limits.get('token')

    return None


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(prog='CoverUp',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('cov_json', type=Path,
                    help='SlipCover JSON output file with coverage information')

    ap.add_argument('source_files', type=Path, nargs='*',
                    help='only process certain source file(s)')

    ap.add_argument('--model', type=str, default=DEFAULT_MODEL,
                    help='OpenAI model to use')

    ap.add_argument('--model-temperature', type=str, default=0,
                    help='Model "temperature" to use')

    # TODO derive this somehow?
    ap.add_argument('--tests-dir', type=Path, default='tests',
                    help='directory where tests reside')

    # TODO derive this somehow?
    ap.add_argument('--source-dir', type=Path, default='src',
                    help='directory where sources reside')

    ap.add_argument('--checkpoint', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=f'whether to save progress to {CKPT_FILE}')

    ap.add_argument('--line-limit', type=int, default=50,
                    help='attempt to keep code segment(s) at or below this limit')

    ap.add_argument('--rate-limit', type=int,
                    help='max. tokens/minute to send in prompts')

    ap.add_argument('--max-backoff', type=int, default=64,
                    help='max. number of seconds for backoff interval')

    ap.add_argument('--dry-run', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'whether to actually prompt the model; used for testing')

    ap.add_argument('--show-details', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'show details of lines/branches after each response')

    ap.add_argument('--check-for-side-effects', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=f'whether to check for side effects; requires running the entire suite for each new test')

    ap.add_argument('--log-file', default=f"{PREFIX}-log",
                    help='log file to use')

    ap.add_argument('--pytest-args', type=str, default='',
                    help='extra arguments to pass to pytest')

    ap.add_argument('--install-missing-modules', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='attempt to install any missing modules')

    ap.add_argument('--write-requirements-to', type=Path,
                    help='append the name of any missing modules to the given file')

    return ap.parse_args()


test_seq = 1
def new_test_file():
    """Creates a new test file, returning its Path."""

    global test_seq, args

    while True:
        p = args.tests_dir / f"test_{PREFIX}_{test_seq}.py"
        try:
            p.touch(exist_ok=False)
            return p
        except FileExistsError:
            pass

        test_seq += 1


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


def clean_error(error: str) -> str:
    """Conservatively removes pytest-generated (and possibly other) output not needed by GPT,
       to cut down on token use.  Conservatively: if the format isn't recognized, leave it alone."""

    if (match := re.search("=====+ (?:FAILURES|ERRORS) ===+\n" +\
                           "___+ [^\n]+ _+___\n" +\
                           "\n?" +\
                           "(.*)", error,
                           re.DOTALL)):
        error = match.group(1)

    if (match := re.search("(.*\n)" +\
                           "===+ short test summary info ===+", error,
                           re.DOTALL)):
        error = match.group(1)

    return error


class CodeSegment:
    """Represents a section of code that is missing coverage."""

    def __init__(self, filename: Path, name: str, begin: int, end: int,
                 lines_of_interest: T.Set[int],
                 missing_lines: T.Set[int],
                 executed_lines: T.Set[int],
                 missing_branches: T.Set[T.Tuple[int, int]],
                 context: T.List[T.Tuple[int, int]]):
        self.filename = filename
        self.name = name
        self.begin = begin
        self.end = end
        self.lines_of_interest = lines_of_interest
        self.missing_lines = missing_lines
        self.executed_lines = executed_lines
        self.missing_branches = missing_branches
        self.context = context

    def __repr__(self):
        return f"CodeSegment(\"{self.filename}\", \"{self.name}\", {self.begin}, {self.end}, " + \
               f"{self.missing_lines}, {self.executed_lines}, {self.missing_branches}, {self.context})"


    def identify(self) -> str:
        return f"{self.filename}:{self.begin}-{self.end-1}"

    def get_excerpt(self):
        excerpt = []
        with open(self.filename, "r") as src:
            code = src.readlines()

            for b, e in self.context:
                for i in range(b, e):
                    excerpt.extend([f"{'':10}  ", code[i-1]])

            if not self.executed_lines:
                for i in range(self.begin, self.end):
                    excerpt.extend([f"{'':10}  ", code[i-1]])

            else:
                for i in range(self.begin, self.end):
                    if i in self.lines_of_interest:
                        excerpt.extend([f"{i:10}: ", code[i-1]])
                    else:
                        excerpt.extend([f"{'':10}  ", code[i-1]])

        return ''.join(excerpt)


    def lines_branches_missing_do(self):
        if not self.executed_lines:
            return 'it does'

        return lines_branches_do(self.missing_lines, self.executed_lines, self.missing_branches)

    def missing_count(self) -> int:
        return len(self.missing_lines)+len(self.missing_branches)


log_file = None
def log_write(seg: CodeSegment, m: str) -> None:
    """Writes to the log file, opening it first if necessary."""

    global log_file
    if not log_file:
        log_file = open(args.log_file, "a", buffering=1)    # 1 = line buffered

    log_file.write(f"---- {seg.identify()} ----\n{m}\n")


def measure_coverage(seg: CodeSegment, test: str):
    """Runs a given test and returns the coverage obtained."""
    import sys
    import tempfile
    global args

    with tempfile.NamedTemporaryFile(prefix=PREFIX + "_tmp_", suffix='.py',
                                     dir=str(args.tests_dir), mode="w") as t:
        t.write(test)
        t.flush()

        with tempfile.NamedTemporaryFile(prefix=PREFIX + "_") as j:
            # -qq to cut down on tokens
            p = subprocess.run((f"{sys.executable} -m slipcover --branch --json --out {j.name} " +
                                f"-m pytest {args.pytest_args} -qq {t.name}").split(),
                               check=True, capture_output=True, timeout=60)
            log_write(seg, str(p.stdout, 'UTF-8'))
            cov = json.load(j)

    return cov["files"]


def run_test_suite():
    # throws subprocess.CalledProcessError in case of problems
    subprocess.run((f"{sys.executable} -m pytest {args.pytest_args} {args.tests_dir}").split(),
                   check=True, capture_output=True)


def get_missing_coverage(jsonfile, line_limit = 100) -> T.List[CodeSegment]:
    """Processes a JSON SlipCover output and generates a list of Python code segments,
    such as functions or classes, which have less than 100% coverage.
    """
    import ast
    global args

    with open(jsonfile, "r") as f:
        cov = json.load(f)

    code_segs = []

    def find_first_line(node):
        return min([node.lineno] + [d.lineno for d in node.decorator_list])

    def find_enclosing(root, line):
        for node in ast.walk(root):
            if node is root:
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "lineno"):
                # skip back to include decorators, as they are really part of the definition
                begin = find_first_line(node)
                if begin <= line <= node.end_lineno:
                    return (node, begin, node.end_lineno+1) # +1 for range() style


    for fname, fcov in cov['files'].items():
        with open(fname, "r") as src:
            tree = ast.parse(src.read(), fname)

        missing_lines = set(fcov['missing_lines'])
        executed_lines = set(fcov['executed_lines'])
        missing_branches = fcov.get('missing_branches', set())

        line_ranges = dict()

        lines_of_interest = missing_lines.union(set(sum(missing_branches,[])))
        lines_of_interest.discard(0)  # may result from N->0 branches
        for line in sorted(lines_of_interest):   # sorted() simplifies tests
            if element := find_enclosing(tree, line):
                node, begin, end = element

                context = []

                while isinstance(node, ast.ClassDef) and end - begin > line_limit:
                    if element := find_enclosing(node, line):
                        context.append((begin, node.lineno+1)) # +1 for range() style
                        node, begin, end = element

                    else:
                        end = begin + line_limit
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
                               hasattr(child, "lineno"):
                                end = min(end, find_first_line(child))
                                break

                if line < end and (begin, end) not in line_ranges:
                    # FIXME handle lines >= end (lines between functions, etc.) somehow
                    #print(f"{fname} line {line} -> {node} {begin}..{end}")
                    line_ranges[(begin, end)] = (node, context)

        if line_ranges:
            for (begin, end), (node, context) in line_ranges.items():
                line_range_set = {*range(begin, end)}
                code_segs.append(CodeSegment(fname, node.name, begin, end,
                                             lines_of_interest=lines_of_interest.intersection(line_range_set),
                                             missing_lines=missing_lines.intersection(line_range_set),
                                             executed_lines=executed_lines.intersection(line_range_set),
                                             missing_branches={tuple(b) for b in missing_branches if b[0] in line_range_set},
                                             context=context))

    return code_segs


token_encoding = None
def count_tokens(completion: dict):
    """Counts the number of tokens in a chat completion request."""

    import tiktoken
    global args

    global token_encoding
    if not token_encoding:
        token_encoding = tiktoken.encoding_for_model(args.model)

    count = 0
    for m in completion['messages']:
        count += len(token_encoding.encode(m['content']))

    return count


def compute_cost(usage: dict, model: str) -> float:
    from math import ceil

    if 'prompt_tokens' in usage and 'completion_tokens' in usage:
        try:
            return llm_utils.calculate_cost(usage['prompt_tokens'], usage['completion_tokens'], model)

        except ValueError:
            pass # unknown model

    return None

def find_imports(python_code: str) -> T.List[str]:
    import ast

    try:
        t = ast.parse(python_code)
    except SyntaxError:
        return []

    modules = []

    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for name in n.names:
                if isinstance(name, ast.alias):
                    modules.append(name.name.split('.')[0])

        elif isinstance(n, ast.ImportFrom):
            modules.append(n.module.split('.')[0])

    return modules


module_available = dict()
def missing_imports(modules: T.List[str]) -> T.List[str]:
    import importlib.util

    for module in modules:
        if module not in module_available:
            spec = importlib.util.find_spec(module)
            module_available[module] = 0 if spec is None else 1

    return [m for m in modules if not module_available[m]]


def install_missing_imports(seg: CodeSegment, modules: T.List[str]) -> bool: 
    all_ok = True
    for module in modules:
        try:
            # FIXME we probably want to limit the module(s) installed to an "approved" list
            p = subprocess.run((f"{sys.executable} -m pip install {module}").split(),
                               check=True, capture_output=True, timeout=60)
            module_available[module] = 2    # originally unavailable, but now added
            print(f"Installed module {module}")
            log_write(seg, f"Installed module {module}")
        except subprocess.CalledProcessError as e:
            log_write(seg, f"Unable to install module {module}:\n{e.output}")
            all_ok = False

    return all_ok


def get_required_modules() -> T.List[str]:
    """Returns a list of the modules originally missing (i.e., even if we installed them)"""
    return [m for m in module_available if module_available[m] != 1]


def get_module_name(src_file: Path, src_dir: Path) -> str:
    try:
        src_file = Path(src_file)
        src_dir = Path(src_dir)
        relative = src_file.resolve().relative_to(src_dir.resolve())
        return ".".join((src_dir.stem,) + relative.parts[:-1] + (relative.stem,))
    except ValueError:
        return None  # not relative to source


token_rate_limit = None
async def do_chat(seg: CodeSegment, completion: dict) -> str:
    """Sends a GPT chat request, handling common failures and returning the response."""

    global token_rate_limit

    sleep = 1
    while True:
        try:
            if token_rate_limit:
                await token_rate_limit.acquire(count_tokens(completion))

            return await openai.ChatCompletion.acreate(**completion)

        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.Timeout) as e:
            import random
            sleep = min(sleep*2, args.max_backoff)
            sleep_time = random.uniform(sleep/2, sleep)
            print(f"{str(e)}; waiting {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)

        except openai.error.InvalidRequestError as e:
            # usually "maximum context length" XXX check for this?
            log_write(seg, f"Received {e}")
            print(e)
            return None

        except (openai.error.APIConnectionError,
                openai.error.APIError) as e:
            # usually a server-side error... just retry
            print(e)


class Progress:
    """Tracks progress, showing a tqdm-based bar."""

    def __init__(self, total, initial, usage):
        import tqdm
        from collections import OrderedDict

        self.postfix = OrderedDict()
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.postfix['usage'] = ''
        self.postfix['G'] = 0
        self.postfix['F'] = 0
        self.postfix['U'] = 0

        self.bar = tqdm.tqdm(total=total, initial=initial)
        self.bar.set_postfix(ordered_dict=self.postfix)

        if usage: self.add_usage(usage)

    def add_usage(self, usage: dict):
        """Signals more tokens were used."""
        for k in self.usage:
            self.usage[k] += usage[k]
        cost = compute_cost(self.usage, args.model)
        self.postfix['usage'] = f'{self.usage["prompt_tokens"]}+{self.usage["completion_tokens"]}' + \
                                (f' (~${cost:.02f})' if cost is not None else '')
        self.bar.set_postfix(ordered_dict=self.postfix)

    def add_failing(self):
        """Signals a failing test."""
        self.postfix['F'] += 1
        self.bar.set_postfix(ordered_dict=self.postfix)

    def add_useless(self):
        """Signals an useless test (that doesn't increase coverage)."""
        self.postfix['U'] += 1
        self.bar.set_postfix(ordered_dict=self.postfix)

    def add_good(self):
        """Signals a 'good', useful test was found."""
        self.postfix['G'] += 1
        self.bar.set_postfix(ordered_dict=self.postfix)

    def update(self):
        """Signals an item completed."""
        self.bar.update()

    def close(self):
        """Closes the underlying tqdm bar."""
        self.bar.close()


def extract_python(response: str) -> str:
    # This regex accepts a truncated code block... this seems fine since we'll try it anyway
    m = re.search('^```python\n(.*?\n)(```\n?)?$', response, re.M|re.S)
    if not m: raise RuntimeError(f"Unable to extract Python code from response {response}")
    return m.group(1)


progress = None
async def improve_coverage(seg: CodeSegment) -> bool:
    """Works to improve coverage for a code segment."""
    global args, progress

    def pl(item, singular, plural = None):
        if len(item) <= 1:
            return singular
        return plural if plural is not None else f"{singular}s"

    module_name = get_module_name(seg.filename, args.source_dir)

    messages = [{"role": "user",
                 "content": f"""
The code below, extracted from {seg.filename},{' module ' + module_name + ',' if module_name else ''} does not achieve full coverage:
when tested, {seg.lines_branches_missing_do()} not execute.
Create a new pytest test function that executes these missing lines/branches, always making
sure that the new test is correct and indeed improves coverage.
Always send entire Python test scripts when proposing a new test or correcting one you
previously proposed.
Be sure to include assertions in the test that verify any applicable postconditions.
Please also make VERY SURE to clean up after the test, so as not to affect tests ran after it
in the same pytest execution.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{seg.get_excerpt()}
```
"""
            }]

    log_write(seg, messages[0]['content'])  # initial prompt

    attempts = 0

    if args.dry_run:
        return True

    while True:
        if (attempts > 5):
            log_write(seg, "Too many attempts, giving up")
            break

        attempts += 1

        if not (response := await do_chat(seg, {'model': args.model, 'messages': messages,
                                                'temperature': args.model_temperature})):
            log_write(seg, "giving up")
            break

        response_message = response["choices"][0]["message"]
        log_write(seg, response_message['content'])

        progress.add_usage(response['usage'])
        log_write(seg, f"total usage: {progress.usage}")

        messages.append(response_message)

        if '```python' in response_message['content']:
            last_test = extract_python(response_message['content'])
        else:
            log_write(seg, "No Python code in GPT response, giving up")
            break

        if missing := missing_imports(find_imports(last_test)):
            if not args.install_missing_modules or not install_missing_imports(seg, missing):
                return False # not finished: needs a missing module

        try:
            result = measure_coverage(seg, last_test)

            new_lines = set(result[seg.filename]['executed_lines']) if seg.filename in result else set()
            new_branches = set(tuple(b) for b in result[seg.filename]['executed_branches']) \
                           if seg.filename in result else set()
            now_missing_lines = seg.missing_lines - new_lines
            now_missing_branches = seg.missing_branches - new_branches

            if args.show_details:
                print(seg.identify())
                print(f"Originally missing: {list(seg.missing_lines)}")
                print(f"                    {list(seg.missing_branches)}")
                print(f"Still missing:      {list(now_missing_lines)}")
                print(f"                    {list(now_missing_branches)}")

            # XXX insist on len(now_missing_lines)+len(now_missing_branches) == 0 ?
            if len(now_missing_lines)+len(now_missing_branches) == seg.missing_count():
                messages.append({
                    "role": "user",
                    "content": f"""
This test still lacks coverage: {lines_branches_do(now_missing_lines, set(), now_missing_branches)} not execute.
Modify it to correct that; respond only with the complete Python code in backticks.
"""
                })
                log_write(seg, messages[-1]['content'])
                progress.add_useless()
                continue

            # the test is good 'nough...
            new_test = new_test_file()
            new_test.write_text(f"# file {seg.identify()}\n" +\
                                f"# lines {sorted(seg.missing_lines)}\n" +\
                                f"# branches {list(format_branches(seg.missing_branches))}\n\n" +\
                                last_test)

            if args.check_for_side_effects:
                try:
                    run_test_suite()
                except subprocess.CalledProcessError as e:
                    progress.add_failing()
                    print(f"Test for {seg.identify()} has side effects")
                    new_test.unlink()
                    messages.append({
                        "role": "user",
                        "content": "Executing the test along with the rest of the test suite yields an error, shown below.\n" +\
                                   "Modify the test to correct it; respond only with the complete Python code in backticks.\n\n" +\
                                   clean_error(str(e.output, 'UTF-8'))
                    })
                    log_write(seg, messages[-1]['content'])
                    continue

            log_write(seg, f"Saved as {new_test}\n")
            progress.add_good()
            break

        except subprocess.TimeoutExpired:
            log_write(seg, "measure_coverage timed out, giving up")
            break

        except subprocess.CalledProcessError as e:
            progress.add_failing()
            messages.append({
                "role": "user",
                "content": "Executing the test yields an error, shown below.\n" +\
                           "Modify the test to correct it; respond only with the complete Python code in backticks.\n\n" +\
                           clean_error(str(e.output, 'UTF-8'))
            })
            log_write(seg, messages[-1]['content'])

    return True # finished


if __name__ == "__main__":
    import sys
    args = parse_args()

    if args.rate_limit or token_rate_limit_for_model(args.model):
        limit = (args.rate_limit, 60) if args.rate_limit else token_rate_limit_for_model(args.model)
        from aiolimiter import AsyncLimiter
        token_rate_limit = AsyncLimiter(*limit)
        # TODO also add request limit, and use 'await asyncio.gather(t.acquire(tokens), r.acquire())' to acquire both

    if not args.tests_dir.exists():
        print(f'Directory "{args.tests_dir}" does not exist. Please specify the correct one or create it.')
        sys.exit(1)

    if 'OPENAI_API_KEY' not in os.environ:
        print("Please place your OpenAI key in an environment variable named OPENAI_API_KEY and try again.")
        sys.exit(1)

    try:
        run_test_suite()
    except subprocess.CalledProcessError:
        print("Running the test suite yields errors; please correct or silence them before running CoverUp.")
        sys.exit(1)

    openai.key=os.environ['OPENAI_API_KEY']
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization=os.environ['OPENAI_ORGANIZATION']

    segments = sorted(get_missing_coverage(args.cov_json, line_limit=args.line_limit),
                      key=lambda seg: seg.missing_count(), reverse=True)

    checkpoint_file = Path(CKPT_FILE)
    done : T.Dict[str, T.Set[T.Tuple[int, int]]] = defaultdict(set)

    usage = None
    if args.checkpoint:
        try:
            with checkpoint_file.open("r") as f:
                ckpt = json.load(f)
                assert ckpt['version'] == 1
                for filename, done_list in ckpt['done'].items():
                    done[filename] = set(tuple(d) for d in done_list)
                usage = ckpt['usage']
        except json.decoder.JSONDecodeError:
            pass
        except FileNotFoundError:
            pass

    async def work_segment(seg: CodeSegment) -> None:
        if await improve_coverage(seg):
            # Only mark done if was able to complete, so that it can be retried
            # after installing any missing modules
            done[seg.filename].add((seg.begin, seg.end))

            if args.checkpoint:
                with checkpoint_file.open("w") as f:
                    json.dump({
                        'version': 1,
                        'done': {k:list(v) for k,v in done.items()},    # cannot serialize sets as-is
                        'usage': progress.usage
                    }, f)

        progress.update()

    worklist = []
    seg_done_count = 0
    for seg in segments:
        if not args.source_dir in Path(seg.filename).parents:
            continue

        if args.source_files and all(seg.filename not in str(s) for s in args.source_files):
            continue

        if (seg.begin, seg.end) in done[seg.filename]:
            seg_done_count += 1
        else:
            worklist.append(work_segment(seg))

    async def runit():
        await asyncio.gather(*worklist)

    progress = Progress(total=len(worklist)+seg_done_count, initial=seg_done_count, usage=usage)
    asyncio.run(runit())

    progress.close()

    if required := get_required_modules():
        # Sometimes GPT outputs 'from your_module import XYZ', asking us to modify
        print(f"Some modules seem missing:  {', '.join(str(m) for m in required)}")
        if args.write_requirements_to:
            with args.write_requirements_to.open("a") as f:
                for module in required:
                    f.write(f"{module}\n")
