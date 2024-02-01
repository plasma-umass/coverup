import asyncio
import openai
import json
import subprocess
from pathlib import Path
import typing as T
import re
import sys
from datetime import datetime
from coverup.llm import *
from coverup.segment import *
from coverup.testrunner import *


PREFIX = 'coverup'
DEFAULT_MODEL='gpt-4-1106-preview'


def parse_args(args=None):
    import argparse

    ap = argparse.ArgumentParser(prog='CoverUp',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('source_files', type=Path, nargs='*',
                    help='only process certain source file(s)')

    ap.add_argument('--tests-dir', type=Path, required=True,
                    help='directory where tests reside')

    ap.add_argument('--source-dir', '--source', type=Path, required=True,
                    help='directory where sources reside')

    ap.add_argument('--checkpoint', type=Path, 
                    help=f'path to save progress to (and to resume it from)')
    ap.add_argument('--no-checkpoint', action='store_const', const=None, dest='checkpoint', default=argparse.SUPPRESS,
                    help=f'disables checkpoint')

    ap.add_argument('--model', type=str, default=DEFAULT_MODEL,
                    help='OpenAI model to use')

    ap.add_argument('--model-temperature', type=str, default=0,
                    help='Model "temperature" to use')

    ap.add_argument('--line-limit', type=int, default=50,
                    help='attempt to keep code segment(s) at or below this limit')

    ap.add_argument('--rate-limit', type=int,
                    help='max. tokens/minute to send in prompts')

    ap.add_argument('--max-attempts', type=int, default=3,
                    help='max. number of prompt attempts for a code segment')

    ap.add_argument('--max-backoff', type=int, default=64,
                    help='max. number of seconds for backoff interval')

    ap.add_argument('--dry-run', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'whether to actually prompt the model; used for testing')

    ap.add_argument('--show-details', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'show details of lines/branches after each response')

    ap.add_argument('--log-file', default=f"{PREFIX}-log",
                    help='log file to use')

    ap.add_argument('--pytest-args', type=str, default='',
                    help='extra arguments to pass to pytest')

    ap.add_argument('--install-missing-modules', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='attempt to install any missing modules')

    ap.add_argument('--write-requirements-to', type=Path,
                    help='append the name of any missing modules to the given file')

    ap.add_argument('--only-disable-interfering-tests', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='rather than try to add new tests, only look for tests causing others to fail and disable them.')

    ap.add_argument('--debug', '-d', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='print out debugging messages.')

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 0: raise argparse.ArgumentTypeError("must be a number >= 0")
        return ivalue

    ap.add_argument('--max-concurrency', type=positive_int, default=50,
                    help='maximum number of parallel requests; 0 means unlimited')

    return ap.parse_args(args)


def test_file_path(test_seq: int) -> Path:
    """Returns the Path for a test's file, given its sequence number."""
    return args.tests_dir / f"test_{PREFIX}_{test_seq}.py"


test_seq = 1
def new_test_file():
    """Creates a new test file, returning its Path."""

    global test_seq, args

    while True:
        p = test_file_path(test_seq)
        if not (p.exists() or (p.parent / ("disabled_" + p.name)).exists()):
            try:
                p.touch(exist_ok=False)
                return p
            except FileExistsError:
                pass

        test_seq += 1


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


log_file = None
def log_write(seg: CodeSegment, m: str) -> None:
    """Writes to the log file, opening it first if necessary."""

    global log_file
    if not log_file:
        log_file = open(args.log_file, "a", buffering=1)    # 1 = line buffered

    log_file.write(f"---- {datetime.now().isoformat(timespec='seconds')} {seg} ----\n{m}\n")


def disable_interfering_tests() -> dict:
    """While the test suite fails, disables any interfering tests.
       If the test suite succeeds, returns the coverage observed."""

    while True:
        print("Checking test suite...  ", end='')
        try:
            coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.source_dir,
                                              pytest_args=args.pytest_args)
            print("tests ok!")
            return coverage

        except subprocess.CalledProcessError as e:
            failing_test = parse_failed_test(args.tests_dir, e)

        btf = BadTestsFinder(tests_dir=args.tests_dir, pytest_args=args.pytest_args,
                             trace=(print if args.debug else None))

        print(f"{failing_test} is failing, looking for culprit(s)...")

        if btf.run_tests({failing_test}) is not None:
            print(f"{failing_test} fails by itself(!)")
            culprits = {failing_test}
        else:
            test_set = None
            while True:
                try:
                    culprits = btf.find_culprit(failing_test, test_set=test_set)
                    break
                except EarlierFailureException as ef:
                    print(f"{ef.failed} also fails, looking into that first...")
                    failing_test = ef.failed
                    test_set = ef.test_set

        for c in culprits:
            print(f"Disabling {c}")
            c.rename(c.parent / ("disabled_" + c.name))


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
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            module_available[module] = 2    # originally unavailable, but now added
            print(f"Installed module {module}")
            log_write(seg, f"Installed module {module}")
        except subprocess.CalledProcessError as e:
            log_write(seg, f"Unable to install module {module}:\n{e.stdout}")
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


PROGRESS_COUNTERS=['G', 'F', 'U', 'R']  # good, failed, useless, retry
class Progress:
    """Tracks progress, showing a tqdm-based bar."""

    def __init__(self, total, initial):
        import tqdm
        from collections import OrderedDict

        self.bar = tqdm.tqdm(total=total, initial=initial)

        self.postfix = OrderedDict()
        for p in ['usage', *PROGRESS_COUNTERS]:
            self.postfix[p] = ''  # to establish order

        self.bar.set_postfix(ordered_dict=self.postfix)

    def update_usage(self, usage: dict):
        """Updates the usage display."""
        if (cost := compute_cost(usage, args.model)) is not None:
            self.postfix['usage'] = f'~${cost:.02f}'
        else:
            self.postfix['usage'] = f'{usage["prompt_tokens"]}+{usage["completion_tokens"]}'

        self.bar.set_postfix(ordered_dict=self.postfix)

    def update_counters(self, counters: dict):
        """Updates the counters display."""
        for k in counters:
            self.postfix[k] = counters[k]

        self.bar.set_postfix(ordered_dict=self.postfix)

    def signal_one_completed(self):
        """Signals an item completed."""
        self.bar.update()

    def close(self):
        """Closes the underlying tqdm bar."""
        self.bar.close()


class State:
    def __init__(self, initial_coverage: dict):
        """Initializes the CoverUp state."""
        from collections import defaultdict

        self.done : T.Dict[str, T.Set[T.Tuple[int, int]]] = defaultdict(set)
        self.coverage = initial_coverage
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.counters = {k:0 for k in PROGRESS_COUNTERS}
        self.bar = None


    def get_initial_coverage(self) -> dict:
        """Returns the coverage initially measured."""
        return self.coverage


    def set_progress_bar(self, bar: Progress):
        """Specifies a progress bar to update."""
        self.bar = bar
        if bar is not None:
            self.bar.update_usage(self.usage)
            self.bar.update_counters(self.counters)


    def add_usage(self, usage: dict):
        """Signals more tokens were used."""
        for k in self.usage:
            self.usage[k] += usage[k]

        if self.bar:
            self.bar.update_usage(self.usage)


    def inc_counter(self, key: str):
        """Increments a progress counter."""
        self.counters[key] += 1

        if self.bar:
            self.bar.update_counters(self.counters)


    def mark_done(self, seg: CodeSegment):
        """Marks a segment done."""
        self.done[seg.filename].add((seg.begin, seg.end))


    def is_done(self, seg: CodeSegment):
        """Returns whether a segment is done."""
        return (seg.begin, seg.end) in self.done[seg.filename]


    @staticmethod
    def load_checkpoint(ckpt_file: Path):   # -> State
        """Loads state from a checkpoint."""
        try:
            with ckpt_file.open("r") as f:
                ckpt = json.load(f)
                assert ckpt['version'] == 1
                assert 'usage' in ckpt
                assert 'coverage' in ckpt, "coverage information missing in checkpoint"

                state = State(ckpt['coverage'])
                for filename, done_list in ckpt['done'].items():
                    state.done[filename] = set(tuple(d) for d in done_list)
                state.add_usage(ckpt['usage'])
                if 'counters' in ckpt:
                    state.counters = ckpt['counters']
                return state

        except FileNotFoundError:
            return None


    def save_checkpoint(self, ckpt_file: Path):
        """Saves this state to a checkpoint file."""
        with ckpt_file.open("w") as f:
            json.dump({
                'version': 1,
                'done': {k:list(v) for k,v in self.done.items() if len(v)},  # cannot serialize 'set' as-is
                'usage': self.usage,
                'counters': self.counters,
                'coverage': self.coverage
                # FIXME save missing modules
            }, f)


state = None
token_rate_limit = None
async def do_chat(seg: CodeSegment, completion: dict) -> str:
    """Sends a GPT chat request, handling common failures and returning the response."""

    global token_rate_limit

    sleep = 1
    while True:
        try:
            if token_rate_limit:
                await token_rate_limit.acquire(count_tokens(args.model, completion))

            return await openai.ChatCompletion.acreate(**completion)

        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.Timeout) as e:

            # This message usually indicates out of money in account
            if 'You exceeded your current quota' in str(e):
                log_write(seg, f"Failed: {type(e)} {e}")
                raise e

            log_write(seg, f"Error: {type(e)} {e}")

            import random
            sleep = min(sleep*2, args.max_backoff)
            sleep_time = random.uniform(sleep/2, sleep)
            state.inc_counter('R')
            await asyncio.sleep(sleep_time)

        except openai.error.InvalidRequestError as e:
            # usually "maximum context length" XXX check for this?
            log_write(seg, f"Error: {type(e)} {e}")
            return None # gives up this segment

        except (openai.error.APIConnectionError,
                openai.error.APIError) as e:
            log_write(seg, f"Error: {type(e)} {e}")
            # usually a server-side error... just retry right away
            state.inc_counter('R')


def extract_python(response: str) -> str:
    # This regex accepts a truncated code block... this seems fine since we'll try it anyway
    m = re.search(r'```python\n(.*?)(?:```|\Z)', response, re.DOTALL)
    if not m: raise RuntimeError(f"Unable to extract Python code from response {response}")
    return m.group(1)


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
        attempts += 1
        if (attempts > args.max_attempts):
            log_write(seg, "Too many attempts, giving up")
            break

        if not (response := await do_chat(seg, {'model': args.model, 'messages': messages,
                                                'temperature': args.model_temperature})):
            log_write(seg, "giving up")
            break

        response_message = response["choices"][0]["message"]
        log_write(seg, response_message['content'])

        state.add_usage(response['usage'])
        log_write(seg, f"total usage: {state.usage}")

        messages.append(response_message)

        if '```python' in response_message['content']:
            last_test = extract_python(response_message['content'])
        else:
            log_write(seg, "No Python code in GPT response, giving up")
            break

        if missing := missing_imports(find_imports(last_test)):
            log_write(seg, f"Missing modules {' '.join(missing)}")
            if not args.install_missing_modules or not install_missing_imports(seg, missing):
                return False # not finished: needs a missing module

        try:
            result = measure_coverage(test=last_test, tests_dir=args.tests_dir, pytest_args=args.pytest_args,
                                      log_write=lambda msg: log_write(seg, msg))

        except subprocess.TimeoutExpired:
            log_write(seg, "measure_coverage timed out")
            return False # try again next time

        except subprocess.CalledProcessError as e:
            state.inc_counter('F')
            messages.append({
                "role": "user",
                "content": "Executing the test yields an error, shown below.\n" +\
                           "Modify the test to correct it; respond only with the complete Python code in backticks.\n\n" +\
                           clean_error(str(e.stdout, 'UTF-8'))
            })
            log_write(seg, messages[-1]['content'])
            continue

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
            state.inc_counter('U')
            continue

        # the test is good 'nough...
        new_test = new_test_file()
        new_test.write_text(f"# file {seg.identify()}\n" +\
                            f"# lines {sorted(seg.missing_lines)}\n" +\
                            f"# branches {list(format_branches(seg.missing_branches))}\n\n" +\
                            last_test)

        log_write(seg, f"Saved as {new_test}\n")
        state.inc_counter('G')
        break

    return True # finished


def add_to_pythonpath(source_dir: Path):
    import os
    parent = str(source_dir.parent)
    os.environ['PYTHONPATH'] = parent + (f":{os.environ['PYTHONPATH']}" if 'PYTHONPATH' in os.environ else "")
    sys.path.insert(0, parent)


def main():
    from collections import defaultdict
    import os

    global args, token_rate_limit, state
    args = parse_args()

    if not args.tests_dir.exists():
        print(f'Directory "{args.tests_dir}" does not exist. Please specify the correct one or create it.')
        return 1

    # add source dir to paths so that the module doesn't need to be installed to be worked on
    add_to_pythonpath(args.source_dir)

    if args.only_disable_interfering_tests:
        disable_interfering_tests()
        return

    if args.rate_limit or token_rate_limit_for_model(args.model):
        limit = (args.rate_limit, 60) if args.rate_limit else token_rate_limit_for_model(args.model)
        from aiolimiter import AsyncLimiter
        token_rate_limit = AsyncLimiter(*limit)
        # TODO also add request limit, and use 'await asyncio.gather(t.acquire(tokens), r.acquire())' to acquire both

    if 'OPENAI_API_KEY' not in os.environ:
        print("Please place your OpenAI key in an environment variable named OPENAI_API_KEY and try again.")
        return 1

    openai.key=os.environ['OPENAI_API_KEY']
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization=os.environ['OPENAI_ORGANIZATION']

    log_write('startup', f"Command: {' '.join(sys.argv)}")

    # --- (1) load or measure initial coverage, figure out segmentation ---

    if args.checkpoint and (state := State.load_checkpoint(args.checkpoint)):
        print("Continuing from checkpoint;  ", end='')
    else:
        try:
            print("Measuring test suite coverage...  ", end='')
            coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.source_dir,
                                              pytest_args=args.pytest_args)
        except subprocess.CalledProcessError as e:
            print("Error measuring coverage:\n" + str(e.stdout, 'UTF-8'))
            return 1

        state = State(coverage)

    coverage = state.get_initial_coverage()

    print(f"initial coverage: {coverage['summary']['percent_covered']:.1f}%")
    # TODO also show running coverage estimate

    segments = sorted(get_missing_coverage(state.get_initial_coverage(), line_limit=args.line_limit),
                      key=lambda seg: seg.missing_count(), reverse=True)

    # save initial coverage so we don't have to redo it next time
    if args.checkpoint:
        state.save_checkpoint(args.checkpoint)

    # --- (2) prompt for tests ---

    print(f"Prompting {args.model} for tests to increase coverage...")

    async def work_segment(seg: CodeSegment) -> None:
        if await improve_coverage(seg):
            # Only mark done if was able to complete (True return),
            # so that it can be retried after installing any missing modules
            state.mark_done(seg)

        if args.checkpoint:
            state.save_checkpoint(args.checkpoint)
        progress.signal_one_completed()

    worklist = []
    seg_done_count = 0
    for seg in segments:
        if not args.source_dir in Path(seg.filename).parents:
            continue

        if args.source_files and all(seg.filename not in str(s) for s in args.source_files):
            continue

        if state.is_done(seg):
            seg_done_count += 1
        else:
            worklist.append(work_segment(seg))

    progress = Progress(total=len(worklist)+seg_done_count, initial=seg_done_count)
    state.set_progress_bar(progress)

    async def run_it():
        if args.max_concurrency:
            semaphore = asyncio.Semaphore(args.max_concurrency)

            async def sem_coro(coro):
                async with semaphore:
                    return await coro

            await asyncio.gather(*(sem_coro(c) for c in worklist))
        else:
            await asyncio.gather(*worklist)

    try:
        asyncio.run(run_it())
    except KeyboardInterrupt:
        print("Interrupted.")
        if args.checkpoint:
            state.save_checkpoint(args.checkpoint)
        return 1

    progress.close()

    # --- (3) check resulting test suite ---

    coverage = disable_interfering_tests()
    print(f"End coverage: {coverage['summary']['percent_covered']:.1f}%")

    # --- (4) final remarks ---

    if required := get_required_modules():
        # Sometimes GPT outputs 'from your_module import XYZ', asking us to modify
        # FIXME move this to 'state'
        print(f"Some modules seem missing:  {', '.join(str(m) for m in required)}")
        if args.write_requirements_to:
            with args.write_requirements_to.open("a") as f:
                for module in required:
                    f.write(f"{module}\n")

    return 0
