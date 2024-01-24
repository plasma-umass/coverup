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
from coverup.delta import BadTestsFinder


PREFIX = 'coverup'
CKPT_FILE = PREFIX + "-ckpt.json"
DEFAULT_MODEL='gpt-4-1106-preview'


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

    ap.add_argument('--initial-test-check', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=f'run test suite at start, aborting in case of failure.')

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

    ap.add_argument('--find-interfering-tests', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='run test suite looking for which test(s) interfere with others')

    return ap.parse_args()


def test_file_path(test_seq: int) -> Path:
    """Returns the Path for a test's file, given its sequence number."""
    return args.tests_dir / f"test_{PREFIX}_{test_seq}.py"


test_seq = 1
def new_test_file():
    """Creates a new test file, returning its Path."""

    global test_seq, args

    while True:
        p = test_file_path(test_seq)
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


def measure_coverage(seg: CodeSegment, test: str):
    """Runs a given test and returns the coverage obtained."""
    import tempfile
    global args

    with tempfile.NamedTemporaryFile(prefix=PREFIX + "_tmp_", suffix='.py',
                                     dir=str(args.tests_dir), mode="w") as t:
        t.write(test)
        t.flush()

        with tempfile.NamedTemporaryFile(prefix=PREFIX + "_") as j:
            # -qq to cut down on tokens
            p = subprocess.run((f"{sys.executable} -m slipcover --branch --json --out {j.name} " +
                                f"-m pytest {args.pytest_args} -qq --disable-warnings {t.name}").split(),
                               check=True, capture_output=True, timeout=60)
            log_write(seg, str(p.stdout, 'UTF-8'))
            cov = json.load(j)

    return cov["files"]


def run_test_with_others(test_file: Path):
    """Runs a test and a few others together, looking for side effects."""
    # For now, we just run the test, twice, along with a few others... would a random selection be better?
    # 'test_file' is run first in the hope that will catch any undesirable side effects.
    # We run it twice because if it leaves out something unclean, it may trip over that itself.

    # TODO Running it twice like this can cause a RecursionError if pytest-asyncio is installed

    if not (m := re.search('_(\\d+)$', test_file.stem)):
        raise RuntimeError(f"Unable to read test sequence number in \"{test_file}\"")
    test_seq = int(m.group(1))

    tests = [test_file, test_file]
    for seq in range(max(1, test_seq-10), test_seq):
        if len(tests) == 5: break

        candidate = test_file_path(seq)
        if candidate.exists():
            tests.append(candidate)

    tests = ' '.join(str(t) for t in tests)

    # Throws subprocess.CalledProcessError in case of problems
    # Throws TimeoutExpired if this takes too long; this is something to watch out for, as taking
    #   too long may cause outstanding chat requests to fail.
    subprocess.run((f"{sys.executable} -m pytest {args.pytest_args} -x {tests}").split(),
                   check=True, capture_output=True, timeout=120)


def run_test_suite():
    # throws subprocess.CalledProcessError in case of problems
    # TODO this can take very long (>10 minutes). Watch out for outstanding requests timing out...
    subprocess.run((f"{sys.executable} -m pytest {args.pytest_args} -x {args.tests_dir}").split(),
                   check=True, capture_output=True)


def find_interfering_tests():
    print("running suite...")
    p = subprocess.run((f"{sys.executable} -m pytest {args.pytest_args} -qq --disable-warnings {args.tests_dir}").split(),
                       check=False, capture_output=True)

    if p.returncode == 0:
        print("suite ok!")
        return

    if not (failed := BadTestsFinder.find_failed_test(str(p.stdout, 'UTF-8'))):
        print("Unable to find which test failed. Output:\n" + str(p.stdout, 'UTF-8'))
        return

    btf = BadTestsFinder(args.tests_dir)
    culprits = btf.find_culprit(failing_test)
    print("Done! Culprit(s):  {list(p.name for p in culprits)}")


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
        self.postfix['R'] = 0

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

    def add_retry(self):
        """Signals a request retry."""
        self.postfix['R'] += 1
        self.bar.set_postfix(ordered_dict=self.postfix)

    def update(self):
        """Signals an item completed."""
        self.bar.update()

    def close(self):
        """Closes the underlying tqdm bar."""
        self.bar.close()


token_rate_limit = None
progress = None
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
            progress.add_retry()
            await asyncio.sleep(sleep_time)

        except openai.error.InvalidRequestError as e:
            # usually "maximum context length" XXX check for this?
            log_write(seg, f"Error: {type(e)} {e}")
            return None # gives up this segment

        except (openai.error.APIConnectionError,
                openai.error.APIError) as e:
            log_write(seg, f"Error: {type(e)} {e}")
            # usually a server-side error... just retry right away
            progress.add_retry()


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
            log_write(seg, f"Missing modules {' '.join(missing)}")
            if not args.install_missing_modules or not install_missing_imports(seg, missing):
                return False # not finished: needs a missing module

        try:
            result = measure_coverage(seg, last_test)

        except subprocess.TimeoutExpired:
            log_write(seg, "measure_coverage timed out")
            return False # try again next time

        except subprocess.CalledProcessError as e:
            progress.add_failing()
            messages.append({
                "role": "user",
                "content": "Executing the test yields an error, shown below.\n" +\
                           "Modify the test to correct it; respond only with the complete Python code in backticks.\n\n" +\
                           clean_error(str(e.output, 'UTF-8'))
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
                run_test_with_others(new_test)

            except subprocess.CalledProcessError as e:
                progress.add_failing()
                new_test.unlink()
                messages.append({
                    "role": "user",
                    "content": "Executing the test along with the rest of the test suite yields an error, shown below.\n" +\
                               "Modify the test to correct it; respond only with the complete Python code in backticks.\n\n" +\
                               clean_error(str(e.output, 'UTF-8'))
                })
                log_write(seg, messages[-1]['content'])
                continue

            except subprocess.TimeoutExpired as e:
                log_write(seg, f"Timed out running {new_test} with others, hope it's ok")

        log_write(seg, f"Saved as {new_test}\n")
        progress.add_good()
        break

    return True # finished


def main():
    from collections import defaultdict
    import os

    global args, token_rate_limit, progress
    args = parse_args()

    if args.find_interfering_tests:
        find_interfering_tests()
        return

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

    if args.initial_test_check:
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

    ckpt = None
    if args.checkpoint:
        try:
            with checkpoint_file.open("r") as f:
                ckpt = json.load(f)
                assert ckpt['version'] == 1
                for filename, done_list in ckpt['done'].items():
                    done[filename] = set(tuple(d) for d in done_list)
                assert 'usage' in ckpt
        except json.decoder.JSONDecodeError:
            pass
        except FileNotFoundError:
            pass

    log_write('startup', f"Command: {' '.join(sys.argv)}" +\
                        (f"\nCheckpoint: True\nPrevious usage: {ckpt['usage']}" if ckpt else ""))

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
                        # XXX save other status, such as G, F, etc. and missing modules
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

    progress = Progress(total=len(worklist)+seg_done_count, initial=seg_done_count, usage=(ckpt['usage'] if ckpt else None))
    asyncio.run(runit())

    progress.close()

    if required := get_required_modules():
        # Sometimes GPT outputs 'from your_module import XYZ', asking us to modify
        print(f"Some modules seem missing:  {', '.join(str(m) for m in required)}")
        if args.write_requirements_to:
            with args.write_requirements_to.open("a") as f:
                for module in required:
                    f.write(f"{module}\n")
