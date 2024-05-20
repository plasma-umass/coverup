import asyncio
import json
import warnings

with warnings.catch_warnings():
    # ignore pydantic warnings https://github.com/BerriAI/litellm/issues/2832
    warnings.simplefilter('ignore')
    import litellm # type: ignore

import logging
import openai
import subprocess
import re
import sys
import typing as T

from pathlib import Path
from datetime import datetime

from .llm import *
from .segment import *
from .testrunner import *
from . import prompt


# Turn off most logging
litellm.set_verbose = False
logging.getLogger().setLevel(logging.ERROR)
# Ignore unavailable parameters
litellm.drop_params=True

def parse_args(args=None):
    import argparse

    ap = argparse.ArgumentParser(prog='CoverUp',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('source_files', type=Path, nargs='*',
                    help='only process certain source file(s)')

    def Path_dir(value):
        path_dir = Path(value).resolve()
        if not path_dir.is_dir(): raise argparse.ArgumentTypeError("must be a directory")
        return path_dir

    ap.add_argument('--tests-dir', type=Path_dir, required=True,
                    help='directory where tests reside')

    ap.add_argument('--source-dir', '--source', type=Path_dir, required=True,
                    help='directory where sources reside')

    ap.add_argument('--checkpoint', type=Path, 
                    help=f'path to save progress to (and to resume it from)')
    ap.add_argument('--no-checkpoint', action='store_const', const=None, dest='checkpoint', default=argparse.SUPPRESS,
                    help=f'disables checkpoint')

    def default_model():
        if 'OPENAI_API_KEY' in os.environ:
            return "openai/gpt-4o-2024-05-13"
        if 'ANTHROPIC_API_KEY' in os.environ:
            return "anthropic/claude-3-sonnet-20240229"
        if 'AWS_ACCESS_KEY_ID' in os.environ:
            return "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"

    ap.add_argument('--model', type=str, default=default_model(),
                    help='OpenAI model to use')

    ap.add_argument('--prompt-family', type=str,
                    choices=list(prompt.prompters.keys()),
                    default='gpt',
                    help='Prompt style to use')

    ap.add_argument('--model-temperature', type=float, default=0,
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

    ap.add_argument('--log-file', default=f"coverup-log",
                    help='log file to use')

    ap.add_argument('--pytest-args', type=str, default='',
                    help='extra arguments to pass to pytest')

    ap.add_argument('--install-missing-modules', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='attempt to install any missing modules')

    ap.add_argument('--prefix', type=str, default='coverup',
                    help='prefix to use for test file names')

    ap.add_argument('--write-requirements-to', type=Path,
                    help='append the name of any missing modules to the given file')

    ap.add_argument('--disable-polluting', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='look for tests causing others to fail and disable them')

    ap.add_argument('--disable-failing', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='look for failing tests and disable them')

    ap.add_argument('--prompt-for-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='prompt LLM for new tests')

    ap.add_argument('--isolate-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='run tests in isolation (to work around any state pollution) when measuring suite coverage')

    ap.add_argument('--debug', '-d', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='print out debugging messages.')

    ap.add_argument('--add-to-pythonpath', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='add (parent of) source directory to PYTHONPATH')

    ap.add_argument('--branch-coverage', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=argparse.SUPPRESS)

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 0: raise argparse.ArgumentTypeError("must be a number >= 0")
        return ivalue

    ap.add_argument('--max-concurrency', type=positive_int, default=50,
                    help='maximum number of parallel requests; 0 means unlimited')

    args = ap.parse_args(args)

    for i in range(len(args.source_files)):
        args.source_files[i] = args.source_files[i].resolve()

    if args.disable_failing and args.disable_polluting:
        ap.error('Specify only one of --disable-failing and --disable-polluting')

    return args


def test_file_path(test_seq: int) -> Path:
    """Returns the Path for a test's file, given its sequence number."""
    global args
    return args.tests_dir / f"test_{args.prefix}_{test_seq}.py"


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


def check_whole_suite() -> None:
    """Check whole suite and disable any polluting/failing tests."""

    pytest_args = args.pytest_args
    if args.disable_polluting:
        pytest_args += " -x"  # stop at first (to save time)

    while True:
        print("Checking test suite...  ", end='', flush=True)
        try:
            btf = BadTestsFinder(tests_dir=args.tests_dir, pytest_args=pytest_args,
                                 branch_coverage=args.branch_coverage,
                                 trace=(print if args.debug else None))
            outcomes = btf.run_tests()
            failing_tests = list(p for p, o in outcomes.items() if o == 'failed')
            if not failing_tests:
                print("tests ok!")
                return

        except subprocess.CalledProcessError as e:
            print(str(e) + "\n" + str(e.stdout, 'UTF-8', errors='ignore'))
            sys.exit(1)

        if args.disable_failing:
            print(f"{len(failing_tests)} test(s) failed, disabling...")
            to_disable = failing_tests

        else:
            print(f"{failing_tests[0]} failed; Looking for culprit(s)...")

            def print_noeol(message):
                # ESC[K clears the rest of the line
                print(message, end='...\033[K\r', flush=True)

            try:
                btf = BadTestsFinder(tests_dir=args.tests_dir, pytest_args=args.pytest_args,
                                     branch_coverage=args.branch_coverage,
                                     trace=(print if args.debug else None),
                                     progress=(print if args.debug else print_noeol))

                to_disable = btf.find_culprit(failing_tests[0])

            except BadTestFinderError as e:
                print(e)
                to_disable = {failing_tests[0]}

        for t in to_disable:
            print(f"Disabling {t}")
            t.rename(t.parent / ("disabled_" + t.name))


def find_imports(python_code: str) -> T.List[str]:
    """Collects a list of packages needed by a program by examining its 'import' statements"""
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
            if n.module and n.level == 0:
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
            log_write(seg, f"Unable to install module {module}:\n{str(e.stdout, 'UTF-8', errors='ignore')}")
            all_ok = False

    return all_ok


def get_required_modules() -> T.List[str]:
    """Returns a list of the modules originally missing (i.e., even if we installed them)"""
    return [m for m in module_available if module_available[m] != 1]


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
        self.final_coverage = None
        self.bar = None


    def get_initial_coverage(self) -> dict:
        """Returns the coverage initially measured."""
        return self.coverage


    def set_final_coverage(self, cov: dict) -> None:
        """Adds the final coverage obtained, so it can be saved in a checkpoint."""
        self.final_coverage = cov


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
        self.done[seg.path].add((seg.begin, seg.end))


    def is_done(self, seg: CodeSegment):
        """Returns whether a segment is done."""
        return (seg.begin, seg.end) in self.done[seg.path]


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
                    state.done[Path(filename).resolve()] = set(tuple(d) for d in done_list)
                state.add_usage(ckpt['usage'])
                if 'counters' in ckpt:
                    state.counters = ckpt['counters']
                return state

        except FileNotFoundError:
            return None


    def save_checkpoint(self, ckpt_file: Path):
        """Saves this state to a checkpoint file."""
        ckpt = {
            'version': 1,
            'done': {str(k):list(v) for k,v in self.done.items() if len(v)},  # cannot serialize 'Path' or 'set' as-is
            'usage': self.usage,
            'counters': self.counters,
            'coverage': self.coverage
            # FIXME save missing modules
        }

        if self.final_coverage:
            ckpt['final_coverage'] = self.final_coverage

        with ckpt_file.open("w") as f:
            json.dump(ckpt, f)


state = None
token_rate_limit = None
async def do_chat(seg: CodeSegment, completion: dict) -> str:
    """Sends a GPT chat request, handling common failures and returning the response."""

    global token_rate_limit

    sleep = 1
    while True:
        try:
            if token_rate_limit:
                try:
                    await token_rate_limit.acquire(count_tokens(args.model, completion))
                except ValueError as e:
                    log_write(seg, f"Error: too many tokens for rate limit ({e})")
                    return None # gives up this segment

            return await litellm.acreate(**completion)

        except (litellm.exceptions.ServiceUnavailableError,
                openai.RateLimitError,
                openai.APITimeoutError) as e:

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

        except openai.BadRequestError as e:
            # usually "maximum context length" XXX check for this?
            log_write(seg, f"Error: {type(e)} {e}")
            return None # gives up this segment

        except openai.APIConnectionError as e:
            log_write(seg, f"Error: {type(e)} {e}")
            # usually a server-side error... just retry right away
            state.inc_counter('R')

        except openai.APIError as e:
            # APIError is the base class for all API errors;
            # we may be missing a more specific handler.
            print(f"Error: {type(e)} {e}; missing handler?")
            log_write(seg, f"Error: {type(e)} {e}")
            return None # gives up this segment

def extract_python(response: str) -> str:
    # This regex accepts a truncated code block... this seems fine since we'll try it anyway
    m = re.search(r'```python\n(.*?)(?:```|\Z)', response, re.DOTALL)
    if not m: raise RuntimeError(f"Unable to extract Python code from response {response}")
    return m.group(1)


async def improve_coverage(seg: CodeSegment) -> bool:
    """Works to improve coverage for a code segment."""
    global args, progress

    def log_prompts(prompts: T.List[dict]):
        for p in prompts:
            log_write(seg, p['content'])
    
    prompter = prompt.prompters[args.prompt_family](args=args, segment=seg)
    messages = prompter.initial_prompt()
    log_prompts(messages)

    attempts = 0

    if args.dry_run:
        return True

    while True:
        attempts += 1
        if (attempts > args.max_attempts):
            log_write(seg, "Too many attempts, giving up")
            break

        completion = {'model': args.model,
                      'messages': messages,
                      'temperature': args.model_temperature}
        
        if "ollama" in args.model:
            completion["api_base"] = "http://localhost:11434"
            
        if not (response := await do_chat(seg, completion)):
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
            result = await measure_test_coverage(test=last_test, tests_dir=args.tests_dir, pytest_args=args.pytest_args,
                                                 branch_coverage=args.branch_coverage,
                                                 log_write=lambda msg: log_write(seg, msg))

        except subprocess.TimeoutExpired:
            log_write(seg, "measure_coverage timed out")
            # FIXME is the built-in timeout reasonable? Do we prompt for a faster test?
            # We don't want slow tests, but there may not be any way around it.
            return True

        except subprocess.CalledProcessError as e:
            state.inc_counter('F')
            prompts = prompter.error_prompt(clean_error(str(e.stdout, 'UTF-8', errors='ignore')))
            messages.extend(prompts)
            log_prompts(prompts)
            continue

        new_lines = set(result[seg.filename]['executed_lines']) if seg.filename in result else set()
        new_branches = set(tuple(b) for b in result[seg.filename]['executed_branches']) \
                       if (seg.filename in result and 'executed_branches' in result[seg.filename]) else set()
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
            state.inc_counter('U')
            prompts = prompter.missing_coverage_prompt(now_missing_lines, now_missing_branches)
            messages.extend(prompts)
            log_prompts(prompts)
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
    if args.add_to_pythonpath:
        add_to_pythonpath(args.source_dir)

    if args.prompt_for_tests:
        if args.rate_limit or token_rate_limit_for_model(args.model):
            limit = (args.rate_limit, 60) if args.rate_limit else token_rate_limit_for_model(args.model)
            from aiolimiter import AsyncLimiter
            token_rate_limit = AsyncLimiter(*limit)
            # TODO also add request limit, and use 'await asyncio.gather(t.acquire(tokens), r.acquire())' to acquire both


        # Check for an API key for OpenAI or Amazon Bedrock.
        if 'OPENAI_API_KEY' not in os.environ and 'ANTHROPIC_API_KEY' not in os.environ:
            if not all(x in os.environ for x in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION_NAME']):
                print("You need a key (or keys) from an AI service to use CoverUp.")
                print()
                print("OpenAI:")
                print("  You can get a key here: https://platform.openai.com/api-keys")
                print("  Set the environment variable OPENAI_API_KEY to your key value:")
                print("    export OPENAI_API_KEY=<your key>")
                print()
                print()
                print("Anthropic:")
                print("  Set the environment variable ANTHROPIC_API_KEY to your key value:")
                print("    export ANTHROPIC_API_KEY=<your key>")
                print()
                print()
                print("Bedrock:")
                print("  To use Bedrock, you need an AWS account.")
                print("  Set the following environment variables:")
                print("    export AWS_ACCESS_KEY_ID=<your key id>")
                print("    export AWS_SECRET_ACCESS_KEY=<your secret key>")
                print("    export AWS_REGION_NAME=us-west-2")
                print("  You also need to request access to Claude:")
                print(
                    "   https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#manage-model-access"
                )
                print()
                return 1

        if not args.model:
            print("Please specify model to use with --model")
            return 1

        log_write('startup', f"Command: {' '.join(sys.argv)}")

        # --- (1) load or measure initial coverage, figure out segmentation ---

        if args.checkpoint and (state := State.load_checkpoint(args.checkpoint)):
            print("Continuing from checkpoint;  coverage: ", end='', flush=True)
            coverage = state.get_initial_coverage()
        else:
            if args.disable_polluting or args.disable_failing:
                # check and clean up suite before measuring coverage
                check_whole_suite()

            try:
                print("Measuring coverage...  ", end='', flush=True)
                coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.source_dir,
                                                  pytest_args=args.pytest_args,
                                                  isolate_tests=args.isolate_tests,
                                                  branch_coverage=args.branch_coverage,
                                                  trace=(print if args.debug else None))
                state = State(coverage)

            except subprocess.CalledProcessError as e:
                print("Error measuring coverage:\n" + str(e.stdout, 'UTF-8', errors='ignore'))
                return 1

        print(f"{coverage['summary']['percent_covered']:.1f}%")
        # TODO also show running coverage estimate

        segments = sorted(get_missing_coverage(state.get_initial_coverage(), line_limit=args.line_limit),
                          key=lambda seg: seg.missing_count(), reverse=True)

        # save initial coverage so we don't have to redo it next time
        if args.checkpoint:
            state.save_checkpoint(args.checkpoint)

        # --- (2) prompt for tests ---

        print(f"Prompting {args.model} for tests to increase coverage...")
        print("(in the following, G=good, F=failed, U=useless and R=retry)")

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
            if not seg.path.is_relative_to(args.source_dir):
                continue

            if args.source_files and seg.path not in args.source_files:
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

    # --- (3) clean up resulting test suite ---

    if args.disable_polluting or args.disable_failing:
        check_whole_suite()

    # --- (4) show final coverage

    if args.prompt_for_tests:
        try:
            print("Measuring coverage...  ", end='', flush=True)
            coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.source_dir,
                                              pytest_args=args.pytest_args,
                                              isolate_tests=args.isolate_tests,
                                              branch_coverage=args.branch_coverage,
                                              trace=(print if args.debug else None))

        except subprocess.CalledProcessError as e:
            print("Error measuring coverage:\n" + str(e.stdout, 'UTF-8', errors='ignore'))
            return 1

        print(f"{coverage['summary']['percent_covered']:.1f}%")

    # --- (5) save state and show missing modules, if appropriate

        if args.checkpoint:
            state.set_final_coverage(coverage)
            state.save_checkpoint(args.checkpoint)

        if required := get_required_modules():
            # Sometimes GPT outputs 'from your_module import XYZ', asking us to modify
            # FIXME move this to 'state'
            print(f"Some modules seem missing:  {', '.join(str(m) for m in required)}")
            if args.write_requirements_to:
                with args.write_requirements_to.open("a") as f:
                    for module in required:
                        f.write(f"{module}\n")

    return 0
