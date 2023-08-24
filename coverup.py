import openai
import json
import subprocess
import os
import shutil
import ast
from collections import defaultdict

COVJSON = 'flask.json'  # FIXME parse arguments

#PATH = '/Users/juan/tmp/flask/'
PATH = ''

#MODEL="gpt-3.5-turbo",
#MODEL="gpt-3.5-turbo-16k",
MODEL="gpt-4"
USE_FUNCTIONS=False

PREFIX = 'coverup-'
COVERUP_TESTS_DIR="./" + PREFIX + "tests/"  # XXX use Path class

shutil.rmtree(COVERUP_TESTS_DIR, ignore_errors=True)
os.mkdir(COVERUP_TESTS_DIR)
log = open(PREFIX + "log", "w", buffering=1)    # 1 = line buffered

openai.key=os.environ['OPENAI_API_KEY']
openai.organization=os.environ['OPENAI_PLASMA_ORG']


test_seq = 1
def current_test():
    return PREFIX + f"test{test_seq}"


def get_missing_coverage(jsonfile, path):
    """Processes a JSON SlipCover output and generates a list of Python code segments,
    such as functions or classes, which have less than 100% coverage.
    """
    with open(jsonfile, "r") as f:
        cov = json.load(f)

    code_segs = []

    for fname in cov['files']:
        with open(path + fname, "r") as src:
            tree = ast.parse(src.read(), path + fname)

        code_this_file = defaultdict(set)

        for line in cov['files'][fname]['missing_lines']:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if hasattr(node, "lineno") and node.lineno <= line <= node.end_lineno:
                        begin = node.lineno
                        for d in node.decorator_list: # skip back to include decorators, in case they matter
                            begin = min(begin, d.lineno)

                        #print(f"{fname} line {line} -> {str(node)} {begin}..{node.end_lineno}")
                        code_this_file[(node.name, begin, node.end_lineno)].add(line)
                        break

        if code_this_file:
            for it in code_this_file:
                name, first, last = it
                missing = code_this_file[it]
                code_segs.append([fname, name, (first, last), list(missing)])

    return code_segs


def measure_coverage(test: str):
    import sys

    fname = current_test()
    fname_test = fname + ".py"
    fname_json = fname + ".json"

    with open(fname_test, "w") as tmp:
        tmp.write(test)

    # note response must be JSON
    p = subprocess.run(f"{sys.executable} -m slipcover --json --out {fname_json} -m pytest {fname_test}".split(),
                       check=True, capture_output=True)
    log.write(str(p.stdout, 'UTF-8') + "\n")
    #return str(p.stdout, 'UTF-8')
    with open(fname_json) as j:
        cov = json.load(j)

    return cov["files"]


def improve_coverage(missing, path):
    fname, objname, line_range, missing_lines = missing
    print(f"\n=== {objname} ({fname}) ===")

    with open(path + fname, "r") as src:
        excerpt = ''.join(src.readlines()[line_range[0]-1:line_range[1]])

    messages = [{"role": "user",
                 "content": f"""
The code below, which starts on line {line_range[0]} of {fname},
does not achieve full line coverage: when tested, {'lines' if len(missing_lines)>1 else 'line'}
{", ".join(map(str, missing_lines))} {'do' if len(missing_lines)>1 else 'does'} not execute.
Create a new pytest test function that executes these missing lines, always checking
using the provided function that the new version is correct and indeed improves
coverage. Do not propose a new version without first checking that it is correct
and that it provides improved coverage. Always send entire Python test scripts
when proposing a new test or correcting one you previously proposed.
```python
{excerpt}
```
"""
            }]

    functions = [
        {
            "name": "measure_coverage",
            "description": "Returns SlipCover coverage results from executing the given test script",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {
                        "type": "string",
                        "description": "The entire Python pytest script to execute, as a single string",
                    },
                },
                "required": ["test"],
            },
        }
    ]

    log.write(messages[0]['content'] + "\n---\n")

    attempts = 0

    while True:
        if (attempts > 5):
            log.write("Too many attempts, giving up\n---\n")
            print("giving up")
            break

        attempts += 1

        print("sent request, waiting on GPT...")
        sleep = 1
        while True:
            try:
                args = {
                    'model': MODEL,
                    'messages': messages,
                    'temperature': 0
                }

                if USE_FUNCTIONS:
                    args['functions'] = functions
                    args['function_call'] = 'auto'

                response = openai.ChatCompletion.create(**args)
                break

            except (openai.error.ServiceUnavailableError,
                    openai.error.RateLimitError) as e:
                print(e)
                print(f"waiting {sleep}s)")
                import time
                time.sleep(sleep)
                sleep *= 2

            except openai.error.InvalidRequestError as e:
                # usually "maximum context length" XXX check for this?
                log.write(f"Received {e}: giving up\n")
                print(f"Received {e}: giving up")
                return

        response_message = response["choices"][0]["message"]

        if response_message['content']:
            print(f"received \"{response_message['content'][:75]}...\"")
            log.write(response_message['content'] + "\n---\n")

        messages.append(response_message)

        if (call := response_message.get("function_call")):
            func_name = call['name']
            func_args = json.loads(call['arguments'])

            log.write(f"calling {func_name}({func_args})\n")
            assert func_name == 'measure_coverage'

            try:
                last_test = func_args['test']

                print(f"calling {func_name} {func_args}")
                result = globals()[func_name](**func_args)
                result = result[fname]
                log.write(f"result: {result}\n")

                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(result)
                })
                log.write(messages[-1]['content'] + "\n---\n")

            except subprocess.CalledProcessError as e:
                messages.append({
                    "role": "user",
                    "content": "Executing that function yields an error:\n\n" + str(e.output, 'UTF-8')
                })
                log.write(messages[-1]['content'] + "\n---\n")

        else:
            if '```python' in response_message['content']:
                import re
                m = re.search('^```python(.*?)^```$', response_message['content'], re.M|re.S)
                if m:
                    last_test = m.group(1)
            else:
                log.write("No Python code in GPT response, giving up\n---\n")
                print("No Python code in GPT response, giving up")
                break

            try:
                result = measure_coverage(last_test)
                result = result[fname]

                orig_missing = set(missing_lines)
                new_covered = set(result['executed_lines'])
                now_missing = orig_missing - new_covered

                print(f"Previously missing: {list(orig_missing)}")
                print(f"Still missing:      {list(now_missing)}")

                if len(now_missing) < len(orig_missing):
                    shutil.copyfile(current_test() + ".py", COVERUP_TESTS_DIR + current_test() + ".py")
                    global test_seq
                    test_seq += 1
                    break

                messages.append({
                    "role": "user",
                    "content": f"""
This test does not improve coverage: {'lines' if len(missing_lines)>1 else 'line'}
{", ".join(map(str, missing_lines))} still {'do' if len(missing_lines)>1 else 'does'} not execute.
"""
                })
                log.write(messages[-1]['content'] + "\n---\n")

            except subprocess.CalledProcessError as e:
                messages.append({
                    "role": "user",
                    "content": "Executing the test yields an error:\n\n" + str(e.output, 'UTF-8')
                })
                log.write(messages[-1]['content'] + "\n---\n")



missing = get_missing_coverage(COVJSON, PATH)
for m in missing:
    fname, objname, line_range, missing_lines = m
    if not fname.startswith('src/'): continue  # FIXME need to derive this somehow, or handle tests somehow
    improve_coverage(m, PATH)
