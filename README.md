# CoverUp: LLM-Powered Python Test Coverage Improver
by [Juan Altmayer Pizzorno](https://jaltmayerpizzorno.github.io) and [Emery Berger](https://emeryberger.com)
at UMass Amherst's [PLASMA lab](https://plasma-umass.org/).

[![pypi](https://img.shields.io/pypi/v/coverup?color=blue)](https://pypi.org/project/coverup/)
![pyversions](https://img.shields.io/pypi/pyversions/coverup)

## About CoverUp
CoverUp adds to your module's test suite so that more of its code gets tested
(i.e., increases its [code coverage](https://en.wikipedia.org/wiki/Code_coverage)).
It can also create a suite from scratch if you don't yet have one.
The new tests are based on your code, making them useful for [regression testing](https://en.wikipedia.org/wiki/Regression_testing).

CoverUp is designed to work closely with the [pytest](https://docs.pytest.org/en/latest/) test framework.
To generate tests, it first measures your suite's coverage using [SlipCover](https://github.com/plasma-umass/slipcover)
and selects portions of the code that need more testing.
It then engages in a conversation with an [LLM](https://en.wikipedia.org/wiki/Large_language_model),
prompting for tests, checking the results and re-prompting for adjustments as necessary.

## Installing
CoverUp is available on PyPI, so you can install simply with
```shell
python3 -m pip install coverup
```

CoverUp currently requires an [OpenAI account](https://platform.openai.com/signup) to run.
Your account will also need to have a [positive balance](https://platform.openai.com/account/usage).
Create an [API key](https://platform.openai.com/api-keys) and store its "secret key"
in an environment variable named `OPENAI_API_KEY`:
```shell
export OPENAI_API_KEY=<...your-api-key...>
```

## Using
If your module's source code is in `src` and your tests in `tests`, you can run CoverUp as
```shell
coverup --source-dir src --tests-dir tests
```
CoverUp then creates tests named `test_coverup_N.py`, where `N` is a number, under the `tests` directory.

 - talk about `PYTHONPATH`... or make this automatic?
 - talk about using docker
 - checkpointing

## Any Disclaimers?
Experimental, your mileage may vary, ask your doctor, ...

## Acknowledgements
This material is based upon work supported by the National Science
Foundation under Grant No. 1955610. Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the author(s) and do not necessarily reflect the views of the National
Science Foundation.
