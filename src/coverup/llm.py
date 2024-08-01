import typing as T
import openai
import logging
import asyncio
import warnings
import textwrap
from .segment import CodeSegment

with warnings.catch_warnings():
    # ignore pydantic warnings https://github.com/BerriAI/litellm/issues/2832
    warnings.simplefilter('ignore')
    import litellm # type: ignore


# Turn off most logging
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger().setLevel(logging.ERROR)

# Ignore unavailable parameters
litellm.drop_params=True


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
    if model_name.startswith('openai/'):
        model_name = model_name[7:]

    if (model_limits := MODEL_RATE_LIMITS.get(model_name)):
        return model_limits.get('token')

    return None


def compute_cost(usage: dict, model_name: str) -> float:
    from math import ceil

    if model_name.startswith('openai/'):
        model_name = model_name[7:]

    if 'prompt_tokens' in usage and 'completion_tokens' in usage:
        if (cost := litellm.model_cost.get(model_name)):
            return usage['prompt_tokens'] * cost['input_cost_per_token'] +\
                   usage['completion_tokens'] * cost['output_cost_per_token']

    return None


_token_encoding_cache = dict()
def count_tokens(model_name: str, completion: dict):
    """Counts the number of tokens in a chat completion request."""
    import tiktoken

    if not (encoding := _token_encoding_cache.get(model_name)):
        model = model_name
        if model_name.startswith('openai/'):
            model = model_name[7:]

        encoding = _token_encoding_cache[model_name] = tiktoken.encoding_for_model(model)

    count = 0
    for m in completion['messages']:
        count += len(encoding.encode(m['content']))

    return count


class ChatterError(Exception):
    pass


class Chatter:
    def __init__(self, model: str, model_temperature: float,
                 log_write: T.Callable, signal_retry: T.Callable):
        Chatter._validate_model(model)

        self._model = model
        self._model_temperature = model_temperature
        self._max_backoff = 64 # seconds
        self.set_token_rate_limit(token_rate_limit_for_model(model))

        self._log_write = log_write
        self._signal_retry = signal_retry
        self._functions = dict()


    @staticmethod
    def _validate_model(model) -> None:
        try:
            _, provider, _, _ = litellm.get_llm_provider(model)
        except litellm.exceptions.BadRequestError:
            raise ChatterError(textwrap.dedent("""\
                Unknown or unsupported model.
                Please see https://docs.litellm.ai/docs/providers for supported models."""
            ))

        result = litellm.validate_environment(model)
        if result['missing_keys']:
            if provider == 'openai':
                raise ChatterError(textwrap.dedent("""\
                    You need an OpenAI key to use {model}.
                    
                    You can get a key here: https://platform.openai.com/api-keys
                    Set the environment variable OPENAI_API_KEY to your key value
                        export OPENAI_API_KEY=<your key>"""
                ))
            elif provider == 'bedrock':
                raise ChatterError(textwrap.dedent("""\
                    To use Bedrock, you need an AWS account. Set the following environment variables:
                       export AWS_ACCESS_KEY_ID=<your key id>
                       export AWS_SECRET_ACCESS_KEY=<your secret key>
                       export AWS_REGION_NAME=us-west-2

                    You also need to request access to Claude:
                       https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#manage-model-access"""
                ))
            else:
                raise ChatterError(textwrap.dedent(f"""\
                    You need a key (or keys) from {provider} to use {model}.
                    Set the following environment variables:
                        {', '.join(result['missing_keys'])}"""
                ))


    def set_token_rate_limit(self, limit: T.Union[T.Tuple[int, int], None]) -> None:
        if limit:
            from aiolimiter import AsyncLimiter
            self.token_rate_limit = AsyncLimiter(*limit)
        else:
            self.token_rate_limit = None


    def set_max_backoff(self, max_backoff: int) -> None:
        self._max_backoff = max_backoff


    def add_function(self, function: T.Callable) -> None:
        """Makes a function availabe to the LLM."""
        if not litellm.supports_function_calling(self._model):
            raise ChatError(f"The {f._model} model does not support function calling.")

        try:
            schema = json.loads(function.__doc__)
            if 'name' not in schema:
                raise ChatterError("Name missing from function {function} schema.")
        except json.decoder.JSONDecodeError as e:
            raise ChatterError(f"Invalid JSON in function docstring: {e}")

        assert schema['name'] not in self._functions, "Duplicated function name {schema['name']}"
        self._functions[schema['name']] = {"function": function, "schema": schema}


    def _completion(self, messages: T.List[dict]) -> dict:
        return {
            'model': self._model,
            'temperature': self._model_temperature,
            'messages': messages,
            **({'api_base': "http://localhost:11434"} if "ollama" in self._model else {}),
            **({'tools': [{'type': 'function', 'function': f['schema']} for f in self._functions.values()]} \
                    if self._functions else {})
        }


    async def chat(self, seg: CodeSegment, messages: list) -> dict:
        """Sends a GPT chat request, handling common failures and returning the response."""

        completion = self._completion(messages)
        sleep = 1
        while True:
            try:
                # TODO also add request limit; could use 'await asyncio.gather(t.acquire(tokens), r.acquire())'
                # to acquire both
                if self.token_rate_limit:
                    try:
                        await self.token_rate_limit.acquire(count_tokens(self._model, completion))
                    except ValueError as e:
                        self._log_write(seg, f"Error: too many tokens for rate limit ({e})")
                        return None # gives up this segment

                return await litellm.acreate(**completion)

            except (litellm.exceptions.ServiceUnavailableError,
                    openai.RateLimitError,
                    openai.APITimeoutError) as e:

                # This message usually indicates out of money in account
                if 'You exceeded your current quota' in str(e):
                    self._log_write(seg, f"Failed: {type(e)} {e}")
                    raise

                self._log_write(seg, f"Error: {type(e)} {e}")

                import random
                sleep = min(sleep*2, self._max_backoff)
                sleep_time = random.uniform(sleep/2, sleep)
                self._signal_retry()
                await asyncio.sleep(sleep_time)

            except openai.BadRequestError as e:
                # usually "maximum context length" XXX check for this?
                self._log_write(seg, f"Error: {type(e)} {e}")
                return None # gives up this segment

            except openai.AuthenticationError as e:
                self._log_write(seg, f"Failed: {type(e)} {e}")
                raise

            except openai.APIConnectionError as e:
                self._log_write(seg, f"Error: {type(e)} {e}")
                # usually a server-side error... just retry right away
                self._signal_retry()

            except openai.APIError as e:
                # APIError is the base class for all API errors;
                # we may be missing a more specific handler.
                print(f"Error: {type(e)} {e}; missing handler?")
                self._log_write(seg, f"Error: {type(e)} {e}")
                return None # gives up this segment
