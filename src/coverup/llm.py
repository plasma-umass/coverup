import typing as T
import litellm


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
