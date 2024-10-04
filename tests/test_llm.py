import pytest
import coverup.llm as llm


def test_compute_cost():
    assert pytest.approx(0.033, abs=.001) == \
           llm.compute_cost({'prompt_tokens':1100, 'completion_tokens':0}, 'gpt-4')

    assert pytest.approx(0.033, abs=.001) == \
           llm.compute_cost({'prompt_tokens':1100, 'completion_tokens':0}, 'openai/gpt-4')

    assert pytest.approx(2.10, abs=.01) == \
           llm.compute_cost({'prompt_tokens':60625, 'completion_tokens':4731}, 'gpt-4')

    assert pytest.approx(2.10, abs=.01) == \
           llm.compute_cost({'prompt_tokens':60625, 'completion_tokens':4731}, 'openai/gpt-4')

    # unknown model
    assert None == llm.compute_cost({'prompt_tokens':60625, 'completion_tokens':4731}, 'unknown')

    # unknown token types
    assert None  == llm.compute_cost({'blue_tokens':60625, 'red_tokens':4731}, 'gpt-4')


def test_token_rate_limit_for_model():
    assert llm.token_rate_limit_for_model('gpt-4') != None
    assert llm.token_rate_limit_for_model('openai/gpt-4') != None

    assert llm.token_rate_limit_for_model('unknown') is None


def test_token_rate_limit_for_model_no_encoding(monkeypatch):
    import tiktoken

    def mock_get_encoding(model: str):
        raise KeyError(f"{model} who?")

    monkeypatch.setattr(tiktoken, 'encoding_for_model', mock_get_encoding)

    assert llm.token_rate_limit_for_model('gpt-4') is None
    assert llm.token_rate_limit_for_model('openai/gpt-4') is None
