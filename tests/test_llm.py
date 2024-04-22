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
