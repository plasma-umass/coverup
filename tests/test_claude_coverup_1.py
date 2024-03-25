# file src/coverup/llm.py:47-51
# lines [48, 49, 51]
# branches ['48->49', '48->51']

import pytest
from unittest.mock import patch
from coverup.llm import token_rate_limit_for_model

@pytest.fixture
def mock_model_rate_limits():
    with patch('coverup.llm.MODEL_RATE_LIMITS', {'model_a': {'token': (10, 20)}, 'model_b': {'token': (30, 40)}}):
        yield

def test_token_rate_limit_for_model(mock_model_rate_limits):
    # Test case when model_name is in MODEL_RATE_LIMITS
    assert token_rate_limit_for_model('model_a') == (10, 20)
    assert token_rate_limit_for_model('model_b') == (30, 40)

    # Test case when model_name is not in MODEL_RATE_LIMITS
    assert token_rate_limit_for_model('unknown_model') is None
