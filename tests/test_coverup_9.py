# file src/coverup/llm.py:47-51
# lines [48, 49, 51]
# branches ['48->49', '48->51']

import pytest
from unittest.mock import patch
from src.coverup.llm import token_rate_limit_for_model

# Assuming MODEL_RATE_LIMITS is a dictionary accessible within the module
# If it's not, you would need to mock it or set it up appropriately.

@pytest.fixture
def model_rate_limits():
    with patch('src.coverup.llm.MODEL_RATE_LIMITS', {'known_model': {'token': (100, 200)}}) as mock_limits:
        yield mock_limits

def test_token_rate_limit_for_model_with_known_model(model_rate_limits):
    # Call the function with the known model
    result = token_rate_limit_for_model('known_model')
    
    # Assert that the correct rate limit is returned
    assert result == (100, 200), "The token rate limit should be returned for a known model"

def test_token_rate_limit_for_model_with_unknown_model(model_rate_limits):
    # Call the function with an unknown model
    result = token_rate_limit_for_model('unknown_model')
    
    # Assert that None is returned for an unknown model
    assert result is None, "None should be returned for an unknown model"
