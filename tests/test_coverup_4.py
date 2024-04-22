# file src/coverup/llm.py:47-51
# lines [48, 49, 51]
# branches ['48->49', '48->51']

import pytest
from coverup.llm import token_rate_limit_for_model

# Assuming MODEL_RATE_LIMITS is a dictionary defined in the same module
from coverup.llm import MODEL_RATE_LIMITS

# Test to cover lines 48-49
def test_token_rate_limit_for_existing_model():
    # Setup: Add a model with token rate limits to MODEL_RATE_LIMITS
    model_name = 'test_model'
    token_limits = (100, 60)
    MODEL_RATE_LIMITS[model_name] = {'token': token_limits}

    # Test
    limits = token_rate_limit_for_model(model_name)
    assert limits == token_limits

    # Cleanup
    del MODEL_RATE_LIMITS[model_name]

# Test to cover line 51
def test_token_rate_limit_for_non_existing_model():
    # Setup: Ensure the model does not exist in MODEL_RATE_LIMITS
    model_name = 'non_existing_model'
    assert model_name not in MODEL_RATE_LIMITS

    # Test
    limits = token_rate_limit_for_model(model_name)
    assert limits is None
