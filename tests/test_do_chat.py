import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import argparse
import openai

import warnings
with warnings.catch_warnings():
    # ignore pydantic warnings https://github.com/BerriAI/litellm/issues/2832
    warnings.simplefilter('ignore')
    import litellm # type: ignore


from coverup.coverup import do_chat


@pytest.mark.asyncio
async def test_do_chat_authentication_error():
    class MockCodeSegment:
        def __init__(self, name):
            self.name = name

    class MockResponse:
        def __init__(self):
            self.request = None
            self.status_code = 401
            self.headers = {}

    seg = MockCodeSegment("test_segment")
    completion = {"prompt": "test prompt"}

    with patch("coverup.coverup.log_write", new=lambda segment, message: print(message)):
        with patch("litellm.acreate",
                    new=AsyncMock(side_effect=openai.AuthenticationError("Authentication failed", response=MockResponse(), body=''))):
                with patch("coverup.coverup.token_rate_limit", None):
                    with patch("coverup.coverup.args", argparse.Namespace(model="gpt-4", max_backoff=10), create=True):
                        with pytest.raises(openai.AuthenticationError):
                            await do_chat(seg, completion)
