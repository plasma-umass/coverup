import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import argparse
import openai

import coverup.llm as llm


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
    messages = [{'role': 'assistant',
                 'content': 'test prompt'}]

    with patch("litellm.acreate",
               new=AsyncMock(side_effect=openai.AuthenticationError("Authentication failed", response=MockResponse(), body=''))):
        with patch.object(llm.Chatter, '_validate_model'):
            chatter = llm.Chatter(model="gpt-4")

            with pytest.raises(openai.AuthenticationError):
                await chatter.chat(messages)
