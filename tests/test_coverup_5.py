# file src/coverup/utils.py:72-97
# lines [91]
# branches ['88->91']

import asyncio
import subprocess
import pytest
from coverup.utils import subprocess_run

@pytest.mark.asyncio
async def test_subprocess_run_timeout(monkeypatch):
    async def mock_create_subprocess_exec(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.returncode = 0

            async def communicate(self):
                raise asyncio.TimeoutError()

            def terminate(self):
                pass

            async def wait(self):
                pass

        return MockProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        await subprocess_run(["echo", "hello"], timeout=1)

    assert exc_info.value.timeout == 1.0

@pytest.mark.asyncio
async def test_subprocess_run_no_timeout(monkeypatch):
    async def mock_create_subprocess_exec(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.returncode = 0

            async def communicate(self):
                raise asyncio.TimeoutError()

            def terminate(self):
                pass

            async def wait(self):
                pass

        return MockProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        await subprocess_run(["echo", "hello"])

    assert exc_info.value.timeout == 0.0
