import pytest


@pytest.fixture(autouse=True)
def explode() -> None:
    raise RuntimeError("boom")
