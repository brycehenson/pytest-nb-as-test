import pytest


@pytest.fixture(autouse=True)
def set_env_for_notebooks(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("notebook") is None:
        yield
        return
    monkeypatch.setenv("PYTEST_NOTEBOOK_FIXTURE", "1")
    yield
