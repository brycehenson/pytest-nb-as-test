import matplotlib


def test_regular_backend_unchanged() -> None:
    assert matplotlib.BACKEND is None
