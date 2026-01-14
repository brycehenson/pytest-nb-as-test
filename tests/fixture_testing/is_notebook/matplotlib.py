BACKEND = None


def use(backend: str) -> None:
    global BACKEND
    BACKEND = backend
