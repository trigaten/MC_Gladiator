class ComatoseBot:
    """Does nothing"""
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, obs):
        return {"camera":[0, 0]}