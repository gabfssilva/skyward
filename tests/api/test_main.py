import skyward as sky


def test_sky_main_marks_function():
    @sky.main
    def train(epochs: int = 10):
        return epochs

    assert hasattr(train, "__sky_main__")
    assert train.__sky_main__ is True


def test_sky_main_callable():
    @sky.main
    def train(epochs: int = 10):
        return epochs

    result = train(epochs=5)
    assert result == 5


def test_sky_main_with_parens():
    @sky.main()
    def train(epochs: int = 10):
        return epochs

    assert hasattr(train, "__sky_main__")
    assert train.__sky_main__ is True


def test_sky_main_preserves_name():
    @sky.main
    def train(epochs: int = 10):
        return epochs

    assert train.__name__ == "train"


def test_sky_main_with_parens_callable():
    @sky.main()
    def train(epochs: int = 10):
        return epochs

    result = train(epochs=3)
    assert result == 3
