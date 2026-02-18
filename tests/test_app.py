from skyward.app import App, get_app


def test_app_stored_in_context():
    assert get_app() is None
    with App() as app:
        assert get_app() is app
    assert get_app() is None


def test_app_default_has_no_spy():
    with App() as app:
        assert app.spy is None


def test_app_console_flag_defaults_true():
    app = App()
    assert app.console is True


def test_app_console_false():
    with App(console=False) as app:
        assert app.spy is None
