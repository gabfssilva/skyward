import tempfile

from vscode.sidecar.discovery import discover_main_functions


def test_discover_sky_main():
    code = '''
import skyward as sky

@sky.function
def fit(data, epochs, lr):
    return model.fit(data)

@sky.main
def train(epochs: int = 10, lr: float = 0.001, dataset: str = "cifar10"):
    result = fit(data, epochs, lr) >> sky
    return result
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = discover_main_functions([f.name])

    assert len(result) == 1
    fn = result[0]
    assert fn["name"] == "train"
    assert fn["line"] == 9
    assert len(fn["params"]) == 3
    assert fn["params"][0] == {"name": "epochs", "type": "int", "default": "10"}
    assert fn["params"][1] == {"name": "lr", "type": "float", "default": "0.001"}
    assert fn["params"][2] == {"name": "dataset", "type": "str", "default": "'cifar10'"}


def test_discover_no_sky_main():
    code = '''
@sky.function
def fit(data):
    return data
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = discover_main_functions([f.name])

    assert len(result) == 0


def test_discover_sky_main_with_parens():
    """@sky.main() with parentheses should also be discovered."""
    code = '''
import skyward as sky

@sky.main()
def train(epochs: int = 10):
    pass
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = discover_main_functions([f.name])

    assert len(result) == 1
    assert result[0]["name"] == "train"


def test_discover_no_params():
    """Function with no parameters."""
    code = '''
import skyward as sky

@sky.main
def run():
    pass
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = discover_main_functions([f.name])

    assert len(result) == 1
    assert result[0]["params"] == []


def test_discover_param_no_default():
    """Parameter without default value."""
    code = '''
import skyward as sky

@sky.main
def train(data_path: str):
    pass
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = discover_main_functions([f.name])

    assert len(result) == 1
    assert result[0]["params"] == [{"name": "data_path", "type": "str"}]


def test_discover_syntax_error_skipped():
    """Files with syntax errors are silently skipped."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("def broken(:\n")
        f.flush()
        result = discover_main_functions([f.name])

    assert result == []
