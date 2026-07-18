import subprocess
import sys

import pytest

from skyward.storage import Storage, presets

pytestmark = pytest.mark.unit


def test_construction_defaults():
    storage = Storage(endpoint="https://example.com")
    assert storage.endpoint == "https://example.com"
    assert storage.access_key is None
    assert storage.secret_key is None
    assert storage.path_style is False


def test_import_does_not_load_aioboto3():
    check = subprocess.run(
        [sys.executable, "-c", "import skyward.storage, sys; sys.exit('aioboto3' in sys.modules)"],
        capture_output=True,
    )
    assert check.returncode == 0, "importing skyward.storage must not import aioboto3"


def test_r2_preset():
    storage = presets.R2(account_id="acct123", access_key="ak", secret_key="sk")
    assert storage.endpoint == "https://acct123.r2.cloudflarestorage.com"
    assert storage.access_key == "ak"
    assert storage.secret_key == "sk"
    assert storage.path_style is False


def test_s3_preset_defaults():
    storage = presets.S3()
    assert storage.endpoint == "https://s3.us-east-1.amazonaws.com"
    assert storage.access_key is None
    assert storage.secret_key is None
    assert storage.path_style is False


def test_s3_preset_region():
    storage = presets.S3(region="eu-west-1", access_key="ak", secret_key="sk")
    assert storage.endpoint == "https://s3.eu-west-1.amazonaws.com"
    assert storage.access_key == "ak"
    assert storage.secret_key == "sk"


def test_gcs_preset():
    storage = presets.GCS(access_key="ak", secret_key="sk")
    assert storage.endpoint == "https://storage.googleapis.com"
    assert storage.access_key == "ak"
    assert storage.secret_key == "sk"
    assert storage.path_style is False


def test_wasabi_preset():
    storage = presets.Wasabi(region="us-west-1", access_key="ak", secret_key="sk")
    assert storage.endpoint == "https://s3.us-west-1.wasabisys.com"
    assert storage.access_key == "ak"
    assert storage.secret_key == "sk"


def test_backblaze_preset():
    storage = presets.Backblaze(region="us-west-004", key_id="kid", app_key="app")
    assert storage.endpoint == "https://s3.us-west-004.backblazeb2.com"
    assert storage.access_key == "kid"
    assert storage.secret_key == "app"


def test_hyperstack_preset_defaults():
    storage = presets.Hyperstack(access_key="ak", secret_key="sk")
    assert storage.endpoint == "https://ca1.obj.nexgencloud.io"
    assert storage.access_key == "ak"
    assert storage.secret_key == "sk"
    assert storage.path_style is True


def test_hyperstack_preset_endpoint_override():
    storage = presets.Hyperstack(access_key="ak", secret_key="sk", endpoint="https://no1.obj.nexgencloud.io")
    assert storage.endpoint == "https://no1.obj.nexgencloud.io"
    assert storage.path_style is True


async def test_resolve_string_credentials_is_identity():
    storage = Storage(endpoint="https://example.com", access_key="ak", secret_key="sk")
    resolved = await storage.resolve()
    assert resolved is storage


async def test_resolve_sync_callable_credentials():
    storage = Storage(endpoint="https://example.com", access_key=lambda: "ak", secret_key=lambda: "sk")
    resolved = await storage.resolve()
    assert resolved.access_key == "ak"
    assert resolved.secret_key == "sk"


async def test_resolve_async_callable_credentials():
    async def make_key() -> str:
        return "async-ak"

    storage = Storage(endpoint="https://example.com", access_key=make_key, secret_key="sk")
    resolved = await storage.resolve()
    assert resolved.access_key == "async-ak"
    assert resolved.secret_key == "sk"
