"""What the user writes about an account, and what the daemon gets back.

A provider is written once and read twice: the client takes it apart into the
credentials and the config a provider row is made of, and the adapter puts it
back together on the other side of the wire. These are the two halves agreeing.
"""

from pathlib import Path

import msgspec
import pytest

import skyward as sky
from skyward.core.provider import resolve
from skyward.providers.registry import REGISTRY
from skyward.shared import providers
from skyward.shared.providers import Provider

pytestmark = pytest.mark.local

ACCOUNTS = tuple(
    value
    for value in vars(providers).values()
    if isinstance(value, type) and issubclass(value, Provider) and value is not Provider
)


@pytest.fixture(autouse=True)
def nowhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No account of the machine running the tests answers for anything."""
    for account in ACCOUNTS:
        for variable in providers.variables(account).values():
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-file"))


def describe_writing_a_provider() -> None:
    def it_keeps_the_secrets_out_of_the_config() -> None:
        credentials, config = resolve(sky.AWS(access_key_id="AKIA", secret_access_key="s3cret", region="eu-west-1"))

        assert credentials == {"access_key_id": "AKIA", "secret_access_key": "s3cret"}
        assert config["region"] == "eu-west-1"
        assert not credentials.keys() & config.keys(), "a field is a secret or a setting, never both"
        assert "name" not in config, "the alias names the row rather than living in it"

    def it_takes_the_environment_only_for_what_was_left_unset(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "from-the-environment")
        assert resolve(sky.RunPod())[0] == {"api_key": "from-the-environment"}
        assert resolve(sky.RunPod(api_key="written-down"))[0] == {"api_key": "written-down"}

    def describe_when_the_environment_is_silent() -> None:
        def it_falls_back_to_the_aws_credentials_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            shared = tmp_path / "credentials"
            shared.write_text("[default]\naws_access_key_id = AKIAFILE\naws_secret_access_key = filesecret\n")
            monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(shared))

            assert resolve(sky.AWS())[0] == {"access_key_id": "AKIAFILE", "secret_access_key": "filesecret"}

    def it_takes_one_region_or_several() -> None:
        assert sky.AWS(region="eu-west-1").regions == ("eu-west-1",)
        assert sky.AWS(region=("eu-west-1", "us-east-2")).regions == ("eu-west-1", "us-east-2")


def describe_handing_a_provider_to_the_daemon() -> None:
    @pytest.mark.parametrize("account", ACCOUNTS, ids=lambda account: account.kind)
    def it_comes_back_as_the_very_object_that_was_written(account: type[Provider]) -> None:
        written = account()
        credentials, config = resolve(written)
        wire = msgspec.json.decode(msgspec.json.encode({**credentials, **config}))

        assert msgspec.convert(wire, account) == written

    def it_is_refused_at_the_wire_when_a_setting_is_not_one_of_the_allowed() -> None:
        _, config = resolve(sky.RunPod())

        with pytest.raises(msgspec.ValidationError):
            msgspec.convert({**config, "cloud_type": "whatever"}, sky.RunPod)


def describe_the_adapters() -> None:
    def each_one_has_the_account_the_user_writes_for_it() -> None:
        kinds = {account.kind for account in ACCOUNTS}

        assert set(REGISTRY) <= kinds, "an adapter with no account is one nobody can configure"
