"""Tests for provider default_options and Options timeout defaults."""

from __future__ import annotations

import pytest

from skyward.api.spec import (
    DEFAULT_BOOTSTRAP_TIMEOUT,
    DEFAULT_PROVISION_TIMEOUT,
    DEFAULT_SSH_TIMEOUT,
    Options,
)
from skyward.providers.aws.config import AWS
from skyward.providers.container.config import Container
from skyward.providers.gcp.config import GCP
from skyward.providers.hyperstack.config import Hyperstack
from skyward.providers.jarvislabs.config import JarvisLabs
from skyward.providers.novita.config import Novita
from skyward.providers.runpod.config import RunPod
from skyward.providers.scaleway.config import Scaleway
from skyward.providers.tensordock.config import TensorDock
from skyward.providers.vastai.config import VastAI
from skyward.providers.verda.config import Verda
from skyward.providers.vultr.config import Vultr

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestNovitaDefaultOptions:
    def test_returns_options_instance(self):
        opts = Novita().default_options()
        assert isinstance(opts, Options)

    def test_provision_timeout(self):
        opts = Novita().default_options()
        assert opts.provision_timeout == 600

    def test_ssh_timeout(self):
        opts = Novita().default_options()
        assert opts.ssh_timeout == 600

    def test_bootstrap_timeout(self):
        opts = Novita().default_options()
        assert opts.bootstrap_timeout == 600


class TestOtherProvidersDefaultOptions:
    @pytest.mark.parametrize(
        "provider",
        [
            AWS(),
            GCP(),
            Hyperstack(),
            VastAI(),
            RunPod(),
            TensorDock(),
            Verda(),
            Vultr(),
            Scaleway(),
            JarvisLabs(),
            Container(),
        ],
        ids=lambda p: type(p).__name__,
    )
    def test_returns_none(self, provider):
        assert provider.default_options() is None


class TestOptionsDefaults:
    def test_provision_timeout_defaults_to_none(self):
        assert Options().provision_timeout is None

    def test_ssh_timeout_defaults_to_none(self):
        assert Options().ssh_timeout is None

    def test_bootstrap_timeout_defaults_to_none(self):
        assert Options().bootstrap_timeout is None

    def test_provision_retry_delay_defaults_to_none(self):
        assert Options().provision_retry_delay is None

    def test_max_provision_attempts_defaults_to_none(self):
        assert Options().max_provision_attempts is None

    def test_ssh_retry_interval_defaults_to_none(self):
        assert Options().ssh_retry_interval is None


class TestOptionsExplicitValues:
    def test_provision_timeout(self):
        opts = Options(provision_timeout=900)
        assert opts.provision_timeout == 900

    def test_ssh_timeout(self):
        opts = Options(ssh_timeout=120)
        assert opts.ssh_timeout == 120

    def test_bootstrap_timeout(self):
        opts = Options(bootstrap_timeout=450)
        assert opts.bootstrap_timeout == 450

    def test_provision_retry_delay(self):
        opts = Options(provision_retry_delay=10.0)
        assert opts.provision_retry_delay == 10.0

    def test_max_provision_attempts(self):
        opts = Options(max_provision_attempts=5)
        assert opts.max_provision_attempts == 5

    def test_ssh_retry_interval(self):
        opts = Options(ssh_retry_interval=4)
        assert opts.ssh_retry_interval == 4


class TestSystemDefaultConstants:
    def test_default_provision_timeout(self):
        assert DEFAULT_PROVISION_TIMEOUT == 300

    def test_default_ssh_timeout(self):
        assert DEFAULT_SSH_TIMEOUT == 300

    def test_default_bootstrap_timeout(self):
        assert DEFAULT_BOOTSTRAP_TIMEOUT == 300
