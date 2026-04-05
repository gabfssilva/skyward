"""Tests for provider default_options and Options timeout defaults."""

from __future__ import annotations

import pytest

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


