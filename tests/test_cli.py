"""The ``sky`` command, driven the way a shell drives it.

Every command here runs as its own process against a daemon of its own, which is
the only arrangement there is: the CLI owns no state and hosts no control plane,
so a command with no daemon to reach is a command with nothing to say.
Nothing is patched — what is asserted is what would have been printed.
"""

import pytest

from tests.conftest import cli, rows

pytestmark = pytest.mark.local


def describe_the_sky_command() -> None:
    def it_reports_the_version_it_is() -> None:
        printed = cli("version").out

        assert printed.startswith("skyward ")
        assert "python" in printed


def describe_when_no_daemon_is_running() -> None:
    def a_command_says_so_rather_than_becoming_one(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKYWARD_URL", raising=False)
        ran = cli("compute", "list", "--url", "http://127.0.0.1:1")

        assert ran.code != 0
        assert "no daemon at http://127.0.0.1:1" in ran.err
        assert "sky server start" in ran.err, "the message has to name the way out"


def describe_looking_at_computes() -> None:
    def describe_when_the_daemon_has_none() -> None:
        def listing_them_is_an_empty_answer_not_a_failure(alone: str) -> None:
            assert rows("compute", "list", "--url", alone) == []

    def describe_when_the_one_named_is_not_there() -> None:
        @pytest.mark.parametrize("verb", ["get", "delete", "view"])
        def it_is_refused_by_name_rather_than_raising_a_traceback(alone: str, verb: str) -> None:
            ran = cli("compute", verb, "absent", "--url", alone)

            assert ran.code != 0
            assert "not_found" in ran.err
            assert "Traceback" not in ran.err, "a refusal is an answer, not a crash"

    def describe_when_the_provider_is_one_nobody_ships() -> None:
        def creating_it_is_refused_before_anything_is_provisioned(alone: str) -> None:
            ran = cli("compute", "create", "--provider", "nowhere", "--url", alone)

            assert ran.code != 0
            assert "unknown provider" in ran.err


def describe_looking_at_offers() -> None:
    def describe_when_no_provider_has_been_registered() -> None:
        def there_are_no_offers_to_show(alone: str) -> None:
            assert rows("offers", "list", "--url", alone) == []

        def it_says_why_rather_than_leaving_an_empty_table(alone: str) -> None:
            ran = cli("offers", "list", "--url", alone)

            assert "no accounts are registered" in ran.err


def describe_registering_a_provider() -> None:
    def it_is_written_with_the_settings_it_was_given(alone: str) -> None:
        written = rows("providers", "set", "container", "--config", "binary=podman", "--url", alone)[0]
        listed = rows("providers", "list", "--url", alone)

        assert written["kind"] == "container"
        assert [row["name"] for row in listed] == [written["name"]]

    def it_changes_the_settings_of_one_already_registered(alone: str) -> None:
        rows("providers", "set", "container", "--config", "binary=podman", "--url", alone)
        rows("providers", "set", "container", "--config", "binary=docker", "--url", alone)

        assert len(rows("providers", "list", "--url", alone)) == 1, "the same account, written twice"

    def it_refuses_a_setting_the_provider_has_no_name_for(alone: str) -> None:
        ran = cli("providers", "set", "runpod", "--config", "cloud_type=nowhere", "--url", alone)

        assert ran.code != 0
        assert "cloud_type" in ran.err

    def it_refuses_a_flag_that_is_not_a_pair(alone: str) -> None:
        ran = cli("providers", "set", "runpod", "--config", "cloud_type", "--url", alone)

        assert ran.code != 0
        assert "key=value" in ran.err

    def it_refuses_a_kind_nobody_ships(alone: str) -> None:
        ran = cli("providers", "set", "nowhere", "--url", alone)

        assert ran.code != 0
        assert "unknown provider" in ran.err


def describe_asking_where_a_command_would_go() -> None:
    def it_reports_the_daemon_it_resolved(alone: str) -> None:
        reported = {row["setting"]: row["value"] for row in rows("config", "show", "--url", alone)}

        assert reported["url"] == alone
        assert reported["source"] == "flag"

    def it_falls_back_to_the_address_the_daemon_binds(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKYWARD_URL", raising=False)
        reported = {row["setting"]: row["value"] for row in rows("config", "show")}

        assert reported["url"] == "http://127.0.0.1:17590"
        assert reported["source"] == "default"

    def it_answers_that_the_daemon_is_reachable(alone: str) -> None:
        assert "ok" in cli("config", "validate", "--url", alone).out.lower()

    def it_answers_that_an_absent_daemon_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKYWARD_URL", raising=False)
        ran = cli("config", "validate", "--url", "http://127.0.0.1:1")

        assert ran.code != 0
        assert "fail" in ran.out
