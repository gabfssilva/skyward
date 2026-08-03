"""The ``sky`` command, driven the way a shell drives it.

Every command here runs as its own process against an embedded daemon on a
database of its own, which is what a user gets when no ``SKYWARD_URL`` is set.
Nothing is patched: what is asserted is what would have been printed.
"""

from pathlib import Path

import pytest

from tests.conftest import cli, rows

pytestmark = pytest.mark.local


def describe_the_sky_command() -> None:
    def it_reports_the_version_it_is() -> None:
        printed = cli("version").out

        assert printed.startswith("skyward ")
        assert "python" in printed


def describe_looking_at_computes() -> None:
    def describe_when_the_daemon_has_none() -> None:
        def listing_them_is_an_empty_answer_not_a_failure(database: Path) -> None:
            assert rows("compute", "list", "--database", str(database)) == []

    def describe_when_the_one_named_is_not_there() -> None:
        @pytest.mark.parametrize("verb", ["get", "delete", "view"])
        def it_is_refused_by_name_rather_than_raising_a_traceback(database: Path, verb: str) -> None:
            ran = cli("compute", verb, "absent", "--database", str(database))

            assert ran.code != 0
            assert "not_found" in ran.err
            assert "Traceback" not in ran.err, "a refusal is an answer, not a crash"

    def describe_when_the_provider_is_one_nobody_ships() -> None:
        def creating_it_is_refused_before_anything_is_provisioned(database: Path) -> None:
            ran = cli("compute", "create", "--provider", "nowhere", "--database", str(database))

            assert ran.code != 0
            assert "unknown provider" in ran.err


def describe_looking_at_offers() -> None:
    def describe_when_no_provider_has_been_registered() -> None:
        def there_are_no_offers_to_show(database: Path) -> None:
            assert rows("offers", "list", "--database", str(database)) == []


def describe_asking_where_a_command_would_go() -> None:
    def it_reports_the_database_it_resolved(database: Path) -> None:
        reported = {row["setting"]: row["value"] for row in rows("config", "path", "--database", str(database))}

        assert reported["database"] == str(database)

    def it_answers_that_an_embedded_daemon_is_reachable(database: Path) -> None:
        assert "ok" in cli("config", "validate", "--database", str(database)).out.lower()
