"""Tests for reauth triggering when Watercare rejects stored credentials."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from custom_components.watercare.api import WatercareAuthError

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.watercare.sensor import WatercareUsageSensor


async def test_get_data_starts_reauth_when_credentials_rejected(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))

    result = await sensor._get_data(
        endpoint="halfhourly", start_date=None, end_date=None
    )

    assert result is None
    sensor._entry.async_start_reauth.assert_called_once_with(sensor.hass)


async def test_get_data_passes_through_response_without_starting_reauth(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor._api.get_data = AsyncMock(return_value="usage-data")

    result = await sensor._get_data(
        endpoint="halfhourly", start_date=None, end_date=None
    )

    assert result == "usage-data"
    sensor._entry.async_start_reauth.assert_not_called()


async def test_get_data_only_starts_reauth_once_per_outage(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """Regression: the retry ladder can call _get_data many times a day."""
    sensor = make_sensor()
    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))

    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._entry.async_start_reauth.assert_called_once_with(sensor.hass)


async def test_get_data_starts_reauth_again_after_an_intervening_success(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """A later success (e.g. after reauth completes) re-arms the guard."""
    sensor = make_sensor()
    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._api.get_data = AsyncMock(return_value="usage-data")
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    assert sensor._entry.async_start_reauth.call_count == 2


async def test_get_data_unrelated_none_response_does_not_rearm_the_guard(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """
    Regression: only a real success may clear _reauth_started.

    An unrelated transient failure (e.g. a non-200 from the usage endpoint)
    also returns None without raising WatercareAuthError -- that must not
    be mistaken for recovery and silently re-arm the reauth spam guard.
    """
    sensor = make_sensor()
    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._api.get_data = AsyncMock(return_value=None)
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._api.get_data = AsyncMock(side_effect=WatercareAuthError("bad creds"))
    await sensor._get_data(endpoint="halfhourly", start_date=None, end_date=None)

    sensor._entry.async_start_reauth.assert_called_once_with(sensor.hass)
