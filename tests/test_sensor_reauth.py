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
