"""Tests for the fixed NZ-time poll scheduling (should_poll=False path)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from custom_components.watercare import sensor as watercare_sensor
from custom_components.watercare.const import NZ_TIMEZONE

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from custom_components.watercare.sensor import WatercareUsageSensor


def nz_dt(year: int, month: int, day: int, hour: int) -> datetime:
    """
    Build a correctly-localized NZ-time datetime.

    NZ_TIMEZONE is a pytz timezone -- passing it directly as `tzinfo=` to the
    datetime() constructor (rather than via .localize()) silently attaches
    the zone's historical LMT offset instead of the current NZST/NZDT one.
    """
    return NZ_TIMEZONE.localize(
        datetime(year, month, day, hour, 0, 0)  # noqa: DTZ001 -- localize() below
    )


def test_should_poll_is_disabled(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()

    assert sensor.should_poll is False


def test_next_poll_time_picks_soonest_hour_same_day(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 1)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 2)


def test_next_poll_time_picks_later_hour_same_day(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 5)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 14)


def test_next_poll_time_rolls_to_next_day_after_last_hour(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 20)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 17, 2)


def test_next_poll_time_skips_immediate_reschedule_at_exact_boundary(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """Regression: firing exactly at 02:00 must not schedule right back at 02:00."""
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 2)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 14)


def test_schedule_next_poll_registers_at_computed_time(
    make_sensor: Callable[..., WatercareUsageSensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = make_sensor()
    fixed_utcnow = nz_dt(2026, 8, 16, 5).astimezone(UTC)
    monkeypatch.setattr(watercare_sensor.dt_util, "utcnow", lambda: fixed_utcnow)

    unsub_sentinel = MagicMock(name="unsub")
    fake_track_point_in_time = MagicMock(return_value=unsub_sentinel)
    monkeypatch.setattr(
        watercare_sensor, "async_track_point_in_time", fake_track_point_in_time
    )

    sensor._schedule_next_poll()

    fake_track_point_in_time.assert_called_once_with(
        sensor.hass,
        sensor._handle_scheduled_poll,
        nz_dt(2026, 8, 16, 14),
    )
    assert sensor._unsub_scheduled_poll is unsub_sentinel


async def test_handle_scheduled_poll_updates_writes_state_and_reschedules(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor.async_update = AsyncMock()
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    await sensor._handle_scheduled_poll(datetime(2026, 8, 16, 14, 0, 0, tzinfo=UTC))

    sensor.async_update.assert_awaited_once()
    sensor.async_write_ha_state.assert_called_once()
    sensor._schedule_next_poll.assert_called_once()


async def test_handle_scheduled_poll_logs_and_still_reschedules_on_failure(
    make_sensor: Callable[..., WatercareUsageSensor],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: a failed scheduled update must not abandon future polling."""
    sensor = make_sensor()
    sensor.async_update = AsyncMock(side_effect=RuntimeError("boom"))
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    with caplog.at_level(logging.ERROR):
        await sensor._handle_scheduled_poll(datetime(2026, 8, 16, 14, 0, 0, tzinfo=UTC))

    assert any(
        "Watercare scheduled update failed" in message for message in caplog.messages
    )
    sensor.async_write_ha_state.assert_called_once()
    sensor._schedule_next_poll.assert_called_once()


async def test_will_remove_from_hass_cancels_scheduled_poll(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    unsub = MagicMock(name="unsub")
    sensor._unsub_scheduled_poll = unsub

    await sensor.async_will_remove_from_hass()

    unsub.assert_called_once()
    assert sensor._unsub_scheduled_poll is None


async def test_will_remove_from_hass_is_a_noop_when_never_scheduled(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()

    await sensor.async_will_remove_from_hass()

    assert sensor._unsub_scheduled_poll is None
