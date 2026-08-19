"""Tests for the fixed NZ-time poll scheduling (should_poll=False path)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from datetime import date as date_type
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


def dst_nz_dt(year: int, month: int, day: int, hour: int, *, is_dst: bool) -> datetime:
    """
    Build an NZ-time datetime for an ambiguous/nonexistent hour.

    The two dates/hours a year that are ambiguous (fall back) or
    nonexistent (spring forward) as naive wall-clock times, forcing the
    side that must be picked explicitly.
    """
    return NZ_TIMEZONE.localize(
        datetime(year, month, day, hour, 0, 0),  # noqa: DTZ001 -- localize() below
        is_dst=is_dst,
    )


def test_should_poll_is_disabled(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()

    assert sensor.should_poll is False


def test_next_poll_time_picks_start_hour_same_day_when_fresh(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """With no halfhourly data seen yet, freshness defaults to True."""
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 1)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 6)


def test_next_poll_time_rolls_to_next_day_when_past_start_hour_and_fresh(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 20)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 17, 6)


def test_next_poll_time_skips_immediate_reschedule_at_start_hour_boundary(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """Regression: firing exactly at 06:00 must not schedule right back at 06:00."""
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 6)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 17, 6)


def test_next_poll_time_retries_hourly_while_stale(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor._latest_reading_date = date_type(2026, 8, 14)  # two days stale
    now_nz = nz_dt(2026, 8, 16, 6)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 7)


def test_next_poll_time_still_retries_at_cutoff_hour_while_stale(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """The cutoff hour itself is still polled -- only hours after it give up."""
    sensor = make_sensor()
    sensor._latest_reading_date = date_type(2026, 8, 14)
    now_nz = nz_dt(2026, 8, 16, 11)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 16, 12)


def test_next_poll_time_gives_up_after_cutoff_hour_while_stale(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """Regression: don't retry hourly forever if Watercare never publishes."""
    sensor = make_sensor()
    sensor._latest_reading_date = date_type(2026, 8, 14)
    now_nz = nz_dt(2026, 8, 16, 12)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 17, 6)


def test_next_poll_time_stops_for_the_day_once_fresh(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """Regression: catching up mid-morning shouldn't schedule another same-day poll."""
    sensor = make_sensor()
    now_nz = nz_dt(2026, 8, 16, 7)
    sensor._latest_reading_date = now_nz.date() - timedelta(days=1)

    assert sensor._next_poll_time(now_nz) == nz_dt(2026, 8, 17, 6)


def test_localize_nz_handles_nonexistent_hour_on_spring_forward(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """
    Regression: 02:00 doesn't exist on NZ's spring-forward day.

    NZ DST starts at 2am on the last Sunday of September. localize(...,
    is_dst=True) on the impossible 02:00 would silently resolve an hour
    *before* the gap (firing early) -- the first valid instant after it,
    03:00 NZDT, must be used instead. None of the current poll hours land
    on 2am, but _localize_nz must still handle it correctly for any hour.
    """
    sensor = make_sensor()

    result = sensor._localize_nz(date_type(2026, 9, 26), 1, 2)

    assert result == nz_dt(2026, 9, 27, 3)
    assert result.utcoffset() == timedelta(hours=13)


def test_localize_nz_handles_ambiguous_hour_on_fall_back(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """
    Regression: 02:00 occurs twice on NZ's fall-back day.

    NZ DST ends at 3am on the first Sunday of April, so 02:00 occurs
    twice that day; the later (std) occurrence must be picked.
    """
    sensor = make_sensor()

    result = sensor._localize_nz(date_type(2026, 4, 4), 1, 2)

    assert result == dst_nz_dt(2026, 4, 5, 2, is_dst=False)


def test_next_poll_time_uses_correct_offset_after_dst_transition(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """
    Regression: a stale pre-transition offset must not leak into candidates.

    The old replace()-based candidates kept `now`'s stale pre-transition
    offset, silently firing the scheduled poll up to an hour off the
    intended wall-clock target once the candidate crossed a DST change.
    """
    sensor = make_sensor()
    # Just past the spring-forward transition; the 06:00 candidate must
    # carry NZDT (+13:00), not the pre-transition NZST (+12:00) offset.
    now_nz = dst_nz_dt(2026, 9, 27, 3, is_dst=True)

    next_poll = sensor._next_poll_time(now_nz)

    assert next_poll == nz_dt(2026, 9, 27, 6)
    assert next_poll.utcoffset() == timedelta(hours=13)


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
        nz_dt(2026, 8, 16, 6),
    )
    assert sensor._unsub_scheduled_poll is unsub_sentinel


def test_schedule_next_poll_cancels_existing_timer_first(
    make_sensor: Callable[..., WatercareUsageSensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: re-scheduling must not leak the previous timer."""
    sensor = make_sensor()
    fixed_utcnow = nz_dt(2026, 8, 16, 5).astimezone(UTC)
    monkeypatch.setattr(watercare_sensor.dt_util, "utcnow", lambda: fixed_utcnow)
    old_unsub = MagicMock(name="old_unsub")
    sensor._unsub_scheduled_poll = old_unsub
    monkeypatch.setattr(
        watercare_sensor,
        "async_track_point_in_time",
        MagicMock(return_value=MagicMock(name="new_unsub")),
    )

    sensor._schedule_next_poll()

    old_unsub.assert_called_once()
    assert sensor._unsub_scheduled_poll is not old_unsub


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


async def test_run_backfill_schedules_poll_only_after_completing(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    """
    Regression: scheduling must wait for backfill, not race it.

    Scheduling the fixed-time poll from async_added_to_hass() (rather than
    from here, after backfill finishes) could let a scheduled poll fire
    mid-backfill and advance the statistics watermark out from under
    async_backfill_halfhourly_history() -- see its append-only warning.
    """
    sensor = make_sensor()
    sensor._entry.data = {}
    sensor.async_backfill_halfhourly_history = AsyncMock()
    sensor.async_update = AsyncMock()
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    await sensor._run_backfill()

    sensor.async_backfill_halfhourly_history.assert_awaited_once()
    sensor._schedule_next_poll.assert_called_once()


async def test_run_backfill_schedules_poll_even_on_failure(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor._entry.data = {}
    sensor.async_backfill_halfhourly_history = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    await sensor._run_backfill()

    sensor._schedule_next_poll.assert_called_once()


async def test_run_initial_update_schedules_poll_after_completing(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor.async_update = AsyncMock()
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    await sensor._run_initial_update()

    sensor.async_update.assert_awaited_once()
    sensor._schedule_next_poll.assert_called_once()


async def test_run_initial_update_schedules_poll_even_on_failure(
    make_sensor: Callable[..., WatercareUsageSensor],
) -> None:
    sensor = make_sensor()
    sensor.async_update = AsyncMock(side_effect=RuntimeError("boom"))
    sensor.async_write_ha_state = MagicMock()
    sensor._schedule_next_poll = MagicMock()

    await sensor._run_initial_update()

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
