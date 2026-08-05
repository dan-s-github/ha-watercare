"""Watercare custom integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.const import Platform

from .api import WatercareApi
from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR]

SERVICE_BACKFILL_HISTORY = "backfill_history"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Watercare from a config entry."""
    # Handle both old (email) and new (username) config formats
    email = entry.data.get("username") or entry.data.get("email")
    password = entry.data.get("password")

    if not email or not password:
        _LOGGER.error("Missing username/email or password in config entry")
        return False

    api = WatercareApi(email, password)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["api"] = api

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    async def _async_handle_backfill_history(call: ServiceCall) -> None:  # noqa: ARG001
        sensor = hass.data.get(DOMAIN, {}).get("sensor")
        if sensor is None:
            _LOGGER.error("Watercare sensor not yet set up; cannot backfill history")
            return
        await sensor.async_backfill_halfhourly_history()

    if not hass.services.has_service(DOMAIN, SERVICE_BACKFILL_HISTORY):
        hass.services.async_register(
            DOMAIN, SERVICE_BACKFILL_HISTORY, _async_handle_backfill_history
        )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop("api")
        hass.data[DOMAIN].pop("sensor", None)

    return unload_ok
