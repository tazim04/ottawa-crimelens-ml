from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import APP_TIMEZONE

logger = logging.getLogger(__name__)


def local_today(timezone_name: str = APP_TIMEZONE) -> date:
    """
    Return today's date in the configured application timezone.
    """
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "Unknown APP_TIMEZONE '%s'; falling back to UTC for local date resolution",
            timezone_name,
        )
        timezone = ZoneInfo("UTC")
    return datetime.now(timezone).date()
