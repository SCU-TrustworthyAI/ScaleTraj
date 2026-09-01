"""Single source of truth for all dataset-specific parameters.

HOLIDAY_CALENDAR is a pre-generated static lookup covering 2007-2027.
"""
from __future__ import annotations

import datetime

DATASET_CONFIG: dict[str, dict] = {
    "tencent_beijing": {
        "raw_load_fn": "load_tencent_stays",

        "anchor_grid_size_deg": 0.005,
        "snap_grid_deg": 0.005,
        "min_step_distance_km": 0.05,

        "day_start_hour": 0,
        "min_observed_days": 5,
        "min_user_day_span_hours": 4,

        "max_median_step_km": 50.0,
        "user_max_n_threshold": None,
        "user_sample_size": 500,
        "user_sample_seed": 42,
    },

    "geolife_beijing": {
        "anchor_grid_size_deg": 0.001,
        "snap_grid_deg": 0.001,
        "min_step_distance_km": 0.05,

        "day_start_hour": 0,
        "min_observed_days": 3,
        "min_user_day_span_hours": 4,

        "max_median_step_km": 50.0,
        "user_max_n_threshold": 20,
        "user_sample_size": None,
        "user_sample_seed": 42,
    },

}


# ---------------------------------------------------------------------------
# Static holiday calendar — generated once at import, covers GeoLife + Tencent
# ---------------------------------------------------------------------------

def _build_holiday_calendar(
    start_year: int = 2007,
    end_year: int = 2025,
) -> dict[str, str]:
    """Pre-generate day_type for every date from start_year-01-01 to end_year-12-31.

    Uses chinese_calendar.get_holiday_detail(date):
      - is_off=False                     → "workday"
      - is_off=True, holiday_name != None → "holiday"
      - is_off=True, holiday_name is None → "weekend"
    """
    import chinese_calendar

    cal: dict[str, str] = {}
    failed_years: list[int] = []

    for year in range(start_year, end_year + 1):
        try:
            chinese_calendar.get_holiday_detail(datetime.date(year, 1, 1))
        except (NotImplementedError, ValueError, Exception):
            failed_years.append(year)
            continue

        d = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        while d <= end:
            is_off, holiday_name = chinese_calendar.get_holiday_detail(d)

            if not is_off:
                cal[d.isoformat()] = "workday"
            elif holiday_name is not None:
                cal[d.isoformat()] = "holiday"
            else:
                cal[d.isoformat()] = "weekend"
            d += datetime.timedelta(days=1)

    if failed_years:
        print(f"[config] WARNING: chinese_calendar does not cover years: {failed_years}")

    return cal


HOLIDAY_CALENDAR: dict[str, str] = _build_holiday_calendar()


def get_day_type(date: datetime.date | str) -> str:
    """Lookup day_type from the static calendar.

    Raises ValueError if the date is outside chinese_calendar coverage
    (currently 2004–2026) to prevent silent misclassification.
    """
    key = date if isinstance(date, str) else date.isoformat()
    dt = HOLIDAY_CALENDAR.get(key)
    if dt is not None:
        return dt
    raise ValueError(
        f"Date {key} is outside chinese_calendar coverage (2004–2026). "
        "Cannot determine day_type without reliable holiday data."
    )
