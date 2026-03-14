"""
Shared constants for the feature engineering layer.
"""

# Default history settings used by both training and daily scoring
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_HISTORY_DAYS = 7

# Source table for raw crime rows used to derive daily grid features
CRIME_RECORDS_TABLE = "crime_records"


# Fixed production offence-category vocabulary to keep feature columns stable
OFFENCE_CATEGORIES = [
    "Arson",
    "Assaults",
    "Attempted Murder",
    "Break and Enter",
    "Criminal Harassment",
    "Homicide",
    "Indecent or Harassing Communications",
    "Mischief",
    "Robbery",
    "Theft $5000 and Under",
    "Theft Over $5000",
    "Theft of Motor Vehicle",
    "Uttering Threats",
]

# Normalized feature-column names corresponding to ``OFFENCE_CATEGORIES``
CATEGORY_FEATURE_COLUMNS = [
    "category_arson",
    "category_assaults",
    "category_attempted_murder",
    "category_break_and_enter",
    "category_criminal_harassment",
    "category_homicide",
    "category_indecent_or_harassing_communications",
    "category_mischief",
    "category_robbery",
    "category_theft_5000_and_under",
    "category_theft_over_5000",
    "category_theft_of_motor_vehicle",
    "category_uttering_threats",
]

# Coarse time-of-day buckets derived from HHMM-style event hours
TIME_BUCKET_COLUMNS = [
    "night_crimes",
    "morning_crimes",
    "afternoon_crimes",
    "evening_crimes",
    "unknown_hour_crimes",
]

# Counters that track when event timestamps had to fall back to reported values
TIME_FALLBACK_COLUMNS = [
    "used_reported_date_fallback_count",
    "used_reported_hour_fallback_count",
]
