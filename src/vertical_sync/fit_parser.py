"""FIT file parsing and activity metric extraction."""

from pathlib import Path

import fitdecode
import numpy as np
import pandas as pd

from .config import FIT_DIR, HR_ZONES


# ---------------------------------------------------------------------------
# Minetti energy cost model (Minetti et al., 2002)
# ---------------------------------------------------------------------------

def minetti_cost(grade: float) -> float:
    """Energy cost of running at a given grade (J/kg/m).

    grade: fraction (0.10 = 10% slope). Clamped to [-0.5, 0.5].
    """
    i = max(-0.5, min(0.5, grade))
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


FLAT_COST = minetti_cost(0.0)  # 3.6 J/kg/m

# Vectorized version for numpy arrays
_minetti_vec = np.vectorize(minetti_cost)


def parse_fit(path: Path) -> dict:
    """Parse a FIT file into records, sessions, and laps."""
    records, sessions, laps = [], [], []
    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            row = {f.name: f.value for f in frame.fields}
            if frame.name == "record":
                records.append(row)
            elif frame.name == "session":
                sessions.append(row)
            elif frame.name == "lap":
                laps.append(row)
    return {"records": records, "sessions": sessions, "laps": laps}


def get_date_from_filename(filename: str) -> int:
    """Extract YYYYMMDD date from FIT filename like '20260415_Name_id.fit'."""
    return int(filename[:8])


def find_fit_files(start: int | None = None, end: int | None = None) -> list[Path]:
    """Find FIT files in data/fit/, optionally filtered by date range."""
    files = sorted(FIT_DIR.glob("*.fit"))
    if start is None and end is None:
        return files
    filtered = []
    for f in files:
        try:
            date = get_date_from_filename(f.name)
        except (ValueError, IndexError):
            continue
        if start and date < start:
            continue
        if end and date > end:
            continue
        filtered.append(f)
    return filtered


def format_pace(speed_ms: float) -> str:
    """Convert m/s to min/km pace string."""
    if not speed_ms or speed_ms <= 0:
        return "N/A"
    pace_s = 1000 / speed_ms
    return f"{int(pace_s // 60)}:{int(pace_s % 60):02d}"


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def compute_hr_zones(hr_series: pd.Series) -> dict:
    """Compute HR zone distribution from a heart rate series.

    Returns dict keyed by zone name (Z1..Z5), each with 'name', 'pct'.
    """
    hr = hr_series.dropna()
    total = len(hr)
    if total == 0:
        return {}

    zones = {}
    for z in HR_ZONES:
        count = int(((hr >= z["min"]) & (hr < z["max"])).sum())
        zones[z["zone"]] = {
            "name": z["name"],
            "pct": round(count / total * 100, 1),
        }
    return zones


def compute_uphill_time(records_df: pd.DataFrame) -> float | None:
    """Estimate time spent climbing from smoothed altitude records (seconds).

    Uses a 30-record rolling mean to filter GPS altitude noise.
    """
    if "enhanced_altitude" not in records_df.columns or "timestamp" not in records_df.columns:
        return None
    df = records_df[["timestamp", "enhanced_altitude"]].dropna().reset_index(drop=True)
    if len(df) < 30:
        return None
    ts = pd.to_datetime(df["timestamp"])
    alt_smooth = df["enhanced_altitude"].rolling(window=30, center=True, min_periods=1).mean()
    deltas = ts.diff().dt.total_seconds()
    climbing = alt_smooth.diff() > 0
    uphill_s = deltas[climbing].sum()
    return uphill_s if uphill_s > 0 else None


def compute_cardiac_drift(records_df: pd.DataFrame) -> float | None:
    """Cardiac drift: % HR increase from first half to second half."""
    if "heart_rate" not in records_df.columns:
        return None
    hr = records_df["heart_rate"].dropna()
    if len(hr) < 60:
        return None
    mid = len(hr) // 2
    first_avg = hr.iloc[:mid].mean()
    second_avg = hr.iloc[mid:].mean()
    if first_avg == 0:
        return None
    return round((second_avg - first_avg) / first_avg * 100, 1)


def enrich_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """Add gradient and GAP speed columns to a records DataFrame.

    Requires: enhanced_altitude, timestamp, enhanced_speed.
    Adds: alt_smooth, dt, dist_delta, gradient, gap_speed.
    """
    required = {"enhanced_altitude", "timestamp", "enhanced_speed"}
    if not required.issubset(records_df.columns):
        return records_df

    df = records_df.copy()

    # Smooth altitude (30-record rolling mean filters GPS noise)
    df["alt_smooth"] = df["enhanced_altitude"].rolling(window=30, center=True, min_periods=1).mean()

    # Time deltas (seconds)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["dt"] = df["timestamp_dt"].diff().dt.total_seconds().fillna(0)

    # Distance deltas
    if "distance" in df.columns:
        df["dist_delta"] = df["distance"].diff().fillna(0)
    else:
        df["dist_delta"] = df["enhanced_speed"] * df["dt"]

    # Altitude deltas from smoothed data
    df["alt_delta"] = df["alt_smooth"].diff().fillna(0)

    # Gradient (only where distance > 0.5m to avoid noise)
    df["gradient"] = np.where(
        df["dist_delta"] > 0.5,
        np.clip(df["alt_delta"] / df["dist_delta"], -0.5, 0.5),
        0.0,
    )

    # GAP speed = actual_speed * cost(grade) / cost(flat)
    cost_factors = _minetti_vec(df["gradient"].values) / FLAT_COST
    df["gap_speed"] = df["enhanced_speed"] * cost_factors

    return df


GRADIENT_BINS = [
    (-0.50, -0.20, "< -20%"),
    (-0.20, -0.10, "-20% a -10%"),
    (-0.10, -0.05, "-10% a -5%"),
    (-0.05, 0.00, "-5% a 0%"),
    (0.00, 0.05, "0% a 5%"),
    (0.05, 0.10, "5% a 10%"),
    (0.10, 0.20, "10% a 20%"),
    (0.20, 0.50, "> 20%"),
]


def compute_gradient_profile(enriched_df: pd.DataFrame) -> list[dict]:
    """Compute pace/HR/GAP stats bucketed by gradient range."""
    if "gradient" not in enriched_df.columns:
        return []

    df = enriched_df.dropna(subset=["gradient", "enhanced_speed"])
    profile = []

    for low, high, label in GRADIENT_BINS:
        mask = (df["gradient"] >= low) & (df["gradient"] < high)
        subset = df[mask]
        if len(subset) < 5:
            continue

        avg_speed = subset["enhanced_speed"].mean()
        avg_gap = subset["gap_speed"].mean() if "gap_speed" in subset.columns else None
        avg_hr = subset["heart_rate"].mean() if "heart_rate" in subset.columns else None
        time_s = subset["dt"].sum() if "dt" in subset.columns else float(len(subset))
        dist_m = subset["dist_delta"].sum() if "dist_delta" in subset.columns else 0.0

        profile.append({
            "gradient_range": label,
            "gradient_pct_low": round(low * 100),
            "gradient_pct_high": round(high * 100),
            "avg_pace": format_pace(avg_speed),
            "avg_gap": format_pace(avg_gap) if avg_gap else "N/A",
            "avg_hr": round(float(avg_hr)) if avg_hr and not pd.isna(avg_hr) else None,
            "time_s": round(float(time_s)),
            "time": format_duration(float(time_s)),
            "distance_m": round(float(dist_m)),
        })

    return profile


# ---------------------------------------------------------------------------
# Fitness-test estimators — transparent, open-formula alternative to Coros's
# proprietary numbers. Every input comes from the athlete's own FIT + config;
# nothing is read back from Coros. Cross-check the watch, don't copy it.
# ---------------------------------------------------------------------------

# Friel running HR zones as fractions of LTHR, collapsed to 5 zones
# (his 5a/5b/5c anaerobic split is irrelevant for ultra trail).
FRIEL_LTHR_PCT = [0.85, 0.90, 0.95, 1.03]


def estimate_threshold(records_df: pd.DataFrame, window_s: int = 1200) -> dict | None:
    """LTHR + threshold pace from the hardest continuous `window_s` window.

    Field-test method (Friel/Coggan 20-min TT): threshold HR is the mean HR of
    a maximal ~20-min sustained effort, and threshold pace is the grade-adjusted
    pace (GAP) over that same window — so it stays valid on rolling terrain,
    unlike Coros's flat-only test. Taking the highest-mean-HR window makes it
    robust to where the warm-up / cool-down sit in the file. Returns None if
    there is no HR or the file is shorter than the window.
    """
    if "heart_rate" not in records_df.columns or "timestamp" not in records_df.columns:
        return None
    df = enrich_records(records_df)
    if "timestamp_dt" not in df.columns:
        df = df.assign(timestamp_dt=pd.to_datetime(df["timestamp"]))
    df = df.dropna(subset=["heart_rate"]).set_index("timestamp_dt").sort_index()
    if df.empty or (df.index[-1] - df.index[0]).total_seconds() < window_s:
        return None

    hr_roll = df["heart_rate"].rolling(f"{window_s}s").mean()
    end = hr_roll.idxmax()
    lthr = round(float(hr_roll.loc[end]))

    thr_pace_s = None
    win = df.loc[end - pd.Timedelta(seconds=window_s):end]
    if "gap_speed" in win.columns:
        gap = win.loc[win["enhanced_speed"] > 0.5, "gap_speed"].dropna()
        if len(gap) and gap.mean() > 0:
            thr_pace_s = round(1000 / gap.mean())

    return {"lthr": lthr, "threshold_pace_s_per_km": thr_pace_s, "window_min": window_s // 60}


def estimate_vo2max(hr_max: float | None, hr_rest: float | None) -> float | None:
    """VO2max estimate, Uth-Sorensen (2004): 15.3 * HRmax / HRrest (ml/kg/min)."""
    if not hr_max or not hr_rest:
        return None
    return round(15.3 * hr_max / hr_rest, 1)


def zones_from_lthr(lthr: int) -> list[dict]:
    """5 HR zones derived from LTHR via Friel's running-zone percentages."""
    b = [round(lthr * p) for p in FRIEL_LTHR_PCT]
    names = ["Recovery", "Aerobic", "Tempo", "Threshold", "Anaerobic"]
    bounds = [0, *b, 999]
    return [
        {"zone": f"Z{i + 1}", "name": names[i], "min": bounds[i], "max": bounds[i + 1]}
        for i in range(5)
    ]


def analyze_fitness_test(fit_data: dict, filename: str, hr_rest: float | None = None) -> dict:
    """Estimate threshold, HR zones and VO2max from a field-test FIT (open formulas)."""
    records_df = pd.DataFrame(fit_data["records"])
    hr = records_df["heart_rate"].dropna() if "heart_rate" in records_df.columns else pd.Series(dtype=float)
    max_hr = int(hr.max()) if len(hr) else None
    result = {
        "filename": filename,
        "max_hr": max_hr,
        "hr_rest": hr_rest,
        "vo2max": estimate_vo2max(max_hr, hr_rest),
    }
    thr = estimate_threshold(records_df)
    if thr:
        result.update(thr)
        result["zones"] = zones_from_lthr(thr["lthr"])
    return result


def analyze_activity(fit_data: dict, filename: str) -> dict | None:
    """Analyze a single activity and return structured metrics."""
    if not fit_data["sessions"]:
        return None

    session = fit_data["sessions"][0]
    records_df = pd.DataFrame(fit_data["records"])

    total_distance = session.get("total_distance", 0) or 0
    total_time = session.get("total_timer_time", 0) or 0
    total_ascent = session.get("total_ascent", 0) or 0
    total_descent = session.get("total_descent", 0) or 0
    avg_hr = session.get("avg_heart_rate", 0) or 0
    max_hr = session.get("max_heart_rate", 0) or 0
    avg_cadence = session.get("avg_running_cadence") or session.get("avg_cadence", 0) or 0
    avg_speed = session.get("enhanced_avg_speed") or session.get("avg_speed", 0) or 0
    start_time = session.get("start_time")

    # Coros stores cadence as half-cycles
    if avg_cadence and avg_cadence < 100:
        avg_cadence = avg_cadence * 2

    distance_km = total_distance / 1000

    # Elevation range from records
    elevation = {}
    if "enhanced_altitude" in records_df.columns:
        alt = records_df["enhanced_altitude"].dropna()
        if len(alt) > 0:
            elevation = {"min": round(float(alt.min())), "max": round(float(alt.max()))}

    # HR zones
    hr_zones = {}
    if "heart_rate" in records_df.columns:
        hr_zones = compute_hr_zones(records_df["heart_rate"])

    # Cardiac drift
    cardiac_drift = compute_cardiac_drift(records_df)

    # GAP (Grade-Adjusted Pace)
    enriched = enrich_records(records_df)
    avg_gap_speed = None
    if "gap_speed" in enriched.columns:
        valid_gap = enriched.loc[enriched["enhanced_speed"] > 0.5, "gap_speed"]
        if len(valid_gap) > 0:
            avg_gap_speed = float(valid_gap.mean())

    # Uphill time and ascent rate
    uphill_time_s = compute_uphill_time(records_df)
    ascent_rate = round(total_ascent / uphill_time_s * 3600) if uphill_time_s and total_ascent > 0 else 0
    vertical_ratio = round(total_ascent / distance_km) if distance_km > 0 else 0
    km_effort = round(distance_km + total_ascent / 100, 1)

    # Date from filename
    try:
        date_int = get_date_from_filename(filename)
        date_str = f"{str(date_int)[:4]}-{str(date_int)[4:6]}-{str(date_int)[6:8]}"
    except (ValueError, IndexError):
        date_str = str(start_time) if start_time else "unknown"
        date_int = 0

    # Laps
    laps = []
    for i, lap in enumerate(fit_data["laps"]):
        lap_speed = lap.get("enhanced_avg_speed") or lap.get("avg_speed", 0) or 0
        lap_cadence = (lap.get("avg_running_cadence") or 0) * 2
        laps.append({
            "lap": i + 1,
            "distance_km": round((lap.get("total_distance", 0) or 0) / 1000, 2),
            "duration": format_duration(lap.get("total_timer_time", 0) or 0),
            "duration_s": lap.get("total_timer_time", 0) or 0,
            "pace": format_pace(lap_speed),
            "avg_hr": lap.get("avg_heart_rate", 0) or 0,
            "max_hr": lap.get("max_heart_rate", 0) or 0,
            "cadence": lap_cadence,
            "ascent_m": lap.get("total_ascent", 0) or 0,
            "descent_m": lap.get("total_descent", 0) or 0,
        })

    return {
        "filename": filename,
        "date": date_str,
        "date_int": date_int,
        "distance_km": round(distance_km, 2),
        "duration": format_duration(total_time),
        "duration_s": total_time,
        "ascent_m": total_ascent,
        "descent_m": total_descent,
        "avg_pace": format_pace(avg_speed),
        "avg_gap": format_pace(avg_gap_speed) if avg_gap_speed else "N/A",
        "avg_gap_speed_ms": round(avg_gap_speed, 2) if avg_gap_speed else None,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "avg_cadence": avg_cadence,
        "elevation": elevation,
        "hr_zones": hr_zones,
        "cardiac_drift_pct": cardiac_drift,
        "ascent_rate_m_h": ascent_rate,
        "vertical_ratio_m_km": vertical_ratio,
        "km_effort": km_effort,
        "laps": laps,
    }


def compute_week_summary(activities: list[dict]) -> dict:
    """Aggregate weekly stats from a list of activity analyses."""
    if not activities:
        return {}
    total_km = sum(a["distance_km"] for a in activities)
    total_dplus = sum(a["ascent_m"] for a in activities)
    total_dminus = sum(a["descent_m"] for a in activities)
    total_time_s = sum(a["duration_s"] for a in activities)
    avg_hr = sum(a["avg_hr"] for a in activities) / len(activities)

    return {
        "runs": len(activities),
        "total_km": round(total_km, 1),
        "total_dplus": total_dplus,
        "total_dminus": total_dminus,
        "total_time": format_duration(total_time_s),
        "total_time_h": round(total_time_s / 3600, 2),
        "avg_hr": round(avg_hr),
        "vertical_ratio": round(total_dplus / total_km) if total_km > 0 else 0,
        "km_effort": round(total_km + total_dplus / 100, 1),
    }


if __name__ == "__main__":
    # self-check for the open-formula estimators
    assert estimate_vo2max(187, 47) == 60.9, estimate_vo2max(187, 47)
    z = zones_from_lthr(175)
    assert [x["max"] for x in z[:4]] == [149, 158, 166, 180], z
    assert z[0]["min"] == 0 and z[-1]["max"] == 999, z
    # synthetic 25-min ramp on flat terrain: LTHR = last-20-min mean, pace = 3 m/s
    n = 25 * 60
    _df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
        "heart_rate": np.linspace(150, 180, n),
        "enhanced_altitude": np.zeros(n),
        "enhanced_speed": np.full(n, 3.0),
    })
    thr = estimate_threshold(_df)
    assert thr and 165 <= thr["lthr"] <= 178, thr
    assert thr["threshold_pace_s_per_km"] == round(1000 / 3.0), thr
    print("fit_parser self-check OK:", estimate_vo2max(187, 47), thr)
