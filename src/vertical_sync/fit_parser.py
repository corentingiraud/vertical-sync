"""FIT file parsing and activity metric extraction."""

from pathlib import Path

import fitdecode
import numpy as np
import pandas as pd

from .config import FIT_DIR, HR_MAX, HR_ZONES


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


def compute_cardiac_drift(records_df: pd.DataFrame) -> dict | None:
    """Cardiac drift: % HR increase from first half to second half.

    Pass an enriched DataFrame (see enrich_records) to also compare the
    terrain profile of the two halves (avg gradient, climb-time share): when
    they differ, the number reflects the course profile, not fatigue — it is
    flagged ``confounded`` with a reason, never hidden.
    """
    if "heart_rate" not in records_df.columns:
        return None
    df = records_df.dropna(subset=["heart_rate"])
    if len(df) < 60:
        return None
    mid = len(df) // 2
    halves = df.iloc[:mid], df.iloc[mid:]
    first_avg = halves[0]["heart_rate"].mean()
    second_avg = halves[1]["heart_rate"].mean()
    if first_avg == 0:
        return None
    result = {
        "pct": round((second_avg - first_avg) / first_avg * 100, 1),
        "confounded": False,
        "note": None,
    }

    if {"gradient", "dt", "alt_delta", "dist_delta"}.issubset(df.columns):
        grads, climbs = [], []
        for h in halves:
            dist = h["dist_delta"].sum()
            time = h["dt"].sum()
            grads.append(h["alt_delta"].sum() / dist if dist > 0 else 0.0)
            climbs.append(h.loc[h["gradient"] > 0.03, "dt"].sum() / time if time > 0 else 0.0)
        # A drift too small to interpret (< 3%) needs no terrain excuse; only
        # flag when the profile asymmetry could explain the observed number
        asymmetric = abs(grads[0] - grads[1]) > 0.03 or abs(climbs[0] - climbs[1]) > 0.20
        if asymmetric and abs(result["pct"]) >= 3:
            result["confounded"] = True
            result["note"] = (
                f"profil asymetrique entre les deux moities "
                f"(pente moy {grads[0] * 100:+.1f}% vs {grads[1] * 100:+.1f}%, "
                f"temps en montee {climbs[0] * 100:.0f}% vs {climbs[1] * 100:.0f}%)"
            )
    return result


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

    # Time deltas (seconds) and elapsed time since start
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["dt"] = df["timestamp_dt"].diff().dt.total_seconds().fillna(0)
    df["elapsed_s"] = (df["timestamp_dt"] - df["timestamp_dt"].iloc[0]).dt.total_seconds()

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


def _time_base(records_df: pd.DataFrame) -> pd.DataFrame:
    """Enriched records, with time columns even when altitude/speed are missing."""
    df = enrich_records(records_df)
    if "elapsed_s" not in df.columns and "timestamp" in df.columns and len(df):
        df = df.copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df["dt"] = df["timestamp_dt"].diff().dt.total_seconds().fillna(0)
        df["elapsed_s"] = (df["timestamp_dt"] - df["timestamp_dt"].iloc[0]).dt.total_seconds()
    return df


def window_stats(enriched_df: pd.DataFrame, start_s: float, end_s: float) -> dict | None:
    """Metrics over an elapsed-time window of records.

    Shared math for fitness-test and effort extraction: mean HR, raw and
    grade-adjusted pace, net elevation, HR slope (last vs first 5 min) and
    continuity (longest pause, moving share). Expects a `_time_base` frame;
    returns None when the window holds no records.
    """
    if "elapsed_s" not in enriched_df.columns:
        return None
    win = enriched_df[(enriched_df["elapsed_s"] >= start_s) & (enriched_df["elapsed_s"] <= end_s)]
    if win.empty:
        return None

    hr = win["heart_rate"].dropna() if "heart_rate" in win.columns else pd.Series(dtype=float)

    hr_slope = hr_range = None
    if len(hr):
        seg = min(300.0, (end_s - start_s) / 2)
        first = win.loc[win["elapsed_s"] < start_s + seg, "heart_rate"].dropna()
        last = win.loc[win["elapsed_s"] >= end_s - seg, "heart_rate"].dropna()
        if len(first) and len(last):
            hr_slope = round(float(last.mean() - first.mean()), 1)
        # spread of minute-by-minute means: catches a ramp-then-fade window
        # whose endpoints happen to match
        minute_means = win.groupby((win["elapsed_s"] - start_s) // 60)["heart_rate"].mean().dropna()
        if len(minute_means) >= 2:
            hr_range = round(float(minute_means.max() - minute_means.min()), 1)

    pace_raw = pace_gap = moving_share = None
    moving = win
    if "enhanced_speed" in win.columns:
        moving = win[win["enhanced_speed"] > 0.5]
        v = moving["enhanced_speed"].dropna()
        if len(v) and v.mean() > 0:
            pace_raw = round(1000 / v.mean())
        if "gap_speed" in moving.columns:
            g = moving["gap_speed"].dropna()
            if len(g) and g.mean() > 0:
                pace_gap = round(1000 / g.mean())

    net_elev = None
    if "alt_smooth" in win.columns:
        alt = win["alt_smooth"].dropna()
        if len(alt) >= 2:
            net_elev = round(float(alt.iloc[-1] - alt.iloc[0]))

    max_pause = None
    if "dt" in win.columns and len(win):
        max_pause = round(float(win["dt"].max()))
        total_dt = float(win["dt"].sum())
        if total_dt > 0 and "enhanced_speed" in win.columns:
            moving_share = round(float(moving["dt"].sum()) / total_dt, 2)

    return {
        "window_start_s": round(start_s),
        "window_end_s": round(end_s),
        "avg_hr": round(float(hr.mean())) if len(hr) else None,
        "pace_raw_s_per_km": pace_raw,
        "pace_gap_s_per_km": pace_gap,
        "net_elevation_m": net_elev,
        "hr_slope_bpm": hr_slope,
        "hr_range_bpm": hr_range,
        "max_pause_s": max_pause,
        "moving_share": moving_share,
    }


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


def estimate_threshold(records_df: pd.DataFrame, window_s: int = 1200,
                       window: tuple[float, float] | None = None) -> dict | None:
    """LTHR + threshold pace from a ~20-min sustained effort window.

    Field-test method (Friel/Coggan 20-min TT): threshold HR is the mean HR of
    a maximal ~20-min sustained effort, and threshold pace is the grade-adjusted
    pace (GAP) over that same window — so it stays valid on rolling terrain,
    unlike Coros's flat-only test. By default the window is the highest-mean-HR
    continuous `window_s` stretch; pass ``window=(start_s, end_s)`` (elapsed
    seconds) to override when the auto pick lands wrong. Steady-state quality
    warnings (HR ramp, net elevation) make a bad window visible instead of
    silently wrong. Returns None if there is no usable HR window.
    """
    if "heart_rate" not in records_df.columns or "timestamp" not in records_df.columns:
        return None
    df = _time_base(records_df)
    if "elapsed_s" not in df.columns or df.empty:
        return None

    if window is not None:
        start_s, end_s = window
    else:
        t0 = df["timestamp_dt"].iloc[0]
        hrdf = df.dropna(subset=["heart_rate"]).set_index("timestamp_dt").sort_index()
        if hrdf.empty or (hrdf.index[-1] - hrdf.index[0]).total_seconds() < window_s:
            return None
        hr_roll = hrdf["heart_rate"].rolling(f"{window_s}s").mean()
        end_s = (hr_roll.idxmax() - t0).total_seconds()
        start_s = max(0.0, end_s - window_s)

    stats = window_stats(df, start_s, end_s)
    if not stats or stats["avg_hr"] is None:
        return None

    warnings = []
    slope = stats["hr_slope_bpm"]
    if slope is not None and slope > 5:
        warnings.append(
            f"HR still rising at end of window ({slope:+.0f} bpm): "
            f"ramp effort, LTHR likely overestimated"
        )
    net = stats["net_elevation_m"]
    if net is not None and abs(net) > 10:
        updown, bias = ("downhill", "flattered") if net < 0 else ("uphill", "penalized")
        warnings.append(f"window is net {updown} ({net:+d} m): raw pace {bias}")

    return {
        "lthr": stats["avg_hr"],
        # backward compat: threshold_pace_s_per_km stays the grade-adjusted value
        "threshold_pace_s_per_km": stats["pace_gap_s_per_km"],
        "threshold_pace_raw_s_per_km": stats["pace_raw_s_per_km"],
        "threshold_pace_gap_s_per_km": stats["pace_gap_s_per_km"],
        "window_min": round((end_s - start_s) / 60),
        "window_start_s": stats["window_start_s"],
        "window_end_s": stats["window_end_s"],
        "window_net_elevation_m": net,
        "hr_slope_bpm": slope,
        "warnings": warnings,
    }


def find_sustained_efforts(records_df: pd.DataFrame, window_minutes: tuple[int, ...] = (20,),
                           top: int = 3) -> list[dict]:
    """Best sustained efforts per window length, ranked by mean HR.

    Opportunistic LTHR/threshold calibration from real outings and races: the
    fitness-test window math (window_stats), minus the zone/VO2max wrapping.
    Each candidate carries a ``steady`` verdict (small HR slope, no pause
    > 30 s, continuous movement); non-steady candidates are kept but flagged.
    """
    if "heart_rate" not in records_df.columns or "timestamp" not in records_df.columns:
        return []
    df = _time_base(records_df)
    if "elapsed_s" not in df.columns or df.empty:
        return []
    t0 = df["timestamp_dt"].iloc[0]
    hrdf = df.dropna(subset=["heart_rate"]).set_index("timestamp_dt").sort_index()
    if hrdf.empty:
        return []

    out = []
    for wmin in window_minutes:
        w_s = wmin * 60
        hr_roll = hrdf["heart_rate"].rolling(f"{w_s}s").mean()
        end_elapsed = (hr_roll.index - t0).total_seconds()
        chosen: list[tuple[float, float]] = []
        for i in np.argsort(-hr_roll.values):
            e = end_elapsed[i]
            s = e - w_s
            if s < 0:
                continue
            if any(e > c_start and s < c_end for c_start, c_end in chosen):
                continue
            chosen.append((s, e))
            if len(chosen) >= top:
                break
        for s, e in chosen:
            stats = window_stats(df, s, e)
            if not stats:
                continue
            steady = bool(
                stats["hr_slope_bpm"] is not None and abs(stats["hr_slope_bpm"]) < 5
                # ponytail: 15 bpm calibrated on real files — race climbs sit
                # at 11-13, a misplaced ramp+fade window at 16+
                and (stats["hr_range_bpm"] is None or stats["hr_range_bpm"] < 15)
                and (stats["max_pause_s"] or 0) <= 30
                and (stats["moving_share"] is None or stats["moving_share"] >= 0.85)
            )
            out.append({"window_min": wmin, **stats, "steady": steady})
    return out


def despike_max_hr(records_df: pd.DataFrame) -> int | None:
    """Max HR excluding sensor-glitch episodes.

    A glitch starts with an implausible jump (> 5 bpm/s — real HR rises
    ~2-3 bpm/s at most) and falls back near the pre-jump baseline within a
    minute; a real max effort keeps HR elevated much longer. Samples inside
    such episodes are excluded from the clean max.
    """
    if "heart_rate" not in records_df.columns:
        return None
    hr = records_df["heart_rate"].dropna().to_numpy(dtype=float)
    if len(hr) == 0:
        return None

    # ponytail: assumes ~1 Hz sampling (true for Coros/Garmin records)
    ok = np.ones(len(hr), dtype=bool)
    jumps = np.flatnonzero(np.diff(hr) > 5) + 1
    for i in jumps:
        baseline = hr[i - 1]
        horizon = min(len(hr), i + 60)
        back = np.flatnonzero(hr[i:horizon] <= baseline + 5)
        if len(back):  # reverted within a minute → glitch episode
            ok[i:i + back[0]] = False
    return round(float(hr[ok].max())) if ok.any() else round(float(hr.max()))


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


def analyze_fitness_test(fit_data: dict, filename: str, hr_rest: float | None = None,
                         window: tuple[float, float] | None = None) -> dict:
    """Estimate threshold, HR zones and VO2max from a field-test FIT (open formulas).

    ``window=(start_s, end_s)`` overrides the auto-picked best-20-min window.
    """
    records_df = pd.DataFrame(fit_data["records"])
    hr = records_df["heart_rate"].dropna() if "heart_rate" in records_df.columns else pd.Series(dtype=float)
    max_hr = int(hr.max()) if len(hr) else None
    result = {
        "filename": filename,
        "max_hr": max_hr,
        "hr_rest": hr_rest,
        "vo2max": estimate_vo2max(max_hr, hr_rest),
    }
    thr = estimate_threshold(records_df, window=window)
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

    sport = session.get("sport")
    sub_sport = session.get("sub_sport")
    is_run = sport in (None, "running")

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

    # GAP (Grade-Adjusted Pace)
    enriched = enrich_records(records_df)
    avg_gap_speed = None
    if "gap_speed" in enriched.columns:
        valid_gap = enriched.loc[enriched["enhanced_speed"] > 0.5, "gap_speed"]
        if len(valid_gap) > 0:
            avg_gap_speed = float(valid_gap.mean())

    # Cardiac drift (with terrain-confound check from the enriched columns)
    cardiac_drift = compute_cardiac_drift(enriched)

    # Max HR sanity: isolated sensor spikes and values beyond the configured
    # physiological max are flagged, never silently replaced
    max_hr_clean = despike_max_hr(records_df)
    max_hr_suspect = bool(max_hr) and (
        (max_hr_clean is not None and max_hr > max_hr_clean + 3)
        or (bool(HR_MAX) and max_hr > HR_MAX + 3)
    )

    # Cadence on flat running segments only (poles on climbs corrupt wrist
    # cadence; > 2 m/s excludes walking/technical bits)
    cadence_flat = None
    if is_run and {"cadence", "gradient", "enhanced_speed"}.issubset(enriched.columns):
        flat = enriched[(enriched["gradient"].abs() < 0.05) & (enriched["enhanced_speed"] > 2.0)]
        cad = flat["cadence"].dropna()
        cad = cad[cad > 0]
        if len(cad) >= 60:
            cadence_flat = round(float(cad.mean()))
            if cadence_flat < 100:  # Coros stores cadence as half-cycles
                cadence_flat *= 2

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

    result = {
        "filename": filename,
        "date": date_str,
        "date_int": date_int,
        "sport": sport,
        "sub_sport": sub_sport,
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
        "max_hr_clean": max_hr_clean,
        "max_hr_suspect": max_hr_suspect,
        "avg_cadence": avg_cadence,
        "cadence_flat": cadence_flat,
        "elevation": elevation,
        "hr_zones": hr_zones,
        "cardiac_drift_pct": cardiac_drift["pct"] if cardiac_drift else None,
        "cardiac_drift_confounded": cardiac_drift["confounded"] if cardiac_drift else None,
        "cardiac_drift_note": cardiac_drift["note"] if cardiac_drift else None,
        "ascent_rate_m_h": ascent_rate,
        "vertical_ratio_m_km": vertical_ratio,
        "km_effort": km_effort,
        "laps": laps,
    }

    if not is_run:
        # run-only metrics are meaningless on a bike (coasting, no step cadence,
        # GAP model is a running cost model)
        result.update({
            "avg_pace": None,
            "avg_gap": None,
            "avg_gap_speed_ms": None,
            "avg_cadence": None,
            "cadence_flat": None,
            "cardiac_drift_pct": None,
            "cardiac_drift_confounded": None,
            "cardiac_drift_note": None,
            "vertical_ratio_m_km": None,
            "km_effort": None,
        })

    return result


def compute_week_summary(activities: list[dict]) -> dict:
    """Aggregate weekly stats from a list of activity analyses."""
    if not activities:
        return {}
    total_km = sum(a["distance_km"] for a in activities)
    total_dplus = sum(a["ascent_m"] for a in activities)
    total_dminus = sum(a["descent_m"] for a in activities)
    total_time_s = sum(a["duration_s"] for a in activities)
    avg_hr = sum(a["avg_hr"] for a in activities) / len(activities)

    # Per-sport split so bike time/D+ is never silently mixed into run totals
    by_sport: dict[str, dict] = {}
    for a in activities:
        sport = a.get("sport") or "running"
        key = {"running": "run", "cycling": "ride"}.get(sport, sport)
        g = by_sport.setdefault(key, {"sessions": 0, "total_km": 0.0, "total_dplus": 0, "_time_s": 0.0})
        g["sessions"] += 1
        g["total_km"] += a["distance_km"]
        g["total_dplus"] += a["ascent_m"]
        g["_time_s"] += a["duration_s"]
    for g in by_sport.values():
        g["total_km"] = round(g["total_km"], 1)
        g["total_time_h"] = round(g.pop("_time_s") / 3600, 2)

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
        "by_sport": by_sport,
    }


if __name__ == "__main__":
    # self-check for the open-formula estimators and window helpers
    assert estimate_vo2max(187, 47) == 60.9, estimate_vo2max(187, 47)
    z = zones_from_lthr(175)
    assert [x["max"] for x in z[:4]] == [149, 158, 166, 180], z
    assert z[0]["min"] == 0 and z[-1]["max"] == 999, z

    # synthetic 40-min flat file @3 m/s: 10-min warm-up, 20-min steady @175, 10-min cool-down
    n = 40 * 60
    _df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
        "heart_rate": np.concatenate([np.full(600, 130.0), np.full(1200, 175.0), np.full(600, 120.0)]),
        "enhanced_altitude": np.zeros(n),
        "enhanced_speed": np.full(n, 3.0),
    })
    thr = estimate_threshold(_df)
    assert thr and 170 <= thr["lthr"] <= 175, thr
    assert thr["threshold_pace_s_per_km"] == round(1000 / 3.0), thr
    assert thr["warnings"] == [], thr

    # explicit window targets the steady block exactly
    thr2 = estimate_threshold(_df, window=(600, 1800))
    assert thr2["lthr"] == 175 and thr2["window_start_s"] == 600, thr2

    # ramp effort → warning fires
    thr3 = estimate_threshold(_df.assign(heart_rate=np.linspace(140, 190, n)))
    assert any("ramp" in w for w in thr3["warnings"]), thr3

    # effort finder lands on the steady block and calls it steady
    eff = find_sustained_efforts(_df, (20,), top=1)
    assert eff and abs(eff[0]["window_start_s"] - 600) <= 30 and eff[0]["steady"], eff

    # isolated HR spike is despiked
    _sp = pd.Series(np.full(600, 150.0))
    _sp.iloc[300] = 205.0
    assert despike_max_hr(pd.DataFrame({"heart_rate": _sp})) == 150

    # drift confounded on a climb-then-descend profile with a large drift,
    # not on flat, and not when the drift is too small to interpret
    alt = np.concatenate([np.linspace(0, 300, n // 2), np.linspace(300, 0, n - n // 2)])
    hr2 = np.concatenate([np.full(n // 2, 130.0), np.full(n - n // 2, 160.0)])
    d = compute_cardiac_drift(enrich_records(_df.assign(enhanced_altitude=alt, heart_rate=hr2)))
    assert d["confounded"] and d["pct"] > 20, d
    d_flat = compute_cardiac_drift(enrich_records(_df))
    assert not d_flat["confounded"], d_flat
    d_small = compute_cardiac_drift(enrich_records(_df.assign(enhanced_altitude=alt, heart_rate=150.0)))
    assert d_small["pct"] == 0 and not d_small["confounded"], d_small

    print("fit_parser self-check OK")
