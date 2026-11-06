"""Time-pairing of drug administrations and simulated null distributions.

This module reproduces the method used in "Timing of hypertonic albumin and co-administered
furosemide in sepsis". It contains no patient data. See README.md for the input format.

Inputs
------
doses, other
    Long tables with columns ``admission_id`` and ``dt`` (datetime). ``doses`` holds the index
    drug (one row = one administration record) and is the denominator. ``other`` holds the
    comparator drug.
cohort
    One row per admission with ``admission_id``, ``adm`` (start of the admission day) and ``dis``
    (end of the discharge day).

Test statistic
--------------
The percentage of index doses with at least one comparator record from ``lo_h`` to ``hi_h`` hours
after the dose, both ends included (negative hours are before the dose). Every null uses this same
statistic. Comparator times are never moved. One comparator record may fall in the windows of
several doses, and a dose with several comparator records in its window counts once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NULL_ANCHORS = ("stay", "day", "clock")
_SECONDS_PER_DAY = 86_400.0


def _dose_hits(doses: pd.DataFrame, other: pd.DataFrame, lo_h: float, hi_h: float) -> pd.DataFrame:
    d = (doses[["admission_id", "dt"]].reset_index(drop=True)
         .rename_axis("dose_row").reset_index())
    if d.empty:
        d["hit"] = pd.Series(dtype=bool)
        return d
    m = d.merge(other, on="admission_id", suffixes=("_a", "_o"))
    m["gap_h"] = (m["dt_o"] - m["dt_a"]).dt.total_seconds() / 3600
    hit = m.loc[(m["gap_h"] >= lo_h) & (m["gap_h"] <= hi_h), "dose_row"]
    d["hit"] = d["dose_row"].isin(set(hit))
    return d


def pair_fraction(doses: pd.DataFrame, other: pd.DataFrame, lo_h: float, hi_h: float) -> tuple[int, int, int, int]:
    """Return (paired doses, all doses, admissions with a paired dose, admissions with a dose)."""
    if doses.empty:
        return (0, 0, 0, 0)
    d = _dose_hits(doses, other, lo_h, hi_h)
    return (int(d["hit"].sum()), len(d),
            int(d.loc[d["hit"], "admission_id"].nunique()), int(d["admission_id"].nunique()))


def per_admission_pairing(doses: pd.DataFrame, other: pd.DataFrame, lo_h: float, hi_h: float) -> pd.DataFrame:
    """Paired doses, doses and their share for each admission (removes the weighting by dose count)."""
    columns = ["admission_id", "paired", "doses", "fraction"]
    if doses.empty:
        return pd.DataFrame(columns=columns)
    d = _dose_hits(doses, other, lo_h, hi_h)
    g = d.groupby("admission_id")["hit"].agg(paired="sum", doses="size").reset_index()
    g["paired"] = g["paired"].astype(int)
    g["fraction"] = g["paired"] / g["doses"]
    return g[columns]


def overlap_summary(doses: pd.DataFrame, other: pd.DataFrame, lo_h: float, hi_h: float) -> dict[str, int]:
    """Count how windows overlap. Records with the same admission and time count as one record."""
    if doses.empty or other.empty:
        return {"other_events_in_a_window": 0, "other_events_in_2plus_windows": 0,
                "doses_with_2plus_other_events": 0}
    d = (doses[["admission_id", "dt"]].reset_index(drop=True).rename_axis("dose_row").reset_index())
    o = other[["admission_id", "dt"]].drop_duplicates().reset_index(drop=True)
    o = o.rename_axis("other_row").reset_index()
    m = d.merge(o, on="admission_id", suffixes=("_a", "_o"))
    gap = (m["dt_o"] - m["dt_a"]).dt.total_seconds() / 3600
    hit = m[(gap >= lo_h) & (gap <= hi_h)]
    per_other = hit.groupby("other_row")["dose_row"].nunique()
    per_dose = hit.groupby("dose_row")["other_row"].nunique()
    return {"other_events_in_a_window": int(len(per_other)),
            "other_events_in_2plus_windows": int((per_other >= 2).sum()),
            "doses_with_2plus_other_events": int((per_dose >= 2).sum())}


def _redraw_bounds(doses: pd.DataFrame, cohort: pd.DataFrame, anchor: str):
    if anchor not in NULL_ANCHORS:
        raise ValueError(f"anchor must be one of {NULL_ANCHORS}, got {anchor!r}")
    stay = cohort.set_index("admission_id")[["adm", "dis"]]
    d = doses[["admission_id", "dt"]].join(stay, on="admission_id")
    d["dt"] = pd.to_datetime(d["dt"])
    if anchor == "stay":
        base, end = d["adm"], d["dis"]
    else:
        off = (d["dt"] - d["adm"]).dt.total_seconds()
        base = d["adm"] + pd.to_timedelta(np.floor(off / _SECONDS_PER_DAY) * _SECONDS_PER_DAY, unit="s")
        end = (base + pd.Timedelta(seconds=_SECONDS_PER_DAY)).clip(upper=d["dis"])
    span = (end - base).dt.total_seconds().to_numpy()
    span = np.where(np.isfinite(span) & (span > 0), span, 0.0)
    return d["admission_id"].to_numpy(), base.to_numpy(), span


def _second_of_day(ts: np.ndarray) -> np.ndarray:
    return np.mod(ts.astype("datetime64[s]").astype("int64"), int(_SECONDS_PER_DAY)).astype(float)


def _draw_clock_offsets(rng, span, base_tod, tod_pool, *, tries: int = 32) -> np.ndarray:
    """Draw a position inside the 24-h block from the observed clock times of all index doses.

    Blocks truncated at discharge use rejection sampling, which conditions the empirical clock-time
    distribution on the available interval. If no observed clock time fits after ``tries`` rounds,
    the position is drawn uniformly from the interval.
    """
    off = np.zeros(len(span))
    todo = span > 0
    for _ in range(tries):
        k = int(todo.sum())
        if not k:
            break
        cand = np.mod(tod_pool[rng.integers(0, len(tod_pool), size=k)] - base_tod[todo], _SECONDS_PER_DAY)
        ok = cand < span[todo]
        idx = np.flatnonzero(todo)
        off[idx[ok]] = cand[ok]
        todo[idx[ok]] = False
    if todo.any():
        off[todo] = rng.random(int(todo.sum())) * span[todo]
    return off


def _null_draws(doses, cohort, anchor, replicates, seed):
    rng = np.random.default_rng(seed)
    adm_ids, base, span = _redraw_bounds(doses, cohort, anchor)
    tod_pool = base_tod = None
    if anchor == "clock":
        tod_pool = _second_of_day(pd.to_datetime(doses["dt"]).to_numpy())
        base_tod = _second_of_day(base)
    for _ in range(replicates):
        off = (_draw_clock_offsets(rng, span, base_tod, tod_pool) if anchor == "clock"
               else rng.random(len(span)) * span)
        yield pd.DataFrame({"admission_id": adm_ids, "dt": base + (off * 1e9).astype("timedelta64[ns]")})


def null_pair_fraction(doses, other, cohort, lo_h, hi_h, *, anchor="stay", replicates=2000, seed=42,
                       percentiles=(2.5, 97.5)) -> tuple[float, float, float]:
    """Mean and percentile range of the statistic when each dose time is redrawn independently.

    ``anchor="stay"`` (within-admission): uniform over the admission.
    ``anchor="day"`` (same-day): uniform within the dose's 24-h block counted from admission,
    truncated at discharge.
    ``anchor="clock"`` (clock-preserving): within that block, at clock times drawn from all index
    doses. Every anchor keeps the admission and its number of doses but not the spacing of doses,
    which can make the ranges too narrow. Clinical state and treatment schedules are not modelled.
    """
    if doses.empty:
        return (float("nan"),) * 3
    rates = []
    for pseudo in _null_draws(doses, cohort, anchor, replicates, seed):
        k, n, _, _ = pair_fraction(pseudo, other, lo_h, hi_h)
        rates.append(100 * k / n if n else float("nan"))
    r = np.asarray(rates, dtype=float)
    lo_p, hi_p = percentiles
    return (float(r.mean()), float(np.percentile(r, lo_p)), float(np.percentile(r, hi_p)))


def schedule_shift_null(doses, other, cohort, lo_h, hi_h, *, max_shift_days, replicates=2000, seed=42,
                        percentiles=(2.5, 97.5)) -> dict[str, float]:
    """Shift each admission's whole dose schedule by a random whole number of days.

    All doses of an admission receive the same shift ``k``, so dose spacing and clock times are kept
    and only the link between that day's doses and the comparator is broken. ``k`` is drawn uniformly
    from the non-zero integers in ``[-max_shift_days, max_shift_days]`` that keep the whole schedule
    inside ``[adm, dis]``. Admissions with no such ``k`` are excluded, and the observed statistic is
    recomputed on the same admissions. What remains conditioned on: the admission, its number of
    doses, their spacing and clock times, and the approximate phase of the stay. Clinical state on the
    shifted day is not modelled.
    """
    if max_shift_days < 1:
        raise ValueError("max_shift_days must be at least 1")
    nan = float("nan")
    out = {"n_admissions_total": int(doses["admission_id"].nunique()), "n_admissions_shiftable": 0,
           "n_doses_total": len(doses), "n_doses_shiftable": 0, "observed_k": 0, "observed_pct": nan,
           "expected_pct": nan, "lo_pct": nan, "hi_pct": nan}
    if doses.empty:
        return out
    d = doses[["admission_id", "dt"]].copy()
    d["dt"] = pd.to_datetime(d["dt"])
    span = d.groupby("admission_id")["dt"].agg(first="min", last="max").join(
        cohort.set_index("admission_id")[["adm", "dis"]])
    day = pd.Timedelta(days=1)
    k_min = np.ceil((span["adm"] - span["first"]) / day).clip(lower=-max_shift_days)
    k_max = np.floor((span["dis"] - span["last"]) / day).clip(upper=max_shift_days)
    feasible = {aid: [k for k in range(int(a), int(b) + 1) if k != 0]
                for aid, a, b in zip(span.index, k_min, k_max) if np.isfinite(a) and np.isfinite(b)}
    feasible = {aid: ks for aid, ks in feasible.items() if ks}
    d = d[d["admission_id"].isin(set(feasible))].reset_index(drop=True).rename_axis("dose_row").reset_index()
    out.update(n_admissions_shiftable=len(feasible), n_doses_shiftable=len(d))
    if d.empty:
        return out
    m = d.merge(other[["admission_id", "dt"]], on="admission_id", suffixes=("_a", "_o"))
    gap = (m["dt_o"] - m["dt_a"]).dt.total_seconds().to_numpy() / 3600
    observed = len(np.unique(m["dose_row"].to_numpy()[(gap >= lo_h) & (gap <= hi_h)]))
    near = (gap >= lo_h - 24 * max_shift_days) & (gap <= hi_h + 24 * max_shift_days)
    rows, gaps = m["dose_row"].to_numpy()[near], gap[near]
    order = sorted(feasible)
    pos = pd.Series(np.arange(len(order)), index=order)
    pair_adm = pos.reindex(m["admission_id"].to_numpy()[near]).to_numpy()
    rng = np.random.default_rng(seed)
    rates = np.empty(replicates)
    for r in range(replicates):
        k = np.array([ks[rng.integers(len(ks))] for ks in (feasible[a] for a in order)], dtype=float)
        g = gaps - 24.0 * k[pair_adm]   # moving the dose by +k days changes the gap by -k days
        rates[r] = 100 * len(np.unique(rows[(g >= lo_h) & (g <= hi_h)])) / len(d)
    lo_p, hi_p = percentiles
    out.update(observed_k=observed, observed_pct=100 * observed / len(d), expected_pct=float(rates.mean()),
               lo_pct=float(np.percentile(rates, lo_p)), hi_pct=float(np.percentile(rates, hi_p)))
    return out


def relative_time_histogram(doses, other, edges) -> np.ndarray:
    """Count comparator records by hours from each dose. The unit is a dose-record pair, not a dose."""
    zeros = np.zeros(len(edges) - 1, dtype=float)
    if doses.empty or other.empty:
        return zeros
    m = doses[["admission_id", "dt"]].merge(other, on="admission_id", suffixes=("_a", "_o"))
    if m.empty:
        return zeros
    gap = (m["dt_o"] - m["dt_a"]).dt.total_seconds().to_numpy() / 3600
    return np.histogram(gap, bins=edges)[0].astype(float)


def cluster_bootstrap_ci(doses, other, lo_h, hi_h, *, replicates=1000, seed=42, percentiles=(2.5, 97.5),
                         cluster: pd.Series | None = None) -> tuple[float, float]:
    """Percentile interval of the statistic, resampling admissions or, with ``cluster``, individuals.

    ``cluster`` maps ``admission_id`` to the individual's identifier.
    """
    if doses.empty:
        return (float("nan"), float("nan"))
    d = _dose_hits(doses, other, lo_h, hi_h)
    key = "admission_id"
    if cluster is not None:
        d["_cluster"] = d["admission_id"].map(cluster)
        if d["_cluster"].isna().any():
            raise ValueError("cluster has no entry for some admission_id")
        key = "_cluster"
    g = d.groupby(key)["hit"].agg(["sum", "size"])
    k, n = g["sum"].to_numpy(dtype=float), g["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(replicates, len(k)))
    ks, ns = k[idx].sum(axis=1), n[idx].sum(axis=1)
    r = np.where(ns > 0, 100 * ks / ns, np.nan)
    lo_p, hi_p = percentiles
    return (float(np.percentile(r, lo_p)), float(np.percentile(r, hi_p)))


def suspected_infection_onset(cultures: pd.DataFrame, antimicrobials: pd.DataFrame,
                              *, antimicrobial_first_h: float = 24, culture_first_h: float = 72) -> pd.Series:
    """Earliest suspected-infection onset for each admission (Seymour et al., JAMA 2016).

    A culture and an antimicrobial form a sequence when the culture follows the antimicrobial within
    24 h or the antimicrobial follows the culture within 72 h. Onset is the earlier of the two events.
    Both inputs have columns ``admission_id`` and ``dt``.
    """
    pairs = cultures.merge(antimicrobials, on="admission_id", suffixes=("_c", "_a"))
    gap = (pairs["dt_a"] - pairs["dt_c"]).dt.total_seconds() / 3600
    ok = pairs[(gap >= -antimicrobial_first_h) & (gap <= culture_first_h)]
    return ok[["dt_c", "dt_a"]].min(axis=1).groupby(ok["admission_id"]).min().rename("infection_onset")
