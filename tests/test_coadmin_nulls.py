"""Hand-counted checks on synthetic data. No patient data are used."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import coadmin_nulls as CO  # noqa: E402

T0 = pd.Timestamp("2020-01-01")


def cohort(ids, days=10):
    return pd.DataFrame({"admission_id": list(ids), "adm": T0, "dis": T0 + pd.Timedelta(days=days)})


def long(pairs):
    return pd.DataFrame({"admission_id": [a for a, _ in pairs],
                         "dt": [T0 + pd.Timedelta(hours=h) for _, h in pairs]})


def test_window_boundaries_are_inclusive():
    assert CO.pair_fraction(long([("a", 10.0)]), long([("a", 8.0)]), -2, 12)[0] == 1
    assert CO.pair_fraction(long([("a", 10.0)]), long([("a", 22.0)]), -2, 12)[0] == 1
    assert CO.pair_fraction(long([("a", 10.0)]), long([("a", 22.5)]), -2, 12)[0] == 0


def test_a_dose_with_two_comparators_counts_once():
    assert CO.pair_fraction(long([("a", 10.0)]), long([("a", 11.0), ("a", 12.0)]), -2, 12) == (1, 1, 1, 1)


def test_shift_breaks_a_same_day_link():
    r = CO.schedule_shift_null(long([("a", 120.0)]), long([("a", 121.0)]), cohort(["a"]), -2, 12,
                               max_shift_days=1, replicates=50, seed=1)
    assert r["observed_pct"] == 100.0 and r["expected_pct"] == 0.0


def test_shift_keeps_a_daily_round():
    other = long([("a", 121.0 + 24 * d) for d in range(-4, 5)])
    r = CO.schedule_shift_null(long([("a", 120.0)]), other, cohort(["a"]), -2, 12,
                               max_shift_days=3, replicates=50, seed=1)
    assert r["observed_pct"] == 100.0 and r["expected_pct"] == 100.0


def test_shift_keeps_spacing():
    # Doses 24 h apart, comparator just after the second. k=+1 pairs the first dose (1 of 2); k=-1 pairs none.
    r = CO.schedule_shift_null(long([("a", 96.0), ("a", 120.0)]), long([("a", 121.0)]), cohort(["a"]), -2, 12,
                               max_shift_days=1, replicates=4000, seed=2)
    assert r["observed_pct"] == 50.0
    assert abs(r["expected_pct"] - 25.0) < 2.0


def test_schedule_that_fills_the_stay_is_excluded_from_both_sides():
    doses = long([("a", 120.0), ("b", 1.0), ("b", 239.0)])
    other = long([("a", 121.0), ("b", 2.0), ("b", 240.0)])
    r = CO.schedule_shift_null(doses, other, cohort(["a", "b"]), -2, 12, max_shift_days=3, replicates=20, seed=1)
    assert r["n_admissions_shiftable"] == 1 and r["n_doses_shiftable"] == 1 and r["observed_pct"] == 100.0


def test_shift_stays_inside_the_stay():
    r = CO.schedule_shift_null(long([("a", 1.0)]), long([("a", 26.0)]), cohort(["a"]), -2, 12,
                               max_shift_days=1, replicates=20, seed=1)
    assert r["observed_pct"] == 0.0 and r["expected_pct"] == 100.0


def test_clock_null_keeps_the_observed_clock_times():
    ids = [f"a{i}" for i in range(20)]
    doses = long([(a, 9.0 + 24 * (i % 5)) for i, a in enumerate(ids)])
    draws = next(CO._null_draws(doses, cohort(ids), "clock", 1, 3))
    assert set(pd.to_datetime(draws["dt"]).dt.hour) == {9}


def test_same_seed_reproduces():
    ids = [f"a{i}" for i in range(12)]
    doses = long([(a, 100.0 + i) for i, a in enumerate(ids)])
    other = long([(a, 101.0 + i) for i, a in enumerate(ids) if i % 2 == 0])
    for anchor in CO.NULL_ANCHORS:
        assert (CO.null_pair_fraction(doses, other, cohort(ids), -2, 12, anchor=anchor, replicates=20, seed=9)
                == CO.null_pair_fraction(doses, other, cohort(ids), -2, 12, anchor=anchor, replicates=20, seed=9))


def test_overlap_summary_hand_count():
    r = CO.overlap_summary(long([("a", 0.0), ("a", 6.0)]),
                           long([("a", 1.0), ("a", 8.0), ("a", 8.0), ("a", 20.0)]), -2, 12)
    assert r == {"other_events_in_a_window": 2, "other_events_in_2plus_windows": 1,
                 "doses_with_2plus_other_events": 1}


def test_person_cluster():
    ids = [f"a{i}" for i in range(30)]
    doses = long([(a, 50.0) for a in ids])
    other = long([(a, 51.0) for i, a in enumerate(ids) if i % 2 == 0])
    pairs = pd.Series({a: f"p{i // 2}" for i, a in enumerate(ids)})
    assert CO.cluster_bootstrap_ci(doses, other, -2, 12, replicates=200, seed=5, cluster=pairs) == (50.0, 50.0)
    with pytest.raises(ValueError):
        CO.cluster_bootstrap_ci(doses, other, -2, 12, cluster=pairs.iloc[:3])


def test_suspected_infection_sequence():
    cultures = long([("first_abx", 20.0), ("first_culture", 10.0), ("too_late", 100.0)])
    abx = long([("first_abx", 0.0), ("first_culture", 80.0), ("too_late", 10.0)])
    onset = CO.suspected_infection_onset(cultures, abx)
    assert onset["first_abx"] == T0
    assert onset["first_culture"] == T0 + pd.Timedelta(hours=10)
    assert "too_late" not in onset.index


def test_empty_inputs():
    assert np.isnan(CO.null_pair_fraction(long([]), long([("a", 1.0)]), cohort(["a"]), -2, 12)[0])
    with pytest.raises(ValueError):
        CO.schedule_shift_null(long([]), long([]), cohort(["a"]), -2, 12, max_shift_days=0)
