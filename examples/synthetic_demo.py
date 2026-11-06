"""Run the null models on simulated admissions. The numbers below are not study results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import coadmin_nulls as CO  # noqa: E402

rng = np.random.default_rng(2026)
start = pd.Timestamp("2020-01-01")
ids = [f"adm{i:03d}" for i in range(200)]
stay_days = rng.integers(10, 40, size=len(ids))
cohort = pd.DataFrame({"admission_id": ids,
                       "adm": start,
                       "dis": [start + pd.Timedelta(days=int(d), hours=23, minutes=59) for d in stay_days]})

dose_rows, other_rows = [], []
for admission, days in zip(ids, stay_days):
    first = int(rng.integers(2, max(3, days - 6)))
    for day in range(first, min(first + int(rng.integers(1, 6)), int(days))):
        when = start + pd.Timedelta(days=day, hours=9, minutes=int(rng.integers(0, 60)))
        dose_rows.append({"admission_id": admission, "dt": when})
        if rng.random() < 0.3:   # simulated comparator shortly after the dose
            other_rows.append({"admission_id": admission, "dt": when + pd.Timedelta(minutes=int(rng.integers(5, 90)))})
    for day in range(int(days)):
        if rng.random() < 0.15:  # simulated comparator on a morning round, unrelated to the dose
            other_rows.append({"admission_id": admission, "dt": start + pd.Timedelta(days=day, hours=9, minutes=int(rng.integers(0, 60)))})

doses, other = pd.DataFrame(dose_rows), pd.DataFrame(other_rows)
LO, HI, REPS = -2, 12, 200

k, n, ka, na = CO.pair_fraction(doses, other, LO, HI)
print(f"Observed: {k}/{n} doses ({100 * k / n:.1f}%); {ka}/{na} admissions")
print("Bootstrap interval, resampling admissions: {:.1f}-{:.1f}".format(*CO.cluster_bootstrap_ci(doses, other, LO, HI)))
for anchor, label in (("stay", "within-admission"), ("day", "same-day"), ("clock", "clock-preserving")):
    mean, lo, hi = CO.null_pair_fraction(doses, other, cohort, LO, HI, anchor=anchor, replicates=REPS)
    print(f"{label}: expected {mean:.1f} [{lo:.1f}-{hi:.1f}]; difference {100 * k / n - mean:+.1f}")
for days in (1, 3, 7):
    r = CO.schedule_shift_null(doses, other, cohort, LO, HI, max_shift_days=days, replicates=REPS)
    print(f"schedule shift up to {days} days: observed {r['observed_pct']:.1f}, expected {r['expected_pct']:.1f} "
          f"[{r['lo_pct']:.1f}-{r['hi_pct']:.1f}], excluded {r['n_admissions_total'] - r['n_admissions_shiftable']}")
