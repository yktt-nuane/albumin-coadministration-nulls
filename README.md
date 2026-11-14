# Time-pairing nulls for drug administrations

Code and aggregate sensitivity tables for the time-pairing analysis in "Timing of first recorded non-operative hypertonic albumin doses and nearby furosemide records in adults with a positive sepsis screen: A single-centre cohort study".

The repository contains no patient data. `results/` holds aggregate tables only (counts, percentages, medians and simulated ranges). `examples/synthetic_demo.py` runs the code on simulated admissions.

## Sensitivity tables

| File | Contents |
|---|---|
| `results/S1_window_sensitivity.csv` | Pairing in each window (−2 to +12 h, −2 to +2 h, 0 to +6 h, −6 to +6 h, −12 to +12 h) with both nulls. Confidence intervals in this file resample admissions. |
| `results/S2_null_models.csv` | Both nulls by population and drug, with schedule shifts of up to 1, 3 and 7 days and the admissions excluded for lack of a feasible shift. |
| `results/S3_aligned_and_after_onset.csv` | Admissions with an aligned screen, and doses at or after suspected-infection onset. |
| `results/S4_theatre_records.csv` | Operating-theatre drug records: albumin and furosemide. |
| `results/S5_same_time_records.csv` | Albumin records sharing an admission and time, and the analysis counting them once. |
| `results/S6_pairing_units.csv` | Pairing with confidence intervals from resampling admissions and from resampling individuals (the article reports individuals). |
| `results/S7_screen_components.csv` | Components of the sepsis screen and the suspected-infection sequence, in all, treated and untreated admissions. |

All analyses in these tables are exploratory. Percentile ranges describe variability under each simulated null and are not confidence intervals for the observed-minus-expected difference. Neither null conditions on clinical state. The pairing window is asymmetric (2 h before to 12 h after a dose), and drugs given on common medication rounds can share clock times, so the clock-preserving null keeps the distribution of dose clock times. A window as wide as the null's 24-h block (−12 to +12 h) leaves the clock-preserving expectation equal to the observed value by construction.

## What it does

For each dose of an index drug, the statistic is whether a comparator drug was recorded in a window around the dose. The percentage of doses paired is compared with four simulated nulls:

| Null | What moves | What is kept |
|---|---|---|
| Within-admission | Each dose, uniformly over the admission | Admission and number of doses |
| Same-day | Each dose, uniformly within its 24-h block from admission, truncated at discharge | Also the day of the dose |
| Clock-preserving | Each dose, within that block at clock times drawn from all index doses | Also the distribution of clock times, conditioned on the available interval in blocks truncated at discharge |
| Schedule shift | The admission's whole schedule, by a random non-zero whole number of days up to a maximum, staying in hospital | Also dose spacing and each dose's clock time |

Comparator times are never moved. Admissions with no feasible schedule shift are excluded, and the observed percentage is recomputed on the same admissions. Boundary rules and the test statistic are in the docstrings of `src/coadmin_nulls.py`.

## Input format

- `doses`, `other`: `admission_id`, `dt` (datetime). One row is one administration record.
- `cohort`: `admission_id`, `adm` (start of the admission day), `dis` (end of the discharge day). Every admission in `doses` must appear here.

## Parameters used in the article

Window from 2 h before to 12 h after the dose. Shorter and symmetric windows: 2 h either side, 0 to 6 h after, 6 h either side, 12 h either side. 2,000 null replicates, 1,000 bootstrap replicates resampling individuals (the `cluster` argument), seed 42, 2.5th and 97.5th percentiles. Schedule shift up to 3 days, with 1 and 7 days as sensitivity analyses. The demo uses 200 null replicates and resamples admissions.

## Running it

```
pip install -r requirements.txt
python examples/synthetic_demo.py
python -m pytest tests
```

## Licence

MIT. See `LICENSE`.
