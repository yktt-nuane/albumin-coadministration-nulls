# Time-pairing nulls for drug administrations

Code for the time-pairing analysis in "Timing of hypertonic albumin and co-administered furosemide in sepsis: A single-centre cohort study".

The repository contains no patient data and no study results. `examples/synthetic_demo.py` runs the code on simulated admissions.

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

Window from 2 h before to 12 h after the dose. Shorter and symmetric windows: 2 h either side, 0 to 6 h after, 6 h either side, 12 h either side. 2,000 null replicates, 1,000 bootstrap replicates, seed 42, 2.5th and 97.5th percentiles. Schedule shift up to 3 days, with 1 and 7 days as sensitivity analyses.

## Running it

```
pip install -r requirements.txt
python examples/synthetic_demo.py
python -m pytest tests
```

## Licence

MIT. See `LICENSE`.
