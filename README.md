# Model Monitoring & Drift Detection Pipeline

Monitors a MovieLens rating-prediction model for three types of drift, using PSI and KS-test statistics, with synthetically injected drift so detection latency is measured against known ground truth.

## What's here

```
src/
  baseline_model.py   # GBT rating predictor (the "production model")
  drift_injector.py   # covariate / concept / data-quality drift injection
  drift_metrics.py    # PSI and KS-test implementations
  alerting.py         # threshold rule engine
tests/
  test_drift_metrics.py   # 11 tests verifying PSI/KS correctness
notebooks/
  run_pipeline.py     # end-to-end demo (saves plots to plots/)
```

## Quick start

```bash
pip install -r requirements.txt
python src/baseline_model.py       # downloads data, trains, prints val RMSE
pytest tests/ -v                   # 11 tests, all pass
python notebooks/run_pipeline.py   # full demo with detection latency table
```

MovieLens 25M is downloaded automatically on first run (~262 MB).

---

## PSI vs KL Divergence

Both PSI and KL divergence measure how much one probability distribution differs from another, and both use the formula `Σ p log(p/q)` as a building block. The difference is in the direction and symmetry:

**KL divergence** (`KL(P||Q) = Σ P(x) log(P(x)/Q(x))`) is asymmetric — it measures how well Q approximates P. If Q puts zero mass where P has non-zero mass, KL diverges to infinity.

**PSI** (`Σ (actual% - expected%) × ln(actual% / expected%)`) is symmetric in structure: it adds both directions together, making it more stable when the current distribution has bins the reference doesn't. In practice PSI = KL(current||reference) + KL(reference||current), which is the Jensen-Shannon divergence up to a constant. This symmetry makes PSI more robust to bin-edge accidents in production.

For monitoring purposes, PSI is preferred because:
1. It has well-established thresholds (<0.10 stable, 0.10-0.20 warning, >0.20 critical) from decades of credit-scoring practice.
2. It handles the zero-mass-bin problem more gracefully via epsilon smoothing.
3. The output is on an interpretable scale across different features.

---

## Covariate Shift vs Concept Drift

**Covariate shift** (also called dataset shift): the distribution of input features `P(X)` changes, but the relationship between features and labels `P(Y|X)` stays the same. Example: Action movies suddenly dominate traffic because a new franchise launched. The model's predictions will be off because it's seeing more of one slice, but if you feed it those inputs it would still predict correctly — the world's rating-to-feature relationship hasn't changed.

Detection: feature PSI fires. Fix: retrain on recent data to recalibrate the model to the new traffic mix.

**Concept drift**: `P(Y|X)` changes — the same features now predict a different label. Example: users in a new cultural context rate comedies consistently 1 star higher than the training population did. The features look identical to training data, so covariate PSI stays quiet, but the model's predictions are now systematically wrong.

Detection: prediction error PSI fires (covariate PSI stays below threshold). This is why you must monitor *both* input distributions and prediction errors, not just inputs. A model that passes all covariate drift checks can still be silently broken.

**Data quality drift**: a feature's null rate increases (or values go out of range) due to an upstream pipeline failure. Detected via null rate monitoring, which is a separate check from distributional PSI.

**The key experiment in this pipeline** (Scenario B): inject concept drift (label offset for Comedy genre), verify that `genre_Comedy` covariate PSI stays below the warning threshold while `error_psi` crosses critical. This proves the two drift types produce distinguishable signatures and that you need both monitoring signals.

---

## Detection latency

Latency = first window where an alert fires − injection window. Reported by `run_pipeline.py` in the console output under `DETECTION LATENCY SUMMARY`. Covariate shift typically fires within 1-2 windows; concept drift (via error PSI) fires within 2-4 windows once true labels are available; data quality drift fires progressively as null rate ramps.
