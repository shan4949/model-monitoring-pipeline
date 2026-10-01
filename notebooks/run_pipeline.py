"""
Full end-to-end demonstration of the drift detection pipeline.
Run from the project root: python notebooks/run_pipeline.py

Produces:
  - Console output with per-window PSI values and alerts
  - plots/psi_covariate.png
  - plots/psi_concept.png
  - plots/psi_quality.png
  - plots/detection_latency_summary.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alerting import AlertEngine
from baseline_model import train
from drift_injector import (
    DriftConfig,
    create_windows,
    inject_covariate_shift,
    inject_concept_drift,
    inject_data_quality,
)
from drift_metrics import monitor_window

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

N_WINDOWS = 20
INJECTION_WINDOW = 8  # drift starts here; windows 0-7 are clean baseline


def run_scenario(name, inject_fn, windows, reference_df, pkg):
    """Run monitoring over all windows with one injection function applied."""
    engine = AlertEngine()
    records = []
    for i, w in enumerate(windows):
        injected = inject_fn(w, i)
        metrics = monitor_window(reference_df, injected, pkg.feature_cols, pkg.model, pkg.fill_values)
        alerts = engine.process(metrics, i)
        records.append({"window": i, "metrics": metrics, "n_alerts": len(alerts)})
        if alerts:
            for a in alerts:
                print(f"  {a}")
    return records, engine


def plot_psi_over_time(records, title, highlight_features, save_path, injection_window):
    windows = [r["window"] for r in records]
    fig, axes = plt.subplots(len(highlight_features) + 2, 1, figsize=(12, 3 * (len(highlight_features) + 2)), sharex=True)

    for ax, feat in zip(axes, highlight_features):
        vals = [r["metrics"]["feature_psi"].get(feat, np.nan) for r in records]
        ax.plot(windows, vals, marker="o", linewidth=2, label=feat)
        ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="warn (0.10)")
        ax.axhline(0.20, color="red", linestyle="--", linewidth=1, label="crit (0.20)")
        ax.axvline(injection_window - 0.5, color="purple", linestyle=":", linewidth=2, label="injection point")
        ax.set_ylabel("PSI")
        ax.set_title(f"Feature PSI — {feat}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    # Prediction PSI
    ax = axes[-2]
    pred_vals = [r["metrics"].get("pred_psi", np.nan) for r in records]
    ax.plot(windows, pred_vals, marker="s", color="teal", linewidth=2)
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1)
    ax.axhline(0.20, color="red", linestyle="--", linewidth=1)
    ax.axvline(injection_window - 0.5, color="purple", linestyle=":", linewidth=2)
    ax.set_ylabel("PSI")
    ax.set_title("Prediction Distribution PSI")
    ax.set_ylim(bottom=0)

    # Error PSI
    ax = axes[-1]
    err_vals = [r["metrics"].get("error_psi", np.nan) for r in records]
    ax.plot(windows, err_vals, marker="^", color="crimson", linewidth=2)
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1)
    ax.axhline(0.20, color="red", linestyle="--", linewidth=1)
    ax.axvline(injection_window - 0.5, color="purple", linestyle=":", linewidth=2)
    ax.set_xlabel("Window (day)")
    ax.set_ylabel("PSI")
    ax.set_title("Prediction Error PSI (concept drift signal)")
    ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def print_latency_table(scenarios):
    print("\n" + "=" * 60)
    print("DETECTION LATENCY SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<22} {'Feature':<22} {'First Alert Window':<20} {'Latency (days)'}")
    print("-" * 72)
    for scenario_name, engine, config in scenarios:
        for feature, label in [
            (f"genre_{config.covariate_genre}", "covariate feature"),
            ("errors", "error PSI"),
            (config.quality_feature, "quality feature"),
        ]:
            w = engine.first_alert_window(feature, "critical")
            if w is not None:
                latency = w - INJECTION_WINDOW
                print(f"{scenario_name:<22} {label:<22} {w:<20} {latency}")


def main():
    print("Step 1 — Train baseline model")
    pkg, train_df, val_df, prod_df = train()
    print(f"  Val RMSE (reference): {pkg.val_rmse:.4f}")

    print("\nStep 2 — Create time windows")
    windows = create_windows(prod_df, N_WINDOWS)
    sizes = [len(w) for w in windows]
    assert all(w1.timestamp.max() <= w2.timestamp.min() for w1, w2 in zip(windows, windows[1:])), \
        "Windows are not chronologically ordered!"
    print(f"  {N_WINDOWS} windows, sizes: min={min(sizes):,} max={max(sizes):,}")
    print("  Non-overlapping chronological order verified.")

    # Reference: first half of training data (dense, uncontaminated)
    reference_df = train_df.sample(min(20_000, len(train_df)), random_state=0)
    config = DriftConfig(start_window=INJECTION_WINDOW)

    print("\nStep 3 — Scenario A: Covariate shift")
    cov_records, cov_engine = run_scenario(
        "covariate",
        lambda w, i: inject_covariate_shift(w, config, i),
        windows, reference_df, pkg,
    )
    plot_psi_over_time(
        cov_records,
        f"Scenario A: Covariate Shift (genre={config.covariate_genre}, injection@w{INJECTION_WINDOW})",
        [f"genre_{config.covariate_genre}", "movie_mean_rating"],
        PLOTS_DIR / "psi_covariate.png",
        INJECTION_WINDOW,
    )

    print("\nStep 4 — Scenario B: Concept drift")
    concept_records, concept_engine = run_scenario(
        "concept",
        lambda w, i: inject_concept_drift(w, config, i),
        windows, reference_df, pkg,
    )
    plot_psi_over_time(
        concept_records,
        f"Scenario B: Concept Drift (genre={config.concept_genre}, offset={config.concept_offset}, injection@w{INJECTION_WINDOW})",
        [f"genre_{config.concept_genre}", "user_mean_rating"],
        PLOTS_DIR / "psi_concept.png",
        INJECTION_WINDOW,
    )

    # KEY RESULT: covariate PSI should stay low but error PSI should rise
    pre_cov_psi = np.nanmean([
        r["metrics"]["feature_psi"].get(f"genre_{config.concept_genre}", np.nan)
        for r in concept_records[:INJECTION_WINDOW]
    ])
    post_cov_psi = np.nanmean([
        r["metrics"]["feature_psi"].get(f"genre_{config.concept_genre}", np.nan)
        for r in concept_records[INJECTION_WINDOW:]
    ])
    post_error_psi = np.nanmean([
        r["metrics"].get("error_psi", np.nan)
        for r in concept_records[INJECTION_WINDOW:]
    ])
    print(f"\n  KEY RESULT (concept drift):")
    print(f"    genre_{config.concept_genre} covariate PSI before: {pre_cov_psi:.4f}")
    print(f"    genre_{config.concept_genre} covariate PSI after:  {post_cov_psi:.4f}  (should stay low)")
    print(f"    Error PSI after injection:                          {post_error_psi:.4f}  (should be high)")
    print(f"    → Covariate PSI {'STABLE' if post_cov_psi < 0.15 else 'UNEXPECTEDLY HIGH'}, "
          f"Error PSI {'DETECTED' if post_error_psi > 0.10 else 'NOT DETECTED'}")

    print("\nStep 5 — Scenario C: Data quality drift")
    quality_records, quality_engine = run_scenario(
        "quality",
        lambda w, i: inject_data_quality(w, config, i),
        windows, reference_df, pkg,
    )
    null_rates = [
        r["metrics"]["null_rates"].get(config.quality_feature, 0)
        for r in quality_records
    ]
    plot_psi_over_time(
        quality_records,
        f"Scenario C: Data Quality Drift (feature={config.quality_feature}, injection@w{INJECTION_WINDOW})",
        [config.quality_feature, "movie_mean_rating"],
        PLOTS_DIR / "psi_quality.png",
        INJECTION_WINDOW,
    )
    print(f"  Null rates for '{config.quality_feature}' — pre: {null_rates[INJECTION_WINDOW-1]:.2%}, "
          f"post final: {null_rates[-1]:.2%}")

    print_latency_table([
        ("covariate_shift", cov_engine, config),
        ("concept_drift", concept_engine, config),
        ("data_quality", quality_engine, config),
    ])
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
