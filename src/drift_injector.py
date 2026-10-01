"""
Simulates three types of drift in time-windowed inference batches.
Drift starts at a configurable window index so pre-injection windows
can serve as a statistical baseline (KS p > 0.05 vs reference).
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class DriftConfig:
    start_window: int = 10

    # Covariate shift: oversample rows matching this genre
    covariate_genre: str = "Action"
    covariate_oversample_factor: float = 4.0

    # Concept drift: add offset to true ratings for this genre
    concept_genre: str = "Comedy"
    concept_offset: float = 1.2

    # Data quality drift: inject NaN into this feature at increasing rate
    quality_feature: str = "user_mean_rating"
    quality_null_rate_per_window: float = 0.07  # 7% more nulls per window after start


def create_windows(prod_df: pd.DataFrame, n_windows: int = 20) -> List[pd.DataFrame]:
    """
    Split prod_df into N chronologically ordered, non-overlapping windows.
    Each window represents a "day" of inference traffic.
    """
    prod_df = prod_df.sort_values("timestamp").reset_index(drop=True)
    indices = np.array_split(np.arange(len(prod_df)), n_windows)
    return [prod_df.iloc[idx].copy() for idx in indices]


def inject_covariate_shift(window: pd.DataFrame, config: DriftConfig, window_idx: int) -> pd.DataFrame:
    """
    Skews the genre distribution by oversampling rows matching config.covariate_genre.
    Simulates a traffic pattern change (e.g., a genre goes viral).
    """
    if window_idx < config.start_window:
        return window
    genre_col = f"genre_{config.covariate_genre}"
    if genre_col not in window.columns:
        return window
    genre_rows = window[window[genre_col] == 1]
    extra = genre_rows.sample(
        frac=config.covariate_oversample_factor,
        replace=True,
        random_state=window_idx,
    )
    return pd.concat([window, extra]).reset_index(drop=True)


def inject_concept_drift(window: pd.DataFrame, config: DriftConfig, window_idx: int) -> pd.DataFrame:
    """
    Perturbs the true rating-generating function by adding an offset to ratings
    for a specific genre. The model's input features don't change, but its
    predictions systematically diverge from the shifted true labels.
    """
    if window_idx < config.start_window:
        return window
    window = window.copy()
    genre_col = f"genre_{config.concept_genre}"
    if genre_col not in window.columns:
        return window
    mask = window[genre_col] == 1
    window.loc[mask, "rating"] = (window.loc[mask, "rating"] + config.concept_offset).clip(0.5, 5.0)
    return window


def inject_data_quality(window: pd.DataFrame, config: DriftConfig, window_idx: int) -> pd.DataFrame:
    """
    Injects NaN into a feature at a rate that increases each window after start.
    Simulates an upstream pipeline failure gradually corrupting a feature.
    """
    if window_idx < config.start_window:
        return window
    window = window.copy()
    windows_elapsed = window_idx - config.start_window + 1
    null_rate = min(config.quality_null_rate_per_window * windows_elapsed, 0.95)
    rng = np.random.RandomState(window_idx)
    null_mask = rng.random(len(window)) < null_rate
    window.loc[null_mask, config.quality_feature] = np.nan
    return window
