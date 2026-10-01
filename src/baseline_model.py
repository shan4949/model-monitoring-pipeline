"""
Trains a GradientBoosting rating predictor on MovieLens 25M.
This model is the "production model" that the monitoring layer watches.
"""
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = Path(__file__).parent.parent / "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
SAMPLE_SIZE = 1_500_000

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Horror", "Sci-Fi", "Animation"]
FEATURE_COLS = (
    ["user_mean_rating", "user_rating_count", "movie_mean_rating", "movie_rating_count"]
    + [f"genre_{g}" for g in GENRES]
)

@dataclass
class ModelPackage:
    model: GradientBoostingRegressor
    feature_cols: list
    fill_values: pd.Series
    val_rmse: float


def download_movielens(data_dir: Path = DATA_DIR) -> Path:
    ml_path = data_dir / "ml-25m"
    if ml_path.exists() and (ml_path / "ratings.csv").exists():
        return ml_path
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-25m.zip"
    print("Downloading MovieLens 25M (~262MB)...")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    zip_path.unlink()
    return ml_path


def _featurize(df: pd.DataFrame, movies: pd.DataFrame, user_stats: pd.DataFrame, movie_stats: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(movies, on="movieId", how="left")
    for g in GENRES:
        df[f"genre_{g}"] = df["genres"].str.contains(g, regex=False).astype(float)
    df = df.join(user_stats, on="userId").join(movie_stats, on="movieId")
    return df


def load_and_prepare(data_dir: Path = DATA_DIR, sample_size: int = SAMPLE_SIZE):
    """
    Returns (train_df, val_df, prod_df) with engineered features.
    60% train / 10% val / 30% prod, chronologically ordered.
    User/movie stats computed from train only — no leakage.
    """
    ml_path = download_movielens(data_dir)
    ratings = pd.read_csv(ml_path / "ratings.csv")
    movies = pd.read_csv(ml_path / "movies.csv")[["movieId", "genres"]]

    if sample_size and len(ratings) > sample_size:
        ratings = ratings.sample(sample_size, random_state=42).reset_index(drop=True)

    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(ratings)
    train_raw = ratings.iloc[: int(n * 0.6)].copy()
    val_raw = ratings.iloc[int(n * 0.6) : int(n * 0.7)].copy()
    prod_raw = ratings.iloc[int(n * 0.7) :].copy()

    user_stats = (
        train_raw.groupby("userId")["rating"]
        .agg(user_mean_rating="mean", user_rating_count="count")
    )
    movie_stats = (
        train_raw.groupby("movieId")["rating"]
        .agg(movie_mean_rating="mean", movie_rating_count="count")
    )

    train_df = _featurize(train_raw, movies, user_stats, movie_stats)
    val_df = _featurize(val_raw, movies, user_stats, movie_stats)
    prod_df = _featurize(prod_raw, movies, user_stats, movie_stats)
    return train_df, val_df, prod_df


def train(data_dir: Path = DATA_DIR, sample_size: int = SAMPLE_SIZE) -> tuple:
    print(f"Loading and preparing data (sample={sample_size:,})...")
    train_df, val_df, prod_df = load_and_prepare(data_dir, sample_size)

    fill_values = train_df[FEATURE_COLS].median()
    X_train = train_df[FEATURE_COLS].fillna(fill_values)
    y_train = train_df["rating"]
    X_val = val_df[FEATURE_COLS].fillna(fill_values)
    y_val = val_df["rating"]

    print(f"Training GBT on {len(X_train):,} samples...")
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, n_iter_no_change=10, validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    print(f"Validation RMSE: {val_rmse:.4f}")

    pkg = ModelPackage(model=model, feature_cols=FEATURE_COLS, fill_values=fill_values, val_rmse=val_rmse)
    return pkg, train_df, val_df, prod_df


if __name__ == "__main__":
    pkg, train_df, val_df, prod_df = train()
    print(f"\nSplit sizes — train: {len(train_df):,}  val: {len(val_df):,}  prod: {len(prod_df):,}")
    print(f"Validation RMSE (reference performance): {pkg.val_rmse:.4f}")
