from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INSTRUMENTS = [f"INSTRUMENT_{i}" for i in range(1, 11)]


def load_data(data_dir: str | Path = "data/2024-12-31") -> pd.DataFrame:
    data_dir = Path(data_dir)
    dfs = [
        pd.read_csv(data_dir / "cash_rate.csv", parse_dates=["date"]),
        pd.read_csv(data_dir / "prices.csv", parse_dates=["date"]),
        pd.read_csv(data_dir / "signals.csv", parse_dates=["date"]),
        pd.read_csv(data_dir / "volumes.csv", parse_dates=["date"]),
    ]

    df = (
        reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return add_periodic_date_features(df)


def add_periodic_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = pd.to_datetime(df["date"])

    day_of_week = date.dt.dayofweek
    day_of_year = date.dt.dayofyear
    month = date.dt.month
    week_of_year = date.dt.isocalendar().week.astype(int)

    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["woy_sin"] = np.sin(2 * np.pi * week_of_year / 52.18)
    df["woy_cos"] = np.cos(2 * np.pi * week_of_year / 52.18)
    return df


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def cap_and_renorm(w: pd.Series, cap: float = 0.25) -> pd.Series:
    w = w.clip(lower=0.0, upper=cap)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / s


def build_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "date"]


def make_labels_next_return(df: pd.DataFrame, inst: str) -> pd.Series:
    x = df[inst].astype(float)
    return x.shift(-1) / x - 1.0


def train_predict_scores(
    df: pd.DataFrame, min_train: int = 200, alpha: float = 10.0
) -> pd.Series:
    df = df.sort_values("date").reset_index(drop=True)
    feat_cols = build_feature_cols(df)
    X = df.loc[:-2, feat_cols].copy()

    preds: dict[str, float] = {}

    for inst in INSTRUMENTS:
        y = make_labels_next_return(df, inst).loc[:-2]
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        X_i = X.loc[mask]
        y_i = y.loc[mask]

        if len(X_i) < min_train:
            preds[inst] = 0.0
            continue

        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=0)),
            ]
        )
        model.fit(X_i.values, y_i.values)

        x_last = df.loc[df.index[-1], feat_cols].astype(float)
        if x_last.isna().any():
            preds[inst] = 0.0
        else:
            preds[inst] = float(model.predict([x_last.values])[0])

    return pd.Series(preds)


def construct_weights(
    df: pd.DataFrame,
    beta: float = 5.0,
    cap: float = 0.25,
    smooth_alpha: float = 0.35,
    prev_weights: pd.Series | None = None,
) -> pd.Series:
    pred = train_predict_scores(df)

    last = df.sort_values("date").iloc[-1]
    vols = pd.Series({inst: float(last.get(f"{inst}_vol", np.nan)) for inst in INSTRUMENTS})
    vols = vols.replace([np.inf, -np.inf], np.nan).fillna(vols.median())
    vols = vols.clip(lower=1e-6)

    scores = pred / vols
    w = pd.Series(softmax(beta * scores.values), index=INSTRUMENTS)
    w = cap_and_renorm(w, cap=cap)

    if prev_weights is not None:
        prev_weights = prev_weights.reindex(INSTRUMENTS).fillna(0.0)
        prev_weights = prev_weights / prev_weights.sum()
        w = smooth_alpha * w + (1.0 - smooth_alpha) * prev_weights
        w = cap_and_renorm(w, cap=cap)

    return w


def write_submission(
    weights: pd.Series, team_name: str, round_n: int, out_path: str | Path = "."
) -> str:
    out = pd.DataFrame({"asset": weights.index, "weight": weights.values})
    fname = f"{team_name}_round_{round_n}.csv"
    out.to_csv(Path(out_path) / fname, index=False)
    return fname


def main() -> None:
    df = load_data()

    # prev = pd.read_csv("myteam_round_3.csv").set_index("asset")["weight"]
    prev = None

    w = construct_weights(df, prev_weights=prev)
    fname = write_submission(w, team_name="myteam", round_n=4, out_path=".")
    print("Wrote:", fname)
    print(w)


if __name__ == "__main__":
    main()
