import pandas as pd
from pandas import DataFrame
from typing import Literal, List
import numpy as np

class FeatureExtractor:
    def __init__(
        self,
        past_knowledge: DataFrame,
        cyclical_feature_names: List[str],
        freq: str = "D",
        lag_size: int = 30,
        window_size: int = 30,
    ):
        self.PAST_KNOWLEDGE = past_knowledge.sort_values(by="datetime")
        self.cyclical_feature_names = cyclical_feature_names
        self.lag_size = lag_size
        self.window_size = window_size
        self.freq = freq

    def transform(self, dates_to_predict: pd.DatetimeIndex) -> DataFrame:
        df = self._get_all_ranges(dates_to_predict)
        self.full_df = df.join(self.PAST_KNOWLEDGE, how="left")

        return (
            df.pipe(self._start_pipeline)
            .pipe(self._add_lag_features)
            .pipe(self._add_rolling_window_features)
            .pipe(self._add_exponential_moving_features)
            .pipe(self._drop_columns_with_same_values)
            .pipe(self._expand_datetime)
            .pipe(self._add_fourier_features)
            .pipe(
                lambda df: df.astype(
                    {
                        col: "int32"
                        for col in df.select_dtypes(["int", "uint32"]).columns
                    }
                )
            )
            .pipe(
                lambda df: df.astype(
                    {col: "float32" for col in df.select_dtypes("float").columns}
                )
            )
            .bfill()
            .loc[dates_to_predict, :]
        )

    def _start_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().sort_index()

    def _get_all_ranges(self, dates_to_predict: pd.DatetimeIndex) -> pd.DataFrame:
        start_date = min(dates_to_predict.min(), self.PAST_KNOWLEDGE.index.min())
        end_date = max(dates_to_predict.max(), self.PAST_KNOWLEDGE.index.max())
        complete_date_range = pd.date_range(
            start=start_date, end=end_date, freq=self.freq
        )
        return pd.DataFrame(index=complete_date_range)

    def _add_lag_features(
        self,
        df: DataFrame,
        fillna_with: Literal["ffill", "bfill"] | None = "bfill",
    ) -> DataFrame:
        columns_to_use = self.PAST_KNOWLEDGE.select_dtypes(
            include=["number", "object"]
        ).columns.tolist()

        created_features = [
            self.full_df[col].shift(i).rename(f"{col}_lag_{i}")
            for i in range(1, self.lag_size + 1)
            for col in columns_to_use
        ]

        lags_df = pd.concat(created_features, axis=1)

        df = df.join(
            lags_df,
            how="left",
        )

        if fillna_with == "ffill":
            df = df.ffill()
        elif fillna_with == "bfill":
            df = df.bfill()

        return df

    def _add_rolling_window_features(
        self,
        df: DataFrame,
        fillna_with: Literal["ffill", "bfill"] | None = "ffill",
    ) -> DataFrame:
        columns_to_use = self.PAST_KNOWLEDGE.select_dtypes(
            include=["float"]
        ).columns.tolist()

        metrics = ["mean", "std", "min", "max", "median", "var"]

        created_features = [
            (
                self.full_df[col]
                .rolling(window=size, min_periods=1)
                .agg(metrics)
                .rename(columns=lambda metric: f"{col}_rw{size}_{metric}")
            )
            for size in range(2, self.window_size + 1)
            for col in columns_to_use
        ]

        window_df = pd.concat(created_features, axis=1)

        df = df.join(
            window_df,
            how="left",
        )

        if fillna_with == "ffill":
            df = df.ffill()
        elif fillna_with == "bfill":
            df = df.bfill()

        return df

    def _drop_columns_with_same_values(self, df: DataFrame, threshold=0.9) -> DataFrame:
        to_drop = [
            col
            for col in df.columns
            if df[col].value_counts(normalize=True, dropna=False).values[0] >= threshold
        ]
        return df.drop(columns=to_drop)

    def _add_exponential_moving_features(
        self, df: pd.DataFrame, up_to: int = 30
    ) -> pd.DataFrame:
        columns_to_use = self.PAST_KNOWLEDGE.select_dtypes(
            include=["float"]
        ).columns.tolist()

        metrics = ["mean", "std", "var"]

        created_features = [
            (
                self.full_df[col]
                .ewm(span=span, adjust=False)
                .agg(metrics)
                .rename(columns=lambda metric: f"{col}_em_{span}_{metric}")
            )
            for span in range(2, up_to + 1)
            for col in columns_to_use
        ]

        exponential_moving_df = pd.concat(created_features, axis=1)

        df = df.join(
            exponential_moving_df,
            how="left",
        )
        return df

    def _expand_datetime(self, df: DataFrame) -> DataFrame:
        return df.assign(
            **{
                "year": lambda a_df: a_df.index.year,
                "month": lambda a_df: a_df.index.month,
                "day": lambda a_df: a_df.index.day,
                "hour": lambda a_df: a_df.index.hour,
                "day_of_year": lambda a_df: a_df.index.dayofyear,
                "week_of_year": lambda a_df: a_df.index.isocalendar().week,
                "quarter": lambda a_df: a_df.index.quarter,
                # "season": lambda a_df: a_df.index.month % 12 // 3 + 1,
                "is_weekend": lambda a_df: np.vectorize({True: 1, False: 0}.get)(
                    a_df.index.weekday >= 5
                ),
            }
        )

    def _add_fourier_features(self, df: pd.DataFrame, num_terms: int = 7) -> DataFrame:
        for col, max_val in self.cyclical_feature_names.items():
            source = self._get_column_source(df, col)

            for i in range(1, num_terms + 1):
                operation = 2 * np.pi * i * source[col] / max_val

                df[f"fourier_sin_{col}_{i}"] = np.sin(operation)
                df[f"fourier_cos_{col}_{i}"] = np.cos(operation)

        return df

    def _get_column_source(self, df: DataFrame, col: str) -> List[str]:
        if col in df.columns:
            source = df
        elif col in self.PAST_KNOWLEDGE.columns:
            source = self.PAST_KNOWLEDGE
        else:
            raise KeyError(f"{col} not found both in df and past knowledge.")
        return source
