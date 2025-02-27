import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from typing import Any, List, Union, Literal, Optional, Protocol, runtime_checkable


@runtime_checkable
class FoldProtocol(Protocol):
    def split(self, X, y=None, groups=None) -> Any:
        pass


class TargetEncoder:
    def __init__(
        self,
        random_state: Optional[int] = None,
        columns: Union[List[str], str, Literal["auto"]] = "auto",
        smoothing: Union[float, int] = 20,
        cv: Union[int, FoldProtocol] = 5,
        drop_after_transform: bool = True,
        fill_zeros_with_mean: bool = True,
        fillna_with_mean: bool = True,
        epsilon: Union[float | int] = 1e-10,
        stats_to_include: List[str] = [
            "mean",
            "median",
            "var",
            "std",
            "skew",
            "min",
            "max",
        ],
    ):
        self.SEED = random_state
        self.smoothing = smoothing
        self.columns = columns
        self.drop_after_transform = drop_after_transform
        self.fillna_with_mean = fillna_with_mean
        self.fill_zeros_with_mean = fill_zeros_with_mean
        self.cv = cv
        self.EPSILON = epsilon
        self.mappings = {}
        self.stats_to_include = stats_to_include

    def _get_column_names_to_process(
        self,
        X: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        if isinstance(self.columns, str) and self.columns == "auto":
            if X is None:
                raise ValueError("X must be provided when columns='auto'")

            LOWER_COUNT_LIMIT = 10
            categorical_columns = X.select_dtypes(include="object").columns
            high_cardinality_columns = [
                col
                for col in categorical_columns
                if X[col].nunique() >= LOWER_COUNT_LIMIT
            ]
            return high_cardinality_columns

        elif isinstance(self.columns, str):
            return [self.columns]
        elif isinstance(self.columns, list):
            return self.columns

    def _get_fold(self) -> FoldProtocol:
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.SEED)
        elif isinstance(self.cv, FoldProtocol):
            return self.cv
        raise ValueError(
            "cv must be an integer or an object implementing a 'split' method"
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "TargetEncoder":
        X = X.copy()
        y = y.copy()
        self.columns = self._get_column_names_to_process(X)
        self.TARGET = y.columns[0]
        fold = self._get_fold()

        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in X")

            encodings = [
                self._generate_encodings(X.iloc[train_idx], y.iloc[train_idx], column)
                for train_idx, _ in fold.split(X, y)
            ]

            self.mappings[column] = self._calculate_weighted_means_of_encodings(
                encodings, column
            )

        return self

    def _generate_encodings(
        self, X_train_fold: pd.DataFrame, y_train_fold, column: str
    ):
        dataset = pd.concat([X_train_fold, y_train_fold], axis=1)
        global_mean = dataset[self.TARGET].agg(self.stats_to_include)

        grouped_target = dataset.groupby(column)[self.TARGET]
        grouped_target_count = grouped_target.count().iloc[0]

        numerator = grouped_target.agg(self.stats_to_include) + (
            (self.smoothing * global_mean) + self.EPSILON
        )
        denominator = (grouped_target_count * self.smoothing) + self.EPSILON

        result = numerator / denominator

        count_by_category = grouped_target.count().rename("count")
        return pd.concat([result, count_by_category], axis=1)

    def _calculate_weighted_means_of_encodings(
        self, encodings: List[pd.DataFrame], column: str
    ) -> dict:
        combined_encodings = pd.concat(encodings)
        group = combined_encodings.groupby(column)

        return {
            stat: group.apply(lambda x: (x[stat] * x["count"]).sum() / x["count"].sum())
            for stat in self.stats_to_include
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for column in self.columns:
            column_mappings = self.mappings[column]

            for stat in self.stats_to_include:
                generated_column = f"te_{column}_{stat}"

                X[generated_column] = (
                    X[column].map(column_mappings[stat]).astype("float32")
                )

                if self.fillna_with_mean:
                    mean_value = column_mappings[stat].mean().astype("float32")
                    X[generated_column] = X[generated_column].fillna(mean_value)

                if self.fill_zeros_with_mean:
                    mean_value = column_mappings[stat].mean().astype("float32")
                    X.loc[X[generated_column] == 0, generated_column] = mean_value

        if self.drop_after_transform:
            X.drop(columns=self.columns, inplace=True)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def get_data(address: str, target: str, SEED: int):
    auto_df = pd.read_csv(address)

    X = auto_df.loc[:, auto_df.columns != target]
    y = auto_df.loc[:, auto_df.columns == target]

    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def main():
    address = "dataset/autos.csv"
    SEED = 76
    TARGET = "price"

    X_train, X_test, y_train, y_test = get_data(address, TARGET, SEED)
    
    te = TargetEncoder()
    te.fit(X_train, y_train)
    print(te.transform(X_train))
    print(te.transform(X_test))


if __name__ == "__main__":
    main()
