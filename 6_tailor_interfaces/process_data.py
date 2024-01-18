import pandas as pd
from abc import abstractmethod, ABC
from typing import TypeVar, Iterable

NumberOrStr = TypeVar("NumberOrStr", int, float, str)


class SeriesImputer(ABC):
    @abstractmethod
    def impute(self, series: pd.Series) -> pd.Series:
        pass


class MostFrequentSeriesImputer(SeriesImputer):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mode()[0])


class MedianSeriesImputer(SeriesImputer):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.median())


class MeanSeriesImputer(SeriesImputer):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mean())


class DataFrameImputer(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class GroupStatisticImputer(DataFrameImputer):
    def __init__(
        self, strategy: SeriesImputer, group_feature: str, target_feature: str
    ):
        self.strategy = strategy
        self.group_feature = group_feature
        self.target_feature = target_feature

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.target_feature] = df.groupby(self.group_feature)[
            self.target_feature
        ].transform(self.strategy.impute)
        return df


class ConstantImputer(DataFrameImputer):
    def __init__(self, features: Iterable[NumberOrStr], fill_value: NumberOrStr):
        self.features = features
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.features] = df[self.features].fillna(self.fill_value)
        return df


class StatisticsImputer(DataFrameImputer):
    def __init__(self, features: Iterable[NumberOrStr], strategy: SeriesImputer):
        self.features = features
        self.strategy = strategy

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.features] = df[self.features].apply(self.strategy.impute)
        return df


def impute_missing_values(
    df: pd.DataFrame, imputers: list[DataFrameImputer]
) -> pd.DataFrame:
    for imputer in imputers:
        df = imputer.impute(df)
    return df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Cat": ["A", None, "B", "B"],
            "Num": [1, None, 3, None],
        }
    )

    numerical_constant_imputer = ConstantImputer(features=["Num"], fill_value=0)
    categorical_constant_imputer = ConstantImputer(
        features=["Cat"], fill_value="Missing"
    )

    imputed_df = impute_missing_values(
        df, [numerical_constant_imputer, categorical_constant_imputer]
    )
    print(imputed_df)
