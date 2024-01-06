import pandas as pd
from abc import abstractmethod, ABC
from typing import Union, Iterable, TypeVar
from utils import check_variables_in_df, check_variables_is_list

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
        group_feature = check_variables_in_df(df, self.group_feature)
        target_feature = check_variables_in_df(df, self.target_feature)

        if df[group_feature].isna().any():
            raise ValueError(
                f"Group feature {self.group_feature} cannot contain NaN values"
            )

        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            self.strategy.impute
        )
        return df


class ConstantImputer(DataFrameImputer):
    def __init__(
        self,
        features: Union[NumberOrStr, Iterable[NumberOrStr]],
        fill_value: NumberOrStr,
    ):
        self.features = features
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = check_variables_in_df(df, self.features)
        df[features] = df[features].fillna(self.fill_value)
        return df


class StatisticsImputer(DataFrameImputer):
    def __init__(
        self,
        features: Union[NumberOrStr, Iterable[NumberOrStr]],
        strategy: SeriesImputer,
    ):
        self.features = check_variables_is_list(features)
        self.strategy = strategy

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = check_variables_in_df(df, self.features)
        df[features] = df[features].apply(self.strategy.impute)
        return df


def impute_missing_values(
    df: pd.DataFrame, imputers: Union[DataFrameImputer, list[DataFrameImputer]]
) -> pd.DataFrame:
    imputers_ = check_variables_is_list(imputers)
    for imputer in imputers_:
        df = imputer.impute(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    group_imputers = [
        GroupStatisticImputer(
            strategy=MostFrequentSeriesImputer(),
            group_feature="MSSubClass",
            target_feature="MSZoning",
        ),
        GroupStatisticImputer(
            strategy=MedianSeriesImputer(),
            group_feature="Neighborhood",
            target_feature="LotFrontage",
        ),
    ]

    constant_imputers = [
        ConstantImputer(features=["Functional"], fill_value="Typ"),
        ConstantImputer(
            features=[
                "Alley",
                "GarageType",
                "GarageFinish",
                "GarageQual",
                "GarageCond",
                "BsmtQual",
                "BsmtCond",
                "BsmtExposure",
                "BsmtFinType1",
                "BsmtFinType2",
                "FireplaceQu",
                "PoolQC",
                "Fence",
                "MiscFeature",
            ],
            fill_value="Missing",
        ),
        ConstantImputer(features=numerical_features, fill_value=0),
    ]

    statistic_imputers = [
        StatisticsImputer(
            features=categorical_features, strategy=MostFrequentSeriesImputer()
        )
    ]
    imputers = group_imputers + constant_imputers + statistic_imputers
    df = impute_missing_values(df, imputers)
    print(f"There are {df.isna().sum().sum()} null values after imputing")
