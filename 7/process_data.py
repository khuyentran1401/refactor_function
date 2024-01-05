import pandas as pd
from abc import abstractmethod, ABC
from typing import Union, Literal
from check_arguments import check_features_in_df, check_features_is_list, check_strategy


def most_frequent_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.mode()[0])


def median_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.median())


def mean_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.mean())


class DataFrameImputer(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class GroupStatisticImputer(DataFrameImputer):
    def __init__(
        self,
        strategy: Literal["most_frequent", "median", "mean"],
        group_feature: str,
        target_feature: str,
    ):
        self.strategy = check_strategy(strategy)
        self.group_feature = group_feature
        self.target_feature = target_feature

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        group_feature = check_features_in_df(df, self.group_feature)
        target_feature = check_features_in_df(df, self.target_feature)

        if self.strategy == "most_frequent":
            df[target_feature] = df.groupby(group_feature)[target_feature].transform(
                most_frequent_series_imputer
            )
        elif self.strategy == "median":
            df[target_feature] = df.groupby(group_feature)[target_feature].transform(
                median_series_imputer
            )
        elif self.strategy == "mean":
            df[target_feature] = df.groupby(group_feature)[target_feature].transform(
                mean_series_imputer
            )
        return df


class ConstantImputer(DataFrameImputer):
    def __init__(self, features: list, fill_value: Union[str, float, int]):
        self.features = check_features_is_list(features)
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = check_features_in_df(df, self.features)
        df[features] = df[features].fillna(self.fill_value)
        return df


class StatisticsImputer(DataFrameImputer):
    def __init__(
        self, features: list, strategy: Literal["most_frequent", "median", "mean"]
    ):
        self.features = check_features_is_list(features)
        self.strategy = check_strategy(strategy)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = check_features_in_df(df, self.features)
        if self.strategy == "most_frequent":
            df[features] = df[features].apply(most_frequent_series_imputer)
        elif self.strategy == "median":
            df[features] = df[features].apply(median_series_imputer)
        elif self.strategy == "mean":
            df[features] = df[features].apply(mean_series_imputer)
        return df


def impute_missing_values(
    df: pd.DataFrame, imputers: list[DataFrameImputer]
) -> pd.DataFrame:
    for imputer in imputers:
        df = imputer.impute(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    group_imputers = [
        GroupStatisticImputer(
            strategy="most_frequent",
            group_feature="MSSubClass",
            target_feature="MSZoning",
        ),
        GroupStatisticImputer(
            strategy="median",
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
        StatisticsImputer(features=categorical_features, strategy="most_frequent")
    ]
    imputers = group_imputers + constant_imputers + statistic_imputers
    df = impute_missing_values(df, imputers)
    print(f"There are {df.isna().sum().sum()} null values after imputing")
