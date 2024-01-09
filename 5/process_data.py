import pandas as pd
from abc import abstractmethod, ABC
from typing import Union


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

class QuantileImputer(SeriesImputer):
    def __init__(self, quantile_value: float):
        self.quantile_value = quantile_value

    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.quantile(self.quantile_value))


def impute_missing_values_with_group_statistic(
    df: pd.DataFrame,
    group_feature: str,
    target_feature: str,
    strategy: SeriesImputer,
) -> pd.DataFrame:
    df[target_feature] = df.groupby(group_feature)[target_feature].transform(
        strategy.impute
    )
    return df


def impute_missing_values_with_constant(
    df: pd.DataFrame, features: list, impute_value: Union[str, int, float]
) -> pd.DataFrame:
    df[features] = df[features].fillna(impute_value)
    return df


def impute_missing_values_with_statistics(
    df: pd.DataFrame, features: list, strategy: SeriesImputer
) -> pd.DataFrame:
    df[features] = df[features].apply(strategy.impute)
    return df


def impute_missing_values(
    df: pd.DataFrame,
    group_imputers_config: list,
    categorical_constant_imputers_config: list,
    numerical_constant_imputer_config: dict,
    categorical_statistic_imputers_config: dict,
) -> pd.DataFrame:
    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    for imputer in group_imputers_config:
        df = impute_missing_values_with_group_statistic(
            df,
            strategy=imputer.get("strategy"),
            group_feature=imputer.get("group_feature"),
            target_feature=imputer.get("target_feature"),
        )

    for imputer in categorical_constant_imputers_config:
        df = impute_missing_values_with_constant(
            df,
            features=imputer.get("features"),
            impute_value=imputer.get("impute_value"),
        )
    df = impute_missing_values_with_statistics(
        df,
        features=categorical_features,
        strategy=categorical_statistic_imputers_config.get("strategy"),
    )
    df = impute_missing_values_with_constant(
        df,
        features=numerical_features,
        impute_value=numerical_constant_imputer_config.get("impute_value"),
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    group_imputers_config = [
        {
            "strategy": MostFrequentSeriesImputer(),
            "group_feature": "MSSubClass",
            "target_feature": "MSZoning",
        },
        {
            "strategy": MedianSeriesImputer(),
            "group_feature": "Neighborhood",
            "target_feature": "LotFrontage",
        },
    ]

    categorical_constant_imputers_config = [
        {
            "impute_value": "Typ",
            "features": ["Functional"],
        },
        {
            "impute_value": "Missing",
            "features": [
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
        },
    ]

    categorical_statistic_imputer_config = {"strategy": MostFrequentSeriesImputer()}

    numerical_constant_imputer_config = {"impute_value": 0}

    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    impute_missing_values(
        df,
        group_imputers_config,
        categorical_constant_imputers_config,
        numerical_constant_imputer_config,
        categorical_statistic_imputer_config,
    )
    print(f"There are {df.isna().sum().sum()} null values after imputing")