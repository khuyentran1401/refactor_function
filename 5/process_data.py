import pandas as pd
from typing import Union


def most_frequent_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.mode()[0])


def median_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.median())


def mean_series_imputer(series: pd.Series) -> pd.Series:
    return series.fillna(series.mean())


def impute_missing_values_with_group_statistic(
    df: pd.DataFrame,
    group_feature: str,
    target_feature: str,
    strategy: str,
) -> pd.DataFrame:
    if df[group_feature].isna().any():
        raise ValueError(f"Group feature {group_feature} has missing values")
    if strategy == "most_frequent":
        df[target_feature] = df.groupby(group_feature)[
            target_feature
        ].transform(most_frequent_series_imputer)
    elif strategy == "median":
        df[target_feature] = df.groupby(group_feature)[
            target_feature
        ].transform(median_series_imputer)
    elif strategy == "mean":
        df[target_feature] = df.groupby(group_feature)[
            target_feature
        ].transform(mean_series_imputer)
    return df


def impute_missing_values_with_constant(
    df: pd.DataFrame, features: list, impute_value: Union[str, int, float]
) -> pd.DataFrame:
    df[features] = df[features].fillna(impute_value)
    return df


def impute_missing_values_with_statistics(
    df: pd.DataFrame, features: list, strategy: str 
) -> pd.DataFrame:
    if strategy == "most_frequent":
        df[features] = df[features].apply(most_frequent_series_imputer)
    elif strategy == "median":
        df[features] = df[features].apply(median_series_imputer)
    elif strategy == "mean":
        df[features] = df[features].apply(mean_series_imputer)
    return df


def impute_missing_values(
    df: pd.DataFrame,
    group_imputers: list,
    categorical_constant_imputers: list,
    numerical_constant_imputer: dict,
    categorical_statistic_imputers: dict,
) -> pd.DataFrame:
    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    for imputer in group_imputers:
        df = impute_missing_values_with_group_statistic(
            df,
            strategy=imputer.get("strategy"),
            group_feature=imputer.get("group_feature"),
            target_feature=imputer.get("target_feature"),
        )

    for imputer in categorical_constant_imputers:
        df = impute_missing_values_with_constant(
            df,
            features=imputer.get("features"),
            impute_value=imputer.get("impute_value"),
        )
    df = impute_missing_values_with_statistics(
        df,
        features=categorical_features,
        strategy=categorical_statistic_imputers.get("strategy"),
    )
    df = impute_missing_values_with_constant(
        df,
        features=numerical_features,
        impute_value=numerical_constant_imputer.get("impute_value"),
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    group_imputers = [
        {
            "strategy": "most_frequent",
            "group_feature": "MSSubClass",
            "target_feature": "MSZoning",
        },
        {
            "strategy": "median",
            "group_feature": "Neighborhood",
            "target_feature": "LotFrontage",
        },
    ]

    categorical_constant_imputers = [
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

    categorical_statistic_imputer = {"strategy": "most_frequent"}

    numerical_constant_imputer = {"impute_value": 0}

    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    impute_missing_values(
        df,
        group_imputers,
        categorical_constant_imputers,
        numerical_constant_imputer,
        categorical_statistic_imputer,
    )
    print(f"There are {df.isna().sum().sum()} null values after imputing")
