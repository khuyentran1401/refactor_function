import pandas as pd
from typing import Literal, Union, Optional


def impute_missing_values_with_group_statistic(
    df: pd.DataFrame,
    group_feature: str,
    target_feature: str,
    strategy: Literal["most_frequent", "median"],
) -> pd.DataFrame:
    if strategy == "most_frequent":
        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            lambda x: x.fillna(x.mode()[0])
        )
    elif strategy == "median":
        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            lambda x: x.fillna(x.median())
        )
    return df


def impute_missing_values_with_constant(
    df: pd.DataFrame, features: list, impute_value: Union[str, int, float]
) -> pd.DataFrame:
    df[features] = df[features].fillna(impute_value)
    return df


def impute_missing_values_with_statistics(
    df: pd.DataFrame,
    features: list,
    strategy: Literal["most_frequent", "median", "mean", "quantile"],
) -> pd.DataFrame:
    if strategy == "most_frequent":
        impute_dict = df[features].mode().to_dict(orient="records")[0]
    elif strategy == "median":
        impute_dict = df[features].median().to_dict()
    elif strategy == "mean":
        impute_dict = df[features].mean().to_dict()
    df[features] = df[features].fillna(impute_dict)
    return df


def impute_missing_values(
    df: pd.DataFrame,
    group_imputers_config: list,
    constant_imputers_config: list,
    statistic_imputer_config: list,
) -> pd.DataFrame:
    for imputer in group_imputers_config:
        df = impute_missing_values_with_group_statistic(
            df,
            strategy=imputer.get("strategy"),
            group_feature=imputer.get("group_feature"),
            target_feature=imputer.get("target_feature"),
        )

    for imputer in constant_imputers_config:
        df = impute_missing_values_with_constant(
            df,
            features=imputer.get("features"),
            impute_value=imputer.get("impute_value"),
        )
    for imputer in statistic_imputer_config:
        df = impute_missing_values_with_statistics(
            df,
            features=imputer.get("features"),
            strategy=imputer.get("strategy"),
        )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns
    group_imputers_config = [
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

    constant_imputers_config = [
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
        {
            "impute_value": 0,
            "features": numerical_features,
        },
    ]

    statistic_imputer_config = [
        {"features": categorical_features, "strategy": "most_frequent"}
    ]

    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    impute_missing_values(
        df,
        group_imputers_config,
        constant_imputers_config,
        statistic_imputer_config,
    )
    print(f"There are {df.isna().sum().sum()} null values after imputing")
