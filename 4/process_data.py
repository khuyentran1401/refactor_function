import pandas as pd
from typing import Literal, Union


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
    df: pd.DataFrame, features: list, strategy: Literal["most_frequent", "median"]
) -> pd.DataFrame:
    if strategy == "most_frequent":
        impute_dict = df[features].mode().to_dict(orient="records")[0]
    elif strategy == "median":
        impute_dict = df[features].median().to_dict()
    df[features] = df[features].fillna(impute_dict)
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
