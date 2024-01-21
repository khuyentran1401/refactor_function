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


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    df = impute_missing_values_with_group_statistic(
        df,
        strategy="most_frequent",
        group_feature="MSSubClass",
        target_feature="MSZoning",
    )
    df = impute_missing_values_with_group_statistic(
        df,
        strategy="median",
        group_feature="Neighborhood",
        target_feature="LotFrontage",
    )

    df = impute_missing_values_with_constant(
        df, features=["Functional"], impute_value="Typ"
    )
    df = impute_missing_values_with_constant(
        df, features=numerical_features, impute_value=0
    )
    df = impute_missing_values_with_constant(
        df,
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
        impute_value="Missing",
    )
    df = impute_missing_values_with_statistics(
        df, features=categorical_features, strategy="most_frequent"
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    impute_missing_values(df)
    print(f"There are {df.isna().sum().sum()} null values after imputing")
