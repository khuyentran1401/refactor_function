import pandas as pd
from typing import Literal, Union, Optional


def impute_missing_values_with_group_statistic(
    df: pd.DataFrame,
    group_feature: str,
    target_feature: str,
    strategy: Literal["most_frequent", "median", "mean"],
) -> pd.DataFrame:
    if strategy == "most_frequent":
        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            lambda x: x.fillna(x.mode()[0])
        )
    elif strategy == "median":
        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            lambda x: x.fillna(x.median())
        )
    elif strategy == "mean":
        df[target_feature] = df.groupby(group_feature)[target_feature].transform(
            lambda x: x.fillna(x.mean())
        )        
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    return df


def impute_missing_values_with_constant(
    df: pd.DataFrame, features: list, impute_value: Union[str, int, float]
) -> pd.DataFrame:
    df[features] = df[features].fillna(impute_value)
    return df


def impute_missing_values_with_statistics(
    df: pd.DataFrame,
    features: list,
    strategy: Literal["most_frequent", "median", "mean"],
) -> pd.DataFrame:
    if strategy == "most_frequent":
        impute_dict = df[features].mode().to_dict(orient="records")[0]
    elif strategy == "median":
        impute_dict = df[features].median().to_dict()
    elif strategy == "mean":
        impute_dict = df[features].mean().to_dict()
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
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
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "target": [1, 2, None, 4, 4],
            "num_constant": [None, 2, 3, None, 5],
            "cat_constant": [None, "A", "B", "C", None],
            "cat_statistic": [None, "A", "A", "C", None],
        }
    )

    # Define the configuration for imputation
    group_imputers_config = [
        {
            "strategy": "most_frequent",
            "group_feature": "group",
            "target_feature": "target",
        }
    ]
    constant_imputers_config = [
        {"features": ["cat_constant"], "impute_value": "missing"},
        {"features": ["num_constant"], "impute_value": 0},
    ]
    statistic_imputers_config = [
        {"features": ["cat_statistic"], "strategy": "most_frequent"}
    ]

    # Call the impute function
    imputed_df = impute_missing_values(
        df,
        group_imputers_config,
        constant_imputers_config,
        statistic_imputers_config,
    )
    print(imputed_df)
