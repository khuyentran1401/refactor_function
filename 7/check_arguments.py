from typing import Union, Literal, Iterable
import pandas as pd

Features = Union[int, str, Iterable[Union[str, int]]]


def check_strategy(strategy: Literal["most_frequent", "median", "mean"]):
    if strategy not in ["most_frequent", "median", "mean"]:
        raise ValueError("Strategy must be one of most_frequent, median, or mean")
    return strategy


def check_features_is_list(features: Features) -> Iterable[Union[str, int]]:
    if isinstance(features, (str, int)):
        return [features]
    else:
        return features


def check_features_in_df(df: pd.DataFrame, features: Features):
    if isinstance(features, (str, int)):
        if features not in df.columns.to_list():
            raise KeyError(f"The feature {features} is not in the dataframe.")
    else:
        if not set(features).issubset(set(df.columns)):
            raise KeyError("Some of the features are not in the dataframe.")
    return features
