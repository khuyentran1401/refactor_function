import pandas as pd
from abc import abstractmethod, ABC
from typing import TypeVar, Iterable, Literal
import numpy as np

NumberOrStr = TypeVar("NumberOrStr", int, float, str)


def get_strategy_function(strategy: Literal["most_frequent", "median", "mean"]):
    strategy_functions = {
        "most_frequent": lambda series: series.fillna(
            series.mode().get(0, default=pd.NA)
        ),
        "median": lambda series: series.fillna(series.median()),
        "mean": lambda series: series.fillna(series.mean()),
    }
    if strategy not in strategy_functions:
        raise ValueError(f"Invalid strategy: {strategy}")

    return strategy_functions[strategy]


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
        self.strategy = strategy
        self.group_feature = group_feature
        self.target_feature = target_feature

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        impute_function = get_strategy_function(self.strategy)
        df[self.target_feature] = df.groupby(self.group_feature)[
            self.target_feature
        ].transform(impute_function)
        return df


class ConstantImputer(DataFrameImputer):
    def __init__(self, features: Iterable[NumberOrStr], fill_value: NumberOrStr):
        self.features = features
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.features] = df[self.features].fillna(self.fill_value)
        return df


class StatisticsImputer(DataFrameImputer):
    def __init__(
        self,
        features: Iterable[NumberOrStr],
        strategy: Literal["most_frequent", "median", "mean"],
    ):
        self.features = features
        self.strategy = strategy

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        impute_function = get_strategy_function(self.strategy)
        df[self.features] = df[self.features].apply(impute_function)
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

    class RandomImputer(DataFrameImputer):
        def __init__(self, features: Iterable[NumberOrStr]):
            self.features = features

        def impute(self, df: pd.DataFrame) -> pd.DataFrame:
            for feature in self.features:
                nan_mask = df[feature].isna()
                non_nan_values = df.loc[~nan_mask, feature].tolist()
                if non_nan_values:  
                    df.loc[nan_mask, feature] = nan_mask.loc[nan_mask].apply(
                        lambda x: np.random.choice(non_nan_values)
                    )
            return df

    imputers = [
        ConstantImputer(features=["Num"], fill_value=0),
        RandomImputer(features=["Cat"]),
    ]

    imputed_df = impute_missing_values(df, imputers=imputers)
    print(imputed_df)
