import pandas as pd
from abc import abstractmethod, ABC
from typing import TypeVar, Iterable, Literal

NumberOrStr = TypeVar("NumberOrStr", int, float, str)


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
        if self.strategy == "most_frequent":
            df[self.target_feature] = df.groupby(self.group_feature)[
                self.target_feature
            ].transform(lambda x: x.fillna(x.mode()[0]))
        elif self.strategy == "median":
            df[self.target_feature] = df.groupby(self.group_feature)[
                self.target_feature
            ].transform(lambda x: x.fillna(x.median()))
        elif self.strategy == "mean":
            df[self.target_feature] = df.groupby(self.group_feature)[
                self.target_feature
            ].transform(lambda x: x.fillna(x.mean()))
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
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
        if self.strategy == "most_frequent":
            impute_dict = df[self.features].mode().to_dict(orient="records")[0]
        elif self.strategy == "median":
            impute_dict = df[self.features].median().to_dict()
        elif self.strategy == "mean":
            impute_dict = df[self.features].mean().to_dict()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        df[self.features] = df[self.features].fillna(impute_dict)
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

    constant_imputers = [
        ConstantImputer(features=["Num"], fill_value=0),
        ConstantImputer(features=["Cat"], fill_value="Missing"),
    ]

    imputed_df = impute_missing_values(df, imputers=constant_imputers)
    print(imputed_df)
