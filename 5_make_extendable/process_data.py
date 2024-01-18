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
            "Cat": ["A", None, "B", "B"],
            "Num": [1, None, 3, None],
        }
    )

    constant_imputers_config = [
        {"impute_value": 0, "features": ["Num"]},
        {"impute_value": "Missing", "features": ["Cat"]},
    ]

    imputed_df = impute_missing_values(
        df,
        constant_imputers_config=constant_imputers_config,
        # need to specify group_imputers_config and statistic_imputers_config
    )
    print(imputed_df)

    """
    TypeError: impute_missing_values() missing 2 required positional arguments: 'group_imputers' and 'categorical_statistic_imputers_config'
    """
