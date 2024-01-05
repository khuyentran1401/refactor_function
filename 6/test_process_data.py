import pandas as pd
from process_data import (
    MostFrequentSeriesImputer,
    MedianSeriesImputer,
    MeanSeriesImputer,
    GroupStatisticImputer,
    ConstantImputer,
    StatisticsImputer,
    impute_missing_values,
)
import pytest


def test_most_frequent_series_imputer():
    series = pd.Series([1, 2, 2, None])
    imputer = MostFrequentSeriesImputer()
    result = imputer.impute(series)
    expected = pd.Series([1, 2, 2, 2])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_median_series_imputer():
    series = pd.Series([1, 2, 3, None])
    imputer = MedianSeriesImputer()
    result = imputer.impute(series)
    expected = pd.Series([1, 2, 3, 2])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_mean_series_imputer():
    series = pd.Series([1, 2, 3, None])
    imputer = MeanSeriesImputer()
    result = imputer.impute(series)
    expected = pd.Series([1, 2, 3, 2])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_group_statistic_imputer():
    df = pd.DataFrame({"Group": ["A", "A", "B", "B"], "Value": [1, None, 3, None]})
    strategy = MeanSeriesImputer()
    imputer = GroupStatisticImputer(strategy, "Group", "Value")
    result = imputer.impute(df)
    expected = pd.DataFrame({"Group": ["A", "A", "B", "B"], "Value": [1, 1, 3, 3]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_group_statistics_imputer_with_nan():
    with pytest.raises(ValueError):
        df = pd.DataFrame({"Group": ["A", "A", None, "B"], "Value": [1, None, 3, None]})
        strategy = MeanSeriesImputer()
        imputer = GroupStatisticImputer(strategy, "Group", "Value")
        imputer.impute(df)


def test_constant_imputer():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
    imputer = ConstantImputer(features=["A", "B"], fill_value=0)
    result = imputer.impute(df)
    expected = pd.DataFrame({"A": [1, 0, 3], "B": ["x", "y", 0]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_statistics_imputer():
    df = pd.DataFrame({"A": [1, 2, 3, None]})
    strategy = MeanSeriesImputer()
    imputer = StatisticsImputer(features=["A"], strategy=strategy)
    result = imputer.impute(df)
    expected = pd.DataFrame({"A": [1, 2, 3, 2]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_impute_missing_values():
    df = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [1, None, 3, None],
            "Constant": [None, None, None, None],
        }
    )
    group_strategy = MeanSeriesImputer()
    constant_value = 0
    group_imputer = GroupStatisticImputer(group_strategy, "Group", "Value")
    constant_imputer = ConstantImputer(features=["Constant"], fill_value=constant_value)
    result = impute_missing_values(df, [group_imputer, constant_imputer])
    expected = pd.DataFrame(
        {"Group": ["A", "A", "B", "B"], "Value": [1, 1, 3, 3], "Constant": [0, 0, 0, 0]}
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

def test_impute_missing_values_empty():
    df = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [1, None, 3, None],
            "Constant": [None, None, None, None],
        }
    )
    group_imputer = GroupStatisticImputer(MeanSeriesImputer(), "Group", "Value")
    result = impute_missing_values(df, group_imputer)
    pd.testing.assert_frame_equal(result, df, check_dtype=False)
