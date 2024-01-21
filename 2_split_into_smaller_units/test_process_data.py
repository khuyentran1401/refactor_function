import pandas as pd
import pytest
from typing import Union, Literal, Optional
from process_data import (
    impute_missing_values_with_group_statistic,
    impute_missing_values_with_constant,
    impute_missing_values_with_statistics,
    impute_missing_values,
)


def test_impute_missing_values_with_constant():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
    result = impute_missing_values_with_constant(
        df, features=["A", "B"], impute_value=0
    )
    expected = pd.DataFrame({"A": [1, 0, 3], "B": ["x", "y", 0]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_impute_missing_values_with_group_statistic():
    df = pd.DataFrame(
        {"group": ["A", "A", "B", "B", "B"], "target": [1, 2, None, 4, 4]}
    )
    result = impute_missing_values_with_group_statistic(
        df, group_feature="group", target_feature="target", strategy="mean"
    )
    expected = pd.DataFrame(
        {"group": ["A", "A", "B", "B", "B"], "target": [1, 2, 4, 4, 4]}
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_impute_missing_values_with_statistics():
    df = pd.DataFrame({"target": [1, 2, None, 1, 0]})
    result = impute_missing_values_with_statistics(
        df, features=["target"], strategy="mean"
    )
    expected = pd.DataFrame({"target": [1, 2, 1, 1, 0]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
