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


def test_impute_missing_values():
    df = pd.DataFrame(
        {
            "A": ["dog", "dog", "cat", "cat", None],
            "B": [1, 2, None, 4, 5],
            "C": [None, 2, 3, None, 5],
        }
    )

    # Define the configuration for imputation
    group_imputers_config = [
        {"strategy": "mean", "group_feature": "A", "target_feature": "B"}
    ]
    categorical_constant_imputers_config = [
        {"features": ["A"], "impute_value": "missing"}
    ]
    numerical_constant_imputer_config = {"impute_value": 0}
    categorical_statistic_imputers_config = {"strategy": "most_frequent"}

    # Call the impute function
    imputed_df = impute_missing_values(
        df,
        group_imputers_config,
        categorical_constant_imputers_config,
        numerical_constant_imputer_config,
        categorical_statistic_imputers_config,
    )
    expected_df = pd.DataFrame(
        {
            "A": ["dog", "dog", "cat", "cat", "missing"],
            "B": [1, 2, 0, 4, 5],
            "C": [0, 2, 3, 0, 5],
        }
    )
    pd.testing.assert_frame_equal(imputed_df, expected_df, check_dtype=False)


def test_impute_missing_values_with_statistic_imputers():
    df = pd.DataFrame(
        {
            "cat": ["A", "B", None, "A", "B", "B", None],
            "num": [1, 2, 3, None, 5, None, 7],
        }
    )

    group_imputers_config = []
    categorical_constant_imputers_config = []
    numerical_constant_imputer_config = {"impute_value": 0}
    categorical_statistic_imputers_config = {"strategy": "most_frequent"}

    expected_df = pd.DataFrame(
        {"cat": ["A", "B", "B", "A", "B", "B", "B"], "num": [1, 2, 3, 0, 5, 0, 7]}
    )

    imputed_df = impute_missing_values(
        df,
        group_imputers_config,
        categorical_constant_imputers_config,
        numerical_constant_imputer_config,
        categorical_statistic_imputers_config,
    )

    # Assert that the resulting DataFrame is as expected
    pd.testing.assert_frame_equal(imputed_df, expected_df, check_dtype=False)
