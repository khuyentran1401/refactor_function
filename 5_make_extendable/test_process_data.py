from process_data import (
    impute_missing_values,
    impute_missing_values_with_statistics,
    SeriesImputer,
)
import pandas as pd


def test_impute_missing_values():
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

    result = impute_missing_values(
        df,
        constant_imputers_config=constant_imputers_config,
        # need to specify group_imputers_config and statistic_imputers_config
    )
    expected = pd.DataFrame(
        {
            "Cat": ["A", "Missing", "B", "B"],
            "Num": [1, 0, 3, 0],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_impute_missing_values_with_statistics_using_quantile():
    df = pd.DataFrame({"num": [1, 2, 3, None, 5, None, 7]})

    class QuantileImputer(SeriesImputer):
        def __init__(self, quantile_value):
            self.quantile_value = quantile_value

        def impute(self, series):
            return series.fillna(series.quantile(self.quantile_value))

    result = impute_missing_values_with_statistics(
        df, features=["num"], strategy=QuantileImputer(0.5)
    )
    expected = pd.DataFrame({"num": [1, 2, 3, 3, 5, 3, 7]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
