import pandas as pd
from process_data import (
    GroupStatisticImputer,
    ConstantImputer,
    StatisticsImputer,
    impute_missing_values,
)
import pytest


class TestGroupStatisticImputer:
    def test_group_statistic_imputer(self):
        df = pd.DataFrame({"Group": ["A", "A", "B", "B"], "Value": [1, None, 3, None]})
        imputer = GroupStatisticImputer(
            strategy="mean", group_feature="Group", target_feature="Value"
        )
        result = imputer.impute(df)
        expected = pd.DataFrame({"Group": ["A", "A", "B", "B"], "Value": [1, 1, 3, 3]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_group_statistics_imputer_with_nan(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame(
                {"Group": ["A", "A", None, "B"], "Value": [1, None, 3, None]}
            )
            imputer = GroupStatisticImputer(
                strategy="mean", group_feature="Group", target_feature="Value"
            )
            imputer.impute(df)

    def test_group_statistics_feature_not_match(self):
        with pytest.raises(
            KeyError, match="The feature Value2 is not in the dataframe"
        ):
            df = pd.DataFrame(
                {"Group": ["A", "A", "B", "B"], "Value": [1, None, 3, None]}
            )
            imputer = GroupStatisticImputer(
                strategy="mean", group_feature="Group", target_feature="Value2"
            )
            imputer.impute(df)


class TestConstantImputer:
    def test_constant_imputer(self):
        df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
        imputer = ConstantImputer(features=["A", "B"], fill_value=0)
        result = imputer.impute(df)
        expected = pd.DataFrame({"A": [1, 0, 3], "B": ["x", "y", 0]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constant_imputer_feature_not_list(self):
        df = pd.DataFrame({"A": [1, None, 3]})
        imputer = ConstantImputer(features="A", fill_value=0)
        result = imputer.impute(df)
        expected = pd.DataFrame({"A": [1, 0, 3]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constant_imputer_feature_not_match(self):
        df = pd.DataFrame({"A": [1, None, 3]})
        imputer = ConstantImputer(features=["A", "B"], fill_value=0)
        imputer.impute(df)


class TestStatisticsImputer:
    def test_statistics_imputer(self):
        df = pd.DataFrame({"A": [1, 2, 3, None]})
        imputer = StatisticsImputer(features=["A"], strategy="mean")
        result = imputer.impute(df)
        expected = pd.DataFrame({"A": [1, 2, 3, 2]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_statistics_imputer_feature_not_list(self):
        df = pd.DataFrame({"A": [1, 2, 3, None]})
        imputer = StatisticsImputer(features="A", strategy="mean")
        result = imputer.impute(df)
        expected = pd.DataFrame({"A": [1, 2, 3, 2]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_statistics_imputer_feature_not_match(self):
        df = pd.DataFrame({"A": [1, 2, 3, None]})
        imputer = StatisticsImputer(features=["A", "B", "C"], strategy="mean")
        imputer.impute(df)


class TestImputeMissingValues:
    def test_impute_missing_values(self):
        df = pd.DataFrame(
            {
                "Cat": ["A", None, "B", "B"],
                "Num": [1, None, 3, None],
            }
        )
        numerical_constant_imputer = ConstantImputer(features=["Num"], fill_value=0)
        categorical_constant_imputer = ConstantImputer(
            features=["Cat"], fill_value="Missing"
        )
        result = impute_missing_values(
            df, [numerical_constant_imputer, categorical_constant_imputer]
        )
        expected = pd.DataFrame(
            {
                "Cat": ["A", "Missing", "B", "B"],
                "Num": [1, 0, 3, 0],
            }
        )
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_impute_missing_values_one_imputer(self):
        df = pd.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Value": [1, None, 3, None],
                "Constant": [None, None, None, None],
            }
        )
        group_imputer = GroupStatisticImputer(
            strategy="mean", group_feature="Group", target_feature="Value"
        )
        result = impute_missing_values(df, group_imputer)
        pd.testing.assert_frame_equal(result, df, check_dtype=False)
