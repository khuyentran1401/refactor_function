from process_data import impute_missing_values
import pandas as pd


def test_impute_missing_values():
    df = pd.DataFrame(
        {
            "Cat": ["A", None, "B", "B"],
            "Num": [1, None, 3, None],
        }
    )
    numerical_constant_imputer_config = {"impute_value": 0}
    categorical_constant_imputer_config = [
        {"impute_value": "Missing", "features": ["Cat"]}
    ]
    result = impute_missing_values(
        df,
        numerical_constant_imputer_config=numerical_constant_imputer_config,
        categorical_constant_imputers_config=categorical_constant_imputer_config,
    )
    expected = pd.DataFrame(
        {
            "Cat": ["A", "Missing", "B", "B"],
            "Num": [1, 0, 3, 0],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
