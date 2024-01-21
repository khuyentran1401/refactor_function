import pandas as pd
from process_data import impute_missing_values_with_constant


def test_impute_missing_values_with_constant():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
    result = impute_missing_values_with_constant(
        df, features=["A", "B"], impute_value=0
    )
    expected = pd.DataFrame({"A": [1, 0, 3], "B": ["x", "y", 0]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
