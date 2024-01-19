import pandas as pd
from process_data import impute_missing_values


def test_impute_missing_values():
    df = pd.DataFrame(
        {
            "MSZoning": [None, "RL", None, "RM"],
            "MSSubClass": [20, 30, 30, 20],
            "LotFrontage": [80.0, None, 75.0, None],
            "Neighborhood": ["CollgCr", "Veenker", "Crawfor", "CollgCr"],
            "Functional": [None, "Typ", "Min1", "Min2"],
            "Alley": [None, "Grvl", None, None],
            "GarageType": [None, "Attchd", None, "Detchd"],
            "GarageFinish": ["Fin", None, "RFn", None],
            "GarageQual": [None, "TA", None, "TA"],
            "GarageCond": ["TA", None, "Fa", None],
            "BsmtQual": [None, "Gd", "TA", "Ex"],
            "BsmtCond": ["TA", "Gd", None, "Fa"],
            "BsmtExposure": ["No", None, "Mn", "Av"],
            "BsmtFinType1": ["GLQ", "ALQ", None, "Unf"],
            "BsmtFinType2": ["Unf", "BLQ", None, "LwQ"],
            "FireplaceQu": [None, "TA", "Gd", "Ex"],
            "PoolQC": ["Ex", None, "Fa", None],
            "Fence": [None, "MnPrv", "GdWo", None],
            "MiscFeature": [None, "Shed", "Gar2", None],
        }
    )

    imputed_df = impute_missing_values(df)

    assert not imputed_df.isnull().any().any()
