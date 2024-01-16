import pandas as pd


def impute_missing_values(df):
    # Fill missing values with group statistics
    df["MSZoning"] = df.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill missing values with constant
    df["Functional"] = df["Functional"].fillna("Typ")

    for col in [
        "Alley",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]:
        df[col] = df[col].fillna("Missing")

    numerical_df = df.select_dtypes(include=["int64", "float64"]).columns
    df[numerical_df] = df[numerical_df].fillna(0)

    # Fill missing values with mode
    categorical_df = df.select_dtypes(include=["object"]).columns
    impute_dict_categorical = df[categorical_df].mode().to_dict(orient="records")[0]
    df[categorical_df] = df[categorical_df].fillna(impute_dict_categorical)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = impute_missing_values(df)
    print(f'There are {df.isna().sum().sum()} null values after imputing')
