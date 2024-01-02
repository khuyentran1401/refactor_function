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

    df["Alley"] = df["Alley"].fillna("Missing")
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        df[col] = df[col].fillna("Missing")

    for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
        df[col] = df[col].fillna("Missing")

    df["FireplaceQu"] = df["FireplaceQu"].fillna("Missing")

    df["PoolQC"] = df["PoolQC"].fillna("Missing")

    df["Fence"] = df["Fence"].fillna("Missing")

    df["MiscFeature"] = df["MiscFeature"].fillna("Missing")

    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            df[i] = df[i].fillna(0)

    # Fill missing values with mode
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["KitchenQual"] = df["KitchenQual"].fillna("TA")
    df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
    df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])
    for i in df.columns:
        if df[i].dtype == object:
            df[i] = df[i].fillna(df[i].mode()[0])
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = impute_missing_values(df)
    print(f'There are {df.isna().sum().sum()} null values after imputing')
