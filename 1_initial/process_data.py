import pandas as pd


def impute_missing_values(df):
    # the data description states that NA refers to typical ('Typ') values
    df["Functional"] = df["Functional"].fillna("Typ")

    # Replace the missing values in each of the columns below with their mode
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["KitchenQual"] = df["KitchenQual"].fillna("TA")
    df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
    df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])

    # Group by MSSubClass and fill in missing value by the mode MSZoning of the MSSubclass
    df["MSZoning"] = df.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )

    # NaN values for the alley feature, means there's no alley
    df["Alley"] = df["Alley"].fillna("Missing")

    # NaN values for these categorical garage df, means there's no garage
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        df[col] = df[col].fillna("Missing")

    # NaN values for these categorical basement df, means there's no basement
    for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
        df[col] = df[col].fillna("Missing")

    # NaN values for the fireplace feature, means there's no fireplace
    df["FireplaceQu"] = df["FireplaceQu"].fillna("Missing")

    # NaN values for the pool feature, means there's no pool
    df["PoolQC"] = df["PoolQC"].fillna("Missing")

    # NaN values for the fence feature, means there's no fence
    df["Fence"] = df["Fence"].fillna("Missing")

    # NaN values for the misc_feature feature, means there's no misc_feature
    df["MiscFeature"] = df["MiscFeature"].fillna("Missing")

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # We have no particular intuition around how to fill in the rest of the categoricaldf
    # So we replace their missing values with mode
    for i in df.columns:
        if df[i].dtype == object:
            df[i] = df[i].fillna(df[i].mode()[0])

    # And we do the same thing for numerical df, but this time with 0s
    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            df[i] = df[i].fillna(0)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = impute_missing_values(df)
    print(f"There are {df.isna().sum().sum()} null values after imputing")
