import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np 
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns 
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split, KFold
    import missingno as msno
    import warnings
    warnings.filterwarnings("ignore")
    return (
        KFold,
        Lasso,
        LinearRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Ridge,
        StandardScaler,
        math,
        mean_absolute_error,
        mean_squared_error,
        mo,
        msno,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Introduction:
        Geely Auto, a Chinese car manufacturer wants to enter the U.S.  market by initiating a local manufacturer to compete with both Amerrican and European automakers. To ensure strong an succeful entry to the automobile market, Geely Auto utilized an automobile consulting firm to analyze the key factors influencing car pricing in the American market, which may differ from the Chinese market. 

        # 1. Problem Statement: 

        Geely Auto seek want to know two things: 

        1. Which feature contribute the most in the prediction of car prices. (lasso?)
        2. How well do these variable explain car pricing trends (R2?)
        """
    )
    return


@app.cell
def _(pd):
    # importing the dataset 
    path = "./data/CarPrice_Assignment.csv"
    df = pd.read_csv(path)
    df.info()
    return df, path


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 2. Data Preparation 

        ## Data Dictionary 
        - Using the provided datasets from kaggle, we notice the data contrain 26 features with following types

        |type|count|
        |:-:|:---:|
        |float64|8|
        |int64|8|
        |String (Objects)|10|

        The description each features are: 

        | #  | Column Name           | Description                                              | Data Type   |
        |:----:|:-----------------------:|:----------------------------------------------------------:|:-------------:|
        | 1  | Car_ID               | Unique id of each observation                            | Integer     |
        | 2  | Symboling            | Insurance risk rating (+3 = risky, -3 = safe)            | Categorical |
        | 3  | carCompany           | Name of car company                                      | Categorical |
        | 4  | fueltype             | Car fuel type (e.g., gas or diesel)                      | Categorical |
        | 5  | aspiration           | Aspiration used in a car                                 | Categorical |
        | 6  | doornumber           | Number of doors in a car                                 | Categorical |
        | 7  | carbody              | Body of car                                              | Categorical |
        | 8  | drivewheel           | Type of drive wheel                                      | Categorical |
        | 9  | enginelocation       | Location of car engine                                   | Categorical |
        | 10 | wheelbase            | Wheelbase of car                                         | Numeric     |
        | 11 | carlength            | Length of car                                            | Numeric     |
        | 12 | carwidth             | Width of car                                             | Numeric     |
        | 13 | carheight            | Height of car                                            | Numeric     |
        | 14 | curbweight           | Weight of a car without occupants or baggage             | Numeric     |
        | 15 | enginetype           | Type of engine                                           | Categorical |
        | 16 | cylindernumber       | Number of cylinders in the car                           | Categorical |
        | 17 | enginesize           | Size of car engine                                       | Numeric     |
        | 18 | fuelsystem           | Fuel system of car                                       | Categorical |
        | 19 | boreratio            | Bore ratio of car                                        | Numeric     |
        | 20 | stroke               | Stroke or volume inside the engine                       | Numeric     |
        | 21 | compressionratio     | Compression ratio of car                                 | Numeric     |
        | 22 | horsepower           | Horsepower of car                                        | Numeric     |
        | 23 | peakrpm              | Car peak RPM                                             | Numeric     |
        | 24 | citympg              | Mileage in city                                          | Numeric     |
        | 25 | highwaympg           | Mileage on highway                                       | Numeric     |
        | 26 | price (Dependent var)| Price of car                                             | Numeric     |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data Cleaning 
        We note the following: 

        1. The dataset contian 26 feautres and 205 observations (rows) no missing values.
        2. Since IDs are not usde in mathematical operation, it would be wise to change `car_ID` from integer into a strings/object.
        3. The `symboling` feature is considered an `int` while it shold be an `categorical` (i.e. ordinal from -3 to +3) as defined in the data dictionary.
        """
    )
    return


@app.cell
def _(df, msno, pd, plt):
    # Missing value visualization 
    msno.matrix(df)
    plt.show()
    # Cast car_id into an object
    df["car_ID"] = df["car_ID"].astype('object')
    # Cast symboling onto a categorical (ordinal) feature
    symboling_order = sorted(df["symboling"].unique())
    df["symboling"] = pd.Categorical(
        df["symboling"], categories=symboling_order, ordered=True
    )
    print(df.dtypes)
    return (symboling_order,)


@app.cell
def _(df):
    # Seperate car name into two columns brand, and fix duplicated names 
    df.insert(0, "brand", df["CarName"].str.split(" ").str[0])
    df.insert(0, "model", df["CarName"].str.split(" ").str[1:].str.join(" "))
    print(df['brand'].unique())
    replace_brand = {
        "maxda": "mazda",
        "Nissan": "nissan",
        "toyouta": "toyota",
        "vokswagen":"volkswagen",
        "vw":"volkswagen"
    }

    df["brand"] = df["brand"].replace(replace_brand)
    print(df['brand'].unique())

    return (replace_brand,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 3. Exploratory Data Analysis (EDA)


        ## Categorical features
        """
    )
    return


@app.cell
def _(df, math, plt):
    # Checking the cordinality of categorical variable and excluding car_ID 
    categorical_features = df.select_dtypes(include=["category", "object"]).columns.to_list()[1:]
    fig1, ax1 = plt.subplots(math.ceil(len(categorical_features)/3),3)
    fig1.set_size_inches(15,15)
    fig1.suptitle("Bar plots of categorical features")
    plt.subplots_adjust(hspace=0.3)
    for _idx, _feat in enumerate(categorical_features):
        _counts = df[_feat].value_counts()
        ax1.flatten()[_idx].bar(_counts.index, _counts.values)
        ax1.flatten()[_idx].legend
        ax1.flatten()[_idx].set(
            xlabel=_feat,
            ylabel="Count", 
        )

    for _idx in range(len(categorical_features), math.ceil(len(categorical_features)/3) * 3):
        fig1.delaxes(ax1.flatten()[_idx])
    plt.show()
    return ax1, categorical_features, fig1


@app.cell
def _(categorical_features, df):
    # Print cardinality of categorical variables 
    print(f"|----------------------------|")
    for _feat in categorical_features:
        print (f"The column {_feat} has {df[_feat].nunique()} unique entries")
    print(f"|----------------------------|")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        From the above analysis we note the following. 

        1. Most car are of medium risk and leaning toward high risk.
        2. Most cars use gas more than diseal as fuel.
        3. More care are naturally aspirated than turbo.
        4. The majoraty of the cars uses overhead camshaft engine (`ohc` engine) and located at the front.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical features

         - Describtive statistics of the dataset.
        """
    )
    return


@app.cell
def _(df):
    # describtive statistics of the dataset
    df.describe()
    return


@app.cell
def _(df):
    # List of numerical features name
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
    print(numerical_features)
    return (numerical_features,)


@app.cell
def _(df, math, numerical_features, plt):
    # Plotting histograms of numerical features

    fig2, ax2 = plt.subplots(math.ceil(len(numerical_features)/3), 3)
    fig2.suptitle("Scatter plots of numerical features")
    fig2.set_size_inches(15,15)
    plt.subplots_adjust(hspace=0.3)
    for _idx, _feat in enumerate(numerical_features):
        ax2.flatten()[_idx].hist(df[_feat], bins=int(df.shape[0]**0.5))
        ax2.flatten()[_idx].set(
            xlabel=f"{_feat}",
            ylabel=f""
        )
    for _idx in range(len(numerical_features), math.ceil(len(numerical_features)/3) * 3):
        fig2.delaxes(ax2.flatten()[_idx])
    plt.show()
    return ax2, fig2


@app.cell
def _(df, np):
    q75 = np.percentile(df["price"], 75)
    q25 = np.percentile(df["price"], 25)
    iqr = q75 - q25
    print(df[df["price"] > q75 + 1.5*iqr].shape)
    return iqr, q25, q75


@app.cell
def _(mo):
    mo.md(
        r"""
        ### From the above analysis we note: 

        1. Most features looks normally distributed except:
               1. `carwidth`, `enginesize`, `horsepower`, __price__ is postively skewed
               2. `compressionration` has a huge gap between 10 and 20.
        2. Using interquartile method we notice there is 15 outliers in price. For now we will leave these outliers.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # 4. Setup

        1. Dropping `car_ID`.
        2. Feature/Target splitting.
        3. Using holdout + CV method for splitting (70% train, 20% validation, 10% testing)
        4. Using standard scalar for numerical features.
        5. Using one hot encoder for categorical features.
        6. Using Ordinal encoder for ordinal features.
        """
    )
    return


@app.cell
def _(df):
    df.drop(columns="car_ID", inplace=True)
    return


@app.cell
def _(
    KFold,
    Lasso,
    LinearRegression,
    OneHotEncoder,
    OrdinalEncoder,
    Ridge,
    StandardScaler,
    df,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    train_test_split,
):
    X, y = df.drop(columns="price", axis=1).copy(), df["price"].copy()


    # Splitting the test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.225,
        random_state=42
    )

    categorical_to_scale = X.select_dtypes(include="object").columns.to_list()
    numerical_to_scale = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
    ordinal_to_scale = X.select_dtypes(include="category").columns.to_list()

    # Define variable before KFold CV
    ols_scores = []
    ridge_scores = []
    lasso_scores = []


    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
        # Fold split
        X_train_fold, X_val_fold = X_train.iloc[train_index].copy(), X_train.iloc[val_index].copy()
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
   
        # Numerical Scaler 
        scaler = StandardScaler()

        # Fit and transform on the training fold 
        X_train_fold[numerical_to_scale] = scaler.fit_transform(X_train_fold[numerical_to_scale])
        # Transform on validation fold 
        X_val_fold[numerical_to_scale] = scaler.transform(X_val_fold[numerical_to_scale])

        # OneHot Encoder
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit and transform on the training fold.
        cat_train = onehot_encoder.fit_transform(X_train_fold[categorical_to_scale])
    
        # Create dataframe of encoded training fold
        cat_cols = onehot_encoder.get_feature_names_out(categorical_to_scale)
        cat_train_df = pd.DataFrame(cat_train, columns=cat_cols, index=X_train_fold.index)

        # Create dataframf of encoded validation fold
        cat_val = onehot_encoder.transform(X_val_fold[categorical_to_scale])
        cat_val_df = pd.DataFrame(cat_val, columns=cat_cols, index=X_val_fold.index)

        # Drop original categorical columns and join the encoded ones
        X_train_fold = X_train_fold.drop(columns=categorical_to_scale).join(cat_train_df)
        X_val_fold = X_val_fold.drop(columns=categorical_to_scale).join(cat_val_df)
    
        # Ordinal Encoding
        ordinal_encoder = OrdinalEncoder()
        # Fit and transform on the training fold.
        X_train_fold[ordinal_to_scale] = ordinal_encoder.fit_transform(X_train_fold[ordinal_to_scale])
        # Transform the validation fold
        X_val_fold[ordinal_to_scale] = ordinal_encoder.transform(X_val_fold[ordinal_to_scale])


        # Modeling 
        models = {
            "ols": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=42),
            "lasso": Lasso(alpha=0.01, random_state=42)
        }
        y_pred = []
        for model in models.values():
            model.fit(X_train_fold, y_train_fold)
        # Evluating metrics MAE, amd RMSE: 
        # OLS
        ols_pred = models["ols"].predict(X_val_fold)
        ols_mae = mean_absolute_error(y_val_fold, ols_pred)
        ols_rmse = np.sqrt(mean_squared_error(y_val_fold, ols_pred))
        ols_scores.append((ols_mae, ols_rmse))
        # Ridge
        ridge_pred = models["ridge"].predict(X_val_fold)
        ridge_mae = mean_absolute_error(y_val_fold, ridge_pred)
        ridge_rmse = np.sqrt(mean_squared_error(y_val_fold, ridge_pred))
        ridge_scores.append((ridge_mae, ridge_rmse))
        # Lasso 
        lasso_pred = models["lasso"].predict(X_val_fold)
        lasso_mae = mean_absolute_error(y_val_fold, lasso_pred)
        lasso_rmse = np.sqrt(mean_squared_error(y_val_fold, lasso_pred))
        lasso_scores.append((lasso_mae, lasso_rmse))

        print(f"Fold {fold_idx+1}/10")
        print(f"OLS (MAE, RMSE): {(ols_mae, ols_rmse)} ")
        print(f"Ridge (MAE, RMSE): {(ridge_mae, ridge_rmse)} ")
        print(f"Lasso (MAE, RMSE): {(lasso_mae, lasso_rmse)} ")


    # After the loop
    print("\nAverage performance across all folds:")
    print(f"OLS - MAE: {np.mean([s[0] for s in ols_scores]):.4f}, RMSE: {np.mean([s[1] for s in ols_scores]):.4f}")
    print(f"Ridge - MAE: {np.mean([s[0] for s in ridge_scores]):.4f}, RMSE: {np.mean([s[1] for s in ridge_scores]):.4f}")
    print(f"Lasso - MAE: {np.mean([s[0] for s in lasso_scores]):.4f}, RMSE: {np.mean([s[1] for s in lasso_scores]):.4f}")

    return (
        X,
        X_test,
        X_train,
        X_train_fold,
        X_val,
        X_val_fold,
        cat_cols,
        cat_train,
        cat_train_df,
        cat_val,
        cat_val_df,
        categorical_to_scale,
        fold_idx,
        kf,
        lasso_mae,
        lasso_pred,
        lasso_rmse,
        lasso_scores,
        model,
        models,
        numerical_to_scale,
        ols_mae,
        ols_pred,
        ols_rmse,
        ols_scores,
        onehot_encoder,
        ordinal_encoder,
        ordinal_to_scale,
        ridge_mae,
        ridge_pred,
        ridge_rmse,
        ridge_scores,
        scaler,
        train_index,
        val_index,
        y,
        y_pred,
        y_test,
        y_train,
        y_train_fold,
        y_val,
        y_val_fold,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
