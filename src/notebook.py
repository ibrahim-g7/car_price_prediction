import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


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
    from sklearn.feature_selection import SelectFromModel
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
        SelectFromModel,
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
def _(
    KFold,
    Lasso,
    LinearRegression,
    OneHotEncoder,
    OrdinalEncoder,
    Ridge,
    StandardScaler,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    r2_score,
    train_test_split,
):
    # Splitting function
    def splitting(X, y):
        # Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.10, 
            random_state=42
        )
        # Calidation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.225,
            random_state=42
        )
        return X_train, X_val, y_train, y_val, X_test, y_test

    # Frequency encoder for high cardinality features 
    def frequency_encoder(data_frame, column):
        # make a copy
        data_frame = data_frame.copy()
        # create map
        frequency_map = data_frame[column].value_counts(normalize=True)
        # make new freq columns
        data_frame[column + "_freq"] = data_frame[column].map(frequency_map)
        data_frame.drop(columns=column, axis=1, inplace=True)
        return data_frame


    def processing(train, val):
        # Define scaler
        scaler = StandardScaler()
        # Make a copy
        X_train = train.copy()
        X_val = val.copy()
        # Define numerical columns
        numerical_to_scale = X_train.select_dtypes(include=["int64", "float64"]).columns.to_list()
        # Fit and transform on the training split
        if len(numerical_to_scale) > 0:
            X_train[numerical_to_scale] = scaler.fit_transform(X_train[numerical_to_scale])
            X_val[numerical_to_scale] = scaler.transform(X_val[numerical_to_scale])

        # OneHot Encoder
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Define categorical columns and check cardinality 
        categorical_to_scale = []
        high_cardinality_to_scale = []
        if len(X_train.select_dtypes(include="object").columns.to_list()) > 0:
            for col in X_train.select_dtypes(include="object").columns.to_list():
                if X_train[col].nunique() > 50: 
                    high_cardinality_to_scale.append(col)
                else:
                    categorical_to_scale.append(col)
            if len(categorical_to_scale) > 0: 
                cat_train = onehot_encoder.fit_transform(X_train[categorical_to_scale])
                # Create dataframe of encoded training 
                cat_cols = onehot_encoder.get_feature_names_out(categorical_to_scale)
                cat_train_df = pd.DataFrame(cat_train, columns=cat_cols, index=X_train.index)
                # Create dataframf of encoded validation 
                cat_val = onehot_encoder.transform(X_val[categorical_to_scale])
                cat_val_df = pd.DataFrame(cat_val, columns=cat_cols, index=X_val.index)
                # Drop original categorical columns and join the encoded ones
                X_train = X_train.drop(columns=categorical_to_scale).join(cat_train_df)
                X_val = X_val.drop(columns=categorical_to_scale).join(cat_val_df)

            # frequency encoding 
            if len(high_cardinality_to_scale) > 0:
                for col in high_cardinality_to_scale:
                    X_train = frequency_encoder(X_train, col)
                    X_val = frequency_encoder(X_val, col)

        ordinal_to_scale = X_train.select_dtypes(include="category").columns.to_list()

        if len(ordinal_to_scale) > 0:

            # Ordinal Encoding
            ordinal_encoder = OrdinalEncoder()
            # Define ordinal columns 
            # Fit and transform on the training .
            X_train[ordinal_to_scale] = ordinal_encoder.fit_transform(X_train[ordinal_to_scale])
            # Transform the validation 
            X_val[ordinal_to_scale] = ordinal_encoder.transform(X_val[ordinal_to_scale])


        return X_train, X_val


    def k_fold(X_train, X_val, y_train, y_val, alpha, low_verbose=False):
        categorical_to_scale = X_train.select_dtypes(include="object").columns.to_list()
        numerical_to_scale = X_train.select_dtypes(include=["int64", "float64"]).columns.to_list()
        ordinal_to_scale = X_train.select_dtypes(include="category").columns.to_list()

        # Define variable before KFold CV
        ols_scores = []
        ridge_scores = []
        lasso_scores = []


        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
            # Fold split
            X_train_fold, X_val_fold = X_train.iloc[train_index].copy(), X_train.iloc[val_index].copy()
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            X_train_fold, X_val_fold = processing(X_train_fold, X_val_fold)
            # Modeling 
            models = {
                "ols": LinearRegression(),
                "ridge": Ridge(alpha=alpha, random_state=42),
                "lasso": Lasso(alpha=alpha, random_state=42)
            }
            y_pred = []
            for model in models.values():
                model.fit(X_train_fold, y_train_fold)
            # Evluating metrics MAE, amd RMSE on both training and validation to detect over/under fitting: 
            # OLS
            ols_pred = models["ols"].predict(X_val_fold)
            old_pred_train = models["ols"].predict(X_train_fold)
            ols_mae = mean_absolute_error(y_val_fold, ols_pred)
            ols_mae_train = mean_absolute_error(y_train_fold, old_pred_train)
            ols_rmse = np.sqrt(mean_squared_error(y_val_fold, ols_pred))
            ols_rmse_train = np.sqrt(mean_squared_error(y_train_fold, old_pred_train))
            ols_r2 = r2_score(y_val_fold, ols_pred)
            ols_r2_train = r2_score(y_train_fold, old_pred_train)
            ols_scores.append((ols_mae,ols_mae_train, ols_rmse, ols_rmse_train, ols_r2, ols_r2_train))
            # Ridge
            ridge_pred = models["ridge"].predict(X_val_fold)
            ridge_pred_train = models["ridge"].predict(X_train_fold)
            ridge_mae = mean_absolute_error(y_val_fold, ridge_pred)
            ridge_mae_train = mean_absolute_error(y_train_fold, ridge_pred_train)
            ridge_rmse = np.sqrt(mean_squared_error(y_val_fold, ridge_pred))
            ridge_rmse_train = np.sqrt(mean_squared_error(y_train_fold, ridge_pred_train))
            ridge_r2 = r2_score(y_val_fold, ridge_pred)
            ridge_r2_train = r2_score(y_train_fold, ridge_pred_train)
            ridge_scores.append((ridge_mae,ridge_mae_train, ridge_rmse, ridge_rmse_train, ridge_r2, ridge_r2_train))
            # Lasso 
            lasso_pred = models["lasso"].predict(X_val_fold)
            lasso_pred_train = models["lasso"].predict(X_train_fold)
            lasso_mae = mean_absolute_error(y_val_fold, lasso_pred)
            lasso_mae_train = mean_absolute_error(y_train_fold, lasso_pred_train)
            lasso_rmse = np.sqrt(mean_squared_error(y_val_fold, lasso_pred))
            lasso_rmse_train = np.sqrt(mean_squared_error(y_train_fold, lasso_pred_train))
            lasso_r2 = r2_score(y_val_fold, lasso_pred)
            lasso_r2_train = r2_score(y_train_fold, lasso_pred_train)
            lasso_scores.append((lasso_mae,lasso_mae_train, lasso_rmse, lasso_rmse_train, lasso_r2, lasso_r2_train))


        # After the loop
        if not low_verbose:
            print("\nAverage performance across all folds:")

            print(f"OLS[Val] - MAE: {np.mean([s[0] for s in ols_scores]):.4f}, RMSE: {np.mean([s[2] for s in ols_scores]):.4f}, R2: {np.mean([s[4] for s in ols_scores]):.4f}")
            print(f"OLS[Train] - MAE: {np.mean([s[1] for s in ols_scores]):.4f}, RMSE: {np.mean([s[3] for s in ols_scores]):.4f}, R2: {np.mean([s[5] for s in ols_scores]):.4f}")
            print(f"OLS[Train/Val] - MAE: {np.mean([s[1] for s in ols_scores])/np.mean([s[0] for s in ols_scores]):.2f}, RMSE:  {np.mean([s[3] for s in ols_scores])/np.mean([s[2] for s in ols_scores]):.2f}")

            print(f"Ridge[Val] - MAE: {np.mean([s[0] for s in ridge_scores]):.4f}, RMSE: {np.mean([s[2] for s in ridge_scores]):.4f}, R2: {np.mean([s[4] for s in ridge_scores]):.4f}")
            print(f"Ridge[Train] - MAE: {np.mean([s[1] for s in ridge_scores]):.4f}, RMSE: {np.mean([s[3] for s in ridge_scores]):.4f}, R2: {np.mean([s[5] for s in ridge_scores]):.4f}")
            print(f"Ridge[Train/Val] - MAE: {np.mean([s[1] for s in ridge_scores])/np.mean([s[0] for s in ridge_scores]):.2f}, RMSE:  {np.mean([s[3] for s in ridge_scores])/np.mean([s[2] for s in ridge_scores]):.2f}")

            print(f"Lasso[Val] - MAE: {np.mean([s[0] for s in lasso_scores]):.4f}, RMSE: {np.mean([s[2] for s in lasso_scores]):.4f}, R2: {np.mean([s[4] for s in lasso_scores]):.4f}")
            print(f"Lasso[Train] - MAE: {np.mean([s[1] for s in lasso_scores]):.4f}, RMSE: {np.mean([s[3] for s in lasso_scores]):.4f}, R2: {np.mean([s[5] for s in lasso_scores]):.4f}")
            print(f"Lasso[Train/Val] - MAE: {np.mean([s[1] for s in lasso_scores])/np.mean([s[0] for s in lasso_scores]):.2f}, RMSE:  {np.mean([s[3] for s in lasso_scores])/np.mean([s[2] for s in lasso_scores]):.2f}")
        else:
            print(f"Ridge[Train/Val] - MAE: {np.mean([s[1] for s in ridge_scores])/np.mean([s[0] for s in ridge_scores]):.2f}, RMSE:  {np.mean([s[3] for s in ridge_scores])/np.mean([s[2] for s in ridge_scores]):.2f}, R2: {np.mean([s[4] for s in ridge_scores]):.2f}")
            print(f"Lasso[Train/Val] - MAE: {np.mean([s[1] for s in lasso_scores])/np.mean([s[0] for s in lasso_scores]):.2f}, RMSE:  {np.mean([s[3] for s in lasso_scores])/np.mean([s[2] for s in lasso_scores]):.2f}, R2: {np.mean([s[4] for s in lasso_scores]):.2f}")
            print("|:--------------------------------------------------------:|")
    return frequency_encoder, k_fold, processing, splitting


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

        1. Dropping `car_ID`, and 'CarName'.
        2. Feature/Target splitting.
        3. Using holdout + CV method for splitting (70% train, 20% validation, 10% testing)
        4. Using standard scalar for numerical features.
        5. Using one hot encoder for categorical features.
        6. Using Ordinal encoder for ordinal features.

        # 5. Modeling

        1. Traing using OLS, ridge, and lasso.
        2. Calculate the metrics and save them for later analysis.
        """
    )
    return


@app.cell
def _(X):
    X.info()
    return


@app.cell
def _(df, k_fold, splitting):
    X, y = df.drop(columns=["car_ID", "CarName","price"], axis=1).copy(), df["price"].copy()

    X_train, X_val, y_train, y_val, X_test, y_test = splitting(X, y)

    k_fold(X_train, X_val, y_train, y_val, 0.9)
    return X, X_test, X_train, X_val, y, y_test, y_train, y_val


@app.cell
def _(mo):
    mo.md(
        r"""
        # 6. Model evaluation 

        After training the model we note the following:

        1. OLS had the highest MAE, and RMSE, while Ridge has the lowest across all folds.
        2. All the models  are overfitting even at high alpha value ($\alpha = 0.9$), and ridge being the least in severity. To solve this problem, we will use lasso's feature selection for further improvement.
        """
    )
    return


@app.cell
def _(
    Lasso,
    SelectFromModel,
    StandardScaler,
    X_train,
    X_val,
    pd,
    processing,
    y_train,
):
    lasso = Lasso(alpha=0.95)
    sfm = SelectFromModel(lasso, prefit=False)

    # Numerical Scaler 
    _scaler = StandardScaler()
    # Fit and transform on the training  
    _X_train = X_train.copy()
    _X_val = X_val.copy()

    # Proccessing the data 
    _X_train, _X_val = processing(_X_train, _X_val)

    sfm.fit(_X_train, y_train)
    X_transformed = sfm.transform(_X_train)

    selected_features = _X_train.columns[sfm.get_support()].tolist()
    print("Selected features:", len(selected_features))
    print("Total number of columns after encoding", len(_X_train.columns.to_list()))

    # Get feature importances (coefficients) from the Lasso model
    feature_importances = pd.Series(
        sfm.estimator_.coef_, index=_X_train.columns
    )  # Access coefficients from the estimator_

    # Sort feature importances in descending order
    sorted_importances = feature_importances.abs().sort_values(ascending=False)  # Use absolute value
    return (
        X_transformed,
        feature_importances,
        lasso,
        selected_features,
        sfm,
        sorted_importances,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Repeat: 5. Modeling 

        - Since we obtained the most important features we can now simplify the model by reducing the number of columns.
        """
    )
    return


@app.cell
def _(pd, selected_features, sorted_importances):
    feature_num = len(selected_features)
    important_features = pd.DataFrame(sorted_importances)
    important_features.insert(0, "features",important_features.index)
    important_features.rename(columns={0:"value"}, inplace=True)
    important_features.reset_index(drop=True, inplace=True)
    print(important_features["features"].unique().tolist()[:50])

    important_features["original_features"] = important_features.apply(lambda row: row["features"].split("_")[0], axis=1)
    return feature_num, important_features


@app.cell
def _(important_features):
    important_features.head(71)["original_features"].value_counts()
    return


@app.cell
def _(important_features):
    important_features.head(71).groupby("original_features")["value"].sum().sort_values(ascending=False)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        From the results above we note: 

        1.  car brand  contribute the most to the model following by engine type.
        2.  Based on the above table, I will only considere the highest contributing 12 features.
        """
    )
    return


@app.cell
def _(df, feature_num, important_features, k_fold, splitting, y):
    sig_features = list(important_features.head(feature_num).groupby("original_features")["value"].sum().sort_values(ascending=False).index[0:13])
    print(sig_features)
    X_sig = df[sig_features].copy()

    X_sig_train, X_sig_val, y_sig_train, y_sig_val, X_sig_test, y_sig_test = splitting(X_sig, y)
    k_fold(X_sig_train, X_sig_val, y_sig_train, y_sig_val, alpha=1)
    return (
        X_sig,
        X_sig_test,
        X_sig_train,
        X_sig_val,
        sig_features,
        y_sig_test,
        y_sig_train,
        y_sig_val,
    )


@app.cell
def _(X_sig_train, X_sig_val, k_fold, y_sig_train, y_sig_val):
    alpha_values = [0.0001,0.001, 0.01, 0.5, 1, 5, 10, 15, 20, 35, 50, 75,80,85,90,95, 100, 125,150,175,200,225, 250, 500]
    for alpha in alpha_values:
        print(f"alpha = {alpha}")
        k_fold(X_sig_train, X_sig_val, y_sig_train, y_sig_val, alpha=alpha, low_verbose=True)
    return alpha, alpha_values


@app.cell
def _(mo):
    mo.md(
        r"""
        Based on the above analysis, I choose ridge to be my model with hyper parameter $\alpha = 15$ due to the balance betweet $R^2$ and the degree of bias. 

        ---
        - After doing cross validation, feature selection, and hyperparameter tunning. I will train the final model with 0.8/0.2 split.
        """
    )
    return


@app.cell
def _(
    Ridge,
    df,
    mean_absolute_error,
    mean_squared_error,
    np,
    processing,
    r2_score,
    sig_features,
    train_test_split,
):
    # Picking the significant features and split features from target
    X_final, y_final = df.drop(columns=["car_ID", "CarName","price"], axis=1)[sig_features].copy(), df["price"].copy()
    # Train/test split
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_final, y_final,
        test_size=0.1,
        random_state=42
    )
    # Scaling and encoding the trainig set
    X_train_final, X_test_final = processing(X_train_final, X_test_final)
    # Defnie the model
    final_model = Ridge(alpha=15, random_state=42)
    # Tring the model
    final_model.fit(X_train_final, y_train_final)
    # Predict the test 
    y_pred_final_train = final_model.predict(X_train_final)
    y_pred_final_test = final_model.predict(X_test_final)

    # Calculate metrics:
    mae_final_train = mean_absolute_error(y_train_final, y_pred_final_train)
    mae_final_test = mean_absolute_error(y_test_final, y_pred_final_test)

    rmse_final_train = np.sqrt(mean_squared_error(y_train_final, y_pred_final_train))
    rmse_final_test = np.sqrt(mean_squared_error(y_test_final, y_pred_final_test))

    r2_final_train = r2_score(y_train_final, y_pred_final_train)
    r2_final_test = r2_score(y_test_final, y_pred_final_test)

    print(f"Evluation metrics:")
    print(f"Train: MAE = {mae_final_train:.2f}, Test: MAE = {mae_final_test:.2f}")
    print("|:-------------------------------------:|")
    print(f"Train: RMSE = {rmse_final_train:.2f}, Test: RMSE = {rmse_final_test:.2f}")
    print("|:-------------------------------------:|")
    print(f"Train: R2 = {r2_final_train:.2f}, Test: R2 = {r2_final_test:.2f}")
    print("|:-------------------------------------:|")
    return (
        X_final,
        X_test_final,
        X_train_final,
        final_model,
        mae_final_test,
        mae_final_train,
        r2_final_test,
        r2_final_train,
        rmse_final_test,
        rmse_final_train,
        y_final,
        y_pred_final_test,
        y_pred_final_train,
        y_test_final,
        y_train_final,
    )


if __name__ == "__main__":
    app.run()
