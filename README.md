# Car Price Regression Model.
[Assignment](./doc/Assignment_1_502.pdf)

[Source Code](./src/notebook.ipynb) jupyter notebook.

[Source Code](./src/notebook.py) marimo notebook.

[Short Report](./doc/index.pdf)

# Running instructions 

1. **Create a python virtual environment:**

```python3 -m venv .venv```

2. Activate the environment 

- On Linux and MacOS:
    
```source .venv/bin/activate```
    
- On windows:

```.venv\Scripts\activate```

3. Install the requirements/dependencies

```pip install -r requirements.txt```

4. Run the code 

5. Deactivate when finished 

```deactivate```

## Introduction:
Geely Auto, a Chinese car manufacturer wants to enter the U.S.  market by initiating a local manufacturer to compete with both Amerrican and European automakers. To ensure strong an succeful entry to the automobile market, Geely Auto utilized an automobile consulting firm to analyze the key factors influencing car pricing in the American market, which may differ from the Chinese market.

## Problem Statement:

Geely Auto seek want to know two things:

1. Which feature contribute the most in the prediction of car prices. (lasso?)
2. How well do these variable explain car pricing trends (R2?)

## Modeling Approach

1. **Data Splitting**:
   - Training set: 70% of data
   - Validation set: 20% of data
   - Test set: 10% of data

2. **Cross-Validation**:
   - Implemented 10-fold cross-validation on the training set
   - Evaluated models using MAE, RMSE, and R² metrics
   - Monitored train/validation performance ratios to detect overfitting

3. **Hyperparameter Tuning**:
   - Tested alpha values ranging from 0.0001 to 500 for Ridge and Lasso models
   - Selected optimal alpha based on validation performance and overfitting control

## Model Selection Justification:

Ridge regression was selected as the final model because it:

   - Provided the best predictive performance (lowest MAE and RMSE)
   - Controlled overfitting more effectively than OLS
   - Maintained high explanatory power (R² ≈ 0.90)
   - Preserved important features while reducing their coefficients appropriately

   The final Ridge model with α=15 achieved:
   
   - Test MAE: ~2000
   - Test RMSE: ~2500
   - Test R²: 0.90

This indicates the model explains approximately 90% of the variance in car prices, providing reliable predictions and insights for Geely Auto's market entry strategy.
