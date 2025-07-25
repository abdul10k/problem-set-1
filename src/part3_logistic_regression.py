'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''



import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV  # type: ignore
from sklearn.model_selection import StratifiedKFold as KFold_strat  # type: ignore
from sklearn.linear_model import LogisticRegression as lr  # type: ignore


import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

def run_logistic_regression():
    # Load data
    df = pd.read_csv('data/df_arrests.csv')

    # Define features and target
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    target = 'y'

    # Train-test split
    df_train, df_test = train_test_split(
        df, 
        test_size=0.3, 
        shuffle=True, 
        stratify=df[target], 
        random_state=42
    )

    # Parameter grid
    param_grid = {'C': [0.01, 1, 100]}

    # Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)

    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
    gs_cv.fit(df_train[features], df_train[target])

    # Best C
    best_C = gs_cv.best_params_['C']
    print("What was the optimal value for C?")
    print(best_C)

    print("Did it have the most or least regularization? Or in the middle?")
    if best_C == 0.01:
        print("Most regularization")
    elif best_C == 100:
        print("Least regularization")
    else:
        print("In the middle")

    # Predict on test set
    df_test['pred_lr'] = gs_cv.predict(df_test[features])

    # Save for Part 4
    df_train.to_csv('data/df_arrests_train.csv', index=False)
    df_test.to_csv('data/df_arrests_test.csv', index=False)

    return df_train, df_test



