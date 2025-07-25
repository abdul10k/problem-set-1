'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''


import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.model_selection import StratifiedKFold as KFold_strat # type: ignore
from sklearn.tree import DecisionTreeClassifier as DTC # type: ignore


def run_decision_tree():
    # Load the train and test datasets from Part 3
    df_train = pd.read_csv('data/df_arrests_train.csv')
    df_test = pd.read_csv('data/df_arrests_test.csv')

    # Set up features and target
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    target = 'y'

    # Set parameter grid for max_depth
    param_grid_dt = {'max_depth': [1, 3, 5]}

    # Initialize decision tree model
    dt_model = DecisionTreeClassifier() # type: ignore

    # Set up GridSearchCV for decision tree
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(df_train[features], df_train[target])

    best_depth = gs_cv_dt.best_params_['max_depth']
    print("What was the optimal value for max_depth?")
    print(best_depth)

    print("Did it have the most or least regularization? Or in the middle?")
    if best_depth == min(param_grid_dt['max_depth']):
        print("Most regularization")
    elif best_depth == max(param_grid_dt['max_depth']):
        print("Least regularization")
    else:
        print("In the middle")

    # Predict on test set
    df_test['pred_dt'] = gs_cv_dt.predict(df_test[features])

    # Save for Part 5
    df_test.to_csv('data/df_arrests_test.csv', index=False)

    return df_test