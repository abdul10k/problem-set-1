from turtle import pd


def run_part_5():
    # Load the test set with predictions
    df_test = pd.read_csv('data/df_arrests_test.csv')

    # Get true labels
    y_true = df_test['y']

    # Calibration plots
    print("Calibration plot for Logistic Regression:")
    calibration_plot(y_true, df_test['pred_lr'], n_bins=5) # type: ignore

    print("Calibration plot for Decision Tree:")
    calibration_plot(y_true, df_test['pred_dt'], n_bins=5) # type: ignore

    # Print calibration comparison
    print("Which model is more calibrated?")
    print("Visually inspect the plots. The closer to the diagonal line, the better the calibration.")

    # Extra Credit
    print("----- EXTRA CREDIT -----")

    # PPV (Positive Predictive Value) in top 50 for Logistic Regression
    top50_lr = df_test.sort_values('pred_lr', ascending=False).head(50)
    ppv_lr = top50_lr['y'].mean()
    print("PPV (top 50) - Logistic Regression:", ppv_lr)

    # PPV in top 50 for Decision Tree
    top50_dt = df_test.sort_values('pred_dt', ascending=False).head(50)
    ppv_dt = top50_dt['y'].mean()
    print("PPV (top 50) - Decision Tree:", ppv_dt)

    # AUC Scores
    auc_lr = roc_auc_score(y_true, df_test['pred_lr']) # type: ignore
    auc_dt = roc_auc_score(y_true, df_test['pred_dt']) # type: ignore
    print("AUC - Logistic Regression:", auc_lr)
    print("AUC - Decision Tree:", auc_dt)

    # Final comparison
    print("Do both metrics agree that one model is more accurate than the other?")
    if auc_lr > auc_dt and ppv_lr > ppv_dt:
        print("Yes — Logistic Regression is more accurate by both AUC and PPV.")
    elif auc_dt > auc_lr and ppv_dt > ppv_lr:
        print("Yes — Decision Tree is more accurate by both AUC and PPV.")
    else:
        print("Not exactly — AUC and PPV disagree on which model is better.")
