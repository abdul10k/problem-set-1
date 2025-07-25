'''
You will run this problem set from main.py, so set things up accordingly
'''

#import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot


# Call functions / instanciate objects from the .py files
def main():
    
    #part1_etl()

    # PART 2: Preprocess and save df_arrests
    part2_preprocessing.preprocess()

    # PART 3: Run logistic regression, save train/test with pred_lr
    part3_logistic_regression.run_logistic_regression()

    # PART 4: Run decision tree, adds pred_dt to test set
    part4_decision_tree.run_decision_tree()

    # PART 5: Plot calibration curves, print comparison, do extra credit
    part5_calibration_plot.run_part_5()

if __name__ == "__main__":
    main()
