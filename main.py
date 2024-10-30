import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# args
import argparse

from src.model.decision_tree import DecisionTree
from src.dataset.preprocessing import preprocessing

def main(args):
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')

    # preprocessing
    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)

    # Train Validation Split
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=0)

    X_train = train_data.drop('Y_LABEL', axis=1).values
    y_train = train_data['Y_LABEL'].values

    X_val = val_data.drop('Y_LABEL', axis=1).values
    y_val = val_data['Y_LABEL'].values

    # Decision Tree training
    decision_tree = DecisionTree(max_depth=4, min_samples_leaf=1, min_information_gain=0)
    decision_tree.train(X_train, y_train)

    # Macro F1 Score
    y_pred = decision_tree.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'Validaiton Macro F1 Score: {f1}')

    # Test Prediction
    X_test = test_data.values

    y_test_pred = decision_tree.predict(X_test)
    
    sample_submission['Y_LABEL'] = y_test_pred

    sample_submission.to_csv(f'submission/{args.submit_name}', index=False)
    print('Submission file is saved!')
    print(f"Output file is saved as {args.submit_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="brAIn_A_Session_Second_Assignment")
    parser.add_argument("--submit_name", type=str, default="submission.csv")
    args = parser.parse_args()

    main(args)