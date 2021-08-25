import os
import warnings
import argparse

import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    
    input_data_path = os.path.join("/opt/ml/processing/input", "articles.csv")

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    print("df.shape:", df.shape)
    
    X_data = df.drop(df.columns[0], axis=1)
    y_data = df.drop(df.columns[1], axis=1)

    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_ratio, random_state=0)
    
    print("Running preprocessing and feature engineering transformations")

    # create the Tfidf vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectors = vectorizer.fit_transform(X_train.iloc[:,0])
    X_test_vectors = vectorizer.transform(X_test.iloc[:,0])
    
    training_dir = "/opt/ml/processing/train"
    testing_dir = "/opt/ml/processing/test"
    print("training_dir: ", training_dir)
    print("testing_dir: ", testing_dir)

    train_features_output_path = os.path.join(training_dir, "feature_vectors.npz")
    train_labels_output_path = os.path.join(training_dir, "labels.csv")

    test_features_output_path = os.path.join(testing_dir, "feature_vectors.npz")
    test_labels_output_path = os.path.join(testing_dir, "labels.csv")
    
    sparse.save_npz(train_features_output_path, X_train_vectors)
    sparse.save_npz(test_features_output_path, X_test_vectors)
    y_train.to_csv(train_labels_output_path, header=False, index=False)
    y_test.to_csv(test_labels_output_path, header=False, index=False)


