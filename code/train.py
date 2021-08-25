import os    
import argparse
import numpy as np
import pandas as pd
import re
import joblib
import json

from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sagemaker_training import environment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-estimators",  type=int, default=100)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    os.environ
    
    training_data_directory = os.environ["SM_CHANNEL_TRAIN"]
    train_features_path = os.path.join(training_data_directory, 'feature_vectors.npz')
    train_labels_path = os.path.join(training_data_directory, 'labels.csv')
    
    X_train_vectors = sparse.load_npz(train_features_path)
    print(X_train_vectors.shape)
    
    y_train = pd.read_csv(train_labels_path, header=None)

    print("Training the Random Forest Classifier")
    hyperparameters = {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_estimators": args.n_estimators,
            "verbose": 1,  
    }
    
    model = RandomForestClassifier()
    model.set_params(**hyperparameters)
    model.fit(X_train_vectors, y_train.iloc[:,0])

    model_dir = os.environ["SM_MODEL_DIR"]
    model_output_directory = os.path.join(model_dir, "model.joblib")
    print("Saving model to {}".format(model_output_directory))
    joblib.dump(model, model_output_directory)
    
    