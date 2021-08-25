import os
import json
import tarfile
import pandas as pd
import joblib

from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

if __name__ == "__main__":
    
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load("model.joblib")

    print("Loading test input data")
    test_features_data = os.path.join("/opt/ml/processing/test", "feature_vectors.npz")
    test_labels_data = os.path.join("/opt/ml/processing/test", "labels.csv")

    X_test_vectors = sparse.load_npz(test_features_data)
    y_test = pd.read_csv(test_labels_data, header=None)
    
    predictions = model.predict(X_test_vectors)

    f1 = f1_score(y_test, predictions, average='macro')
    print('%s: \n  f1 = %s' % (model, f1))

    print("Creating classification evaluation report")
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_test, predictions)

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
        