import os
import numpy as np
import joblib

# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return np.array(predictions)

