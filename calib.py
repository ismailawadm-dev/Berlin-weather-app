import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib

class Calibrator:
    def __init__(self, method="isotonic"):
        self.method = method
        self.model = None

    def fit(self, p_raw, y):
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y)
        else:
            self.model = LogisticRegression().fit(p_raw.reshape(-1,1), y)
        return self

    def predict(self, p_raw):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(p_raw.reshape(-1,1))[:,1]
        return self.model.predict(p_raw)

    def save(self, path):
        joblib.dump({"method": self.method, "model": self.model}, path)

    @staticmethod
    def load(path):
        o = joblib.load(path)
        c = Calibrator(o["method"]); c.model = o["model"]; return c
