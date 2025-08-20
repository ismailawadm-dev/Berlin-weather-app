import joblib
from lightgbm import LGBMClassifier

FEATURES = ["apcp_mean", "apcp_std", "apcp_p90", "apcp_prob_gt01"]

class ProbModel:
    def __init__(self, params=None):
        self.clf = LGBMClassifier(**(params or {}))

    def fit(self, X, y):
        self.clf.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X[FEATURES])[:,1]

    def save(self, path):
        joblib.dump(self.clf, path)

    @staticmethod
    def load(path):
        m = ProbModel()
        m.clf = joblib.load(path)
        return m
