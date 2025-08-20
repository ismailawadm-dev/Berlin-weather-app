import joblib
from lightgbm import LGBMRegressor

FEATURES = ["apcp_mean", "apcp_std", "apcp_p90", "apcp_prob_gt01"]

class QuantileModel:
    def __init__(self, quantiles=(0.5, 0.75, 0.9), params=None):
        self.quantiles = quantiles
        self.models = {q: LGBMRegressor(objective="quantile", alpha=q, **(params or {})) for q in quantiles}

    def fit(self, X, y):
        for q, m in self.models.items():
            m.fit(X[FEATURES], y)
        return self

    def predict(self, X):
        out = {}
        for q, m in self.models.items():
            out[q] = m.predict(X[FEATURES])
        return out

    def save(self, path):
        joblib.dump({"qs": self.quantiles, "models": self.models}, path)

    @staticmethod
    def load(path):
        o = joblib.load(path)
        qm = QuantileModel(o["qs"]); qm.models = o["models"]; return qm
