import yaml
from pathlib import Path

class Cfg:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.path = Path(path)

    def __getitem__(self, k):
        return self.cfg[k]

    @property
    def cache(self):
        return Path(self.cfg["paths"]["cache"]).expanduser()

    @property
    def models_dir(self):
        return Path(self.cfg["paths"]["models"]).expanduser()
