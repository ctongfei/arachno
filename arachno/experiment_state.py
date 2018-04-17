import json

class ExperimentState:

    def __init__(self,
                 epoch: int = 0,
                 best_validation_score: float = 0.0,
                 best_performing_epoch: int = 0
                 ):

        self.epoch = epoch
        self.best_validation_score = best_validation_score
        self.best_performing_epoch = best_performing_epoch

    def states(self):
        return {
            "epoch": self.epoch,
            "dev_score": self.best_validation_score,
            "best_performing_epoch": self.best_performing_epoch
        }

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)

    def load(self, filename: str):
        with open(filename, "r") as f:
            j = json.load(f)
            self.epoch = int(j["epoch"])
            self.best_validation_score = float(j["best_validation_score"])
            self.best_performing_epoch = int(j["best_performing_epoch"])
