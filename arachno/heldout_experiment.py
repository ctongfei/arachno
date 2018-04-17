from typing import *
from abc import ABC, ABCMeta, abstractmethod
import torch as th
import tqdm
import sys
import shutil
import os
from arachno import AverageBuilder, ExperimentState


class HeldOutExperiment(ABC):
    """
    Base class for doing any held-out experiment.

    """
    __metaclass__ = ABCMeta

    BEST_MODEL_FILENAME = "best.model"
    LAST_MODEL_FILENAME = "last.model"
    LAST_OPTIM_FILENAME = "last.optim"
    LOG_FILENAME = "log"
    STATE_FILENAME = "state"

    working_dir: str = NotImplemented
    max_num_epochs: int = NotImplemented
    minimizing_dev_score: bool = False

    training_module: th.nn.Module = NotImplemented
    optimizer: th.optim.Optimizer = NotImplemented

    @abstractmethod
    def training_data(self):
        pass

    @abstractmethod
    def validation_data(self):
        pass

    @abstractmethod
    def get_validation_score(self, x) -> float:
        pass

    @abstractmethod
    def initialize_modules(self):
        pass

    def __init__(self):

        self.best_model_path = f"{self.working_dir}/{HeldOutExperiment.BEST_MODEL_FILENAME}"
        self.last_model_path = f"{self.working_dir}/{HeldOutExperiment.LAST_MODEL_FILENAME}"
        self.last_optim_path = f"{self.working_dir}/{HeldOutExperiment.LAST_OPTIM_FILENAME}"
        self.log_path = f"{self.working_dir}/{HeldOutExperiment.LOG_FILENAME}"
        self.state_path = f"{self.working_dir}/{HeldOutExperiment.STATE_FILENAME}"

        self.log_file = open(self.log_path, "a")
        self.state = ExperimentState(
            epoch=0,
            best_validation_score=(float("inf") if self.minimizing_dev_score else float("inf")),
            best_performing_epoch=0
        )

    def __print(self, msg: str):
        tqdm.tqdm.write(msg)
        print(msg, file=self.log_file)

    def __read_state(self):
        if os.path.exists(self.state_path):
            self.state.load(self.state_path)

    def __save_state(self):
        self.state.save(self.state_path)

    def run(self):

        training_average_builder = AverageBuilder()
        validation_average_builder = AverageBuilder()

        self.__print("Check if there is a checkpoint...")
        if os.path.exists(f"{self.working_dir}/{HeldOutExperiment.LAST_MODEL_FILENAME}"):
            self.__print("Last checkpoint found.")
            self.training_module.load_state_dict(th.load(self.last_model_path))
            self.__print("Model loaded from last checkpoint.")
        else:
            self.__print("No checkpoint found.")
            self.initialize_modules()
            self.__print("Model parameters initialized.")

        self.__read_state()
        self.__print(f"Last checkpoint: Epoch={self.state.epoch}; DevScore={self.state.best_validation_score}; "
                     f"BestPerformingEpoch={self.state.best_performing_epoch}")

        while self.state.epoch < self.max_num_epochs:

            self.state.epoch += 1
            self.__print(f"Training session for epoch {self.state.epoch} started.")
            training_average_builder.clear()

            for batch in tqdm.tqdm(self.training_data()):

                loss = self.training_module(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = loss.data[0]
                tqdm.tqdm.write(f"Loss for this batch = {loss_value}")
                training_average_builder.add(loss_value)

            training_loss = training_average_builder.average()
            self.__print(f"Training session for epoch {self.state.epoch} completed. TrainingLoss={training_loss}.")

            th.save(self.training_module.state_dict(), self.last_model_path)
            self.__print("Model saved.")

            th.save(self.optimizer.state_dict(), self.last_optim_path)
            self.__print("Optimizer states saved.")

            self.__print(f"Validation session for epoch {self.state.epoch} started.")
            validation_average_builder.clear()

            for batch in tqdm.tqdm(self.validation_data()):

                score = self.get_validation_score(batch)
                validation_average_builder.add(score)

            validation_score = validation_average_builder.average()
            self.__print(f"Validation session for epoch {self.state.epoch} completed. ValidationScore={validation_score}.")
            if (self.minimizing_dev_score and validation_score < self.state.best_validation_score) or \
                    ((not self.minimizing_dev_score) and validation_score > self.state.best_validation_score):

                self.state.best_performing_epoch = self.state.epoch
                self.state.best_validation_score = validation_score
                shutil.copyfile(self.last_model_path, self.best_model_path)
                self.__print(f"New best model found. Saved this new checkpoint as the best to date.")

            self.__save_state()
            self.log_file.flush()

        self.__print("Max number of epochs reached. Training stopped.")

