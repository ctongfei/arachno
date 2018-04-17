import arachno
import torch as th
import torch.nn.functional as thnf
import numpy as np
from typing import *

class Module(th.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.l1 = th.nn.Linear(2, 2)
        self.l2 = th.nn.Linear(2, 2)
        self.loss = th.nn.MSELoss()

    def forward(self, xy):
        x, y = xy
        x1 = th.sigmoid(self.l1(x))
        y_ = thnf.softmax(self.l2(x1), dim=0)
        return self.loss(y_, y)


class ExperimentTest(arachno.HeldOutExperiment):

    working_dir: str = "/Users/tongfei/my/sandbox/experiment-test"
    max_num_epochs: int = 4000
    minimizing_dev_score: bool = True

    training_module: th.nn.Module = Module()
    validation_module: th.nn.Module = training_module
    optimizer: th.optim.Optimizer = th.optim.Adam(training_module.parameters(), lr=0.01)

    def xs(self):
        return (
            th.autograd.Variable(th.from_numpy(np.array(x, dtype=np.float32)))
            for x in [
                [0., 0.],
                [1., 0.],
                [0., 1.],
                [1., 1.]
            ]
        )

    def ys(self):
        return (
            th.autograd.Variable(th.from_numpy(np.array([1., 0.], dtype=np.float32)) if i == 0 else th.from_numpy(np.array([0., 1.], dtype=np.float32)))
            for i in [0, 1, 1, 0]
        )

    def training_data(self):
        return zip(self.xs(), self.ys())

    def validation_data(self):
        return self.training_data()

    def get_validation_score(self, x) -> float:
        return self.validation_module(x).data[0]

    def initialize_modules(self):
        pass


ExperimentTest().run()
