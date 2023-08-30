"""
A loss function measures how good our predictions are.
We can use it to adjust the parameters of our network.
"""

import numpy as np
from znet.tensor import Tensor


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error.
    We are going to do total squared error.
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return float(np.sum((predicted - actual) ** 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
