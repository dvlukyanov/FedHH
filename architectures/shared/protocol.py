from enum import Enum
from typing import List
import numpy as np
from dataclasses import dataclass


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class CommandAction(Enum):
    TRAIN = 1
    STOP = 2

@dataclass
class Command():
    action: CommandAction
    model_type: str
    folder: str
    model_src: str
    model_target: str
    epochs: int
    batch_size: int
    items: List[int]
    test_ratio: float
    seed: int


class CommandResult(Enum):
    ACCEPTED = 1
    DONE = 2
    FAILED = 3


@dataclass
class Metric():
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    # cmatrix: np.ndarray

@dataclass
class CommandResponse():
    result: CommandResult
    train_history: any
    test_history: any