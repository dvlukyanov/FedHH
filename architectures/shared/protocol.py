from enum import Enum
from typing import List
from dataclasses import dataclass


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class CommandAction(Enum):
    TRAIN = 1
    STOP = 2


class Command():
    action: CommandAction
    model_type: str
    folder: str
    model_src: str
    model_target: str
    epochs: int
    batch_size: int
    items: List[int]
    seed: int


class CommandResult(Enum):
    ACCEPTED = 1
    DONE = 2
    FAILED = 3


@dataclass
class CommandResponse():
    result: CommandResult