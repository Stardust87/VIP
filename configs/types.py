from enum import Enum


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"
    SGD = "sgd"


class Scheduler(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"
    FLAT_ANNEAL = "flat_anneal"


class Model(str, Enum):
    I2PA = "i2pa"
    FAME = "fame"
