from enum import Enum


class CEPNoiseType(Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    UNIFORM = "UNIFORM"

    def __str__(self):
        return self.value
