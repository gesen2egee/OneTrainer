from enum import Enum


class LossPreset(Enum):
    CUSTOM = "CUSTOM"
    CWMI = "CWMI"

    def __str__(self):
        return self.value
