from enum import Enum


class DOPPolicy(Enum):
    ALWAYS_ON = "ALWAYS_ON"
    PERIODIC = "PERIODIC"
    ADAPTIVE = "ADAPTIVE"
    MANUAL = "MANUAL"

    def __str__(self):
        return self.value
