from typing import NewType, Union

__ReLU = "ReLU"
__GeLU = "GeLU"
Activation = NewType("Activation", Union[__ReLU, __GeLU])
