from typing import Sequence, Union, Optional, Any, Dict, Tuple

import numpy as np
import torch
from tianshou.utils.net.continuous import Actor
from tianshou.utils.net.common import MLP
from torch import nn


class MyActor(Actor):
    def forward(self, s: Union[np.ndarray, torch.Tensor], state: Any = None, info: Dict[str, Any] = {}) -> Tuple[
        torch.Tensor, Any]:
        logits, h = self.preprocess(s, state)
        logits = self._max * torch.relu(self.last(logits))
        return logits, h