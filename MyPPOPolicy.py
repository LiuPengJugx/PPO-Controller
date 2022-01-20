from typing import Optional, Union, Any
import torch
import numpy as np
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
class MyPPOPolicy(PPOPolicy):
    # def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
    #     logits, h = self.actor(batch.obs, state=state)
    #     if isinstance(logits, tuple):
    #         dist = self.dist_fn(*logits)
    #     else:
    #         dist = self.dist_fn(logits)
    #     act = dist.sample()
    #     return Batch(logits=logits, act=act, state=h, dist=dist)

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        # adapt the discrete action space
        return act

