"""
Copyright 2024 Novartis Institutes for BioMedical Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.optim as optim
from typing import cast, List, Optional, Dict, Tuple

# from https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch

class NoamOpt(optim.Optimizer):
    "Optim wrapper that implements rate."
    def __init__(self, params, lr, betas, eps, warmup, model_size, factor, 
                 weight_decay=0,
                 amsgrad=False, *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: bool = False
                 ):
        self.optimizer = optim.Adam(params, lr, betas, eps)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size    
        self._rate = lr # should be set to 0???
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)        
        super(NoamOpt, self).__init__(params, defaults)

    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self,closure):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        loss = self.optimizer.step(closure)
        #print("[STEP " +
        #      str(self._step) +
        #      "] LR: " +
        #      str(self.rate()) +
        #      " loss: " +
        #      str(loss)
        #      )
        return loss
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
