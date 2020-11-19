#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import torch
from torch_optim import *

class Algo1(Adafactor):
	"""docstring for Algo1"""
	def __init__(self,
		params,
		lr=None,
		eps=(1e-30, 1e-3),
		clip_threshold=1.0,
		decay_rate=-0.8,
		beta1=0.9,
		weight_decay=0.0,
		scale_parameter=True,
		relative_step=True,
		warmup_init=False,
	):
		super(Algo1, self).__init__(params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
		)