#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import ipdb
import math
import torch

from torch_optim import Adafactor

class Algo5(Adafactor):
	"""docstring for Algo5"""
	def __init__(self,
		params,
		lr=None,
		eps=(1e-30, 1e-3),
		clip_threshold=1.0,
		decay_rate=-0.8,
		beta1=0.9,
		beta2=0.999,
		epsilon=1e-8,
		luc=False,
		luc_trust=1e-3,
		luc_eps=1e-8,
		weight_decay=0.0,
		scale_parameter=True,
		relative_step=True,
		warmup_init=False,
	):
		super(Algo5, self).__init__(params,
			lr=lr,
			eps=eps,
			clip_threshold=clip_threshold,
			decay_rate=decay_rate,
			beta1=beta1,
			beta2=beta2,
			epsilon=epsilon,
			luc=False,
			luc_trust=1e-3,
			luc_eps=1e-8,
			weight_decay=weight_decay,
			scale_parameter=scale_parameter,
			relative_step=relative_step,
			warmup_init=warmup_init,
		)


	def _get_options(self, param_group, param_shape):
		factored = len(param_shape) >= 2
		use_first_moment = param_group["beta1"] is not None
		# return factored, use_first_moment
		return factored, True

	def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
		r_factor = (
			(exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
			.unsqueeze(-1)
		)
		c_factor = exp_avg_sq_col.unsqueeze(-2)
		return torch.mul(r_factor, c_factor)


	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.dtype in {torch.float16, torch.bfloat16}:
					grad = grad.float()
				if grad.is_sparse:
					raise RuntimeError("Adafactor does not support sparse gradients.")

				state = self.state[p]
				grad_shape = grad.shape

				factored, use_first_moment = self._get_options(group, grad_shape)

				# State Initialization
				if len(state) == 0:
					state["step"] = 0
					if use_first_moment: # Always True
						# Exponential moving average of gradient values
						state["exp_avg"] = torch.zeros_like(grad)
					if factored:
						state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
						state["exp_avg_sq_col"] = torch.zeros(
							grad_shape[:-2] + grad_shape[-1:]
						).to(grad)
					else:
						state["exp_avg_sq"] = torch.zeros_like(grad)

					state["RMS"] = 0
				else:
					if use_first_moment:
						state["exp_avg"] = state["exp_avg"].to(grad)
					if factored:
						state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
						state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
					else:
						state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

				p_data_fp32 = p.data
				if p.data.dtype in {torch.float16, torch.bfloat16}:
					p_data_fp32 = p_data_fp32.float()

				state["step"] += 1
				state["RMS"] = self._rms(p_data_fp32)
				group["lr"] = self._get_lr(group, state)

				beta2t = 1.0 - math.pow(state["step"], group["decay_rate"]) # Increasing Decay Parameter

				beta1 = group["beta1"]
				beta2 = group["beta2"]

				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']

				update = (grad ** 2) + group["eps"][0]
				if factored:
					exp_avg_sq_row = state["exp_avg_sq_row"]
					exp_avg_sq_col = state["exp_avg_sq_col"]

					exp_avg_sq_row.mul_(beta2).add_(
						update.mean(dim=-1), alpha=1.0 - beta2
					)
					exp_avg_sq_col.mul_(beta2).add_(
						update.mean(dim=-2), alpha=1.0 - beta2
					)

					# Approximation of exponential moving average of square of gradient
					update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col) / bias_correction2
					update = grad/(update.norm() + group["epsilon"])
				else:
					exp_avg_sq = state["exp_avg_sq"]

					exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
					update = exp_avg_sq.rsqrt().mul_(grad)

				update.div_(
					(self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
				)

				update.mul_(group["lr"])

				if use_first_moment:
					exp_avg = state["exp_avg"]
					exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
					update = exp_avg / bias_correction1

				if group["weight_decay"] != 0:
					p_data_fp32.add_(
						p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
					)

				p_data_fp32.add_(-update)

				if p.data.dtype in {torch.float16, torch.bfloat16}:
					p.data.copy_(p_data_fp32)

		return loss