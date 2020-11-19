#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import torch

from torch_optim import Adafactor

class Algo3(Adafactor):
	"""docstring for Algo3"""
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
		super(Algo3, self).__init__(params,
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

	def _approx_sq_grad_1(self, exp_avg_sq_row, exp_avg_sq_col):
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

				# State Initialization: Zero Initialization
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
					state["second_moment"] = torch.zeros(1, dtype=torch.float32).to(grad)
				else:
					if use_first_moment:
						state["exp_avg"] = state["exp_avg"].to(grad)
					if factored:
						state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
						state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
					else:
						state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

					state["second_moment"] = state["second_moment"].to(grad)


				p_data_fp32 = p.data
				if p.data.dtype in {torch.float16, torch.bfloat16}:
					p_data_fp32 = p_data_fp32.float()

				state["step"] += 1
				state["RMS"] = self._rms(p_data_fp32)
				group["lr"] = self._get_lr(group, state)

				# beta2t = 1.0 - math.pow(state["step"], group["decay_rate"]) # Increasing Decay Parameter
				# beta1t = 1.0 - math.pow(state["step"], group["decay_rate"]) # Increasing Decay Parameter
				beta1 = group["beta1"]
				beta2 = group["beta2"]
				# update = (grad ** 2) + group["eps"][0]
				# update = grad + group["eps"][0] 
				update = grad
				if factored:
					exp_avg_sq_row = state["exp_avg_sq_row"]
					exp_avg_sq_col = state["exp_avg_sq_col"]

					# exp_avg_sq_row.mul_(beta2t).add_(
					#	 update.mean(dim=-1), alpha=1.0 - beta2t
					# )
					# exp_avg_sq_col.mul_(beta2t).add_(
					#	 update.mean(dim=-2), alpha=1.0 - beta2t
					# )

					exp_avg_sq_row.mul_(beta1).add_(
						update.mean(dim=-1), alpha=1.0 - beta1
					)
					exp_avg_sq_col.mul_(beta1).add_(
						update.mean(dim=-2), alpha=1.0 - beta1
					)			   

					# Approximation of exponential moving average of square of gradient
					update = self._approx_sq_grad_1(exp_avg_sq_row, exp_avg_sq_col)
					# update.mul_(grad)
				else:
					exp_avg_sq = state["exp_avg_sq"]
					# exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
					# update = exp_avg_sq.rsqrt().mul_(grad)
					exp_avg_sq.mul_(beta1).add_(update, alpha=1.0 - beta1)

				# bias correction
				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']

				# first moment
				update.div_(bias_correction1)

				# second moment
				norm = grad.norm().pow(2)
				state["second_moment"].mul_(beta2).add_(norm, alpha=1.0 - beta2).div_(bias_correction2)


				denom = state["second_moment"].sqrt().add_(group["epsilon"])

				update.div_(denom)



				# Not sure what this is
				update.div_(
					 (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
				)


				# if use_first_moment:
				#	 exp_avg = state["exp_avg"]
				#	 exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
				#	 update = exp_avg

				if group["weight_decay"] != 0:
					p_data_fp32.add_(
						p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
					)

				if group["luc"]:
					# Clip update so that updates are less than eta*weights
					data_norm = torch.norm(p.data)
					grad_norm = torch.norm(update.data)
					luc_factor = group["luc_trust"] * data_norm / (grad_norm + group["luc_eps"])
					luc_factor = min(luc_factor, group["lr"])
					p_data_fp32.add_(update, alpha=-luc_factor)
				else:
					p_data_fp32.add_(update, alpha=-group["lr"])

				if p.data.dtype in {torch.float16, torch.bfloat16}:
					p.data.copy_(p_data_fp32)

		return loss	