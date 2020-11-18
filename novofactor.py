# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim

from math import sqrt
from typing import List
from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II


@dataclass
class FairseqNovoFactorConfig(FairseqDataclass):
    novofactor_betas: str = field(
        default="(0.95, 0.25)",
        metadata={"help": "betas for NovoFactor optimizer"}
    )
    novofactor_eps: str = field(
        default="(1e-30, 1e-8)",
        metadata={"help": "epsilon for NovoFactor optimizer"}
    )
    weight_decay: float = field(
        default=1e-4, metadata={"help": "weight decay"}
    )
    clip_threshold: float = field(
        default=1.0, metadata={"help": "clip threshold"}
    )
    grad_avg: bool = field(
        default=False, metadata={"help": "Use beta1 in first moments"}
    )
    amsgrad: bool = field(
        default=False, metadata={"help": "Use AMSgrad"}
    )
    # TODO common vars below in parent
    tpu: bool = II("params.common.tpu")
    lr: List[float] = II("params.optimization.lr")


@register_optimizer('novofactor', dataclass=FairseqNovoFactorConfig)
class FairseqNovoFactor(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = NovoFactor(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        Note : Convergence issues empirically observed with fp16 on.
               Might require search for appropriate configuration.
        """
        return {
            "lr": self.args.lr[0],
            "betas": eval(self.args.novofactor_betas),
            "eps": eval(self.args.novofactor_eps),
            "weight_decay": self.args.weight_decay,
            "clip_threshold": self.args.clip_threshold,
        }


class NovoFactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98),
                 eps=(1e-30, 1e-8), weight_decay=0,
                 clip_threshold=1.0,
                 grad_avg=False, amsgrad=False):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, clip_threshold=clip_threshold,
            grad_avg=grad_avg, amsgrad=amsgrad
            )
        super(NovoFactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_row, exp_avg_col):
        r_factor = (
            exp_avg_row / exp_avg_row.mean(dim=-1, keepdim=True)
        ).unsqueeze(-1)
        c_factor = exp_avg_col.unsqueeze(-2)
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
                    raise RuntimeError(
                        "NovoGrad does not support sparse gradients, please consider SparseNovoGrad instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]
                p_data_fp32_shape = p_data_fp32.shape

                factored = self._get_options(group, p_data_fp32_shape)
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of squared gradient values
                    if factored:
                        state['exp_avg_row'] = torch.zeros(
                            p_data_fp32_shape[:-1]).to(p_data_fp32)
                        state['exp_avg_col'] = torch.zeros(
                            p_data_fp32_shape[:-2] + p_data_fp32_shape[-1:]
                        ).to(p_data_fp32)
                    else:
                        state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros([]).to(p_data_fp32)
                else:
                    if factored:
                        state['exp_avg_row'] = state['exp_avg_row'].to(
                            p_data_fp32
                        )
                        state['exp_avg_col'] = state['exp_avg_col'].to(
                            p_data_fp32
                        )
                    else:
                        state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg_sq = state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = grad.norm().pow(2)
                # Decay the first and second moment running average coefficient
                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1.0 - beta2)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"][1])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"][1])

                # grad.div_(denom)
                # if group["weight_decay"] != 0:
                #     grad.add_(p_data_fp32, alpha=group["weight_decay"])
                # if group["grad_avg"]:
                #     grad.mul_(1 - beta1)
                update = grad  # + group['eps'][0]
                if factored:
                    exp_avg_row = state['exp_avg_row']
                    exp_avg_col = state['exp_avg_col']

                    exp_avg_row.mul_(beta1).add_(update.mean(dim=-1), alpha=(1/denom))
                    exp_avg_col.mul_(beta1).add_(update.mean(dim=-2), alpha=(1/denom))

                    # Approximation of exponential moving average of square of gradient
                    exp_avg = self._approx_sq_grad(exp_avg_row, exp_avg_col)
                else:
                    exp_avg = state["exp_avg"]
                    grad.div_(denom)
                    exp_avg.mul_(beta1).add_(grad)

                if group["weight_decay"] != 0:
                    exp_avg.add_(p_data_fp32, alpha=group["weight_decay"])

                # exp_avg.div_(
                #     (self._rms(exp_avg) / group['clip_threshold']).clamp_(min=1.0)
                # )

                p_data_fp32.add_(exp_avg, alpha=-group["lr"])

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
