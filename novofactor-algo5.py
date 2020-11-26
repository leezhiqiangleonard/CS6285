# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.optim

from . import LegacyFairseqOptimizer, register_optimizer

@register_optimizer("algo5")
class FairseqAlgo5(LegacyFairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Algo5(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adafactor-eps', default='(1e-30, 1e-3)', metavar="E",
                            help='epsilons for Adafactor optimizer')
        parser.add_argument('--clip-threshold', type=float, default=1.0, metavar="C",
                            help='threshold for clipping update root mean square')
        parser.add_argument('--decay-rate', type=float, default=-0.8, metavar="D",
                            help='decay rate of the second moment estimator')
        parser.add_argument('--beta1', type=float, default=0.9, metavar="B",
                            help='beta for first moment estimator. Optional')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--scale-parameter', action='store_true',
                            help='scale learning rate by root mean square of parameter')
        parser.add_argument('--relative-step', action='store_true',
                            help='set learning rate to inverse square root of timestep,'
                                 'otherwise use external learning rate')
        parser.add_argument('--warmup-init', action='store_true',
                            help='use relative step for warm-up learning rate schedule')
        # fmt: on

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
            "eps": eval(self.args.adafactor_eps),
            "clip_threshold": self.args.clip_threshold,
            "decay_rate": self.args.decay_rate,
            "beta1": self.args.beta1,
            "weight_decay": self.args.weight_decay,
            "scale_parameter": self.args.scale_parameter,  # defaults to False
            "relative_step": self.args.relative_step,  # defaults to False
            "warmup_init": self.args.warmup_init,
        }

class Algo5(torch.optim.Optimizer):
    """docstring for Algo5"""
    def __init__(self,
        params,
        lr=0.0,
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
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        defaults = dict(
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
        super(Algo5, self).__init__(params,defaults)


    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]

        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        # return factored, use_first_moment
        return factored, True

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

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
                #print(group["lr"])
                
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"]) # Increasing Decay Parameter

                beta1 = group["beta1"]
                beta2 = group["beta2"]

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                second_moment = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(
                        second_moment.mean(dim=-1), alpha=1.0 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        second_moment.mean(dim=-2), alpha=1.0 - beta2
                    )

                    # Approximation of exponential moving average of square of gradient
                    second_moment = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col) / bias_correction2
                    denominator = second_moment.norm().sqrt() + group["epsilon"]
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).add_(second_moment, alpha=1.0 - beta2)
                    denominator = exp_avg_sq.norm().sqrt() + group["epsilon"]

                nominator = grad

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(nominator, alpha=1 - group["beta1"])
                    nominator = exp_avg / bias_correction1

                nominator.mul_(group["lr"])

                update = nominator/denominator

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                # Update Clipping
                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
