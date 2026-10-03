import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import List
from .chained_optimizer import ChainedOptimizer, OptimizerSpec

from modules.commons.common_layers import AdamWLinear, AdamWConv1d


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 3 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    X = G.to(torch.float32)

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)
    
    X = X.to(torch.float16)
    
    # Perform the NS iterations
    if X.size(-2) < X.size(-1):
        for _ in range(steps):
            A = torch.bmm(X, X.mT)
            A = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, A, X, beta=a, alpha=1)
    else:
        for _ in range(steps):
            A = torch.bmm(X.mT, X)
            A = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, X, A, beta=a, alpha=1)
            
    return X


def gram_newton_schulz(G: Tensor, steps: int) -> Tensor:
    """
    Refer to: 
    Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon
    Authors: Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao
    Blogpost: https://dao-ailab.github.io/blog/2026/gram-newton-schulz/
    
    Gram Newton-Schulz iteration to compute the orthogonalization of G.
    Mathematically identical to standard Newton-Schulz but computes iterating 
    on the smaller NxN Gram matrix to save up to 50% FLOPs.
    """
    assert G.ndim == 3
    reset_iterations = [2]
    original_shape = G.shape
    dtype = G.dtype

    X = G.to(torch.float32)
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)
    should_transpose = X.size(-2) > X.size(-1)
    if should_transpose:
        X = X.mT
    X = X.to(torch.float16)

    a, b, c = (3.4445, -4.7750,  2.0315)

    if X.size(-2) != X.size(-1):
        R = torch.bmm(X, X.mT)
        Q = None
        for i in range(steps):
            if i in reset_iterations and i != 0:
                X = torch.bmm(Q, X)
                R = torch.bmm(X, X.mT)
                Q = None
            Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
            if i != 0 and i not in reset_iterations:
                Q = torch.baddbmm(Q, Q, Z, beta=a, alpha=1.0)
            else:
                Q = Z.clone()
                Q.diagonal(dim1=-2, dim2=-1).add_(a)
            if i < steps - 1 and (i + 1) not in reset_iterations:
                RZ = torch.baddbmm(R, R, Z, beta=a, alpha=1.0)
                R = torch.baddbmm(RZ, Z, RZ, beta=a, alpha=1.0)
        X = torch.bmm(Q, X) if not should_transpose else torch.bmm(X.mT, Q)
    else:
        for _ in range(steps):
            A = torch.bmm(X, X.mT)
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    return X.to(dtype).view(original_shape)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in float16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["buffers"].append(state["momentum_buffer"])
            for key in shape_groups:
                group_data = shape_groups[key]
                p, g, buf, m = group_data["params"], group_data["grads"], group_data["buffers"], group["momentum"]
                torch._foreach_lerp_(buf, g, 1-m)
                if group["nesterov"]:
                    torch._foreach_lerp_(g, buf, m)
                    g = torch.stack(g)
                else:
                    g = torch.stack(buf)
                original_shape = g.shape
                if g.ndim >= 4:  # for the case of conv filters
                    g = g.view(g.size(0), g.size(1), -1)
                g = gram_newton_schulz(g, steps=group["ns_steps"])
                
                if group["weight_decay"] > 0:
                    torch._foreach_mul_(p, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(p, g.view(original_shape).unbind(0), alpha=-group["lr"] * max(g[0].size()) ** 0.5)


def get_params_for_muon(model) -> List[Parameter]:
    """
    Filter parameters of a module into two groups: those that can be optimized by Muon,
    and those that should be optimized by a standard optimizer.
    Args:
        module: The module to filter parameters for.
    Returns:
        A list of parameters that should be optimized with muon.
    """
    excluded_module_classes = (nn.Embedding, AdamWLinear, AdamWConv1d)
    muon_params = []
    # BFS through all submodules and exclude parameters from certain module types
    queue = collections.deque([model])
    while queue:
        module = queue.popleft()
        if isinstance(module, excluded_module_classes):
            continue
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
        queue.extend(list(module.children()))
    return muon_params


class Muon_AdamW(ChainedOptimizer):
    def __init__(self, model, lr=0.0005, weight_decay=0.0, muon_args=None, adamw_args=None, verbose=False):
        muon_args = {} if muon_args is None else muon_args
        adamw_args = {} if adamw_args is None else adamw_args
        muon_params_id_set = set(id(p) for p in get_params_for_muon(model))
        spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
        spec_adamw = OptimizerSpec(torch.optim.AdamW, adamw_args, None)
        specs = [spec_muon, spec_adamw]
        callback = None
        if verbose:
            callback = lambda p, spec_idx: print(
            f"Adding param {p.shape} to optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
        )
        super().__init__(model.parameters(), specs, lr=lr, weight_decay=weight_decay, optimizer_selection_callback=callback)
