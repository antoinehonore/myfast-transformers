#
# Copyright (c) 2025 KTH Royal Institute of Technology https://www.kth.se
# Written by Antoine Honoré <honore@kth.se>,
# Antoine Honoré <honore@kth.se>
#

import torch
from  multiprocessing.dummy import Pool

from .cross_causal_product_cpu import cross_causal_dot_product as cross_causal_dot_product_cpu, \
                                 cross_causal_dot_backward as cross_causal_dot_backward_cpu


cross_causal_dot_product_cuda, cross_causal_dot_backward_cuda=None,None

if torch.cuda.is_available():
    from .cross_causal_product_cuda import \
        cross_causal_dot_product as cross_causal_dot_product_cuda, \
        cross_causal_dot_backward as cross_causal_dot_backward_cuda


def causal_dot_product(Q, K, V, tq, tkv, pool=None):
    N, H, L = V.shape[:-1]
    Vdummy = torch.ones((N, H, L, 1), device=V.device)
    
    if not (pool is None):
        product, normalization = pool.starmap(cross_causal_dot_product, [(Q, K, V, tq, tkv),(Q, K, Vdummy, tq, tkv)])
    else:
        product = cross_causal_dot_product(Q, K, V, tq, tkv)
        normalization = cross_causal_dot_product(Q, K, Vdummy, tq, tkv)

    return product / (normalization + 1e-6)


class CrossCausalDotNumerator(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot_numerator = {
        "cpu": cross_causal_dot_product_cpu,
        "cuda": cross_causal_dot_product_cuda
    }
    dot_numerator_backward = {
        "cpu": cross_causal_dot_backward_cpu,
        "cuda": cross_causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V, tq, tkv):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V, tq, tkv)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        
        product = torch.zeros((N, H, L, M), device=device)

        # Actually perform the numerator of dot product
        CrossCausalDotProduct.dot_numerator[device.type](
            Q.data,
            K.data,
            V.data,
            tq, tkv,
            product
        )

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V, tq, tkv = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        # Actually compute the gradients
        CrossCausalDotProduct.dot_numerator_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            tq, tkv,
            grad_out,
            grad_Q,
            grad_K,
            grad_V
        )

        return grad_Q, grad_K, grad_V, None, None


# Alias the autograd functions to python style snake case naming
cross_causal_dot_product = CrossCausalDotProduct.apply

