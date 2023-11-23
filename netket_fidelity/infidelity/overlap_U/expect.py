from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from netket import jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.vqs import MCState, expect, expect_and_grad, get_local_kernel_arguments
from netket.utils import mpi

from netket_fidelity.utils import expect_2distr

from .operator import InfidelityOperatorUPsi


@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int = None):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
        op.cv_coeff,
        return_grad=False,
        chunk_size=chunk_size,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: MCState,
    op: InfidelityOperatorUPsi,
    chunk_size: int = None,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
        op.cv_coeff,
        return_grad=True,
        chunk_size=chunk_size,
    )

def process_local_arguments(args, σ, n_samples):
    """ args = operator (jax), or xp and mels ()"""
    N = σ.shape[-1]
    if isinstance(args, DiscreteJaxOperator):
        xp, mels = args.get_conn_padded(σ)
        n_xp = args.max_conn_size
    else:
        xp = args[0].reshape(σ.shape[0], -1, N)
        mels = args[1].reshape(σ.shape[0], -1)
        n_xp = args[0].shape[-2]
    xp_splitted = [c.reshape(n_samples, N) for c in jnp.split(xp, n_xp, axis=-2)]
    xp_ravel = jnp.vstack(xp_splitted) # n_samples is 2nd axis now
    return xp_ravel, mels, n_xp

def local_fidelity_kernel(
        afun, params, σ, xp_ravel, mels, n_xp,
        afun_t, params_t, σ_t, xp_t_ravel, mels_t, n_xp_t, 
        cv_coeff=None, 
        model_state=None, model_state_t=None
    ):
    """ we can do the chunking over the samples, since we only take gradients through the params ! """
    model_state = model_state or {}
    model_state_t = model_state_t or {}
    
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}

    logpsi_t_xp = jnp.split(afun_t(W_t, xp_ravel), n_xp, axis=0)
    logpsi_t_xp = jnp.stack(logpsi_t_xp, axis=1)

    logpsi_xp_t = jnp.split(afun(W, xp_t_ravel), n_xp_t, axis=0)
    logpsi_xp_t = jnp.stack(logpsi_xp_t, axis=1)

    log_val = (
        logsumexp(logpsi_t_xp, axis=-1, b=mels)
        + logsumexp(logpsi_xp_t, axis=-1, b=mels_t)
        - afun(W, σ)
        - afun_t(W_t, σ_t)
    )
    res = jnp.exp(log_val).real
    if cv_coeff is not None:
        res = res + cv_coeff * (jnp.exp(2 * log_val.real) - 1)
    return res


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad", "chunk_size"))
def infidelity_sampling_MCState(
    afun,
    afun_t,
    params,
    params_t,
    model_state,
    model_state_t,
    sigma,
    args,
    sigma_t,
    args_t,
    cv_coeff,
    return_grad,
    chunk_size,
):
    N = sigma.shape[-1]
    n_chains_t = sigma_t.shape[-2]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)
    n_samples = σ.shape[0]

    xp_ravel, mels, n_xp = process_local_arguments(args, σ, n_samples)
    xp_t_ravel, mels_t, n_xp_t = process_local_arguments(args_t, σ_t, n_samples)

    def expect_kernel(params):

        def kernel_fun(params, params_t, σ, σ_t):
            return local_fidelity_kernel(
                afun, params, σ, xp_ravel, mels, n_xp,
                afun_t, params_t, σ_t, xp_t_ravel, mels_t, n_xp_t,
                cv_coeff=cv_coeff, 
                model_state=model_state, 
                model_state_t=model_state_t
            )
        log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        log_pdf_t = (
            lambda params, σ: 2 * afun_t({"params": params, **model_state_t}, σ).real
        )

        return expect_2distr(
            log_pdf,
            log_pdf_t,
            kernel_fun,
            params,
            params_t,
            σ,
            σ_t,
            n_chains=n_chains_t,
            chunk_size=chunk_size,
        )

    if not return_grad:
        F, F_stats = expect_kernel(params)
        return F_stats.replace(mean=1 - F)

    out = nkjax.vjp(
        expect_kernel, 
        params,
        has_aux=True, conjugate=True
    )
    F, F_vjp_fun, F_stats = out # (primals, vjp_fun, aux)
    F_grad = F_vjp_fun(jnp.ones_like(F))[0]

    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
