from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from netket import jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.vqs import MCState, expect, expect_and_grad, get_local_kernel_arguments, get_local_kernel
from netket.utils import mpi

from netket_fidelity.utils import expect_2distr

from .operator import InfidelityOperatorUPsi

def validate_arguments(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int):
    """ Function to check the restrictions we imposed in this file. """
    if not isinstance(op._U, DiscreteJaxOperator):
        raise ValueError("Only works with discrete operators.")
    if not isinstance(op._U_dagger, DiscreteJaxOperator):
        raise ValueError("Only works with discrete operators.")
    

@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: int = None):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    validate_arguments(vstate, op, chunk_size)

    # for DiscreteJaxOperator: args = operator itself!
    
    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        chunk_size,
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

    validate_arguments(vstate, op, chunk_size)

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)


    return infidelity_sampling_MCState(
        chunk_size,
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
    )

def process_local_arguments(args, σ, n_samples):
    """ args = operator (jax), or xp and mels ()"""
    N = σ.shape[-1]
    if isinstance(args, DiscreteJaxOperator):
        op = args
        xp, mels = op.get_conn_padded(σ)
        n_xp = op.max_conn_size
    else:
        raise Exception("no longer supported, since it messes up the chunking")
        xp = args[0].reshape(σ.shape[0], -1, N)
        mels = args[1].reshape(σ.shape[0], -1)
        n_xp = args[0].shape[-2]
    xp_splitted = [c.reshape(n_samples, N) for c in jnp.split(xp, n_xp, axis=-2)]
    xp_ravel = jnp.vstack(xp_splitted) # n_samples is 2nd axis now
    return xp_ravel, mels, n_xp

def local_fidelity_kernel(
        afun, params, σ, args,
        afun_t, params_t, σ_t, args_t, 
        cv_coeff=None, 
        model_state=None, model_state_t=None
    ):
    """ Part handling the local fidelity kernel. """
    
    # to be sure, we block some gradients
    σ = jax.lax.stop_gradient(σ)
    σ_t = jax.lax.stop_gradient(σ_t)
    args = jax.lax.stop_gradient(args)
    args_t = jax.lax.stop_gradient(args_t)
    
    model_state = model_state or {}
    model_state_t = model_state_t or {}
    
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}

    n_samples = σ.shape[0]
    xp_ravel, mels, n_xp = process_local_arguments(args, σ, n_samples)
    xp_t_ravel, mels_t, n_xp_t = process_local_arguments(args_t, σ_t, n_samples)

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


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad", "chunk_size",))
def infidelity_sampling_MCState(
    chunk_size,
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
):
    N = sigma.shape[-1]
    n_chains_t = sigma_t.shape[-2]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)


    def expect_kernel(params):

        def kernel_fun(params, params_t, σ, σ_t):
            return local_fidelity_kernel(
                afun, params, σ, args,
                afun_t, params_t, σ_t, args_t,
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
    F_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
