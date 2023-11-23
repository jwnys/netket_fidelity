from typing import Callable, Tuple
from functools import partial
import jax.numpy as jnp
import jax
from netket.utils.types import PyTree
from netket.jax import vjp as nkvjp
from netket.stats import statistics as mpi_statistics, Stats
import netket.jax as nkjax
from netket.utils import mpi

def expect_2distr(
    log_pdf_new: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_pdf_old: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars_new: PyTree,
    pars_old: PyTree,
    σ_new: jnp.ndarray,
    σ_old: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
    chunk_size: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """

    return _expect_2distr(
        n_chains,
        chunk_size,
        log_pdf_new,
        log_pdf_old,
        expected_fun,
        pars_new,
        pars_old,
        σ_new,
        σ_old,
        *expected_fun_args,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _expect_2distr(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    *expected_fun_args,
):
    pad_axes = [0]*len(expected_fun_args)
    L_σ = nkjax.apply_chunked(expected_fun, in_axes=(None, None, 0, 0, *pad_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ.T)

    return L̄_σ.mean, L̄_σ


def _expect_fwd_fid(
    n_chains,
    chunk_size,
    log_pdf_new,
    log_pdf_old,
    expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    *expected_fun_args,
):
    pad_axes = [0]*len(expected_fun_args)
    L_σ = nkjax.apply_chunked(expected_fun, in_axes=(None, None, 0, 0, *pad_axes), chunk_size=chunk_size)(pars_new, pars_old, σ_new, σ_old, *expected_fun_args)
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars_new, pars_old, σ_new, σ_old, expected_fun_args, ΔL_σ)

def _expect_bwd_fid(n_chains, chunk_size, log_pdf_new, log_pdf_old, expected_fun, residuals, dout):
    pars_new, pars_old, σ_new, σ_old, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout
    log_p_old = nkjax.apply_chunked(log_pdf_old, in_axes=(None, 0), chunk_size=chunk_size)(pars_old, σ_old)

    def f(pars_new, pars_old, σ_new, σ_old, ΔL_σ, log_p_old, *cost_args):
        log_p = log_pdf_new(pars_new, σ_new) + log_p_old
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars_new, pars_old, σ_new, σ_old, *cost_args)
        out = term1 + term2
        out = out.mean()  # now only mean over the chunk size
        if chunk_size is None:
            return out
        else:
            return jnp.repeat(out, chunk_size) # needed for chunking
    
    chunk_argnums = tuple([2, 3, 4, 5] + [i+5 for i in range(len(cost_args))])
    pb = nkjax.vjp_chunked(
        f, 
        pars_new, pars_old, σ_new, σ_old, ΔL_σ, log_p_old, *cost_args, # primals
        chunk_size=chunk_size, chunk_argnums=chunk_argnums,
        nondiff_argnums=(4,5), # (ΔL_σ, log_p_old) are not differentiable
        # nondiff_argnums=chunk_argnums+(1,),
        has_aux=False
    )

    # netket is a bit annoying here, since it automatically chunks the cotangent
    if chunk_size is not None:
        n_samples = log_p_old.shape[0]
        # internally we have:
        # mean over the chunk size, but summed chunk_size times --> just the sum
        dL̄ = jnp.repeat(dL̄ / n_samples, n_samples) # correct the mean factor !
    grad_f = pb(dL̄)
    # move the mpi_mean to the external functions

    return grad_f


_expect_2distr.defvjp(_expect_fwd_fid, _expect_bwd_fid)


def expect_onedistr(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """
    return _expect_onedistr(
        n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _expect_onedistr(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ.T)
    # L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ


def _expect_onedistr_fwd(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean
    # L̄_σ = L_σ.mean(axis=0)

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars, σ, expected_fun_args, ΔL_σ)


def _expect_onedistr_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    def f(pars, σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars, σ, *cost_args)
        out = term1 + term2
        out = out.mean()
        return out

    _, pb = nkvjp(f, pars, σ, *cost_args)

    grad_f = pb(dL̄)

    return grad_f


_expect_onedistr.defvjp(_expect_onedistr_fwd, _expect_onedistr_bwd)
