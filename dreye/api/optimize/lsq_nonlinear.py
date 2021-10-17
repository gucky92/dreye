"""
Non-linear least squares

Currently uses scipy.optimize, but in the future will use
jax optimization to increase speed.
"""

import warnings 
import numpy as np
from scipy import optimize
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, grad, vmap

from dreye.api.optimize.parallel import batch_arrays, batched_iteration
from dreye.api.optimize.utils import prepare_parameters_for_linear, FAILURE_MESSAGE, replace_numpy


# B is assumed to have the affine transform already applied
# jacfwd uses forward-mode automatic differentiation, 
# which is more efficient for “tall” Jacobian matrices 
# (many functions/residuals/channels), 
# while jacrev uses reverse-mode, which is more efficient 
# for “wide” (many parameters to fit) Jacobian matrices.


class LeastSquaresObjective:

    def __init__(self, nonlin, nonlin_prime=None, jac_prime=False):
        self.nonlin = nonlin
        self.nonlin_prime = nonlin_prime
        self.jac_prime = jac_prime  # nonlin prime returns the jacobian

    def objective(self, x, A, e, w, baseline):
        return w * (self.nonlin(A @ x + baseline) - e)

    def objective_jac(self, x, A, e, w, baseline):
        if self.jac_prime:
            # TODO check if this is correct
            return w[..., None] * self.nonlin_prime(A @ x + baseline) @ A
        else:
            return w[..., None] * self.nonlin_prime(A @ x + baseline)[..., None] * A


def lsq_nonlinear(
    A, B,
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    nonlin=None, 
    nonlin_prime=None,
    jac_prime=False,
    error='raise', 
    n_jobs=None, 
    batch_size=1,
    autodiff=True,
    verbose=0, 
    linopt_kwargs={},
    **opt_kwargs
):
    """
    Nonlinear least-squares. 

    A (channels x inputs)
    B (samples x channels)
    K (channels) or (channels x channels)
    baseline (channels)
    ub (inputs)
    lb (inputs)
    w (channels)
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)

    # setup function and jacobian
    if autodiff and (nonlin is not None):
        if nonlin_prime is None:
            jnp_nonlin = replace_numpy(jnp, nonlin)

            if not jac_prime:
                jnp_nonlin_prime = jit(vmap(grad(jnp_nonlin)))
            
            elif (A.shape[1] * 2) > B.shape[1]:
                jnp_nonlin_prime = jit(jacrev(jnp_nonlin))
            
            else:
                jnp_nonlin_prime = jit(jacfwd(jnp_nonlin))

            def nonlin_prime(x):
                return np.asarray(jnp_nonlin_prime(x))

        lsq = LeastSquaresObjective(nonlin, nonlin_prime, jac_prime)
        opt_kwargs['jac'] = lsq.objective_jac
    elif nonlin is not None:
        lsq = LeastSquaresObjective(nonlin)

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    if nonlin is None:
        E = B
    else:
        E = nonlin(B)
    X = np.zeros((B.shape[0], A.shape[-1]))
    count_failure = 0
    for idx, (e, b, w), (A_, baseline_, lb_, ub_) in batched_iteration(E.shape[0], (E, B, W), (A, baseline, lb, ub), batch_size=batch_size):
        # TODO parallelizing
        # TODO test using sparse matrices when batching
        # TODO fit with lsq_linear and skip if perfect?
        # TODO substitute with faster algorithm
        # TODO substitute linear algorithm with faster version
        result = optimize.lsq_linear(
            A_ * w[:, None], (b - baseline_) * w, bounds=(lb_, ub_), 
            **linopt_kwargs
        )
        idx_slice = slice(idx * batch_size, (idx+1) * batch_size)
        
        if nonlin is None:
            X[idx_slice] = result.x.reshape(-1, A.shape[-1])
        else:
            # reshape resulting x
            x0 = result.x.reshape(-1, A.shape[-1])
            # skip zero residual solutions
            res = nonlin(x0 @ A.T + baseline) - E[idx_slice]
            in_gamut = np.isclose(res, 0).all(axis=-1)

            if np.all(in_gamut):
                # if all within gamut just assign to x0
                X[idx_slice] = x0
            
            elif ~np.any(in_gamut):
                # fit in nonlinear case
                result = optimize.least_squares(
                    lsq.objective, 
                    result.x, 
                    args=(A_, e, w, baseline_), 
                    bounds=(lb_, ub_),
                    **opt_kwargs
                )
            
            else:
                # assign within gamut x and fit the rest
                X[idx_slice][in_gamut] = x0[in_gamut]
                n_out = np.sum(~in_gamut)

                # rebatch out of gamut samples
                if n_out == 1:
                    A_, baseline_, lb_, ub_ = A, baseline, lb, ub
                else:
                    A_, baseline_, lb_, ub_ = batch_arrays([A, baseline, lb, ub], n_out)
                e, w = E[idx_slice][~in_gamut].ravel(), W[idx_slice][~in_gamut].ravel()

                result = optimize.least_squares(
                    lsq.objective, 
                    x0[~in_gamut].ravel(), 
                    args=(A_, e, w, baseline_), 
                    bounds=(lb_, ub_),
                    **opt_kwargs
                )

                X[idx_slice][~in_gamut] = result.x.reshape(-1, A.shape[-1])
                
        count_failure += int(result.status <= 0)

    if count_failure:
        if error == "ignore":
            pass
        elif error == "warn":
            warnings.warn(FAILURE_MESSAGE.format(count=count_failure), RuntimeWarning)
        else:
            raise RuntimeError(FAILURE_MESSAGE.format(count=count_failure)) 

    return X