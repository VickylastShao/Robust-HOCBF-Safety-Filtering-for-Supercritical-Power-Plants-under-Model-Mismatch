"""Multi-constraint HOCBF for CCS with multiple safety constraints.

Stacks multiple HOCBF constraints into a single QP:
  min ||u - u_rl||^2  s.t.  A_stacked u <= b_stacked

where A_stacked = [A1; A2; ...; AK] and b_stacked = [b1; b2; ...; bK].
"""
import jax
import jax.numpy as jnp
import numpy as np

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.gp.gp_residual import GPResidual


class MultiConstraintHOCBF:
    """Stack multiple HOCBF constraints into a single QP.

    Each constraint h_i(x) has its own relative degree and k_gains.
    The QP constraint matrix A and RHS b are stacked vertically.

    Parameters
    ----------
    hocbf_list : list of HOCBF
        One HOCBF per constraint.
    """

    def __init__(self, hocbf_list: list):
        self.hocbf_list = hocbf_list
        self.n_constraints = len(hocbf_list)

    def qp_matrices(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return stacked (A, b) for all constraints.

        Returns
        -------
        A : (K, n_u) stacked constraint matrix
        b : (K,) stacked constraint RHS
        """
        A_rows, b_vals = [], []
        for hocbf in self.hocbf_list:
            A_i, b_i = hocbf.qp_matrices(x)
            A_rows.append(A_i)
            b_vals.append(b_i)
        return jnp.concatenate(A_rows, axis=0), jnp.concatenate(b_vals)


class MultiConstraintRobustHOCBF:
    """Stack multiple RobustHOCBF constraints with per-constraint epsilon.

    For each constraint i: A_i u <= b_i - epsilon_i(x)

    Supports three epsilon modes:
      - "compositional": state-dependent epsilon(x) via recursive sigma chain
      - "constant_mean": constant epsilon_0 = mean of epsilon(x) over sampled states
      - "constant_max": constant epsilon_0 = max of epsilon(x) over sampled states

    Parameters
    ----------
    robust_hocbf_list : list of RobustHOCBF
        One RobustHOCBF per constraint.
    epsilon_mode : str
        "compositional" (default), "constant_mean", or "constant_max".
    epsilon_constant_values : list of float or None
        Pre-computed constant epsilon values for each constraint.
        Required when epsilon_mode != "compositional".
    """

    def __init__(self, robust_hocbf_list: list,
                 epsilon_mode: str = "compositional",
                 epsilon_constant_values: list = None):
        self.epsilon_mode = epsilon_mode
        self.n_constraints = len(robust_hocbf_list)

        if epsilon_mode == "compositional":
            self.robust_hocbf_list = robust_hocbf_list
        elif epsilon_mode in ("constant_mean", "constant_max"):
            if epsilon_constant_values is None:
                raise ValueError(
                    f"epsilon_constant_values required for mode '{epsilon_mode}'")
            # Replace each RobustHOCBF with ConstantEpsilonRobustHOCBF
            self.robust_hocbf_list = []
            for hocbf, eps_val in zip(robust_hocbf_list, epsilon_constant_values):
                const_hocbf = ConstantEpsilonRobustHOCBF(
                    h_fn=hocbf.h_fn,
                    f_fn=hocbf.f_nominal,
                    g_fn=hocbf.g_fn,
                    relative_degree=hocbf.m,
                    k_gains=list(hocbf.k_gains),
                    gp_residual=hocbf.gp_residual,
                    epsilon_constant=eps_val,
                    u_max=hocbf.u_max,
                    op_norm_estimate=hocbf.op_norm_estimate,
                    u0=hocbf.u0,
                    epsilon_kappa=hocbf.epsilon_kappa,
                    epsilon_floor=hocbf.epsilon_floor,
                    use_mean_correction=hocbf.use_mean_correction,
                )
                self.robust_hocbf_list.append(const_hocbf)
        else:
            raise ValueError(f"Unknown epsilon_mode: {epsilon_mode}")

    @staticmethod
    def compute_constant_epsilons(robust_hocbf_list, sample_states,
                                  mode="mean"):
        """Pre-compute constant epsilon values from sampled states.

        Parameters
        ----------
        robust_hocbf_list : list of RobustHOCBF
            Compositional epsilon constraints to sample from.
        sample_states : array-like, shape (N, n_x)
            States at which to evaluate epsilon(x).
        mode : str
            "mean" or "max".

        Returns
        -------
        epsilon_constants : list of float
            One constant value per constraint.
        """
        epsilon_constants = []
        for hocbf in robust_hocbf_list:
            eps_values = []
            for x in sample_states:
                x_jax = jnp.array(x)
                eps = float(hocbf.compute_epsilon(x_jax))
                eps_values.append(eps)
            eps_arr = np.array(eps_values)
            if mode == "mean":
                epsilon_constants.append(float(np.mean(eps_arr)))
            elif mode == "max":
                epsilon_constants.append(float(np.max(eps_arr)))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return epsilon_constants

    def compute_epsilon(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute per-constraint robustness margins.

        Returns
        -------
        epsilons : (K,) robustness margin for each constraint
        """
        epsilons = []
        for hocbf in self.robust_hocbf_list:
            epsilons.append(hocbf.compute_epsilon(x))
        return jnp.array(epsilons)

    def qp_matrices(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return stacked (A, b - epsilon) for all robust constraints.

        Returns
        -------
        A : (K, n_u) stacked constraint matrix
        b_robust : (K,) stacked constraint RHS with epsilon subtracted
        """
        A_rows, b_vals = [], []
        for hocbf in self.robust_hocbf_list:
            A_i, b_i = hocbf.qp_matrices(x)  # b_i already has epsilon subtracted
            A_rows.append(A_i)
            b_vals.append(b_i)
        return jnp.concatenate(A_rows, axis=0), jnp.concatenate(b_vals)
