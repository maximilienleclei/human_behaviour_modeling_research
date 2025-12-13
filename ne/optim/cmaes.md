# CMA-ES Optimizer

Covariance Matrix Adaptation Evolution Strategy with diagonal approximation for neuroevolution.

Contains CMAESState maintaining algorithm state: mean [num_params] (search center, weighted average of top performers), sigma (global step size via cumulative step-size adaptation), C_diag [num_params] (diagonal covariance for coordinate-wise variances), p_c/p_sigma [num_params] (evolution paths for covariance/step-size adaptation), generation counter. Learning rates: c_c (covariance path 4/(n+4)), c_1 (rank-1 update), c_mu (rank-mu update), c_sigma (step-size), damps (damping), chi_n (expected length). optimize_cmaes() uses base.optimize() with select_cmaes() performing soft weighted selection and parameter averaging. Samples from N(mean, sigma²·C_diag). Fully implemented with diagonal approximation for efficiency. Used by ne/eval/supervised.py and ne/eval/environment.py.
