"""
Utilities to compute hypothesis tests.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np
from scipy.stats import norm


def hypothesis_test_related_genes(max_candidates, positive_bics):
    """
    Given a sorted list of BIC scores (decreasing order), it tests the
    likelihood that the first n nodes belong to one distribution and the
    rest belong to another one. Finally, it returns the value of n that
    maximizes the likelihood.

    Parameters
    ----------
    max_candidates : int
        Maximum possible value for the number of nodes that belong to the first
        distribution.

    positive_bics : list
        Sorted list of positive BIC scores.

    Returns
    -------
    int
        The number of nodes in the first distribution that maximizes the
        likelihood.
    """

    min_candidates = 2
    num_candidates = 0

    if len(positive_bics) <= min_candidates:
        return len(positive_bics)
    else:
        max_loglik = float('-inf')
        _, dist_all = _gaussian_loglik(positive_bics)

        for i in range(min_candidates, max_candidates):
            subset_0_i = positive_bics[:i]
            subset_i_n = positive_bics[i:max_candidates]

            logliks_0_i = np.sum(_gaussian_loglik(subset_0_i)[0])
            if len(subset_i_n) == 1:
                log_lik_i_n = np.sum(
                    _gaussian_loglik(subset_i_n, dist_all)[0])
            else:
                log_lik_i_n = np.sum(_gaussian_loglik(subset_i_n)[0])
            log_likelihood = logliks_0_i + log_lik_i_n

            if log_likelihood > max_loglik:
                max_loglik = log_likelihood
                num_candidates = i + 1

    return num_candidates


def _gaussian_loglik(instances, gaussian_dist=None):
    """
    Returns the log-likelihood of each element in `instances`. If
    `gaussian_dist` is not provided, it uses the data in `instances` to build
    a Gaussian distribution and then computes the log-likelihood of each
    element.

    Parameters
    ----------
    instances : list
        Elements whose log-likelihood has to be calculated.

    gaussian_dist : norm
        Gaussian distribution followed by `instances`

    Returns
    -------
    (array_like, norm)
        The log-likelihood of each element in ´instances´ and the Gaussian
        distribution used to calculate them.
    """

    instances = np.array(instances, dtype=np.float64)

    if gaussian_dist is None:
        mean, var = norm.fit(instances)
        if var == 0.0:
            var = 0.0001
        gaussian_dist = norm(mean, var)

    loglik = gaussian_dist.logpdf(instances)

    return loglik, gaussian_dist
