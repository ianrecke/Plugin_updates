import pandas as pd
import scipy.stats as scipy_stats
import numpy as np


def chi_square_test(model, x, y, zs):
    if isinstance(zs, (frozenset, list, set, tuple,)):
        zs = list(zs)
    else:
        zs = [zs]

    # ---Test whether is correct to do the chi square test---
    num_params = ((len(model.states_names[x]) - 1) *
                  (len(model.states_names[y]) - 1) *
                  np.prod([len(model.states_names[Z]) for Z in zs]))
    sufficient_data = len(model.data) >= num_params * 5
    if not sufficient_data:
        raise Exception(
            "Insufficient data for testing {0} _|_ {1} | {2}. ".format(x, y,
                                                                       zs) +
            "At least {0} samples recommended, {1} present.".format(
                5 * num_params, len(model.data)))
    # ----------------------------------------------------------

    # compute actual frequency/state_count table:
    # = P(x,y,Zs)
    columns_table_counts = [model.data[y]] + [model.data[Z] for Z in zs]
    xy_z_state_counts = pd.crosstab(index=model.data[x],
                                    columns=columns_table_counts)

    chi2_statistic, p_value, degrees_freedom, xyz_expected_test = scipy_stats.chi2_contingency(
        xy_z_state_counts)

    """
    #Manual method:
    
    # compute the expected frequency/state_count table if x _|_ y | Zs:
    # = P(x|Zs)*P(y|Zs)*P(Zs) = P(x,Zs)*P(y,Zs)/P(Zs)
    if Zs:
        xZ_state_counts = xyZ_state_counts.sum(axis=1, level=Zs).fillna(0)  # marginalize out y
        yZ_state_counts = xyZ_state_counts.sum().unstack(Zs).fillna(0)  # marginalize out x
    else:
        xZ_state_counts = xyZ_state_counts.sum(axis=1)
        yZ_state_counts = xyZ_state_counts.sum()
    Z_state_counts = yZ_state_counts.sum()  # marginalize out both
    
    xyZ_expected = pd.DataFrame(index=xyZ_state_counts.index, columns=xyZ_state_counts.columns)

    for x_val in xyZ_expected.index:
        if Zs:
            for y_val in xyZ_expected.columns.levels[0]:
                value = (xZ_state_counts.loc[x_val] *
                         yZ_state_counts.loc[y_val] /
                         Z_state_counts).values
                xyZ_expected.loc[x_val, y_val] = (xZ_state_counts.loc[x_val] *
                                                  yZ_state_counts.loc[y_val] /
                                                  Z_state_counts).values
        else:
            for y_val in xyZ_expected.columns:
                xyZ_expected.loc[x_val, y_val] = (xZ_state_counts.loc[x_val] *
                                                  yZ_state_counts.loc[y_val] /
                                                  float(Z_state_counts))

    observed = xyZ_state_counts.values.flatten()
    expected = xyZ_expected.fillna(0).values.flatten()
    # remove elements where the expected value is 0;
    # this also corrects the degrees of freedom for chisquare
    observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

    chi2_statistic, p_value = scipy_stats.chisquare(observed, expected)
    """

    return chi2_statistic, p_value


def is_independent_chi_square_test(model, x, y, z):
    chi2_statistic, p_value = chi_square_test(model, x, y, z)

    is_independent = p_value >= model.significance_level

    return is_independent


def hypothesis_test_related_genes(max_candidates, positive_bics_values,
                                  heuristic=False):
    min_candidates = 2
    num_candidates = 0

    if len(positive_bics_values) <= min_candidates:
        num_candidates = len(positive_bics_values)
    else:
        max_q = float("-inf")
        p_all, gaussian_dist_all = gaussian_likelihood(positive_bics_values)

        for i in range(min_candidates, max_candidates):
            print("Backward intersect {} of {}".format(i, max_candidates))
            bics_subset_0_k = positive_bics_values[0:i]

            if heuristic:
                log_likelihood = gaussian_likelihood(bics_subset_0_k)[
                                     0] - p_all[0:i]
                q = np.sum(log_likelihood)
            else:
                bics_subset_k_n = positive_bics_values[i:max_candidates]

                log_lik_0_k = np.sum(gaussian_likelihood(bics_subset_0_k)[0])
                if len(bics_subset_k_n) == 1:
                    log_lik_k_n = np.sum(gaussian_likelihood(bics_subset_k_n,
                                                             gaussian_dist_all)[
                                             0])
                else:
                    log_lik_k_n = np.sum(
                        gaussian_likelihood(bics_subset_k_n)[0])
                log_likelihood = log_lik_0_k + log_lik_k_n

                q = 2 * (log_likelihood - np.sum(p_all))

            if q > max_q:
                max_q = q
                num_candidates = i + 1

    return num_candidates


def gaussian_likelihood(instances, gaussian_dist=None):
    instances = np.array(instances, dtype=np.float64)

    if gaussian_dist is None:
        mean, var = scipy_stats.norm.fit(instances)
        if var == 0.0:
            var = 0.0001
        gaussian_dist = scipy_stats.norm(mean, var)

    log_likelihood = gaussian_dist.logpdf(instances)

    return log_likelihood, gaussian_dist
