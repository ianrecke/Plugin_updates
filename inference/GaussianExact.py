import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import nquad


def joint(order, nodes_ids, parameters):
    """
    Takes the factorized distribution implied by the Bayesian network and
        creates the joint Gaussian distribution.
    @param order: Topological order of the Bayesian network nodes.
    @param nodes_ids: IDs of the nodes of the network
    @param parameters: Parameters of the Bayesian network.
    @return: Joint probability distribution.
    """

    n = len(nodes_ids)
    mu = np.zeros((n,))
    sigma = np.zeros((n, n))

    for node in order:
        node_name = nodes_ids[node]
        params = parameters[node_name]
        mean, var = params.mean, params.var
        parents_coeffs, parents = params.parents_coeffs, params.parents_names
        parents = [nodes_ids.index(i) for i in parents]
        mu[node] = mean + sum(
            [mu[i] * j for i, j in zip(parents, parents_coeffs)])
        if parents:
            cov_parents_involved = sigma[:, parents]
            cov_parents = cov_parents_involved[parents, :]
            sigma[:, node] = np.dot(cov_parents_involved,
                                    np.array(parents_coeffs))
            sigma[node, node] = var + np.dot(np.array(parents_coeffs),
                                             np.dot(cov_parents,
                                                    np.array(parents_coeffs)))
            sigma[node, :] = sigma[:, node]
        else:
            sigma[node, node] = var

    joint_dist = {"mu": mu, "sigma": sigma}

    return joint_dist


def condition_on_evidence(joint_dist, nodes_ids, evidence):
    mu = joint_dist["mu"]
    sigma = joint_dist["sigma"]

    # Conditions a multivariate gaussian on some evidences
    evidences_nodes = evidence.get_names()
    evidences_vals = [evidence.get(node) for node in evidences_nodes]
    not_evidence_nodes_order = [x for x in nodes_ids if
                                 x not in evidences_nodes]

    """Divide the covariance matrix into blocks xx, xy, yy for xx: nodes with 
    evidences, yy:nodes wo evidences"""
    indices = list(range(sigma.shape[0]))
    idx_sigma = np.array([nodes_ids[i] in evidences_nodes for i in indices])

    sigma_xy = sigma[idx_sigma, :][:, ~idx_sigma]
    sigma_xx = sigma[idx_sigma, :][:, idx_sigma]
    sigma_inv = np.linalg.solve(sigma_xx, sigma_xy)
    sigma_yy = sigma[~idx_sigma, :][:, ~idx_sigma]

    # Compute conditional distribution
    mu_y = mu[~idx_sigma] + np.dot(sigma_inv.T,
                                   (evidences_vals - mu[idx_sigma]))
    sigma_y = sigma_yy - np.dot(sigma_xy.T, sigma_inv)

    joint_dist_cond = {"mu": mu_y, "sigma": sigma_y}

    return joint_dist_cond, not_evidence_nodes_order


def marginal(joint_dist, all_nodes_ids, marginal_nodes_ids=None,
             only_one_marginal=False, multivariate=False):
    mu_joint = joint_dist["mu"]
    sigma_joint = joint_dist["sigma"]

    if multivariate:
        idx_mu = [all_nodes_ids.index(node_name) for node_name in
                  marginal_nodes_ids]
        indices = list(range(sigma_joint.shape[0]))
        idx_sigma = np.array(
            [all_nodes_ids[i] in marginal_nodes_ids for i in indices])

        mu_marginal = mu_joint[idx_mu]
        sigma_marginal = sigma_joint[idx_sigma, :][:, idx_sigma]

        if len(marginal_nodes_ids) == 1:
            mu_marginal = mu_marginal
            sigma_marginal = sigma_marginal.item()
    else:
        if only_one_marginal:
            mu_marginal = mu_joint[
                all_nodes_ids.index(marginal_nodes_ids[0])].item()
            sigma_marginal = sigma_joint[
                all_nodes_ids.index(marginal_nodes_ids[0])].item()
            return mu_marginal, sigma_marginal
        else:
            indices_marginals = [all_nodes_ids.index(node_name) for node_name
                                 in marginal_nodes_ids]
            mu_marginal = mu_joint[indices_marginals]
            sigma_marginal = sigma_joint[indices_marginals]

    return mu_marginal, sigma_marginal


def condition_on_ineq(mu, sigma, evidence, nodes_ids, mode="greater"):
    # Takes greater than or lower than as modes. Conditions a joint
    # distribution on inequality evidence and returns the posterior.
    # P(Y|X>x) = P(X>x|Y)P(Y)/P(X>x)
    if mode not in ["greater", "lower"]:
        raise ValueError("Conditioning mode can only be greater or lower")

    evidences_nodes = evidence.get_names()
    evidences_vals = [evidence[node] for node in evidences_nodes]
    not_evidence_nodes_order = [x for x in nodes_ids if
                                 x not in evidence.get_names()]

    # Get the marginals for X and Y
    mu_x, sigma_x = marginal(mu, sigma, nodes_ids, evidences_nodes)
    p_x = mvn(mean=mu_x, cov=sigma_x)
    cdf_x = p_x.cdf(np.array(evidences_vals))
    if mode is "greater":
        cdf_x = 1 - cdf_x

    mu_y, sigma_y = marginal(mu, sigma, nodes_ids, not_evidence_nodes_order)
    p_y = mvn(mean=mu_y, cov=sigma_y)

    def posterior(y, post_mode="equal"):
        if post_mode not in ["map", "equal", "greater", "lower"]:
            raise ValueError(
                "Mode can only be map, equal (pdf), lower or greater (cdf)")

        if post_mode is "map":
            # I think this is just mu_y, but need to check
            return mu_y

        elif post_mode is "equal":
            # Get inverse conditional P(X|Y)
            evidence_inverse = {name: y[i] for i, name in
                                enumerate(not_evidence_nodes_order)}
            mu_xy, sigma_xy, _ = condition_on_evidence(mu, sigma,
                                                       evidence_inverse)
            cdf_x_if_y = mvn(mean=mu_xy, cov=sigma_xy).cdf(
                np.array(evidences_vals))
            if mode is "greater":
                cdf_x_if_y = 1 - cdf_x_if_y

            # Return pdf for P(Y=y|X>x)
            return p_y.pdf(np.array(y)) * cdf_x_if_y / cdf_x

        else:
            def integrand(*args):
                value_y = [arg for arg in args]
                return posterior(value_y)

            # Use pdf mode and integrate to obtain cdf
            if post_mode is "lower":
                ranges = [(mu_y[i] - 10 * sigma_y[i, i], y[i]) for i in
                          range(len(y))]
                cdf = nquad(integrand, ranges)

            else:
                ranges = [(y[i], mu_y[i] + 10 * sigma_y[i, i]) for i in
                          range(len(y))]
                cdf = nquad(integrand, ranges)

            return cdf

    # Returns a function from where you can query the pdf, cdf and map of the
    # posterior.
    return posterior
