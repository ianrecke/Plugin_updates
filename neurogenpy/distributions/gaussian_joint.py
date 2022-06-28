"""
Gaussian joint probability distribution module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np

from .joint_distribution import JointDistribution
from ..parameters.gaussian_mle import GaussianNode


class GaussianJointDistribution(JointDistribution):
    """
    Gaussian Joint distribution class.

    Parameters
    ----------
    mu : numpy.array
        Mean vector.

    sigma : numpy.array
        Covariance matrix.
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def from_product(self, mu, sigmas, parents_coeffs, parents, order):
        n = len(order)
        self.mu = mu
        self.sigma = np.zeros((n, n))

        for i, node in enumerate(order):
            mean, var = mu[i], sigmas[i]
            betas = parents_coeffs[i]
            node_parents = parents[i]
            node_parents = [order.index(p) for p in node_parents]

            if node_parents:
                cov_parents_involved = self.sigma[:, node_parents]
                self.sigma[:, i] = np.dot(cov_parents_involved,
                                          np.array(betas))
                self.sigma[i, i] = var
                self.sigma[i, :] = self.sigma[:, i]
            else:
                self.sigma[i, i] = var

        return self

    def from_parameters(self, parameters, order):
        """
        Takes the topological order of some nodes and their parameters and
        computes the joint Gaussian distribution.

        Parameters
        ----------
        parameters : dict[Any, GaussianNode]
            Parameters of the nodes.

        order : list
            Topological order of the nodes.

        Returns
        -------
        self : GaussianJointDistribution
            Joint probability distribution given by the mean vector and
            the covariance matrix.
        """

        n = len(order)
        self.mu = np.zeros((n,))
        self.sigma = np.zeros((n, n))

        for i, node in enumerate(order):
            node_params = parameters[node]
            mean, var = node_params["mu"], node_params["sigma"]
            betas = node_params["parents_coeffs"]
            parents = [order.index(p) for p in node_params["parents"]]
            self.mu[i] = mean + sum(
                [self.mu[p] * b for p, b in zip(parents, betas)])
            if parents:
                cov_parents_involved = self.sigma[:, parents]
                cov_parents = cov_parents_involved[parents, :]
                self.sigma[:, i] = np.dot(cov_parents_involved,
                                          np.array(betas))
                self.sigma[i, i] = var + np.dot(np.array(betas),
                                                np.dot(cov_parents,
                                                       np.array(betas)))
                self.sigma[i, :] = self.sigma[:, i]
            else:
                self.sigma[i, i] = var

        return self

    def marginal(self, nodes, marginal_nodes):
        """
        Retrieves the marginal distribution parameters for a set of nodes.

        Parameters
        ----------
        nodes : list
            Full set of nodes in the joint distribution.

        marginal_nodes : list
            Set of nodes whose marginal distribution will be computed.

        Returns
        -------
        dict
            The marginal distribution parameters.
        """

        idx_marginal = [nodes.index(node_name) for node_name in marginal_nodes]

        mu_marginal = self.mu[idx_marginal]
        sigma_marginal = self.sigma[idx_marginal, :][:, idx_marginal]

        if len(marginal_nodes) == 1:
            sigma_marginal = sigma_marginal.item()
            mu_marginal = mu_marginal.item()

        return {'mu': mu_marginal, 'sigma': sigma_marginal}

    def condition(self, order, evidence):
        """
        Conditions a Multivariate Gaussian joint distribution on some evidence.

        Parameters
        ----------
        order : list
            Topological order of the nodes.

        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values the observed value for them.
        """

        indices = list(range(self.sigma.shape[0]))
        evidence_mask = np.array(
            [order[i] in evidence.keys() for i in indices])

        evidence_values = [evidence[node] for node in order if
                           node in evidence.keys()]

        # xx: nodes with evidence, yy: nodes without evidence
        sigma_xx = self.sigma[evidence_mask, :][:, evidence_mask]
        sigma_yy = self.sigma[~evidence_mask, :][:, ~evidence_mask]
        sigma_xy = self.sigma[evidence_mask, :][:, ~evidence_mask]
        sigma_inv = np.linalg.solve(sigma_xx, sigma_xy)

        mu_y = self.mu[~evidence_mask] + np.dot(sigma_inv.T, np.array(
            evidence_values) - np.array(self.mu[evidence_mask]))
        sigma_y = sigma_yy - np.dot(sigma_xy.T, sigma_inv)

        self.mu = mu_y
        self.sigma = sigma_y

    def all_cpds(self, nodes):
        idx_marginal = [nodes.index(node_name) for node_name in nodes]

        return {node: {'mu': self.mu[i], 'sigma': self.calc(i)} for i, node in
                enumerate(nodes)}

    def calc(self, node_pos):
        array_node = np.array([node_pos])
        sigma_yy = self.sigma[node_pos, node_pos]
        sigma_xy = self.sigma[~array_node, :][:, node_pos]
        sigma_xx = self.sigma[~array_node, :][:, ~array_node]
        sigma_inv = np.linalg.solve(sigma_xx, sigma_xy)

        return sigma_yy - np.dot(sigma_xy.T, sigma_inv)
