"""
Gaussian joint probability distribution module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from tempfile import NamedTemporaryFile

import numpy as np

from .joint_distribution import JointDistribution
from ..parameters.gaussian_mle import GaussianNode


class GaussianJointDistribution(JointDistribution):
    """
    Gaussian Joint distribution class. If the size of the distribution is big
    (over 1000 nodes), it saves the distribution in a temporary file and loads
    it only when it has to be used.

    Parameters
    ----------
    order : list
        Order of the nodes in the distribution.

    mu : numpy.array, optional
        Mean vector.

    sigma : numpy.array, optional
        Covariance matrix.
    """

    def __init__(self, order, mu=None, sigma=None):
        super().__init__(order)
        self.mu = mu
        self.sigma = sigma

        self.save_dist = self.mu.shape[0] > 1000
        self.file_path = NamedTemporaryFile(
            delete=False, suffix='.npz').name if self.save_dist else None
        self._save()

    def from_parameters(self, parameters):
        """
        Takes the topological order of some nodes and their parameters and
        computes the joint Gaussian distribution.

        Parameters
        ----------
        parameters : dict[Any, GaussianNode]
            Parameters of the nodes.

        Returns
        -------
        self : GaussianJointDistribution
            Joint probability distribution given by the mean vector and
            the covariance matrix.
        """

        n = len(self.order)
        self.mu = np.zeros((n,))
        self.sigma = np.zeros((n, n))

        for i, node in enumerate(self.order):
            node_params = parameters[node]
            uncond_mean, cond_var = node_params["uncond_mean"], node_params[
                "cond_var"]
            betas = node_params["parents_coeffs"]
            parents = [self.order.index(p) for p in node_params["parents"]]
            self.mu[i] = uncond_mean
            if parents:
                cov_parents_involved = self.sigma[:, parents]
                cov_parents = cov_parents_involved[parents, :]
                self.sigma[:, i] = np.dot(cov_parents_involved,
                                          np.array(betas))
                self.sigma[i, i] = cond_var + np.dot(np.array(betas),
                                                     np.dot(cov_parents,
                                                            np.array(betas)))
                self.sigma[i, :] = self.sigma[:, i]
            else:
                self.sigma[i, i] = cond_var

        self._save()
        return self

    def marginal(self, nodes):
        """
        Retrieves the marginal joint distribution for a set of nodes.

        Parameters
        ----------
        nodes : list
            Set of nodes whose marginal distribution will be computed.

        Returns
        -------
        dict
            The marginal distribution parameters.
        """

        self._load()
        idx_marginal = [self.order.index(node_name) for node_name in nodes]

        mu_marginal = self.mu[idx_marginal]
        sigma_marginal = self.sigma[idx_marginal, :][:, idx_marginal]

        if len(nodes) == 1:
            sigma_marginal = sigma_marginal.item()
            mu_marginal = mu_marginal.item()

        result = {'mu': mu_marginal, 'sigma': sigma_marginal}
        self._save()
        return result

    def condition(self, evidence):
        """
        Conditions a Multivariate Gaussian joint distribution on some evidence.

        Parameters
        ----------
        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values represent the observed value for them.

        Returns
        -------
        dict
            CPDs of the unobserved nodes.
        """

        self._load()
        new_order = [node for node in self.order if
                     node not in evidence.keys()]
        evidence_mask = np.array(
            [node not in new_order for i, node in enumerate(self.order)])

        # xx: nodes with evidence, yy: nodes without evidence
        sigma_xx = self.sigma[evidence_mask, :][:, evidence_mask]
        sigma_yy = self.sigma[~evidence_mask, :][:, ~evidence_mask]
        sigma_xy = self.sigma[evidence_mask, :][:, ~evidence_mask]
        sigma_inv = np.linalg.solve(sigma_xx, sigma_xy)

        mu_y = self.mu[~evidence_mask] + np.dot(sigma_inv.T, np.array(
            list(evidence.values())) - np.array(self.mu[evidence_mask]))
        sigma_y = sigma_yy - np.dot(sigma_xy.T, sigma_inv)

        result = {node: {'mu': mu_y[i], 'sigma': _cond_var(sigma_y, i)} for
                  i, node in enumerate(new_order)}
        self._save()
        return result

    def all_cpds(self):
        """
        Retrieves all the conditional distributions, each one represented by
        its mean and variance.

        Returns
        -------
        dict
            Dictionary with the nodes IDs as keys and distribution parameters
            as values.
        """

        self._load()
        result = {node: {'mu': self.mu[i], 'sigma': _cond_var(self.sigma, i)}
                  for i, node in enumerate(self.order)}
        self._save()
        return result

    def get_cpd(self, node):
        """
        Retrieves the conditional probability distribution of a particular
        node.

        Parameters
        ----------
        node :
            Node whose cpd will be computed.

        Returns
        -------
        dict
            Conditional probability distribution of the node given by its
            Parameters.
        """

        self._load()
        try:
            idx_node = self.order.index(node)

            return {'mu': self.mu[idx_node],
                    'sigma': _cond_var(self.sigma, idx_node)}
        except ValueError:
            print(f'{node} is not present in the network.')
        self._save()

    def is_set(self):
        """
        Checks if the distribution parameters are set.

        Returns
        -------
        bool
            Whether the distribution parameters are set or not.
        """

        return bool(self.mu) and bool(self.sigma)

    def size(self):
        """
        Retrieves the number of nodes in the distribution.

        Returns
        -------
        int
            Number of nodes in the distribution.
        """

        self._load()
        size = self.mu.shape[0]
        self._save()
        return size

    def _save(self):
        """Saves the joint probability distribution in a .npz file."""

        if self.save_dist:
            np.savez(self.file_path, {'order': self.order, 'mu': self.mu,
                                      'sigma': self.sigma})

    def _load(self):
        """Loads the distribution."""

        if self.save_dist:
            config = np.load(self.file_path)
            for k, v in config.items():
                setattr(self, k, v)


def _cond_var(sigma, node_pos):
    """Calculates the conditional variance of a node given the covariance
    matrix."""

    not_node_mask = np.array([True for _ in range(sigma.shape[0])])
    not_node_mask[node_pos] = False
    sigma_yy = sigma[node_pos, node_pos]
    sigma_xy = sigma[not_node_mask, :][:, node_pos]
    sigma_xx = sigma[not_node_mask, :][:, not_node_mask]
    sigma_inv = np.linalg.solve(sigma_xx, sigma_xy)

    return sigma_yy - np.dot(sigma_xy.T, sigma_inv)
