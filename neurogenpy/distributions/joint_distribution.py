"""
Joint probability distribution base module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from abc import ABCMeta, abstractmethod


class JPD(metaclass=ABCMeta):
    """
    Abstract class for joint probability distributions.

    Parameters
    ----------
    order :
        Order of the nodes in the distribution.
    """

    def __init__(self, order):
        self.order = order

    @abstractmethod
    def from_parameters(self, parameters):
        """
        Computes the joint probability distribution given the topological order
        of some nodes and their parameters.

        Parameters
        ----------
        parameters : dict
            Parameters of the nodes.

        Returns
        -------
        self : JPD
            The joint probability distribution.
        """

    def all_marginals(self):
        """
        Retrieves the marginal distribution for each node in the distribution.

        Returns
        -------
        dict
            Marginal distribution for each node.
        """

        return {node: self.marginal(node) for node in self.order}

    @abstractmethod
    def marginal(self, node):
        """
        Retrieves the marginal distribution parameters for a node

        Returns
        -------
            The marginal distribution.
        """

    @abstractmethod
    def condition(self, evidence):
        """
        Conditions a joint probability distribution on some evidence and
        retrieves the distributions f(U|E=e) where 'U' represents unobserved
        variables and 'E=e', the evidence.

        Parameters
        ----------
        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values the observed value for them.

        Returns
        -------
        dict
            Distributions f(U|E=e).
        """

    @abstractmethod
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
            Conditional probability distribution of the node.
        """

    def relabel_nodes(self, mapping):
        """
        Relabel the nodes of the distribution according to a given mapping.

        Parameters
        ----------
        mapping : dict
            A dictionary with the old labels as keys and new labels as values.
            It can be a partial mapping.
        """

        self.order = [mapping[node] if node in mapping.keys() else node
                      for node in self.order]

    @abstractmethod
    def is_set(self):
        """
        Checks if the distribution parameters are set.

        Returns
        -------
        bool
            Whether the distribution parameters are set or not.
        """

    @abstractmethod
    def size(self):
        """
        Retrieves the number of nodes in the distribution.

        Returns
        -------
        int
            Number of nodes in the distribution.
        """

    def get_order(self):
        """
        Retrieves the order of the nodes in the distribution.

        Returns
        -------
            Order of the nodes in the distribution.
        """

        return self.order
