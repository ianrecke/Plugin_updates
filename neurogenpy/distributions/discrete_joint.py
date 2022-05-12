"""
Discrete joint probability distribution module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from .joint_distribution import JointDistribution


# TODO: Implement discrete inference.
class DiscreteJointDistribution(JointDistribution):
    """
    Discrete joint probability distribution class.
    """

    def condition(self, evidence, order):
        """
        Conditions a discrete joint probability distribution on some evidence.

        Parameters
        ----------
        order : list
            Topological order of the nodes.

        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values the observed value for them.
        """

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
        JointDistribution
            The marginal distribution.
        """

    def from_parameters(self, parameters, order):
        """
        Computes the joint probability distribution given the topological order
        of some nodes and their parameters.

        Parameters
        ----------
        parameters : dict
            Parameters of the nodes.

        order : list
            Topological order of the nodes.

        Returns
        -------
        self : JointDistribution
            The joint probability distribution.
        """
