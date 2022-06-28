"""
Discrete joint probability distribution module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import copy

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianNetwork as PGMPY_BN

from .joint_distribution import JointDistribution


class DiscreteJointDistribution(JointDistribution):
    """
    Discrete joint probability distribution class.
    """

    def __init__(self, cpds=None):
        super().__init__()
        self.cpds = {cpd.variable: cpd for cpd in cpds}
        self.joint = None

    def condition(self, evidence, order, *, graph=None,
                  algorithm='variable_elimination'):
        """
        Conditions a discrete joint probability distribution on some evidence.

        Parameters
        ----------
        order : list
            Topological order of the nodes.

        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values the observed value for them.

        graph : networkx.DiGraph
            Graph structure of the BN model

        algorithm : str, default='variable_elimination'
            Exact inference algorithm used. Available methods are
            'variable_elimination' and 'message_passing'.

        Raises
        ------
        ValueError
            If the algorithm is not supported.
        """

        model = PGMPY_BN(graph).add_cpds(list(self.cpds.values()))

        if algorithm == 'variable_elimination':
            inference = VariableElimination(model)
        elif algorithm == 'message_passing':
            inference = BeliefPropagation(model)
        else:
            raise ValueError('Algorithm not supported.')

        unobserved = list(set(order) - set(evidence.keys()))

        # TODO: Check elimination order
        factors = {node: inference.query(node, evidence) for node in
                   unobserved}

        return {k: {v['state_names'][k][i]: value} for k, v in factors.items()
                for i, value in enumerate(v['values'])}

    def marginal(self, *, node=None):
        """
        Retrieves the marginal distribution parameters for a set of nodes.

        Parameters
        ----------
        node : list
            Node whose marginal distribution will be computed.

        Returns
        -------
        dict
            The marginal distribution.
        """

        cpd = self.cpds[node]
        if len(cpd.variables) > 1:
            list_vars = copy.deepcopy(cpd.variables)
            list_vars.remove(node)
            cpd.marginalize(list_vars)

        values = list(cpd.get_values().flat)
        states = cpd.state_names[node] if cpd.state_names else states = [
            f'State {i}' for i in range(values)]

        return {states[i]: value for i, value in enumerate(values)}

    def from_parameters(self, parameters, order):
        """
        Computes the joint probability distribution given the topological order
        of some nodes and their parameters. However, it is not recommended
        using the joint probability distribution in the discrete case due to
        exponential blow up. For that reason, inference is implemented using
        the message passing algorithm.

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
        pass
