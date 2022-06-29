"""
Discrete joint probability distribution module. It uses `pgmpy`
implementations.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import copy

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianNetwork as PGMPY_BN

from .joint_distribution import JointDistribution


class DiscreteJointDistribution(JointDistribution):
    """
    Discrete joint probability distribution class. Conditional probability
    distributions are stored rather than the full joint distribution. Inference
    is performed using variable elimination or message passing algorithms.

    Parameters
    ----------
    order : list
        Elimination order of the nodes in the distribution.

    cpds : dict[pgmpy.factors.discrete.TabularCPD]
        Conditional probability distributions of the nodes.
    """

    def __init__(self, order, cpds=None):
        super().__init__(order)
        self.cpds = {cpd.variable: cpd for cpd in cpds}

    def condition(self, evidence, *, graph=None,
                  algorithm='variable_elimination'):
        """
        Conditions a discrete joint probability distribution on some evidence.

        Parameters
        ----------
        evidence : dict
            The evidence to use for conditioning the distribution. The keys are
            nodes and the values the observed value for them.

        graph : networkx.DiGraph
            Graph structure of the BN model.

        algorithm : str, default='variable_elimination'
            Exact inference algorithm used. Available methods are
            'variable_elimination' and 'message_passing'.

        Returns
        -------
        dict
            CPDs of the unobserved nodes.

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

        unobserved = list(set(self.order) - set(evidence.keys()))

        factors = {node: inference.query(node, evidence) for node in
                   unobserved}

        return {k: {v['state_names'][k][i]: value} for k, v in factors.items()
                for i, value in enumerate(v['values'])}

    def marginal(self, nodes):
        """
        Retrieves the marginal distribution for a set of nodes.

        Parameters
        ----------
        nodes : list
            Nodes whose marginal cpds will be computed.

        Returns
        -------
        dict[pgmpy.factors.discrete.TabularCPD]
            The new CPDs.
        """

        result = {}

        for node in nodes:
            cpd = self.cpds[node]
            if len(cpd.variables) > 1:
                list_vars = copy.deepcopy(cpd.variables)
                for keep_node in nodes:
                    if keep_node in list_vars:
                        list_vars.remove(node)
                cpd.marginalize(list_vars)

            values = list(cpd.get_values().flat)
            if cpd.state_names:
                states = cpd.state_names[node]
            else:
                states = [f'State {i}' for i in range(values)]
            result[node] = {states[i]: value for i, value in enumerate(values)}

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
        pgmpy.factors.discrete.TabularCPD
            Conditional probability distribution of the node given by its
            TabularCPD.
        """

        return self.cpds[node]

    def all_cpds(self):
        """
        Retrieves all the conditional distributions, represented by
        `pgmpy.TabularCPD` objects.

        Returns
        -------
        dict[pgmpy.factors.discrete.TabularCPD]
            Dictionary with the nodes IDs as keys and distributions as values.
        """

        return self.cpds

    def from_parameters(self, parameters):
        """
        Sets the conditional probability distributions for the nodes.

        Parameters
        ----------
        parameters : dict[pgmpy.factors.discrete.TabularCPD]
            CPDs of the nodes.

        Returns
        -------
        self : JointDistribution
            The joint probability distribution.
        """

        self.cpds = parameters
        return self

    def is_set(self):
        """
        Checks if the distribution parameters are set.

        Returns
        -------
        bool
            Whether the distribution parameters are set or not.
        """

        return bool(self.cpds)

    def size(self):
        """
        Retrieves the number of nodes in the distribution.

        Returns
        -------
        int
            Number of nodes in the distribution.
        """

        return len(self.cpds)
