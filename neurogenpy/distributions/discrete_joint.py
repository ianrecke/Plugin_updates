"""
Discrete joint probability distribution module. It uses `pgmpy`
implementations.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import copy
import logging

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianNetwork as PGMPY_BN

from .joint_distribution import JPD

logger = logging.getLogger(__name__)


class DiscreteJPD(JPD):
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
        self.cpds = cpds

    def condition(self, evidence, *, graph=None,
                  algorithm='variable_elimination'):
        """
        Conditions a discrete joint probability distribution on some evidence
        and retrieves the distributions P(U|E=e) where 'U' represents
        unobserved variables and 'E=e', the evidence.

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
            Distributions P(U|E=e).

        Raises
        ------
        ValueError
            If the algorithm is not supported.
        """

        model = PGMPY_BN(graph)

        model.add_cpds(*list(self.cpds.values()))

        if algorithm == 'variable_elimination':
            inference = VariableElimination(model)
        elif algorithm == 'message_passing':
            inference = BeliefPropagation(model)
        else:
            raise ValueError('Algorithm not supported.')

        unobserved = list(set(self.order) - set(evidence.keys()))

        factors = {node: inference.query([node], evidence) for node in
                   unobserved}

        return {
            k: {v.state_names[k][i]: value for i, value in enumerate(v.values)}
            for k, v in factors.items()}

    def marginal(self, node):
        """
        Retrieves the marginal distribution for a node.

        Parameters
        ----------
        node : list
            Nodes used to compute the marginal distribution.

        Returns
        -------
        dict
            The marginal distribution for each node.
        """

        try:
            cpd = self.cpds[node].copy()
            if len(cpd.variables) > 1:
                list_vars = copy.deepcopy(cpd.variables)
                list_vars.remove(node)
                cpd.marginalize(list_vars)

            values = list(cpd.get_values().flat)
            if cpd.state_names:
                states = cpd.state_names[node]
            else:
                states = [f'State {i}' for i in range(values)]

            return {states[i]: value for i, value in enumerate(values)}
        except KeyError:
            logger.error(f'{node} is not present in the distribution.')

    def get_cpd(self, node, **kwargs):
        """
        Retrieves the conditional probability distribution of a particular
        node.

        Parameters
        ----------
        node :
            Node whose CPDs will be retrieved.

        Returns
        -------
        pgmpy.factors.discrete.TabularCPD
            Conditional probability distribution of the node given by its
            TabularCPD.
        """
        try:
            return self.cpds[node]
        except KeyError:
            logger.error(f'{node} is not present in the distribution.')

    def from_parameters(self, parameters):
        """
        Sets the conditional probability distributions for the nodes.

        Parameters
        ----------
        parameters : dict[pgmpy.factors.discrete.TabularCPD]
            CPDs of the nodes.

        Returns
        -------
        self : JPD
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

    def to_serializable(self, **kwargs):
        """
        Retrieves a serializable dictionary with the parameters of the
        distribution. It converts `pgmpy.TabularCPD` objects to get this
        serializable representation.

        Returns
        -------
        dict
            Dictionary with serializable objects that represent the parameters
            of the distribution as values.
        """

        str_cpds = {}
        for node in self.order:
            cpd = self.cpds[node]
            values = cpd.get_values().tolist()
            evidence = cpd.get_evidence()
            variable = cpd.variable
            state_names = cpd.state_names
            card = int(cpd.get_cardinality([variable])[variable])
            evidence_dict = {k: int(v) for k, v in
                             cpd.get_cardinality(evidence).items()}
            str_cpds[node] = {'variable': variable, 'card': card,
                              'values': values, 'evidence_dict': evidence_dict,
                              'state_names': state_names}

        return str_cpds


def parse_discrete(parameters):
    """Converts dict representation of pgmpy.TabularCPD objects to TabularCPD
    objects."""

    cpds = {}

    for node, dict_cpd in parameters.items():
        evidence_dict = dict_cpd['evidence_dict']
        cpds[node] = TabularCPD(dict_cpd['variable'], dict_cpd['card'],
                                dict_cpd['values'],
                                evidence=list(evidence_dict.keys()),
                                evidence_card=list(evidence_dict.values()),
                                state_names=dict_cpd['state_names'])

    return cpds
