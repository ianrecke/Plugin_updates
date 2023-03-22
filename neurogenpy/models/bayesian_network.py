"""
Bayesian network module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import inspect
import logging
from operator import itemgetter

import networkx
import numpy as np
from community import best_partition
from networkx.algorithms.centrality import betweenness
from sklearn.metrics import roc_auc_score, average_precision_score

from ..distributions import JPD, GaussianJPD, DiscreteJPD
from ..io.adjacency_matrix import AdjacencyMatrix
from ..io.bif import BIF
from ..io.gexf import GEXF
from ..io.json import JSON as JSON_BN
from ..parameters.discrete_be import DiscreteBE
from ..parameters.discrete_mle import DiscreteMLE
from ..parameters.gaussian_mle import GaussianMLE
from ..parameters.learn_parameters import LearnParameters
from ..score.base import confusion_matrix, accuracy, f1_score, mcc_score, \
    confusion_hubs
from ..structure.cl import CL
from ..structure.fast_iamb import FastIamb
from ..structure.fges import FGES
from ..structure.fges_merge import FGESMerge
from ..structure.genie3 import GENIE3
from ..structure.graphical_lasso import GraphicalLasso
from ..structure.grow_shrink import GrowShrink
from ..structure.hc_tabu import HcTabu
from ..structure.hill_climbing import HillClimbing
from ..structure.hiton_pc import HitonPC
from ..structure.iamb import Iamb
from ..structure.inter_iamb import InterIamb
from ..structure.learn_structure import LearnStructure
from ..structure.lr import Lr
from ..structure.mbc import MBC
from ..structure.mi_continuous import MiContinuous
from ..structure.mmhc import MMHC
from ..structure.mmpc import MMPC
from ..structure.naive_bayes import NB
from ..structure.pc import PC
from ..structure.pearson import Pearson
from ..structure.sparsebn import SparseBn
from ..structure.tan import Tan
from ..util.data_structures import get_data_type

logger = logging.getLogger(__name__)

_DISTRIBUTIONS = {'discrete': DiscreteJPD,
                  'continuous': GaussianJPD}


class BayesianNetwork:
    """
    Bayesian network class. 

    Parameters
    ----------
    graph : networkx.DiGraph, optional
        Graph structure of the network.

    parameters : dict, optional
        Parameters of the nodes in the graph structure (variables). The nodes
        IDs are the keys and the values depend on the `data_type`.

            - Continuous case: the value of a node is a dictionary with
                unconditional mean, conditional variance, parents and parents
                coefficients as explained in GaussianMLE.

            - Discrete case: the value of a node is a
                :class:`~pgmpy.TabularCPD` object.

        If `graph` is not set, this argument is ignored.

    joint_dist : JPD, optional
        Joint probability distribution of the variables in the network. It must
        have two keys: 'distribution' and 'nodes_order'. If `parameters` is set
        or `graph` is not set, this argument is ignored.

    data_type : {'discrete', 'continuous'}, optional
        The type of data we are dealing with.

    Examples
    --------
    If you already have a graph structure and the network parameters (or
    joint probability distribution) in the right formats, it is posible
    to use the constructor for building the network. See
    :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.fit` and
    :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.load`
    methods for other ways of creating Bayesian networks.

    >>> from neurogenpy import BayesianNetwork
    >>> from networkx import DiGraph
    >>> graph = DiGraph()
    >>> graph.add_nodes_from([1, 2])
    >>> graph.add_edges_from([(1, 2)])
    >>> ps = {1: {0, 1, [], []} 2: {0, 1, [1], [0.8]}}
    >>> bn = BayesianNetwork(graph=graph, parameters=ps)
    """

    def __init__(self, *, graph=None, parameters=None, joint_dist=None,
                 data_type=None):

        self.graph = graph
        self.num_nodes = 0 if self.graph is None else len(self.graph)
        self.data_type = data_type

        if self.graph is not None:
            logger.info(graph)
            self._check_parameters(parameters)
            self.parameters = parameters

            if parameters is not None and self.data_type:
                # Topological order is only important in the continuous case.
                # In the discrete one, it just represents the IDs of the nodes.
                logger.info(parameters)
                logger.info(self.data_type)
                self.joint_dist = _DISTRIBUTIONS[self.data_type](
                    self._topological_order()).from_parameters(self.parameters)

            elif self._check_dist(joint_dist):
                self.joint_dist = joint_dist

        elif self.data_type:
            logger.info(self.data_type)
            self.joint_dist = _DISTRIBUTIONS[self.data_type]([])

        self.evidence = {}

    def _topological_order(self):
        """
        Sets a topological order between nodes, i.e., if there is an edge
        u -> v between two nodes, u appears before v in the ordering.
        """
        try:
            return list(networkx.topological_sort(self.graph))
        except networkx.NetworkXUnfeasible:
            return list(self.graph.nodes())
        except AttributeError:
            return []

    def _important_nodes_degrees(self, min_degree, full_list):
        important = {node: adjacencies for node in list(self.graph.nodes()) if
                     (adjacencies := len(
                         self.adjacencies(node))) >= min_degree}
        if not full_list:
            return {k: len(v) for k, v in important}
        else:
            return important

    def _reachable(self, start, observed, end=None):
        """
        Algorithm for finding nodes reachable from `start` given `observed` via
        active trails described in :code:`koller`. If `end` is provided, this
        function returns whether `start` and `end` are d-separated given
        `observed`.

        Parameters
        ----------
        start : list

        observed : list

        end : list, optional

        Returns
        -------
        list
            All the reachable nodes.
        """

        # Phase I: Insert ancestors of observations on a list
        visit_nodes = observed.copy()
        obs_ancestors = set()

        while visit_nodes:
            node = visit_nodes.pop()
            if node not in obs_ancestors:
                visit_nodes += self.parents(node)
                obs_ancestors.add(node)

        # Phase II: traverse active trails starting from X
        via_nodes = [(node, 'up') for node in start]
        visited = set()
        result = set() if end is None else True

        while via_nodes:
            node, direction = via_nodes.pop()
            if (node, direction) not in visited:
                visited.add((node, direction))

                if node not in observed:
                    if end is not None:
                        if node in end:
                            return False
                    else:
                        result.add(node)

                if direction == 'up' and node not in observed:
                    for parent in self.parents(node):
                        via_nodes.append((parent, 'up'))
                    for child in self.children(node):
                        via_nodes.append((child, 'down'))
                elif direction == 'down':
                    if node not in observed:
                        for child in self.children(node):
                            via_nodes.append((child, 'down'))
                    if node in obs_ancestors:
                        for parent in self.parents(node):
                            via_nodes.append((parent, 'up'))
        print("fin de comprobación del camino")
        return result

    def _check_graph(self):
        if not isinstance(self.graph, networkx.DiGraph):
            raise Exception('There is no network structure.')

    def _check_parameters(self, parameters):
        if parameters is not None and set(parameters.keys()) != set(
                self.graph.nodes()):
            raise ValueError(
                'Parameters and graph structure must have the same nodes.')

    def _check_dist(self, dist):
        if dist is not None and (not isinstance(
                dist, JPD) or dist.size() != self.num_nodes or
                                 set(dist.order()) != set(self.graph.nodes())):
            raise ValueError(
                'Wrong Joint Probability Distribution.')
        return dist is not None

    def _check_params_fitted(self):
        if not self.joint_dist.is_set():
            raise Exception('Network parameters are not set.')

    def _check_is_fitted(self):
        """Checks if the Bayesian network is fitted."""
        self._check_graph()
        self._check_params_fitted()

    def _check_node(self, node):
        if node not in self.graph:
            raise ValueError(f'Node {node} is not part of the network.')

    def _check_nodes_warn(self, nodes):
        wrong_nodes = [k for k in nodes if k not in self.graph]
        if wrong_nodes:
            logger.warning(f'Nodes {*wrong_nodes,} are not in the network '
                           'and they have been ignored.')

        return wrong_nodes

    def nodes(self):
        """
        Retrieves the nodes that form the Bayesian network.

        Returns
        -------
        list
            Nodes of the network.
        """

        return list(self.graph.nodes())

    def merge_graph(self, new_graph):
        """
        Adds each edge present in another graph if it is not already present in
        the network.

        Parameters
        ----------
        new_graph : networkx.DiGraph
            The graph to add the edges from.
        """
        self._check_graph()

        for (x, y) in new_graph.edges():
            if not self.graph.has_edge(x, y):
                self.graph.add_edge(x, y)
            edge_data = self.graph.get_edge_data(x, y)
            if 'weight' not in edge_data:
                edge_data['weight'] = 1

    def get_stats(self):
        """
        Returns the number of nodes and edges of the Bayesian network.

        Returns
        -------
        (int, int)
            The number of nodes and the number of edges in the network.
        """
        self._check_graph()

        return self.num_nodes, self.graph.number_of_edges()

    def relabel_nodes(self, mapping):
        """
        Modifies the names of the nodes in the graph according to the names
        received.

        Parameters
        ----------
        mapping : dict
            The new names to be set.
        """
        self._check_is_fitted()
        wrong_nodes = self._check_nodes_warn(list(mapping.keys()))
        for node in wrong_nodes:
            mapping.pop(node)

        self.graph = networkx.relabel_nodes(self.graph, mapping)
        self.joint_dist.relabel_nodes(mapping)

    def markov_blanket(self, node):
        """
        Returns the Markov blanket of a node. Under faithfulness, the Markov
        blanket of a node is formed by its parents, children and the parents of
        its children.

        Parameters
        ----------
        node : int, optional
            ID of the node whose Markov blanket has to be retrieved.

        Returns
        -------
            The IDs of the nodes that form the chosen node's Markov blanket.
        """

        self._check_graph()
        self._check_node(node)

        markov_blanket = [node]
        parents = self.parents(node)
        children = self.children(node)
        markov_blanket += parents + children

        for child in children:
            child_parents = self.graph.predecessors(child)
            for child_parent in child_parents:
                if child_parent not in markov_blanket:
                    markov_blanket.append(child_parent)

        return markov_blanket

    def parents(self, node):
        """
        Returns the parents of a node.

        Parameters
        ----------
        node : int
            ID of the node whose parents have to be retrieved.

        Returns
        -------
            The IDs of the chosen node's parents.
        """
        self._check_node(node)
        return list(self.graph.predecessors(node))

    def children(self, node):
        """
        Returns the children of a node.

        Parameters
        ----------
        node: int
            ID of the node whose children have to be retrieved.


        Returns
        -------
            The IDs of the chosen node's children.
        """
        self._check_node(node)
        return list(self.graph.successors(node))

    def adjacencies(self, node):
        """
        Returns the direct neighbors (parents and children) of a node.

        Parameters
        ----------
        node: int
            ID of the node whose direct neighbors have to be retrieved.

        Returns
        -------
        Iterable, Sized
            The IDs of the chosen node's direct neighbors.
        """

        self._check_node(node)
        return self.parents(node) + self.children(node)

    def important_nodes(self, criteria='degrees', min_degree=0,
                        full_list=False):
        """
        Retrieves the set of important nodes according to some criteria.

        Parameters
        ----------
        criteria : {'degrees', 'betweenness-centrality'}, default='degrees'
            The method used to get the set of important nodes.

        min_degree : int, default=0
            For `degrees` criteria, it only takes into account the nodes with
            a degree greater or equal than `min_degree`.

        full_list : bool, default=False
            For `degrees` criteria, whether the full list of adjacent nodes
            of every important node has to be retrieved or not.

        Returns
        -------
        dict
            Important nodes and their values according to the selected
            `criteria`.

        Raises
        ------
        ValueError
            If `criteria` is not supported.
        """

        self._check_graph()

        if criteria == 'degrees':
            return self._important_nodes_degrees(min_degree, full_list)
        elif criteria == 'betweenness-centrality':
            nodes_importance = betweenness.betweenness_centrality(self.graph)

            return {k: v for k, v in nodes_importance if v > 0}
        else:
            raise ValueError(f'{criteria} criteria is not supported.')

    def communities(self, method='louvain'):
        """
        Retrieves the nodes that belong to a community with more than one node.

        Parameters
        ----------
        method : str, default='louvain'
            The method used to calculate the communities.

        Returns
        -------
        dict
            The nodes as keys and the community each one belongs to as values.

        Raises
        ------
        ValueError
            If `method` is not supported.
        """

        self._check_graph()

        if method == 'louvain':
            undirected_graph = self.graph.to_undirected()
            return best_partition(undirected_graph)
        else:
            raise ValueError(f'{method} method is not supported.')

    def evidence(self):
        """
        Returns the known evidence.

        Returns
        -------
        dict
            Known evidence elements.
        """

        return self.evidence

    def condition(self):
        """
        Retrieves the marginal distribution for each unobserved node in the
        network given the evidence already set, i.e., the distributions of the
        form f(U|E=e) where 'U' represents unobserved variables and 'E=e', the
        evidence.

        Returns
        -------
        dict
            CPDs of the unobserved nodes.
        """

        kwargs = {'graph': self.graph} if self.data_type == 'discrete' else {}

        return self.joint_dist.condition(self.evidence, **kwargs)

    def set_evidence(self, nodes_values, evidence_scale='scalar'):
        """
        Sets some evidence for some nodes (variables) in the Bayesian network.

        Parameters
        ----------
        nodes_values : dict
            Nodes and values to be set.

        evidence_scale : {'scalar', 'num_std_deviations'}, default='scalar'
            How the evidence is provided. Available options are the number of
            standard deviations or the final value for the node.

        Raises
        ------
        ValueError
            If `evidence_scale` is not supported.
        """

        self._check_is_fitted()
        wrong = self._check_nodes_warn(list(nodes_values.keys()))

        for k in wrong:
            nodes_values.pop(k)

        if self.data_type == 'continuous' and \
                evidence_scale == 'num_std_deviations':
            for node, num_devs in nodes_values.keys():
                mean, std_dev = self.joint_dist.marginal(nodes=[node],
                                                         initial=True)

                evidence_value_node = mean + std_dev * num_devs
                self.evidence[node] = evidence_value_node
        elif self.data_type == 'discrete' or evidence_scale == 'scalar':
            self.evidence = {**self.evidence, **nodes_values}
        else:
            raise ValueError(
                f'Evidence scale \'{evidence_scale}\' is not supported.')

    def clear_evidence(self, nodes=None):
        """
        Clears the evidence previously set.

        Parameters
        ----------
        nodes : optional
            If set, it determines for which nodes the evidence has to be
            cleared.
        """

        self._check_is_fitted()

        if nodes is None:
            self.evidence.clear()
        else:
            wrong_nodes = self._check_nodes_warn(nodes)

            nodes = [node for node in nodes if node not in wrong_nodes]

            for node in nodes:
                try:
                    self.evidence.pop(node)
                except KeyError:
                    pass

    def has_evidence(self, node):
        """
        Checks if there is evidence about a node.

        Parameters
        ----------
        node
            ID of the node to be checked.

        Returns
        -------
        bool
            Whether there is evidence about the selected node or not.
        """
        self._check_node(node)
        return node in self.evidence.keys()

    # TODO: Check weights assignment for each algorithm and networkx.
    def sum_weights(self):
        """
        Retrieves the total sum of the edges in the network. If there is no
        weight for an edge, it considers the weight is 1.

        Returns
        -------
        float
            Sum of the edges in the network.
        """

        self._check_graph()

        all_weights = []
        for (x, y, edge_data) in self.graph.edges(data=True):
            if 'weight' not in edge_data:
                edge_data['weight'] = 1
            all_weights.append(edge_data['weight'])

        return np.sum(np.array(all_weights, dtype=np.float64))

    def edges_info(self, node):
        """
        Retrieves information about the edges in which a node participates.

        Parameters
        ----------
        node :
            ID of the node to be explored.

        Returns
        -------
        tuple
            Number of parents and children, top 3 parents (by `weight`) and top
            3 children (by `weight`).
        """
        parents = self.parents(node)
        children = self.children(node)

        nump, numc = len(parents), len(children)

        top_parents = [self.graph.get_edge_data(parent, node)['weight'] for
                       parent in parents]

        top_parents.sort(key=itemgetter(1), reverse=True)
        top_parents = top_parents[:min(3, nump)]

        top_children = [self.graph.get_edge_data(node, child)['weight'] for
                        child in children]

        top_children.sort(key=itemgetter(1), reverse=True)
        top_children = top_children[:min(3, numc)]

        return nump, numc, top_parents, top_children

    def edge_weight(self, source, target):
        """
        Retrieves the weight of an edge in the network. If there is no edge
        between `source` and `target`, the result is -1.

        Parameters
        ----------
        source :
            Source node ID.
        target:
            Target node ID.

        Returns
        -------
        float
            The weight for the desired edge.
        """
        self._check_node(source)
        self._check_node(target)
        if self.graph.has_edge(source, target):
            return self.graph.get_edge_data(source, target)['weight']
        else:
            return -1

    def get_evidence_nodes(self):
        """
        Retrieves the nodes for which there is evidence.

        Returns
        -------
        list
            The set of evidence nodes.
        """
        return list(self.evidence.keys())

    def get_cpds(self, nodes):
        """
        Returns the Conditional Probability Distributions of the desired set
        of nodes.

        Parameters
        ----------
        nodes : list
            A set of nodes in the network.

        Returns
        -------
        dict
            CPDs of the nodes.
        """

        self._check_params_fitted()
        wrong_nodes = self._check_nodes_warn(nodes)

        nodes = [node for node in nodes if node not in wrong_nodes]

        return {node: self.joint_dist.get_cpd(node, graph=self.graph) for node
                in nodes}

    def marginal(self, nodes):
        """
        Retrieves the initial marginal probability distribution of each node in
        the provided set.

        Parameters
        ----------
        nodes : list
            Set of nodes.

        Returns
        -------
        dict
            Marginal distribution of each node.
        """

        self._check_params_fitted()
        wrong_nodes = self._check_nodes_warn(nodes)

        nodes = [node for node in nodes if node not in wrong_nodes]

        return {node: self.joint_dist.marginal(node) for node in nodes}

    def all_marginals(self):
        """
        Retrieves the initial marginal probability distributions of each node
        in the network.

        Returns
        -------
        dict
            Initial marginal distributions of the nodes.
        """

        self._check_params_fitted()

        return self.joint_dist.all_marginals()

    def is_dseparated(self, start, end, observed):
        """
        Checks if two sets of nodes (`start` and `end`) are D-separated given
        another one (`observed`). It follows the algorithm proposed in
        :cite:`koller`, page 75.

        Parameters
        ----------
        start : list

        observed: list

        end: list

        Returns
        -------
        bool
            Whether `start` and `end` are D-separated by `observed` or not.

        """

        start = [node for node in start if
                 node not in self._check_nodes_warn(start)]
        end = [node for node in end if
               node not in self._check_nodes_warn(end)]
        observed = [node for node in observed if
                    node not in self._check_nodes_warn(observed)]

        return self._reachable(start, observed, end)

    def reachable_nodes(self, start):
        """
        Returns the reachable nodes from `start` given the current evidence via
        active trails. It follows the algorithm proposed in
        :cite:`koller`, page 75.

        Parameters
        ----------
        start : list
            Set of nodes for which the reachable nodes have to be retrieved.

        Returns
        -------
        list
            All the reachable nodes.
        """

        self._check_nodes_warn(start)

        return self._reachable(start, list(self.evidence.keys()))

    def subgraph_edges(self, nodes):
        """
        Retrieves all the edges in the subgraph formed by a set of nodes.

        Parameters
        ----------
        nodes : list
            Set of nodes that form the subgraph.

        Returns
        -------
        list
            Set of edges in the subgraph.
        """
        nodes = [node for node in nodes if
                 node not in self._check_nodes_warn(nodes)]
        return [(x, y) for y in nodes for x in nodes if y in self.children(x)]

    def filter_edges(self, min_weight, max_weight):
        """
        Retrieves the graph after removing all the edges that have weights out
        of some range.

        Parameters
        ----------
        min_weight : float
            Minimum value allowed for an edge weight.

        max_weight : float
            Maximum value allowed for an edge weight.

        Returns
        -------
        networkx.DiGraph
            The graph with the edges that satisfy the range condition.
        """

        graph = self.graph.copy()

        for (x, y, edge_data) in self.graph.edges(data=True):
            edge_weight = edge_data['weight']
            if edge_weight < min_weight or edge_weight > max_weight:
                graph.remove_edge(x, y)
        return graph

    def fit(self, df=None, data_type='continuous', estimation=None,
            algorithm='FGESMerge', skip_structure=False, **kwargs):
        """
        Builds a Bayesian network using the input data.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Data set used to learn the network structure and parameters.

        data_type : {'continuous, 'discrete'}, default='continuous
            Type of data in `df`.

        estimation : str or LearnParameters, default='mle'
            Estimation type to be used for learning the parameters of the
            network.
            Supported estimation approaches are:

                - Discrete Bayesian estimation
                - Discrete maximum likelihood estimation
                - Gaussian maximum likelihood estimation

        algorithm : str or LearnStructure, default='FGESMerge'
            Algorithm to be used for learning the structure of the network.
            Supported structure learning algorithms are:

                1. Statistical based:
                    - Pearson Correlation ('pearson_correlation')
                    - Mutual information ('mutual_information')
                    - Linear regression ('lr')
                    - Graphical lasso ('graphical_lasso')
                    - GENIE3 ('genie3')
                2. Constraint based:
                    - PC ('pc')
                    - Grow shrink ('grow_shrink')
                    - iamb ('iamb')
                    - Fast.iamb ('fast_iamb')
                    - Inter.iamb ('inter_iamb')
                3. Score and search:
                    - Hill climbing ('hc')
                    - Hill climbing with tabu search ('hc_tabu')
                    - Chow-Liu tree ('cl')
                    - Hiton Parents and Children ('hiton_pc')
                    - sparsebn ('sparsebn')
                    - FGES ('fges')
                    - FGES-Merge ('fges_merge')
                4. Hybrid:
                    - MMHC ('mmhc')
                    - MMPC ('mmpc')
                5. Tree structure:
                    - Naive Bayes ('nb')
                    - Tree augmented Naive Bayes ('tan')
                6. Multidimensional Bayesian network classifier ('mbc')

        skip_structure : bool, default=False
            Whether to skip the structure learning step. If it is set to
            `True`, a graph structure should have been previously provided.

        **kwargs :

            Keyword arguments are used to set extra arguments for structure
            learning algorithms and parameters estimation methods. They are
            only taken into account if used with the appropriate algorithm or
            estimation method. Valid keyword arguments are:

            prior : {'BDeu', 'K2'}, default='BDeu'
                Prior distribution type used for Discrete Bayesian parameter
                estimation.

            equivalent_size : int or dict
                In Discrete Bayesian parameter estimation, if
                `prior` is 'BDeu', `equivalent_size` must be specified. See
                pgmpy documentation for more information.

            alpha : float, default=0.05
                The target nominal type I error rate for some algorithms and
                the regularization parameter in the Graphical lasso case.

            penalty : int, default=45
                Penalty hyperparameter for the FGES and FGES-Merge structure
                learning methods.

            tol : float, default=1e-4
                The tolerance to declare convergence in the Graphical lasso
                case.

            max_iter : int, default=100
                The maximum number of iterations for some structure learning
                algorithms.

            maxp : int, default=100
                The maximum number of parents for a node in some structure
                learning algorithms.

            class_variable : str
                Name of the class variable for unidimensional classifiers.

            class_variables : list
                Names of the class variables for multidimensional classifiers.

            n_jobs : int, default=1
                Number of threads used for running an algorithm. It is only
                available for GENIE3, FGES and FGES-Merge.

        Returns
        -------
        self : BayesianNetwork
            Fitted Bayesian Network.

        Raises
        ------
        Exception:
            If there is no network structure when parameter learning step is
            going to be performed.
        ValueError:
            If the structure learning algorithm or parameter estimation method
            is not supported.

        Examples
        --------
        Learning the structure and parameters of a Bayesian network from the
        data in a CSV file.

        - Set the structure and parameter learning methods with arguments:

        >>> import pandas as pd
        >>> from neurogenpy import BayesianNetwork
        >>> data = pd.read_csv('file.csv')
        >>> bn = BayesianNetwork().fit(data, estimation='mle', algorithm='PC')

        Additional parameters for the structure learning or parameters
        estimation algorithm can be provided too:

        >>> bn = BayesianNetwork()
        >>> bn = bn.fit(df, algorithm='FGESMerge', penalty=45)

        - Instance a particular
            :class:`~neurogenpy.structure.learn_structure.LearnStructure`
            or
            :class:`~neurogenpy.parameters.learn_parameters.LearnParameters`
            subclass:

        >>> from neurogenpy import BayesianNetwork, FGESMerge, GaussianMLE

        """

        self.data_type = data_type if data_type else get_data_type(df)

        type_conv = 'category' if self.data_type == 'discrete' else 'float'
        df = df.apply(lambda x: x.astype(type_conv))

        estimators = {'bayesian_discrete': DiscreteBE,
                      'mle_discrete': DiscreteMLE,
                      'mle_continuous': GaussianMLE}
        algorithms = {'cl': CL, 'fast_iamb': FastIamb, 'fges': FGES,
                      'fges_merge': FGESMerge, 'genie3': GENIE3,
                      'graphical_lasso': GraphicalLasso,
                      'grow_shrink': GrowShrink, 'hc': HillClimbing,
                      'hc_tabu': HcTabu,
                      'hiton_pc': HitonPC, 'iamb': Iamb,
                      'inter_iamb': InterIamb, 'Lr': Lr, 'mbc': MBC,
                      'mutual_information': MiContinuous, 'mmhc': MMHC,
                      'mmpc': MMPC, 'nb': NB, 'pc': PC,
                      'pearson_correlation': Pearson, 'sparsebn': SparseBn,
                      'tan': Tan}

        if not skip_structure:
            if isinstance(algorithm, LearnStructure):
                self.graph = algorithm.run()

            if algorithm in algorithms.keys():
                alg = algorithms[algorithm]
                params = {k: v for k, v in kwargs.items() if
                          k in inspect.getfullargspec(alg).kwonlyargs}
                self.graph = alg(df, self.data_type, **params).run()

            else:
                raise ValueError('Structure learning is only available for the'
                                 f' following methods: {*algorithms.keys(),}.')
            logger.info(f'Learnt graph structure: {self.graph}')
            self.num_nodes = len(self.graph)

        if self.graph is None:
            raise Exception('The Bayesian Network does not have a structure.')
        elif estimation is None:
            pass
        elif isinstance(estimation, LearnParameters):
            self.parameters = estimation.run()
        else:
            try:
                est = estimators[f'{estimation}_{self.data_type}']
                est_params = {k: v for k, v in kwargs.items() if
                              k in inspect.getfullargspec(est).kwonlyargs}
                self.parameters = est(df, graph=self.graph, **est_params).run()

                self.joint_dist = _DISTRIBUTIONS[data_type](
                    self._topological_order()).from_parameters(self.parameters)
            except KeyError:
                raise ValueError('Parameter learning is only available for the'
                                 f' following methods: {*estimators.keys(),}.')
        logger.info(f'Learned parameters: {self.parameters}')

        return self

    def save(self, file_path='bn.gexf', **kwargs):
        """
        Exports the Bayesian network in a specific format.

        Parameters
        ----------
        file_path : str, default='bn.gexf'
            Path for the generated output file. Supported file extensions are:
                1. '.gexf'
                2. '.csv'
                3. '.gzip'
                4. '.bif'
                5. '.json'

        **kwargs :
            Additional arguments for the layout. If the extension is no '.gexf'
            they are not considered. Valid arguments are:

            layout_name : str, optional
                Layout used for calculating the positions of the nodes in the
                graph.

                - 'circular'
                - 'Dot'
                - 'ForceAtlas2'
                - 'Grid'
                - 'FruchtermanReingold'
                - 'Image'
                - 'Sugiyama'

            communities : bool, default=False
                Whether to assign different colors to the nodes and edges
                belonging to different communities of Louvain.

            sizes_method : {'mb', 'neighbors'}, default='mb'
                The method used to calculate the sizes of the nodes. It can be
                the size of the Markov blanket of each node or the amount of
                neighbors they have.

        Raises
        ------
        ValueError
            If the extension of the file is not supported.

        Examples
        --------
        Saving a model. The available formats are:
            - BIF (pgmpy), which stores the full network.
            - GEXF, for saving a visual representation of the graph structure.
                In this case, it is posible to set the desired layout for the
                graph.
            - CSV and parquet, for saving the adjacency matrix of the graph.
            - JSON, for saving a JSON representation of the graph structure.

        >>> from neurogenpy import BayesianNetwork
        >>> import pandas as pd
        >>> df = pd.read_csv('file.csv')
        >>> bn = BayesianNetwork.fit(df)
        >>> bn.save('bn.gexf', layout_name='circular')

        """

        file_path = file_path.lower()

        if file_path.endswith('.gexf'):
            GEXF(self).write_file(file_path, **kwargs)
        elif file_path.endswith('.gzip') or file_path.endswith('.csv'):
            AdjacencyMatrix(self).write_file(file_path)
        elif file_path.endswith('.bif'):
            BIF(self).write_file(file_path)
        elif file_path.endswith('.json'):
            JSON_BN(self).write_file(file_path)
        else:
            raise ValueError('File extension not supported.')

    def load(self, file_path):
        """
        Loads the Bayesian network stored in a file. It always loads the
        network structure. For some types of files, it also loads its
        parameters

        Parameters
        ----------
        file_path : str
            Path for the input file. Supported file extensions are:
                1. '.gexf'
                2. '.csv'
                3. '.gzip'
                4. '.bif'
                5. '.json'

        Returns
        -------
        self : BayesianNetwork
            Loaded Bayesian network.

        Raises
        ------
        ValueError
            If the extension of the file is not supported.

        Examples
        --------
        Loading an already saved Bayesian network:

        - BIF file (pgmpy): it loads the graph structure and the parameters of
            the network.

        >>> from neurogenpy import BayesianNetwork
        >>> bn = BayesianNetwork().load('bn.bif')

        - GEXF (graph stored with .gexf extension), CSV (adjacency matrix
            stored with '.csv') or parquet (adjacency matrix stored with
            '.gzip' extension) file, it only loads the graph structure of the
            network. The parameters can be learnt according to this graph and
            the initial data.

        >>> import pandas as pd
        >>> from neurogenpy import BayesianNetwork
        >>> bn = BayesianNetwork.load('bn.gexf')
        >>> df = pd.read_csv('file.csv')
        >>> bn = bn.fit(df, estimation='mle', skip_structure=True)
        """

        file_path = file_path.lower()

        if file_path.endswith('.gexf'):
            self.graph = GEXF().read_file(file_path)
        elif file_path.endswith('.gzip') or file_path.endswith('csv'):
            self.graph = AdjacencyMatrix().read_file(file_path)
        elif file_path.endswith('.bif'):
            self.graph, self.parameters = BIF().read_file(file_path)
            self.data_type = 'discrete'
            self.joint_dist = DiscreteJPD(
                self.graph.nodes()).from_parameters(self.parameters)
        elif file_path.endswith('.json'):
            self.graph, self.parameters, self.data_type = JSON_BN().read_file(
                file_path)
            if self.data_type == 'continuous':
                self.joint_dist = GaussianJPD(
                    self._topological_order()).from_parameters(self.parameters)
            elif self.data_type == 'discrete':
                self.joint_dist = DiscreteJPD(
                    self.graph.nodes()).from_parameters(self.parameters)
            else:
                raise ValueError('Data type not supported.')
        else:
            raise ValueError('File extension not supported.')
        self.num_nodes = len(self.graph)

        return self

    def compare(self, real_graph, nodes_order=None, metric='all', *,
                undirected=False, threshold=0, h_method='out_degree',
                hubs_threshold=2):
        """
        Compares the Bayesian network graph structure with another one provided
        using some performance measures.

        Parameters
        ----------
        real_graph : numpy.array, optional
            Adjacency matrix used to compare with the Bayesian network graph
            structure.

        nodes_order : list, optional
            Order of the nodes in `real_graph`. If not provided, the order
            determined by self.graph.nodes() is considered.

        metric : str or list, default='all'
            Metric used to compare the Bayesian network graph structure with
            the actual graph. If one or all values are wanted, it can be a
            string with the name of the metric or 'all'. In other case, a list
            of strings is needed. Possible values are:

                - 'confusion': confusion matrix.
                - 'accuracy'
                - 'f1'
                - 'roc_auc': area under the ROC curve.
                - 'mcc'
                - 'avg_precision'
                - 'confusion_hubs': confusion matrix for the hubs.

        undirected : bool, default=False
            Whether the confusion matrix is calculated after making the graph
            undirected.

        threshold :

        h_method : str, default='out_degree'
            Method used to calculate the hubs.

        hubs_threshold : default=2
            Threshold associated to `h_method` used to calculate the hubs.

        Returns
        -------
        dict
            A dictionary with the metrics names as keys and the value for each
            one as value.

        Raises
        ------
        ValueError
            If the introduced configuration is not valid.

        Examples
        --------

        >>> from numpy import array
        >>> from networkx import DiGraph
        >>> matrix = array([[0,0], [1,0]])
        >>> graph = DiGraph()
        >>> graph.add_nodes_from([1, 2])
        >>> graph.add_edges_from([(1, 2)])
        >>> parameters = {1: {0, 1, [], []}, 2: {0, 1, [1], [0.8]}}
        >>> bn = BayesianNetwork(graph=graph, parameters=parameters)
        >>> res = bn.compare(matrix, nodes_order=[1, 2], metric='all')
        >>> acc = res['accuracy']
        >>> print(f'Model accuracy: {acc}')
        >>> conf_matrix = res['confusion']
        >>> print(conf_matrix)

        """

        self._check_graph()

        available_metrics = {'confusion': confusion_matrix,
                             'confusion_hubs': confusion_hubs,
                             'accuracy': accuracy, 'f1': f1_score,
                             'roc_auc': roc_auc_score,
                             'avg_precision': average_precision_score,
                             'mcc': mcc_score}

        if isinstance(metric, str):
            metric = [metric] if metric != 'all' else list(
                available_metrics.keys())

        if real_graph.shape != (self.num_nodes, self.num_nodes):
            raise ValueError('\'real_graph\' does not have the right shape.')

        if nodes_order is not None and set(nodes_order) != set(
                list(self.graph.nodes())):
            raise ValueError('\'nodes_order\' and the nodes in the Bayesian '
                             'network are not the same.')

        adj_matrix = networkx.to_numpy_array(self.graph,
                                             nodelist=nodes_order).astype(int)

        if {'roc_auc', 'avg_precision'}.intersection(metric):
            flat_pred, flat_true = adj_matrix.flatten(), real_graph.flatten()
        else:
            flat_pred, flat_true = None, None

        available_params = {'m_pred': adj_matrix, 'm_true': real_graph,
                            'undirected': undirected, 'threshold': threshold,
                            'method': h_method, 'y_true': flat_true,
                            'y_score': flat_pred,
                            'hubs_threshold': hubs_threshold}

        if set(metric) - {'roc_auc', 'avg_precision', 'confusion_hubs'}:
            c = confusion_matrix(adj_matrix, real_graph, undirected=undirected,
                                 threshold=threshold)
            available_params['confusion'] = c

        try:
            metric.remove('confusion')
        except ValueError:
            result = {}
        else:
            result = {'confusion': available_params['confusion']}

        for m in metric:
            func = available_metrics[m]
            argspec = inspect.getfullargspec(func)
            params = {p: available_params[p] for p in
                      argspec.args + argspec.kwonlyargs if
                      p in available_params.keys()}
            result = {**result, m: func(**params)}

        return result
