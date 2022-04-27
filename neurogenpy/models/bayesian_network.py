"""
Bayesian network module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import warnings
from operator import itemgetter

import networkx
import networkx as nx
import numpy as np
from community import best_partition
from networkx.algorithms.centrality import betweenness
from sklearn.metrics import roc_auc_score, average_precision_score

from ..distributions.modifiable_joint import ModifiableJointDistribution
from ..io.adjacency_matrix import AdjacencyMatrix
from ..io.bif import BIF
from ..io.gexf import GEXF
from ..learn_parameters.discrete_be import DiscreteBE
from ..learn_parameters.discrete_mle import DiscreteMLE
from ..learn_parameters.gaussian_mle import GaussianNode, GaussianMLE
from ..utils.data_structures import get_data_type
from ..utils.score import confusion_matrix, accuracy, f1_score, mcc_score, \
    confusion_hubs


# TODO: Functions or attributes? Decide
# TODO: Manage hybrid case.


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

            - Continuous case: the value of a node is a GaussianNode object.

            - Discrete case: the value of a node is a pgmpy.TabularCPD object.

            - Hybrid case: mixture

        If `graph` is not set, this argument is ignored.

    joint_dist : dict, optional
        Joint probability distribution of the variables in the network. It must
        have two keys: 'distribution' and 'nodes_order'. The value for the
        first one must be a JointDistribution object and, for the second one a
        list of the nodes order used in the distribution.
        If `parameters` is set or `graph` is not set, this argument is ignored.

    data_type : {'discrete', 'continuous', 'hybrid'}, default='continuous'
        The type of data we are dealing with.
    """

    def __init__(self, *, graph=None, parameters=None, joint_dist=None,
                 data_type='continuous'):

        self.graph = graph
        self.num_nodes = 0 if self.graph is None else len(self.graph)

        self.data_type = data_type

        if self.graph is not None:
            self._check_parameters(parameters)
            self.parameters = parameters

            if self._check_dist(joint_dist):
                dist, order = (joint_dist['distribution'],
                               joint_dist['nodes_order'])
            else:
                dist, order = (None, None)

            self.joint_dist = ModifiableJointDistribution(
                initial=dist,
                save_dist=self.num_nodes > 300,
                data_type=self.data_type,
                nodes_order=order)
            if parameters is not None and self.data_type == 'continuous':
                self.joint_dist.from_params(
                    params=self.parameters, save_dist=self.num_nodes > 300,
                    data_type=self.data_type,
                    nodes_order=self._topological_order())

        self.evidence = {}
        self.cont_nodes = None

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

    def _get_random_cont_params(self):
        """
        Obtains some continuous random parameters for the current graph
        structure.

        Returns
        -------
        dict[GaussianNode]
            The parameters associated with each node.
        """
        model_parameters = {}
        for node in list(self.graph.nodes()):
            node_parents = list(self.graph.predecessors(node))

            mean = np.random.randint(-10, 10)
            variance = np.random.randint(1, 10)

            parents_coeffs = np.random.rand(len(node_parents))

            model_parameters[node] = GaussianNode(mean, variance, node_parents,
                                                  parents_coeffs)

        return model_parameters

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
        active trails described in [1]_. If `end` is provided, this function
        returns whether `start` and `end` are d-separated given `observed`.

        Parameters
        ----------
        start : list

        observed : list

        end : list, optional

        Returns
        -------
        list
            All the reachable nodes.

        References
        ----------
        .. [1] Koller, Daphne, and Nir Friedman. "Probabilistic graphical
           models: principles and techniques". MIT press, 2009, pp. 75.
        """
        # Phase I: Insert ancestors of observations on a list
        visit_nodes = observed.copy()
        obs_ancestors = set()

        while visit_nodes:
            node = visit_nodes.pop()
            if node not in obs_ancestors:
                visit_nodes += list(set(self.parents(node)))
                obs_ancestors.update(node)

        # Phase II: traverse active trails starting from X
        via_nodes = [(node, 'up') for node in start]
        visited = set()

        if end is None:
            result = set()
        else:
            result = True

        while via_nodes:
            node, direction = via_nodes.pop()
            if (node, direction) not in visited:
                visited.add((node, direction))

                if node not in observed:
                    if end is not None:
                        if node in end:
                            return False
                    else:
                        result.update(node)

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
        return result

    def _check_graph(self):
        if not isinstance(self.graph, nx.DiGraph):
            raise Exception('There is no network structure.')

    def _check_parameters(self, parameters):
        if parameters is not None and set(parameters.keys) != set(
                self.graph.nodes()):
            raise ValueError(
                'Parameters and graph structure must have the same nodes.')

    def _check_dist(self, dist):
        if self.parameters is None and dist is not None and (
                len(dist['distribution']) == self.num_nodes or set(
            dist['nodes_order']) != set(self.graph.nodes())):
            raise ValueError(
                'Joint distribution and graph must have the same nodes.')
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
            warnings.warn(f'Nodes {*wrong_nodes,} are not in the network '
                          'and they have been ignored.')

        return wrong_nodes

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
        Returns the Markov blanket of a node. The Markov blanket of a node is
        formed by its parents, children and the parents of its children.

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

        return self.parents(node) + self.children(node)

    # TODO: Check what to return with the important nodes.
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

        Raises
        ------
        ValueError
            If `criteria` is not supported.
        """

        self._check_graph()

        if criteria == 'degrees':
            return self._important_nodes_degrees(min_degree,
                                                 full_list)
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

    def save_dist(self, initial=False, path='joint_dist.npz'):
        """
        Saves the current joint probability distribution in a '.npz' file with
        the path provided.

        Parameters
        ----------
        initial : bool, default=False
            Whether the distribution to save is the initial (without evidence)
            or not.

        path : str, default='joint_dist.npz'
            The path of the file where the distribution has to be saved.
        """
        self.joint_dist.save(initial=initial, path=path)

    def get_evidence(self):
        """
        Returns the known evidence.

        Returns
        -------
        dict
            Known evidence elements.
        """
        return self.evidence

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

        if self.data_type == 'continuous':

            wrong = self._check_nodes_warn(list(nodes_values.keys()))

            for k in wrong:
                nodes_values.pop(k)

            if evidence_scale == 'scalar':
                self.evidence = {**self.evidence, **nodes_values}

            elif evidence_scale == 'num_std_deviations':
                for node, num_devs in nodes_values.keys():
                    mean, std_dev = self.joint_dist.marginal(nodes=[node],
                                                             initial=True)

                    evidence_value_node = mean + std_dev * num_devs
                    self.evidence[node] = evidence_value_node
            else:
                raise ValueError(
                    f'Evidence scale \'{evidence_scale}\' is not supported.')
            self.joint_dist.condition(self.evidence)
        else:
            raise Exception('Discrete and hybrid cases are not supported.')

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

        if self.data_type == 'continuous':
            if nodes is None:
                self.evidence.clear()
            else:
                wrong_nodes = self._check_nodes_warn(nodes)

                nodes -= wrong_nodes

                for node in nodes:
                    try:
                        self.evidence.pop(node)
                    except KeyError:
                        pass

            if not self.evidence:
                self.joint_dist.restart()
            else:
                self.joint_dist.condition(self.evidence)

        else:
            raise Exception('Discrete case not supported.')

    def marginal(self, nodes, initial=False):
        """
        Retrieves the marginal distribution for a set of nodes.

        Parameters
        ----------
        nodes : list
            The set of nodes for which the marginal distribution has to be
            retrieved.

        initial : bool, default=False
            Whether the distribution used for marginalization is the initial
            one (without evidence) or the current one.

        Returns
        -------
            The desired marginal distribution.
        """
        self._check_params_fitted()
        wrong_nodes = self._check_nodes_warn(nodes)

        nodes -= wrong_nodes
        return self.joint_dist.marginal(nodes, initial=initial)

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
            The set of evidence nodes.
        """
        return self.evidence.keys()

    def is_dseparated(self, start, end, observed):
        """
        Checks if two sets of nodes (`start` and `end`) are D-separated given
        another one (`observed`).

        Parameters
        ----------
        start : list

        observed: list

        end: list

        Returns
        -------
        bool
            Whether `start` and `end` are D-separated by `observed` or not.

        References
        ----------
        .. [1] Koller, Daphne, and Nir Friedman. "Probabilistic graphical
           models: principles and techniques". MIT press, 2009, pp. 75.
        """

        start -= self._check_nodes_warn(start)
        end -= self._check_nodes_warn(end)
        observed -= self._check_nodes_warn(observed)

        return self._reachable(start, observed, end)

    def reachable_nodes(self, start):
        """
        Returns the reachable nodes from `start_nodes` given the current
        evidence via active trails.

        Parameters
        ----------
        start : list
            Set of nodes for which the reachable nodes have to be retrieved.

        Returns
        -------
        list
            All the reachable nodes.

        References
        ----------
        .. [1] Koller, Daphne, and Nir Friedman. "Probabilistic graphical
           models: principles and techniques". MIT press, 2009, pp. 75.
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
        nodes -= self._check_nodes_warn(nodes)
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

    def fit(self, df, estimation='mle', algorithm='FGESMerge',
            skip_structure=False, **kwargs):
        """
        Builds a Bayesian network using the input data.

        Parameters
        ----------
        df : pandas.DataFrame
            Data set used to learn the network structure and parameters.

        estimation : {'bayesian', 'mle', 'random'} default='mle'
            Estimation type to be used for learning the parameters of the
            network.
            Supported estimation approaches are:

                - Discrete Bayesian estimation
                - Discrete maximum likelihood estimation
                - Gaussian maximum likelihood estimation
                - Random parameters for the continuous case.

        algorithm : str, default='FGESMerge'
            Algorithm to be used for learning the structure of the network.
            Supported structure learning algorithms are:

                1. Statistical based:
                    - Pearson Correlation ('PC')
                    - Mutual information ('MiContinuous')
                    - Linear regression ('Lr')
                    - Graphical lasso ('Glasso')
                    - GENIE3 ('GENIE3')
                2. Constraint based:
                    - PC ('PC')
                    - Grow shrink ('Gs')
                    - iamb ('Iamb')
                    - Fast.iamb ('FastIamb')
                    - Inter.iamb ('InterIamb')
                3. Score and search:
                    - Hill climbing ('Hc')
                    - Hill climbing with tabu search ('HcTabu')
                    - Chow-Liu tree ('CL')
                    - Hiton Parents and Children ('HitonPC')
                    - sparsebn ('SparseBn')
                    - FGES ('FGES')
                    - FGES-Merge ('FGES-Merge')
                4. Hybrid:
                    - MMHC ('MMHC')
                    - MMPC ('MMPC')
                5. Tree structure:
                    - Naive Bayes ('NB')
                    - Tree augmented Naive Bayes ('Tan')
                6. Multidimensional Bayesian network classifier ('MBC')

        skip_structure : bool, default=False
            Whether to skip the structure learning step. If it is set to
            `True`, a graph structure should have been previously provided.

        **kwargs :

            Valid keyword arguments are:

            prior : {'BDeu', 'K2'}, default='BDeu'
                Prior distribution type used for Discrete Bayesian parameter
                estimation. It is not taken into account if `algorithm` is not
                'FGESMerge'.

            equivalent_size :

            alpha : float, default=0.05

            penalty : int, default=45
                Penalty hyperparameter for the FGES and FGES-Merge structure
                learning methods. It is not taken into account if `algorithm`
                is not 'FGES' or 'FGESMerge'.

            tol :

            max_iter :

            maxp :

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
        """
        # discrete_cols :
        #     Numerical variables in the data set that are discrete. Any
        #     numerical variable not explicitly specified is considered
        #     continuous.

        # TODO: Provide options: estimator and algorithm as objects or strings.
        # TODO: Check df. Rethink data structure (numpy array, pandas
        #  DataFrame or what?)

        # if discrete_cols is not None:
        #     df[discrete_cols] = df[discrete_cols].apply(
        #         lambda col: pd.Categorical(col))
        self.data_type, self.cont_nodes = get_data_type(df)

        # TODO: Check parameters and algorithm are OK in each case.
        # TODO: Check parameters are complete and OK for bnlearn.
        pl_options = ['prior', 'equivalent_size']
        sl_options = ['alpha', 'penalty', 'mode', 'tol', 'max_iter', 'maxp']

        pl_kwargs = {k: v for k, v in kwargs.items() if k in pl_options}
        sl_kwargs = {k: v for k, v in kwargs.items() if k in sl_options}

        estimators = ['bayesian', 'mle']
        algorithms = ['CL', 'FastIamb', 'FGES', 'FGESMerge', 'Genie', 'Glasso',
                      'Gs', 'Hc', 'HcTabu', 'HitonPC', 'Iamb', 'InterIamb',
                      'Lr', 'MBC', 'MiContinuous', 'MMHC', 'MMPC', 'NB', 'PC',
                      'Pearson', 'SparseBn', 'Tan']

        if not skip_structure:
            if algorithm in algorithms:
                structure_learning = globals()[algorithm]
                self.graph = structure_learning(df, self.data_type,
                                                **sl_kwargs).run()
                self.num_nodes = len(self.graph)
            else:
                raise ValueError('Structure learning is only available for the'
                                 f' following methods: {*algorithms,}.')

        if estimation in estimators:
            if self.graph is None:
                raise Exception(
                    'The Bayesian Network does not have a structure.')
            if self.data_type == 'continuous':
                if estimation == 'bayesian':
                    raise ValueError(
                        'Bayesian estimation is not supported in the'
                        ' continuous case.')
                else:
                    params_learning = GaussianMLE(df, self.data_type,
                                                  self.graph)
            else:
                if estimation == 'mle':
                    params_learning = DiscreteMLE(df, self.data_type,
                                                  self.graph)
                else:
                    params_learning = DiscreteBE(df, self.data_type,
                                                 self.graph, **pl_kwargs)
            self.parameters = params_learning.run()
        elif estimation == 'random':
            self.parameters = self._get_random_cont_params()
        else:
            raise ValueError('Parameter learning is only available for the'
                             f' following methods: {*estimators,}.')

        if self.data_type == 'continuous':
            self.joint_dist.from_params(
                data_type=self.data_type,
                save_dist=self.num_nodes > 300, params=self.parameters,
                nodes_order=self._topological_order())
        return self

    def save(self, file_path='bn.gexf', layout=None):
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

        layout: str, optional
            A layout for the GEXF case. If the extension of file is not
            '.gexf', this parameter is not considered. Supported `layout`
            values are:

                - 'Circular'
                - 'Dot'
                - 'ForceAtlas2'
                - 'Grid'
                - 'FruchtermanReingold'
                - 'Image'
                - 'Sugiyama'

        Raises
        ------
        ValueError
            If the extension of the file is not supported.
        """

        file_path = file_path.lower()

        if file_path.endswith('.gexf'):
            GEXF().write_file(self, file_path, layout)
        elif file_path.endswith('.gzip') or file_path.endswith('csv'):
            AdjacencyMatrix().write_file(file_path, self)
        elif file_path.endswith('.bif'):
            BIF().write_file(file_path, self)
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

        Returns
        -------
        self : BayesianNetwork
            Loaded Bayesian network.

        Raises
        ------
        ValueError
            If the extension of the file is not supported.
        """
        file_path = file_path.lower()

        if file_path.endswith('.gexf'):
            self.graph = GEXF().read_file(file_path)
        elif file_path.endswith('.gzip') or file_path.endswith('csv'):
            self.graph = AdjacencyMatrix().read_file(file_path)
        elif file_path.endswith('.bif'):
            self.graph, self.parameters = BIF().read_file(file_path)
            self.data_type = 'discrete'
            self.joint_dist.from_params(
                save_dist=len(self.graph) > 300,
                data_type=self.data_type, params=self.parameters,
                nodes_order=self._topological_order())
        else:
            raise ValueError('File extension not supported.')
        self.num_nodes = len(self.graph)

        return self

    def compare(self, real_graph, nodes_order=None, metric='all',
                undirected=False, threshold=0, h_method='out_degree',
                h_threshold=2):
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

        h_threshold : default=2
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
        """

        self._check_graph()

        if isinstance(metric, str):
            if metric != 'all':
                metric = [metric]
            else:
                metric = ['confusion', 'confusion_hubs', 'acc', 'f1',
                          'roc_auc', 'avg_precision', 'mcc']

        if real_graph.shape != (self.num_nodes, self.num_nodes):
            raise ValueError('\'real_graph\' does not have the right shape.')

        if nodes_order is not None and set(nodes_order) != set(
                list(self.graph.nodes())):
            raise ValueError('\'nodes_order\' and the nodes in the Bayesian '
                             'network are not the same.')

        adj_matrix = nx.to_numpy_matrix(self.graph, nodelist=nodes_order)

        result = {}

        confusion = None
        if 'confusion' in metric:
            confusion = confusion_matrix(adj_matrix, real_graph,
                                         undirected=undirected,
                                         threshold=threshold)
            result = {'confusion': confusion}
        if 'accuracy' in metric:
            result = {**result, 'accuracy': accuracy(adj_matrix, real_graph,
                                                     undirected=undirected,
                                                     threshold=threshold,
                                                     confusion=confusion)}
        if 'f1' in metric:
            result = {**result, 'f1': f1_score(adj_matrix, real_graph,
                                               undirected=undirected,
                                               threshold=threshold,
                                               confusion=confusion)}
        if 'mcc' in metric:
            result = {**result,
                      'mcc': mcc_score(adj_matrix, real_graph,
                                       undirected=undirected,
                                       threshold=threshold,
                                       confusion=confusion)}
        if 'roc_auc' in metric:
            result = {**result, 'roc_auc': roc_auc_score(adj_matrix.flatten(),
                                                         real_graph.flatten())}
        if 'avg_precision' in metric:
            result = {**result, 'avg_precision': average_precision_score(
                adj_matrix.flatten(),
                adj_matrix.flatten())}
        if 'confusion_hubs' in metric:
            result = {**result,
                      'confusion_hubs': confusion_hubs(adj_matrix, real_graph,
                                                       method=h_method,
                                                       threshold=h_threshold)}

        return result

# TODO: Decide about get_parameters, _get_parameters_cont and
#  _get_parameters_disc.
# def _get_parameters_cont(self, node):
#     evidence_value = self.evidence[node]
#     mean, std_dev = self.joint_dist.marginal(node)
#
#     init_mean, init_std_dev = self.joint_dist.marginal(
#        node, initial=True)
#     if init_mean != mean and init_std_dev != std_dev:
#         gaussian_pdf_plot = density_functions_bn(
#             [init_mean, mean], [init_std_dev, std_dev], evidence_value)
#     else:
#         gaussian_pdf_plot = density_functions_bn(mean,
#                                                  std_dev,
#                                                  evidence_value)
#
#     return gaussian_pdf_plot, evidence_value
#
# def _get_parameters_disc(self, node):
#     found = False
#     node_parameters_states, node_parameters_values = None, None
#     for cpd in self.parameters:
#         if cpd.variable == node:
#             if len(cpd.variables) > 1:
#                 list_vars = copy.deepcopy(cpd.variables)
#                 list_vars.remove(node)
#                 cpd.marginalize(list_vars)
#
#             node_parameters_values = list(
#                 np.round(cpd.get_values().flat, decimals=3))
#             if cpd.state_names:
#                 node_parameters_states = cpd.state_names[node]
#             else:
#                 node_parameters_states = []
#                 for i, state in enumerate(node_parameters_values):
#                     node_parameters_states.append('State ' + str(i))
#             found = True
#             break
#
#     if not found:
#         raise Exception('Node parameters not found')
#
#     return node_parameters_states, node_parameters_values

# def get_parameters(self, node):
#     """
#
#     Parameters
#     ----------
#     node
#     """
#     self._check_node(node)
#     if self.data_type == 'continuous' or (
#             self.data_type == 'hybrid' and node in self.cont_nodes):
#         self._get_parameters_cont(node)
#     else:
#         self._get_parameters_disc(node)

# def get_probabilities_effect(self, group_categories=None):
#     means, std_devs = self.marginal_params(
#         group_categories=group_categories, evidences=self.evidence)
#     start_means, start_std_devs = means[0], std_devs[0]
#     current_means, current_std_devs = means[1], std_devs[1]
#
#     return start_means, start_std_devs, current_means, current_std_devs

# def groups(self):
#     groups = []
#
#     if self.additional_parameters:
#         groups = list(
#             self.additional_parameters['discrete_features'].keys())
#
#     return groups

# def get_info_nodes_by_group(self, group_id):
#     info_nodes = {}
#
#     if self.additional_parameters:
#         if group_id in self.additional_parameters['discrete_features']:
#             categories_in_group = \
#                 self.additional_parameters['discrete_features'][group_id]
#             for node_key, node_val in self.additional_parameters[
#                 'nodes'].items():
#                 if group_id in node_val['discrete_features']:
#                     category_node = \
#                         node_val['discrete_features'][group_id][
#                             0]  # Multiple elements not supported yet
#                     category_color = categories_in_group[category_node][
#                         'color']
#                     info_nodes[node_key] = {
#                         'category': category_node,
#                         'color': category_color,
#                     }
#
#     return info_nodes

# def get_categories_in_group(self, group_id):
#     categories_in_group = []
#
#     if self.additional_parameters:
#         if group_id in self.additional_parameters['discrete_features']:
#             categories_in_group = list(
#                 self.additional_parameters['discrete_features'][
#                     group_id].keys())
#
#     return categories_in_group

# def get_category_node(self, group_id, node):
#     category_id = None
#     if self.additional_parameters:
#         category_id = \
#             self.additional_parameters['nodes'][node][
#                 'discrete_features'][
#                 group_id][0]
#
#     return category_id

# def get_nodes_in_category(self, group_id, category_id, structure_id,
#                           show_neighbors):
#     nodes_in_category = []
#     color_category = ''
#     neighbors = set()
#
#     if self.additional_parameters:
#         color_category = \
#             self.additional_parameters['discrete_features'][group_id][
#                 category_id]['color']
#
#         for node_key, node_val in self.additional_parameters[
#             'nodes'].items():
#             if group_id in node_val['discrete_features'].keys():
#                 node_group = node_val['discrete_features'][group_id]
#                 if category_id == node_group[0]:
#                     nodes_in_category.append(node_key)
#
#     if show_neighbors:
#         for node in nodes_in_category:
#             neighbors_node = self.neighbors(node)
#             neighbors_not_in_category = [nd for nd in neighbors_node
#                                          if nd not in nodes_in_category]
#             neighbors.update(neighbors_not_in_category)
#
#     neighbors = list(neighbors)
#
#     return nodes_in_category, color_category, neighbors

# def get_nodes_in_categories(self, group_id, categories_ids):
#     nodes_in_categories = []
#
#     if self.additional_parameters:
#         for node_key, node_val in self.additional_parameters[
#             'nodes'].items():
#             if group_id in node_val['discrete_features'].keys():
#                 node_categories = node_val['discrete_features'][group_id]
#                 if any(x in categories_ids for x in node_categories):
#                     nodes_in_categories.append(node_key)
#
#     return nodes_in_categories
# @staticmethod
# def get_saved_bn_by_model_name(data_client_json, session):
#     if "model_name" in data_client_json:
#         model_name = data_client_json["model_name"]
#     else:
#         model_name = "ml_bayesian_network"
#     bn = session[model_name]
#
#     return bn

# def feature_to_str(self, feature):
#     str_result = feature
#     if feature.isdigit():
#         str_result = f'x_{feature}_'
#
#     return str_result

# def set_additional_parameters(self, additional_parameters):
#     mapping_alt_names = {}
#     for group_key, group_val in additional_parameters[
#         'discrete_features'].items():
#         for category_key, category_val in group_val.items():
#             if 'color' not in category_val:
#                 random_color_hex = generate_random_color()
#                 category_val['color'] = random_color_hex
#
#     additional_parameters_nodes = list(
#         additional_parameters['nodes'].items())
#
#     for node_key, node_val in additional_parameters_nodes:
#         if 'alternative_name' not in node_val:
#             mapping_alt_names[node_key] = node_key
#             continue
#         alt_name = node_val['alternative_name']
#
#         if node_key in self.nodes:
#             mapping_alt_names[node_key] = node_val['alternative_name']
#             node_val['alternative_name'] = node_key
#         elif node_val['alternative_name'] in self.nodes:
#             mapping_alt_names[alt_name] = alt_name
#
#         del additional_parameters['nodes'][node_key]
#         additional_parameters['nodes'][alt_name] = node_val
#         if self.additional_parameters:
#             self.additional_parameters['nodes'][alt_name][
#                 'discrete_features'].update(node_val['discrete_features'])
#
#     if mapping_alt_names:
#         self.graph = networkx.relabel_nodes(
#         self.graph, mapping_alt_names)
#
#     if self.additional_parameters:
#         self.additional_parameters['discrete_features'].update(
#             additional_parameters['discrete_features'])
#
#     if not self.additional_parameters:
#         self.additional_parameters = additional_parameters

# def set_random_additional_parameters(self):
#     discrete_features = {
#         'brain_region': {
#             'hippocampus': {
#                 'color': '#14cb72'
#             },
#             'visual_cortex': {
#                 'color': '#98ca12'
#             },
#             'hypothalamus': {
#                 'color': '#511a85'
#             }
#         },
#         'disease': {
#             'parkinson': {
#                 'color': '#401b72'
#             },
#             'alzheimer': {
#                 'color': '#81ca12'
#             }
#         }
#     }
#     additional_parameters = {
#         'miscellaneous_parameters': {
#             'prob_params_type': 'continuous'
#         },
#         'discrete_features': discrete_features,
#         'nodes': {
#         },
#     }
#
#     groups = list(discrete_features.keys())
#
#     for node in self.nodes:
#         node_config = {
#             'prob_params': {
#                 'mean': 0.5,
#                 'sd': 1.2
#             },
#             'discrete_features': {
#             }
#         }
#         for group_i in groups:
#             categories_group = list(discrete_features[group_i].keys())
#             random_num_categories = random.randint(1,
#                                                    len(categories_group))
#             random_categories = random.sample(set(categories_group),
#                                               random_num_categories)
#
#             node_config['discrete_features'][group_i] = random_categories
#
#         additional_parameters['nodes'][node] = node_config
#
#     self.set_additional_parameters(additional_parameters)
# def get_additional_parameters(self):
#     return self.additional_parameters

# def add_new_additional_parameters(self, communities_nodes, com,
#                                   custom_group_name='group ',
#                                   group_color=None):
#     communities_groups = {}
#     communities_colors = {}
#     groups_number = 1
#
#     for node in communities_nodes:
#         if communities_nodes[node] not in communities_colors.values():
#             if custom_group_name == 'group ':
#                 communities_colors[
#                     custom_group_name + str(groups_number)] = \
#                     communities_nodes[node]
#                 groups_number += 1
#                 communities_groups[node] = custom_group_name + str(
#                     len(communities_colors))
#             else:
#                 communities_colors[custom_group_name] = {
#                     'color': group_color}
#                 communities_groups[node] = custom_group_name
#         else:
#             communities_colors_values = list(communities_colors.values())
#             group_index = 0
#             found = False
#             while not found:
#                 if communities_colors_values[group_index] == \
#                         communities_nodes[node]:
#                     found = True
#                 group_index += 1
#             if custom_group_name == 'group ':
#                 group_name = custom_group_name + str(group_index)
#             else:
#                 group_name = custom_group_name
#             communities_groups[node] = group_name
#
#     additional_parameters = self.get_additional_parameters()
#     if additional_parameters is None:
#         additional_parameters = {'discrete_features': {}, 'nodes': {}}
#         for node in communities_groups:
#             additional_parameters['nodes'][node] = {
#                 'discrete_features': {}}
#     if com in additional_parameters['discrete_features']:
#         additional_parameters['discrete_features'][com].update(
#             communities_colors)
#     else:
#         additional_parameters['discrete_features'][
#             com] = communities_colors
#
#     for node in communities_groups:
#         if node in additional_parameters['nodes']:
#             if com in additional_parameters['nodes'][node][
#                 'discrete_features']:
#                 variable = \
#                     additional_parameters['nodes'][node][
#                         'discrete_features'][
#                         com]
#                 variable.append(communities_groups[node])
#             else:
#                 additional_parameters['nodes'][node]['discrete_features'][
#                     com] = [communities_groups[node]]
#         else:
#             additional_parameters['nodes'][node] = {
#                 'discrete_features': {}}
#             additional_parameters['nodes'][node]['discrete_features'][
#                 com] = [communities_groups[node]]
#
#     return additional_parameters

# def restore_additional_parameters(self, selection_option):
#     additional_parameters = self.get_additional_parameters()
#     print(additional_parameters['discrete_features'][selection_option])
#     additional_parameters['discrete_features'].pop(selection_option)
#     for node in additional_parameters['nodes']:
#         additional_parameters['nodes'][node]['discrete_features'].pop(
#             selection_option)
#
#     return additional_parameters
