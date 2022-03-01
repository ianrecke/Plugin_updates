from rpy2.robjects.packages import importr
import networkx as nx
from itertools import combinations
from pgmpy.estimators.PC import PC as PGMPY_PC

from .LearnStructure import LearnStructure
from ..utils import bn_utils


class PC(LearnStructure):
    """PC structure learning class."""

    def __init__(self, data, data_type, max_number_parents, max_adjacency_size,
                 alpha, states_names=None):
        """
        PC structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param max_number_parents:
        @param max_adjacency_size:
        @param alpha:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(PC, self).__init__(data, data_type, states_names)
        self.max_number_parents = max_number_parents
        self.max_adjacency_size = max_adjacency_size
        self.alpha = alpha

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        nodes = list(self.data.columns.values)

        model = None
        if backend == "neurosuites":
            model = self.run_pc_neurosuites()
        elif backend == "bnlearn":
            model = self.run_bnlearn(importr("bnlearn").pc_stable,
                                     self.alpha)

        return model

    def run_pc_neurosuites(self):
        """

        @return:
        """
        nodes = list(self.data.columns.values)
        # 1st part PC (Create undirected Graph)
        skeleton, separating_sets = self._estimate_skeleton(nodes)
        # 2nd part PC (Direct all graph edges to create a DAG)
        pdag = PGMPY_PC.skeleton_to_pdag(
            skeleton, separating_sets)
        dag = pdag.to_dag(pdag)

        return dag

    # TODO: check implementation
    def _estimate_skeleton(self, nodes):
        # Start with a complete undirected graph G on the set V of all vertices
        graph = nx.Graph(combinations(nodes, 2))
        # return graph, []

        lim_neighbors = 0
        separating_sets = dict()
        while not all([len(list(graph.neighbors(node))) <
                       lim_neighbors for node in nodes]):
            for node in nodes:
                for neighbor in list(graph.neighbors(node)):
                    # search if there is a set of neighbors (of size
                    # lim_neighbors) that makes X and Y independent:
                    for separating_set in combinations(
                            set(list(graph.neighbors(node))) - set([neighbor]),
                            lim_neighbors):
                        if bn_utils.is_independent_chi_square_test(
                                self, node, neighbor, separating_set):
                            separating_sets[frozenset(
                                (node, neighbor))] = separating_set
                            graph.remove_edge(node, neighbor)
                            break
            lim_neighbors += 1

        return graph, separating_sets

    # def direct_edges(self, nodes):
    #     graph = DAG(nodes)
    #
    #     return graph
