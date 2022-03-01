import numpy as np
import networkx as nx
from rpy2.robjects import pandas2ri
from abc import abstractmethod, ABCMeta


class LearnStructure(metaclass=ABCMeta):
    """Base class for all learn structure classes."""

    def __init__(self, data, data_type=None, states_names=None):
        """
        LearnStructure constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        self.data = data
        self.data_type = data_type
        variables = list(data.columns.values)

        if not isinstance(states_names, dict):
            self.states_names = {var: self._collect_states_names(var) for var
                                 in variables}
        else:
            self.states_names = dict()
            for var in variables:
                if var in states_names:
                    if not set(self._collect_states_names(var)) <= set(
                            states_names[var]):
                        raise ValueError(
                            f"Data contains unexpected states for variable '{var}'.")
                    self.states_names[var] = sorted(states_names[var])
                else:
                    self.states_names[var] = self._collect_states_names(var)

    def _collect_states_names(self, variable):
        """Return a list of states that the variable takes in the data."""
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states

    @abstractmethod
    def run(self):
        """
        Learn the structure of the Bayesian network.
        @return: A NetworkN graph with the learnt structure.
        """

    # TODO: add max_parents for HC, tabu, etc.
    def run_bnlearn(self, bnlearn_function, alpha=None):
        """
        Run a structure learning algorithm using its implementation in R's
        bnlearn.
        @param bnlearn_function: bnlearn algorithm function to execute.
        @param alpha: The target nominal type I error rate if needed.
        @return: A NetworkX graph with the learnt structure.
        """
        dataframe = pd2r(self.data, self.data_type)
        nodes = list(self.data.columns.values)

        if alpha is None:
            output_raw_r = bnlearn_function(x=dataframe)
        else:
            output_raw_r = bnlearn_function(x=dataframe,
                                            alpha=alpha)

        graph = parse_output_structure_bnlearn(nodes, output_raw_r)

        return graph


def pd2r(data, data_type):
    """
    Converts a pandas DataFrame into a R dataframe.
    @param data: Dataframe to be transformed.
    @type data: pandas DataFrame.
    @param data_type: type of the data the dataframe contains (continuous,
        discrete or hybrid).
    @return: R dataframe with the same information as the input dataframe.
    """
    # TODO: check pandas DataFrame to R dataframe conversion with activate
    pandas2ri.activate()
    dataframe = data

    # TODO: move data type checking
    if data_type == "hybrid":
        raise Exception(
            "This algorithm still does not support hybrid bayesian networks")

    return dataframe


# TODO: move nx_graph_from_adj_matrix to utils/helpers
def nx_graph_from_adj_matrix(adj_matrix, nodes_names):
    """
    Creates a NetworkX graph from an adjacency matrix.
    @param adj_matrix: 2-Dimensional numpy array with the adjacency matrix.
    @param nodes_names: Names of the nodes of the network.
    @return: A NetworkX graph representing the network.
    """
    nx_graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

    mapping = {}
    for i, node in enumerate(nx_graph.nodes()):
        mapping[node] = nodes_names[i]

    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    return nx_graph


# TODO: move parse_output_structure_bnlearn to utils/helpers
def parse_output_structure_bnlearn(nodes, output_raw_r):
    """
    Converts R's bnlearn output into a NetworkX graph.
    @param nodes: Nodes that form the network.
    @param output_raw_r: R's bnlearn output.
    @return: A NetworkX graph representing the network.
    """
    arcs_r = output_raw_r.rx2("arcs")
    edges_from = np.array(arcs_r.rx(True, 1))
    edges_to = np.array(arcs_r.rx(True, 2))
    edges = zip(edges_from, edges_to)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
