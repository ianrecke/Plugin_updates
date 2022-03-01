from abc import ABCMeta, abstractmethod
import networkx
from pgmpy import models as pgmpy_models


class IOBN(metaclass=ABCMeta):
    """
    Base class for all input/ouput Bayesian network classes.
    """

    def __init__(self):
        pass

    @abstractmethod
    def import_file(self, file_path):
        """Load a bayesian network from a file.
        @param file_path: Path of the file to load the Bayesian network from.
        """

    @abstractmethod
    def export_file(self, file_path, bn):
        """Export the bayesian network to a file.
        @param file_path: Path of the file to store the Bayesian network in.
        @param bn: Bayesian network to be stored.
        """


# TODO: move pgmpy_model_to_nx_graph_parameters
def pgmpy_model_to_nx_graph_parameters(bn_model_pgmpy):
    graph = networkx.DiGraph()
    graph.add_nodes_from(bn_model_pgmpy.nodes())
    graph.add_edges_from(bn_model_pgmpy.edges())
    parameters = bn_model_pgmpy.cpds

    return graph, parameters


# TODO: move nx_graph_parameters_to_pgmpy_model
def nx_graph_parameters_to_pgmpy_model(graph, parameters):
    pgmpy_model = pgmpy_models.BayesianModel()
    pgmpy_model.add_nodes_from(graph.nodes())
    pgmpy_model.add_edges_from(graph.edges())
    if parameters:
        pgmpy_model.add_cpds(*parameters)

    return pgmpy_model
