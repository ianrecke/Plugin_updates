"""
Adjacency matrix input/output module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from tempfile import NamedTemporaryFile

import networkx
import numpy as np
import pandas as pd

from .bnio import BNIO
from ..utils.data_structures import matrix2nx


def save_tmp(graph, nodes, directory, delete=False):
    """
    Saves the numpy array that represents the adjacency matrix of a Bayesian
    network graph structure in a temporary .npz file.

    Parameters
    ----------
    graph : numpy.array
        The graph represented by its adjacency matrix.

    nodes : list
        IDs of the nodes present in the graph.

    directory : str
        The directory used to store the file.

    delete : bool, default=False
        Whether to delete the file after closing it or not.

    Returns
    -------
    str
        The name of the file where the graph is stored.
    """

    outfile = NamedTemporaryFile(delete=delete, dir=directory)
    np.savez(outfile, nodes=np.array(nodes), graph=graph)
    return outfile.name


class AdjacencyMatrix(BNIO):
    """
    Adjacency matrix input/output class.
    """

    def read_file(self, file_path):
        """
        Returns the graph structure of a Bayesian network whose adjacency
        matrix is stored in a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored. It can be a CSV or
            parquet file.

        Returns
        -------
        networkx.DiGraph
            The graph structure of the loaded Bayesian network.
        """

        if file_path.endswith('.csv'):
            pd_adj_matrix = pd.read_csv(file_path, na_filter=False,
                                        dtype=np.float64, low_memory=False)
        elif file_path.endswith('.gzip'):
            pd_adj_matrix = pd.read_parquet(file_path,
                                            engine='fastparquet').astype(
                np.float64)
        else:
            raise ValueError('File extension not supported.')

        nodes = pd_adj_matrix.columns.values[1:]
        adj_matrix = pd_adj_matrix.iloc[:, 1:].values

        return matrix2nx(adj_matrix, nodes)

    def write_file(self, file_path, bn):
        """
        Writes a Bayesian network structure adjacency matrix in a file.

        Parameters
        ----------
        file_path: str
            Path of the file to store the Bayesian network in. It can be a CSV
            or parquet file.

        bn : BayesianNetwork
            Bayesian network to be stored.
        """

        nodes = bn.graph.nodes()
        adj_matrix = networkx.to_numpy_matrix(bn.graph, nodes)
        pd_adj_matrix = pd.DataFrame(adj_matrix, columns=nodes)
        if file_path.endswith('.csv'):
            pd_adj_matrix.to_csv(file_path)
        elif file_path.endswith('.gzip'):
            pd_adj_matrix.to_parquet(file_path)
