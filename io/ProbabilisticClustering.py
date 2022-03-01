import numpy as np
import networkx
import pandas as pd
import os
import tempfile

from .IOBN import IOBN


class ProbabilisticClustering(IOBN):
    def __init__(self, file_extension):
        super(ProbabilisticClustering, self).__init__()
        self.extension = file_extension

    # TODO: join multiple import_file functions
    def import_file(self, file_path):
        pass

    def import_file_precision_matrix(self, path_file):
        precision_matrix = self.import_file_numpy_matrix(
            path_file)  # precision_matrix
        adj_matrix = np.array(precision_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        bn_graph = networkx.from_numpy_matrix(adj_matrix,
                                              create_using=networkx.DiGraph)

        return bn_graph

    def import_file_numpy_matrix(self, path_file):
        if self.extension == "csv":
            pd_matrix = pd.read_csv(path_file, na_filter=False,
                                    dtype=np.float64, low_memory=False,
                                    header=None)
        elif self.extension == "gzip":
            pd_matrix = pd.read_parquet(path_file,
                                        engine="fastparquet").astype(
                np.float64)

        np_matrix = pd_matrix.values

        return np_matrix

    def export_file(self, file_path, bn):
        file_path = file_path + "." + self.extension
        file_path = os.path.join(tempfile.gettempdir(), file_path)

        adj_matrix = networkx.to_numpy_matrix(bn.graph, bn.graph.nodes())
        pd_cov_matrix = pd.DataFrame(adj_matrix, columns=bn.graph.nodes())
        pd_cov_matrix.to_csv(file_path)

        with open(file_path) as file_exported:
            file_exported_memory = file_exported.read()

        return file_exported_memory
