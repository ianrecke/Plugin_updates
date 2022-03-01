from rpy2.robjects.packages import importr
import time
from celery import current_task
import numpy as np
import networkx as nx

from .LearnStructure import LearnStructure, pd2r
from ..helpers.helpers import update_progress_worker


class SparseBn(LearnStructure):
    """sparsebn structure learning class."""

    def __init__(self, data, data_type, states_names=None):
        """
        sparsebn structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(SparseBn, self).__init__(data, data_type, states_names)

    def run(self, backend="sparsebn"):
        """

        @param backend:
        @return:
        """
        nodes = list(self.data.columns.values)

        model = None
        if backend == "neurosuites":
            model = self.run_sparse_bn_neurosuites(nodes)
        elif backend == "sparsebn":
            model = self.run_sparse_bn_sparsebn()

        return model

    def run_sparse_bn_sparsebn(self):
        """

        @return:
        """
        update_progress_worker(current_task, 5)  # ----

        dataframe = pd2r(self.data, self.data_type)

        start_time = time.time()
        sparsebn = importr("sparsebn")
        sparsebn_utils = importr("sparsebnUtils")
        data_r = sparsebn_utils.sparsebnData(dataframe, type=self.data_type)
        dags_r = sparsebn.estimate_dag(data_r)
        solution_idx_r = sparsebn_utils.select_parameter(dags_r, data_r)
        output_raw_r = sparsebn_utils.select(dags_r, index=solution_idx_r)
        output_raw_r = dict(output_raw_r.items())

        update_progress_worker(current_task, 60)  # ----

        edges_r = np.array(output_raw_r["edges"])
        nodes = output_raw_r["edges"].names
        edges_python = []
        for i, parents in enumerate(edges_r):
            parents_node_i = list(parents)
            for parent in parents_node_i:
                edge = (nodes[parent], nodes[i])
                edges_python.append(edge)

        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        print("-------TOTAL TIME sparsebn:--------", total_time)

        update_progress_worker(current_task, 95)  # ----

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges_python)

        return graph

    def r_vector_to_list(self, r_vector):
        return list(r_vector)

    def run_sparse_bn_neurosuites(self, nodes):

        return 0
