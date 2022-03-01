"""
-------Notes----------

Original algorithm in Java:
https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Fges.java

Python wrapper for the original Java code:
https://github.com/bd2kccd/py-causal

Alternative implementation in pure Python:
https://github.com/eberharf/fges-py
------------------------
"""

import shutil
import sys
from mpi4py import MPI
from operator import itemgetter
import math
from numpy.linalg import inv
from os import environ as env
import warnings
from functools import wraps
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from line_profiler import LineProfiler
from pympler import asizeof as aof
import numba
from decorator import decorator
import os
import mkl
import networkx as nx
import numpy as np
from itertools import combinations, permutations
import time
import pandas as pd
import matplotlib

from ..utils.arrows_utils import create_arrow_str, get_i_j, get_j, \
    get_all_vals, get_bic
from ..utils.graph_utils import get_list_hubs, force_hubs_directions, \
    union_graphs, remove_not_related_nodes, remove_edges_children_hubs, \
    intersect_graphs, add_children, adjacencies, remove_undirected, parents, \
    children, undirecteds, intersection_graphs_tops, add_undirected, \
    exists_unblocked_semi_directed_path, is_undirected, is_parent, \
    remove_parent, is_children, remove_children
from ..utils import bn_utils
from ..utils.score_utils import run_scores
from .LearnStructure import LearnStructure

matplotlib.use('agg')

# import my_numba_funcs


# -----MPI setup------------
COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()
node_name = MPI.Get_processor_name()
mpi_info = MPI.Info.Create()
mpi_version = MPI.get_vendor()
print(mpi_version)
# universe_size = COMM.Get_attr(MPI.UNIVERSE_SIZE)
print("########### MPI -- Node: {}; process {} of {} #############".format(
    node_name, rank, size - 1))
# ---------------------------

# -----Numba setup--------------
# https://numba.pydata.org/numba-doc/dev/reference/envvars.html
# https://numba.pydata.org/numba-doc/dev/user/threading-layer.html#numba-threading-layer

print("NUMPY CONFIG: ")
np.show_config()

env['NUMBA_NUM_THREADS'] = str(numba.config.NUMBA_DEFAULT_NUM_THREADS)
numba.config.reload_config()
# torch.set_num_threads(8)  # This sets the OpenMP threads and also the mkl max threads
# mkl.set_num_threads(int(numba.config.NUMBA_DEFAULT_NUM_THREADS / 2))
# #Max = num cores (usually in Intel: num threads = num_cores*2 (because
# of the hyperthreading))

numba.config.THREADING_LAYER = "omp"  # OpenMP
numba.config.NUMBA_WARNINGS = 1  # Enabled

print("MKL_NUM_THREADS: ", mkl.get_max_threads())
print("NUMBA_NUM_THREADS: ", numba.npyufunc.parallel.get_thread_count())


class FGES(LearnStructure):
    def __init__(self, data, algorithm_parameters, data_type,
                 states_names=None, session_id=None):
        super(FGES, self).__init__(data, data_type, states_names)
        self.penalty = int(algorithm_parameters["fges_penalty"])
        self.mode = algorithm_parameters["fges_mode"]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        graphs_dir = os.path.join(current_dir, "local_graphs")
        if not os.path.exists(graphs_dir):
            os.mkdir(graphs_dir)
        self.path_save_graphs = os.path.join(graphs_dir, session_id)
        if not os.path.exists(self.path_save_graphs):
            os.mkdir(self.path_save_graphs)

    def run(self, backend="neurosuites"):
        nodes = list(self.data.columns.values)

        if backend == "neurosuites":
            if self.mode == "global":
                graph = self.run_fges_neurosuites(nodes)
            elif self.mode == "local-global":
                graph = self.run_local_global_fges(nodes)
            else:
                raise Exception("Mode {} not found".format(self.mode))

        mapping = {}
        for i, node in enumerate(nodes):
            mapping[i] = node
        graph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph())
        graph = nx.relabel_nodes(graph, mapping)

        # os.rmdir(self.path_save_graphs) # Clean csv graphs folder for this
        # session
        shutil.rmtree(self.path_save_graphs, ignore_errors=True)

        return graph

    # @profile_each_line
    # @timecall(immediate=True)
    # @profile(immediate=True)
    def run_fges_neurosuites(self, nodes):
        global GRAPH
        # helper_workers.update_progress_worker(current_task, 5) #----

        # ---Run algorithm---
        fges = FGESAlgorithm(self.data, self.penalty, self.path_save_graphs,
                             use_mpi=True)
        fges.forward_equivalence_search()
        if rank == 0:
            fges.reevaluate_backward(NODES)
            fges.backward_equivalence_search()
            fges.orient_graph()
            GRAPH = fges.final_bics_of_edges()
            save_graph_as_csv(graph_name="global_graph")

        print("Process: {} finished!".format(rank))
        """
        #-------------------
        #helper_workers.update_progress_worker(current_task, 95) #----
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(fges.edges)
        """

        return GRAPH

    def candidate_nodes(self, node, nodes, nodes_names, all_data, bics):
        positive_bics = [(j, bics[j]) for j in nodes if bics[j] > 0]
        positive_bics.sort(key=itemgetter(1), reverse=True)
        print("-------- Positive bics for {}: {}-----------".format(node,
                                                                    positive_bics))
        positive_bics_values = [bic[1] for bic in positive_bics]

        max_candidates = min(100, len(positive_bics_values))
        num_candidates = bn_utils.hypothesis_test_related_genes(
            max_candidates, positive_bics_values)

        nodes_selection = [candidate[0] for candidate in
                           positive_bics[0:num_candidates]]
        print(
            "************* Num Positive bics for {}: {} *************".format(
                node, len(positive_bics)))
        print("************* Num Candidates for {}: {} *************".format(
            node, len(nodes_selection)))
        nodes_selection.append(node)

        nodes_names_selection = list(np.array(nodes_names)[nodes_selection])
        data_selection = all_data.loc[:, nodes_names_selection]

        sys.stdout.flush()
        return nodes_selection, data_selection

    def run_local_global_fges(self, nodes_names, train=True,
                              grid_search=False):
        global GRAPH
        global NODES_NAMES
        n = len(nodes_names)

        if rank == 0:
            start_time = time.time()
            time_str = time.ctime(start_time)
        else:
            time_str = None
        time_str = COMM.bcast(time_str, root=0)

        all_data = self.data
        nodes = list(range(n))

        if train:
            all_BICS = np.zeros((n, n))
            all_BICS = init_bics_mpi(all_data.values, nodes, all_BICS,
                                     self.penalty, save_csv=False)

            # -----------MAP----------------------
            if rank == 0:  # Master
                start_time = time.time()
                time_str = time.ctime(start_time)
                chunks_nodes_bics = []
                chunks_nodes = [
                    nodes[int(i * n / size):int((i + 1) * n / size)] for i in
                    range(size)]

                for chunk in chunks_nodes:
                    this_chunk = {}
                    for node in chunk:
                        this_chunk[node] = all_BICS[node, :]
                    chunks_nodes_bics.append(this_chunk)
            else:
                chunks_nodes_bics = None
                time_str = None

            time_str = COMM.bcast(time_str, root=0)
            chunk_nodes_bics = COMM.scatter(chunks_nodes_bics, root=0)
            print(
                "************* Rank {} local subnetworks with nodes: {} *****************".format(
                    rank, chunk_nodes_bics))
            sys.stdout.flush()

            # ----------COMPUTE in each computing node-----------
            for node, bics in chunk_nodes_bics.items():
                nodes_selection, data_selection = self.candidate_nodes(
                    node, nodes, nodes_names, all_data, bics)

                # Construct local network with node+candidate neighbors:
                fges = FGESAlgorithm(data_selection, self.penalty,
                                     self.path_save_graphs, use_mpi=False)
                fges.forward_equivalence_search()
                fges.reevaluate_backward(NODES)
                fges.backward_equivalence_search()
                fges.orient_graph()
                GRAPH = fges.final_bics_of_edges()

                save_graph_as_csv(graph_name=str(node))

        # --------------REDUCE-----------------------
        all_combinations_config = []
        methods = ["union", "intersection"]
        b_intersects = [True, False]
        b_intersects_global = [True, False]
        remove_global = [True, False]
        force_hubs_directions_before = [True, False]
        force_hubs_directions_after = [True, False]
        remove_edges_children_hubs_before = [True, False]
        remove_edges_children_hubs_after = [True, False]
        hubs_methods = ["degree", "betweenness", "degree-betweenness"]
        disconnected_parents_range = [3, 5, 8]
        i = 0

        # --------------COMBINE SUBGRAPHS-----------------------
        combination_config_default = {
            'name': "default_config",
            'primary_combine_method': "union",
            'backward_intersect': True,
            'backward_intersect_global_bad': False,
            'remove_not_related_global': True,
            'force_hubs_directions_before': True,
            'force_hubs_directions_after': True,
            "remove_edges_children_before": True,
            "remove_edges_children_after": True,
            "threshold_neighbors_percentile": 91,
            "hubs_method": "degree",
            "n_parents": 3,
            "counting": False,
        }

        if not grid_search:
            self.run_combination_graphs(all_combinations_config,
                                        combination_config_default, i, n,
                                        nodes_names, time_str)
        else:
            for m in methods:
                for b in b_intersects:
                    for b_global in b_intersects_global:
                        for rm_global in remove_global:
                            for force_hub_before in force_hubs_directions_before:
                                for remove_edges_children_before in remove_edges_children_hubs_before:
                                    for force_hub_after in force_hubs_directions_after:
                                        for remove_edges_children_after in remove_edges_children_hubs_after:
                                            for hubs_method in hubs_methods:
                                                for n_parents in disconnected_parents_range:
                                                    combination_config = {
                                                        'primary_combine_method': m,
                                                        'backward_intersect': b,
                                                        'backward_intersect_global_bad': b_global,
                                                        'remove_not_related_global': rm_global,
                                                        'force_hubs_directions_before': force_hub_before,
                                                        'force_hubs_directions_after': force_hub_after,
                                                        "remove_edges_children_before": remove_edges_children_before,
                                                        "remove_edges_children_after": remove_edges_children_after,
                                                        "threshold_neighbors_percentile": 98,
                                                        "hubs_method": hubs_method,
                                                        "n_parents": n_parents,
                                                        "counting": True,
                                                    }

                                                    self.run_combination_graphs(
                                                        all_combinations_config,
                                                        combination_config, i,
                                                        n, nodes_names,
                                                        time_str)

                                                    i += 1

                                                    # return 1

        print("Rank: {} finished!".format(rank))

        if rank == 0:
            end_time = time.time()
            time_func = end_time - start_time
            print(
                "....................................................................................")
            print(
                "....................................................................................")
            print(
                "............ run_local_global_fges finished in: {} ............".format(
                    time_func))
            print(
                "....................................................................................")
            print(
                "....................................................................................")
            all_combinations_config_pd = pd.DataFrame(all_combinations_config)
            all_combinations_config_pd.to_csv(
                os.path.join(PATH_SAVE_GRAPHS, 'combinations_config.csv'))

        return GRAPH

    def run_combination_graphs(self, all_combinations_config,
                               combination_config, i, n, nodes_names,
                               time_str):
        # -----------MAP----------------------
        if rank == 0:
            # subgraphs = next(os.walk("./local_graphs"))[2]
            # subgraphs = [i[:-4] for i in subgraphs]
            subgraphs = list(range(n))
            chunks_subgraphs_names = [
                subgraphs[int(i * n / size):int((i + 1) * n / size)]
                for i in range(size)]

        else:
            chunks_subgraphs_names = None
        chunks_subgraphs_names = COMM.scatter(chunks_subgraphs_names, root=0)
        graph_global = np.zeros((n, n), dtype=np.float64)
        if size > 1:
            # -----------COMPUTE on each node----------------------
            self.combine_all_subgraphs(graph_global, chunks_subgraphs_names,
                                       nodes_names, combination_config,
                                       graph_name="local_graph_rank_" + str(
                                           rank))
            COMM.barrier()

            # -----------REDUCE----------------------
            if rank == 0:
                # Join all subgraphs generated by the MPI nodes:
                subgraphs_names = ['local_graph_rank_' + str(i) for i in
                                   range(size)]
                graph_global = np.zeros((n, n), dtype=np.float64)

                self.combine_all_subgraphs(
                    graph_global,
                    subgraphs_names,
                    nodes_names,
                    combination_config,
                    graph_name="local_global_graph_" +
                               str(i))
                all_combinations_config.append(combination_config)
                print(
                    "==============================================DONE combination MPI: {}===========================================".format(
                        i))
            COMM.barrier()
        else:
            self.combine_all_subgraphs(
                graph_global,
                chunks_subgraphs_names,
                nodes_names,
                combination_config,
                graph_name="global_modified_graph_" +
                           str(i),
                n=i)
            all_combinations_config.append(combination_config)
            print(
                "==============================================DONE combination one node: {}===========================================".format(
                    i))
            # draw_net("graph_global")

        return 0

    def combine_all_subgraphs(self, graph_global, subgraphs_names, nodes_names,
                              combination_config, only_one_graph=False,
                              graph_name="local_graph_rank_" + str(rank),
                              n=None):
        global NODES_NAMES
        global GRAPH

        if only_one_graph:
            subgraphs_names = [subgraphs_names[0]]
            # if n >= 0:
            #    graph_name = subgraphs_names[0] + "_modified_"+str(n)
            # else:
            PATH = "DREAM tests/Net 1 global graphs penalty 4_34/"
            graph_name = os.path.join(PATH, "test")
        for name in subgraphs_names:
            pd_adj_matrix = pd.read_csv(
                os.path.join(PATH_SAVE_GRAPHS, '{}.csv'.format(str(name))))
            print("Combining subgraph {} rank {}".format(name, rank))

            nodes_names_local = pd_adj_matrix.columns.values[1:]
            local_graph = pd_adj_matrix.iloc[:, 1:].values
            if combination_config["counting"]:
                local_graph = (local_graph > 0).astype(np.int64)
            n = local_graph.shape[0]
            nodes_local = list(range(n))

            nodes_global_indices = [nodes_names.index(node) for node in
                                    nodes_names_local]
            local_global_indices = dict(zip(nodes_local, nodes_global_indices))

            # if combination_config["remove_not_related_global"]:
            #    remove_not_related_nodes(local_graph)

            if combination_config["force_hubs_directions_before"]:
                list_hubs = get_list_hubs(local_graph, combination_config[
                    "threshold_neighbors_percentile"],
                                          method=combination_config[
                                              "hubs_method"], show_plots=False)
                force_hubs_directions(local_graph, list_hubs)
                if combination_config["remove_edges_children_before"]:
                    remove_edges_children_hubs(local_graph, list_hubs,
                                               combination_config["n_parents"])

            if combination_config["primary_combine_method"] == "union":
                graph_global = union_graphs(graph_global, local_graph,
                                            local_global_indices)
            else:
                graph_global = intersect_graphs(graph_global, local_graph,
                                                local_global_indices)

        if combination_config["backward_intersect"]:
            for name in subgraphs_names:
                print("Backward intersect subgraph {} rank {}".format(name,
                                                                      rank))
                pd_adj_matrix = pd.read_csv(
                    os.path.join(PATH_SAVE_GRAPHS, '{}.csv'.format(str(name))))

                nodes_names_local = pd_adj_matrix.columns.values[1:]
                local_graph = pd_adj_matrix.iloc[:, 1:].values
                n = local_graph.shape[0]
                nodes_local = list(range(n))

                nodes_global_indices = [nodes_names.index(node) for node in
                                        nodes_names_local]
                local_global_indices = dict(
                    zip(nodes_local, nodes_global_indices))

                intersection_graphs_tops(graph_global, local_graph,
                                         local_global_indices,
                                         combination_config[
                                             "backward_intersect_global_bad"])

        if combination_config["counting"]:
            graph_global = graph_global / len(
                subgraphs_names)  # np.max(graph_global)

        if combination_config["remove_not_related_global"]:
            remove_not_related_nodes(graph_global)

        if combination_config["force_hubs_directions_after"]:
            list_hubs = get_list_hubs(graph_global, combination_config[
                "threshold_neighbors_percentile"],
                                      method=combination_config["hubs_method"])
            force_hubs_directions(graph_global, list_hubs)
            if combination_config["remove_edges_children_after"]:
                remove_edges_children_hubs(graph_global, list_hubs,
                                           combination_config["n_parents"])

        NODES_NAMES = nodes_names
        GRAPH = graph_global
        save_graph_as_csv(graph_name)

        if only_one_graph:
            run_scores()

        return 0


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response

    return inner


# ---------------------Profilers---------------------------------------------------------------------------------------

@decorator
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()

        return profiled_func

    return inner


@decorator
def profile_each_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        profiled_func(*args, **kwargs)
    finally:
        profiler.print_stats()


# -----------------------------------------------------------------------------------------------------------------
# ===============================================================================================================
# =====================================FGES_Alternative_github code=======
def recursive_partial_corr(x, y, Z):
    if len(Z) == 0:
        return CORRCOEFS[x, y]
    else:
        z0 = min(Z)
        Z1 = Z - {z0}

        term1 = recursive_partial_corr(x, y, Z1)
        term2 = recursive_partial_corr(x, z0, Z1)
        term3 = recursive_partial_corr(y, z0, Z1)

        return (term1 - (term2 * term3)) / math.sqrt(
            (1 - (term2 * term2)) * (1 - (term3 * term3)))


def local_score_diff_parents(node1, node2, parents):
    parents = frozenset(parents)
    n = DATA.shape[0]
    r = recursive_partial_corr(node1, node2, parents)

    return -n * math.log(1.0 - r * r) - PENALTY * math.log(n)
    # return self.local_score(node2, parents + [node1]) -
    # self.local_score(node2, parents)


def local_score_diff(node1, node2):
    return local_score_diff_parents(node1, node2, [])


def init_bics_alternative(BICS):
    n = len(NODES)
    for i in range(n):
        for j in range(n):
            if i != j:
                BICS[i, j] = local_score_diff(i, j)

    return BICS


# ===============================================================================================================
# ===============================================================================================================


# cc = CC('my_numba_funcs')

# @timecall(immediate=True)
# @cc.export('BIC_numba', 'float64(float64[:], float64[:], int32)')
# @numba.jit(nopython=True, fastmath=True)
@numba.jit(nopython=True, fastmath=True)
def BIC_numba(X, y, penalty):
    n = X.shape[0]
    X = np.ascontiguousarray(X).reshape(n, -1)
    k = X.shape[1]
    y = np.ascontiguousarray(y).reshape(n, -1)

    if k == 0:
        return -n * np.log(np.sum(np.square(y - np.mean(y)) / n))

    A = np.ones((n, k + 1),
                dtype=np.float64)  # Bias is the last column of ones
    A[:, :k] = X

    # One liner to speedup some microseconds (not creating temporary
    # variables):
    return -n * np.log(np.sum(np.square(y - np.dot(np.linalg.lstsq(A, y)
                                                   [0].T,
                                                   A.T).T)) / n) - penalty * k * np.log(
        n)


# @profile_each_line
def get_BIC(X, y, data_x, data_y, penalty):
    global NUM_CACHED
    global NUM_NOT_CACHED

    key = (frozenset(X), y)
    if key in CACHE_BICS:
        NUM_CACHED += 1
        BIC_score = CACHE_BICS[key]
    else:
        BIC_score = BIC_no_numba(data_x, data_y, penalty)
        CACHE_BICS[key] = BIC_score
        NUM_NOT_CACHED += 1

    return BIC_score


@numba.jit(nopython=True, fastmath=True)
def BIC_initial(y):
    n = y.shape[0]
    return np.round(-n * np.log(np.sum(np.square(y - np.mean(y))) / n), 5)


@ignore_warnings
def BIC_no_numba(X, y, penalty):
    n = X.shape[0]
    X = np.ascontiguousarray(X).reshape(n, -1)
    k = X.shape[1]
    y = np.ascontiguousarray(y).reshape(n, -1)

    if k == 0:
        return -n * np.log(np.sum(np.square(y - np.mean(y)) / n))

    A = np.ones((n, k + 1),
                dtype=np.float64)  # Bias is the last column of ones
    A[:, :k] = X

    result_lr = np.linalg.lstsq(A, y)
    w = result_lr[0]

    y_predicted_numba = np.dot(w.T, A.T)
    residuals_numba = y - y_predicted_numba.T
    mean_sq_error_numba = np.sum(np.square(residuals_numba)) / n
    BIC_score_numba = -n * np.log(mean_sq_error_numba) - penalty * k * np.log(
        n)

    return BIC_score_numba


def bic(X, y, penalty=1.0):
    """
    @param X: Set of variables in the model that might be parents of y.
    @param y: Variable we are checking against.
    @param penalty: Hyperparameter of the model.
    @return: A Numpy array with the BIC scores.
    """
    if not (isinstance(X, np.ndarray)):
        n = y.shape[0]
        return -n * np.log(np.sum(np.square(y - np.mean(y))) / n)

    if len(X.shape) < 2:
        X = X.reshape(-1, 1)

    n, k = X.shape
    regression = LinearRegression().fit(X, y)
    y_predicted = regression.predict(X)
    residuals = y - y_predicted
    mean_sq_error = np.sum(np.square(residuals)) / n
    BIC_score = -n * np.log(mean_sq_error) - penalty * k * np.log(n)

    return BIC_score


# @timecall(immediate=True)
@numba.jit(nopython=True,
           parallel=True)  # Caching is not available with parallel
def init_bics_numba(data, nodes, BICS, penalty, compile):
    if compile:
        return BICS

    n = len(nodes)
    for i in numba.prange(n):
        # for i in range(n):
        X = data[:, i]
        for j in numba.prange(i + 1, n):
            # for j in range(i + 1, n):
            y = data[:, j]
            BIC_X = BIC_numba(X, y, penalty)
            BIC_y = BIC_initial(y)
            BICS[i, j] = BIC_X - BIC_y
            BICS[j, i] = BIC_X - BIC_y

    return BICS


# @timecall(immediate=True)
def init_bics(data, nodes, BICS, penalty):
    n = len(nodes)
    for i in range(n):
        y = data[:, i]
        for j in range(i + 1, n):
            X = data[:, j]
            BICS[i, j] = bic(X, y, penalty)
            BICS[j, i] = BICS[i, j]

    return BICS


@numba.jit(nopython=True,
           parallel=True)  # Caching is not available with parallel
def init_bics_numba_chunk_mpi(data, n, nodes_chunk, bics_chunk, penalty,
                              compile):  # n: total number of nodes
    if compile:
        return bics_chunk

    for i in numba.prange(len(nodes_chunk)):
        node_i = nodes_chunk[i]
        X = data[:, node_i]
        # print('Rank: ', rank, " ---- node_i {}".format(node_i))
        j = 0
        for node_j in numba.prange(node_i + 1, n):
            y = data[:, node_j]
            BIC_X = BIC_numba(X, y, penalty)
            BIC_y = BIC_initial(y)
            bics_chunk[i, j] = BIC_X - BIC_y
            j += 1
            # print('Rank: ', rank, " ---- node_i {}, j {}".format(node_i, node_j))

    return bics_chunk


def init_bics_mpi(data, nodes, BICS, penalty, save_csv=False):
    n = len(nodes)

    # ===============MPI scatter==========================

    print("init_BICS_mpi rank", rank)
    if rank == 0:  # Master
        start_time = time.time()
        ops_per_chunk = np.floor((n ** 2 - n) / 2 / size)
        nodes_chunks = []
        running_count = 0
        limiter = 0
        for i in range(n):
            running_count += n - 1 - i
            if running_count >= ops_per_chunk:
                nodes_chunks.append(list(range(limiter, i + 1)))
                limiter = i + 1
                running_count = 0
        if limiter != n:
            nodes_chunks.append(list(range(limiter, n)))

        if size == 1:
            nodes_chunks = [list(range(n))]
    else:
        nodes_chunks = None

    nodes_chunk = COMM.scatter(nodes_chunks, root=0)
    print('Rank: ', rank, ', num nodes_chunk received: ', len(nodes_chunk))
    sys.stdout.flush()
    # ===============Calculate BICS (this is done in parallel in every node wi
    start_time_node = time.time()

    bics_chunk = np.zeros([len(nodes_chunk), (n - nodes_chunk[0]) - 1],
                          dtype=np.float64)

    compile = init_bics_numba_chunk_mpi(data, n, nodes_chunk, bics_chunk,
                                        penalty, compile=True)
    bics_chunk = init_bics_numba_chunk_mpi(data, n, nodes_chunk, bics_chunk,
                                           penalty, compile=False)

    split_size = int(len(bics_chunk) / 2)
    bics_chunk_a = bics_chunk[0:split_size, :]
    bics_chunk_b = bics_chunk[split_size:, :]
    """
    print('Rank: ', rank, ', bics_chunk_a computed: ', bics_chunk_a)
    print('Rank: ', rank, ', bics_chunk_b computed: ', bics_chunk_b)
    sys.stdout.flush()
    """
    end_time_node = time.time()
    time_func_node = (end_time_node - start_time_node) / 60
    print("==============")
    print(
        "======== init_bics_MPI finished for rank: {} ; Time: {}=====".format(
            rank, time_func_node))
    print("==============")
    sys.stdout.flush()

    # ===============MPI gather results==========================
    nodes_all_chunks = COMM.gather(nodes_chunk, root=0)
    bics_all_chunks_a = COMM.gather(bics_chunk_a, root=0)
    bics_all_chunks_b = COMM.gather(bics_chunk_b, root=0)
    """
    print('Rank: ', rank, ', bics_all_chunks_a received: ', bics_all_chunks_a)
    print('Rank: ', rank, ', bics_all_chunks_b received: ', bics_all_chunks_b)
    sys.stdout.flush()
    """
    if rank == 0:
        bics_all_chunks = []
        for i in range(len(bics_all_chunks_a)):
            bics_all_chunks.append(
                np.vstack((bics_all_chunks_a[i], bics_all_chunks_b[i])))
        """
        print('Rank: ', rank, ', nodes_all_chunks received: ', nodes_all_chunks)
        print('Rank: ', rank, ', bics_all_chunks received: ', bics_all_chunks)
        """
        for i, bics_part in enumerate(bics_all_chunks):
            x = nodes_all_chunks[i]
            # print("X: ", x, "bic_part shape: ", bics_part.shape)
            for i, node_x in enumerate(x):
                y_indices_chunk = np.arange(0, bics_part.shape[1] - i)
                y_indices_global = np.arange(node_x, n - 1) + 1
                """
                print("node_x: ", node_x)
                print("y_indices_chunk: ", y_indices_chunk)
                print("y_indices_global: ", y_indices_global)
                print("bics_parts_i: ", bics_part[i, :])
                """
                BICS[node_x, y_indices_global] = bics_part[i, y_indices_chunk]
                BICS[y_indices_global, node_x] = bics_part[i, y_indices_chunk]

        if save_csv:
            np.savetxt("./data/test_bics.csv", BICS, delimiter=",")

        end_time = time.time()
        time_func = (end_time - start_time) / 60
        print(
            "........................................................................")
        print(
            "........................................................................")
        print("............ init_BICS MPI finished in: {} ............".format(
            time_func))
        print(
            "........................................................................")
        print(
            "........................................................................")
        sys.stdout.flush()

    sys.stdout.flush()
    return BICS


def save_graph_as_csv(graph_name="0"):
    graph_adj_matrix = pd.DataFrame(GRAPH, columns=NODES_NAMES)

    path_graph = os.path.join(PATH_SAVE_GRAPHS, graph_name + ".csv")
    graph_adj_matrix.to_csv(path_graph)

    return 0


def orient_node_to_y(node, y):
    remove_undirected(GRAPH, node, y)
    add_children(GRAPH, node, y)

    return 0


@numba.jit(nopython=True)
def array_to_set_numba(array):
    array = array.reshape(array.shape[0], -1)

    myset = set(array.flatten())
    myset.add(INIT_ELEM_NUMBA)
    myset.remove(INIT_ELEM_NUMBA)

    return myset


@numba.jit(nopython=True)
def check_clique_numba(node_set, GRAPH_NUMBA):
    # Check if a subgraph is fully connected.

    node_set = array_to_set_numba(node_set)
    for node in node_set:
        set_adjs = adjacencies(GRAPH_NUMBA, node)
        set_all = (set_adjs.intersection(node_set)).union({node})
        if set_all < node_set:
            return False
    return True


def check_clique(node_set):
    # Check if a subgraph is fully connected.
    for node in node_set:
        if (adjacencies(GRAPH, node) & node_set).union({node}) < node_set:
            return False
    return True


def parts_of(node_set):
    # Return an iterator over all the subsets of a given set.
    for size in range(len(node_set) + 1):
        yield from combinations(node_set, size)


@numba.jit(nopython=True)
def num_comb_leq_than_numba(n, k):
    total = 0
    for i in range(k + 1):
        total += num_combinations_numba(n, i)

    return total


@numba.jit(nopython=True)
def num_combinations_numba(n, k):
    # Returns the number of sets of k elements that can be chosen from n
    # elements (any order)
    return int(
        factorial_numba(n) / (factorial_numba(k) * factorial_numba(n - k)))


@numba.jit(nopython=True)
def factorial_numba(n):
    if n < 20:
        return LOOKUP_TABLE_FACTORIAL[n]
    else:
        return math.gamma(n + 1)


# @numba.jit(nopython=True, cache=True)
def remove_inits_list_set_numba():
    lists = [adjacencies, undirecteds, parents, children]
    for list in lists:
        for set in list:
            set.remove(INIT_ELEM_NUMBA)

    return 0


# @numba.jit(nopython=True, cache=True)
def init_list_set_numba():
    lists = [adjacencies, undirecteds, parents, children]
    for list in lists:
        for set in list:
            set.add(INIT_ELEM_NUMBA)

    return 0


@numba.jit(nopython=True)
def all_permutations(A, k):
    # From https://github.com/numba/numba/issues/3599
    r = [[i for i in range(0)]]
    for i in range(k):
        r = [[a] + b for a in A for b in r if (a in b) == False]

    # Remove repetitions:
    sets = []
    for combo_r in r:
        set_r = set(combo_r)
        append = True
        for set_saved in sets:
            if set_r == set_saved:
                append = False
                break
        if append:
            sets.append(set_r)

    return sets


@numba.jit(nopython=True)
def parts_of_numba(node_set, max_size=0):
    if max_size == 0:
        max_size = len(node_set)

    max_size = min(max_size, len(node_set))
    remove_init_set_numba(node_set)

    combos = [set([INIT_ELEM_NUMBA])]
    combos[0].remove(INIT_ELEM_NUMBA)
    node_set = np.array(list(node_set), dtype=np.int64)

    if len(node_set) > 0:
        for size in range(1, max_size + 1):
            combos.extend(all_permutations(node_set, size))

    # TODO: implement this as yield (generator) instead of return the whole
    # list
    return combos


def calculate_arrows_forward(x, y, use_cache=False):
    # This checks all possible edge additions to y from x depending on the adjacent nodes.
    # Then adds them to the list of arrows if they are valid and with positive score.
    # Not sure why we do it this way
    arrows_to_insert = []

    unds_y = undirecteds(GRAPH, y)
    adjs_x = adjacencies(GRAPH, x)
    T = undirecteds(GRAPH, y) - adjacencies(GRAPH, x)
    NaYX = undirecteds(GRAPH, y) & adjacencies(GRAPH, x)

    for subset in parts_of(T):
        """
        if VISUAL_DEBUG:
            draw_net("calculate_arrows_forward x({}), y({}) - For subset of T".format(x, y), ("T (Adjs to Y but not to X)", list(T)))
            draw_net("calculate_arrows_forward x({}), y({}) - For subset of T".format(x, y), ("Subset of T (Adjs to Y but not to X)", list(subset)))
            draw_net("calculate_arrows_forward x({}), y({}) - For subset of T".format(x, y), ("NaYx (Adjs to Y and X)", list(NaYX)))
        """
        S = NaYX.union(set(subset))
        if check_clique(S):

            S = S.union(parents(GRAPH, y))
            data_y = DATA[:, y].reshape(DATA.shape[0], -1)
            X_0 = list(S)
            X = list(S) + [x]
            data_X_0 = DATA[:, X_0].reshape(DATA.shape[0], -1)
            data_X = DATA[:, X].reshape(DATA.shape[0], -1)

            if use_cache:
                bic_X_0 = get_BIC(X_0, y, data_X_0, data_y, PENALTY)
                bic_X = get_BIC(X, y, data_X, data_y, PENALTY)
            else:
                bic_X_0 = BIC_no_numba(data_X_0, data_y, PENALTY)
                bic_X = BIC_no_numba(data_X, data_y, PENALTY)

            b = bic_X - bic_X_0
            # b = local_score_diff_parents(x, y ,list(S)) #With
            # fges_alternative_github
            """
            if VISUAL_DEBUG:
                draw_net("calculate_arrows_forward x({}), y({})".format(x, y), ("S (NaYX + subset T + Pa(Y)", list(S)))
                draw_net("calculate_arrows_forward x({}), y({}) - BIC X ({})".format(x, y, round(bic_X, 2)), ("X (S + X)", list(S) + [x]))
                draw_net("calculate_arrows_forward x({}), y({}) - BIC X_0 ({})".format(x, y, round(bic_X_0, 2)), ("X_0 (S)", list(S)))
                draw_net("calculate_arrows_forward x({}), y({}) - BIC X - BIC X_0 ({})".format(x, y, round(b, 2)), ("X + X_0 + y", list(S) + [x, y]))
            """
            if b > 0:
                tuple_to_insert = ((x, y), NaYX, set(subset), b)
                arrow_insert = create_arrow_str(tuple_to_insert)
                arrows_to_insert.append(arrow_insert)

    return arrows_to_insert


def calculate_arrows_backward(x, y):
    arrows_to_insert = []
    NaYX = undirecteds(GRAPH, y) & adjacencies(GRAPH, x)
    for subset in parts_of(NaYX):
        subset = set(subset)
        S = NaYX - subset
        """
        if VISUAL_DEBUG:
            draw_net("calculate_arrows_backward x({}), y({}) - For subset of NaYX".format(x, y),
                     ("NaYx (Adjs to Y and X)", list(NaYX)))
            draw_net("calculate_arrows_backward x({}), y({}) - For subset of NaYX".format(x, y),
                     ("Subset of NaYX", list(subset)))
            draw_net("calculate_arrows_backward x({}), y({}) - For subset of NaYX".format(x, y),
                     ("S = NaYX - subset", list(S)))
        """
        if check_clique(S):
            S = S.union(parents(GRAPH, y)) - {x}
            data_y = DATA[:, y].reshape(DATA.shape[0], -1)
            X_0 = DATA[:, list(S)].reshape(DATA.shape[0], -1)
            X = DATA[:, list(S) + [x]].reshape(DATA.shape[0], -1)
            BIC_X0 = BIC_numba(X_0, data_y, PENALTY)
            BIC_X = BIC_numba(X, data_y, PENALTY)
            b = BIC_X0 - BIC_X
            """
            if VISUAL_DEBUG:
                draw_net("calculate_arrows_backward x({}), y({})".format(x, y),
                         ("S (NaYX -subset(NaYX) + Pa(Y) - x", list(S)))
                draw_net("calculate_arrows_backward x({}), y({}) - BIC X ({})".format(x, y, round(BIC_X, 2)),
                         ("X (S + X)", list(S) + [x]))
                draw_net("calculate_arrows_backward x({}), y({}) - BIC X_0 ({})".format(x, y, round(BIC_X0, 2)),
                         ("X_0 (S)", list(S)))
                draw_net("calculate_arrows_backward x({}), y({}) - BIC X_0 - BIC X ({})".format(x, y, round(b, 2)),
                         ("X + X_0 + y", list(S) + [x, y]))
             """

            if b > 0:
                tuple_to_insert = ((x, y), NaYX, subset, b)
                arrow_insert = create_arrow_str(tuple_to_insert)
                arrows_to_insert.append(arrow_insert)
    return arrows_to_insert


@numba.jit(nopython=True)
def remove_init_set_numba(set):
    set.remove(INIT_ELEM_NUMBA)

    return 0


def calculate_arrows_forward_for_mpi(node_set, GRAPH_NUMBA, BICS_NUMBA):
    if USE_MPI:
        COMM.barrier()
    sys.stdout.flush()

    # ===============MPI scatter==========================
    if rank == 0:  # Master
        start_time = time.time()
        all_new_arrs_master = []

        chunks_nodes_pos_bics = []
        chunks_node_set = np.array_split(node_set, size)
        range_nodes = np.arange(len(GRAPH_NUMBA))
        for i in range(size):
            chunk = {}
            for node in chunks_node_set[i]:
                positive_bics = [j for j in range_nodes if
                                 BICS_NUMBA[j, node] > 0]
                chunk[node] = positive_bics
            chunks_nodes_pos_bics.append(chunk)

        sys.stdout.flush()
    else:
        chunks_nodes_pos_bics = None

    chunk_nodes_pos_bics = COMM.scatter(chunks_nodes_pos_bics, root=0)
    # print('------------- calculate_arrows_forward_for_mpi Rank: {}, node_set: {}; pos_bics: {}'' ---------"'.format(rank, chunk_nodes_pos_bics.keys(), chunk_nodes_pos_bics.values()))
    sys.stdout.flush()

    all_new_arrs_in_chunk = []
    for y, positive_bics in chunk_nodes_pos_bics.items():

        # ===============Calculate arrows forward (this is done in parallel in
        if len(positive_bics) > 0:
            new_arrs_chunk = arrows_forward_numba_mpi(y, positive_bics,
                                                      GRAPH_NUMBA, DATA)

            all_new_arrs_in_chunk.extend(new_arrs_chunk)

    # ===============MPI gather results==========================
    nodes_pos_bics_all_chunks = COMM.gather(chunk_nodes_pos_bics, root=0)
    new_arrs_all_chunks = COMM.gather(all_new_arrs_in_chunk, root=0)
    if rank == 0:
        # print('Rank: ', rank, ', pos_bics_all_chunks received: ', nodes_pos_bics_all_chunks)
        # print('Rank: ', rank, ', new_arrs_all_chunks received: ', new_arrs_all_chunks)

        for new_arr_chunk in new_arrs_all_chunks:
            for arr_i in range(len(new_arr_chunk)):
                for arr_j in range(len(new_arr_chunk[arr_i])):
                    if new_arr_chunk[arr_i][arr_j][0]:
                        ar = new_arr_chunk[arr_i][arr_j][1]
                        arrow_insert = create_arrow_str(ar)
                        # TODO: We could append  the arrows to the ARROWS
                        # object
                        all_new_arrs_master.append(arrow_insert)

        end_time = time.time()
        time_func = end_time - start_time
        print(
            "............ calculate_arrows_forward_for_mpi finished in: {} ............".format(
                time_func))
        sys.stdout.flush()
    else:
        all_new_arrs_master = None

    all_new_arrs = COMM.bcast(all_new_arrs_master, root=0)

    return all_new_arrs


@numba.jit(nopython=True, parallel=True)
def arrows_forward_numba_mpi(y, positive_bics, GRAPH_NUMBA, DATA_NUMBA):
    init_tuple_numba = (False, ((0, 0), set({np.int64(0)}), set({np.int64(0)}),
                                0.0))  # First value is whether it's inserted or not
    max_size = 3
    # ---Preallocation of possible new arrows (for Numba to fill them in the p
    new_arrs_chunk = []
    n = len(positive_bics)
    for x in positive_bics:
        inside_arr = []

        T = undirecteds(GRAPH_NUMBA, y) - adjacencies(GRAPH_NUMBA, x)
        T.add(INIT_ELEM_NUMBA)
        num_subsets = min(2 ** len(T), num_comb_leq_than_numba(len(T),
                                                               min(len(T),
                                                                   max_size)))
        for xj in range(num_subsets):
            inside_arr.append(init_tuple_numba)

        new_arrs_chunk.append(inside_arr)

    for i in numba.prange(n):
        x = positive_bics[i]
        # This checks all possible edge additions to y from x depending on the adjacent nodes.
        # Then adds them to the list of arrows if they are valid and with positive score.
        # Not sure why we do it this way

        T = undirecteds(GRAPH_NUMBA, y) - adjacencies(GRAPH_NUMBA, x)
        NaYX = undirecteds(GRAPH_NUMBA, y) & adjacencies(GRAPH_NUMBA, x)

        T.add(INIT_ELEM_NUMBA)
        subsets = parts_of_numba(T, max_size)
        for j in numba.prange(len(subsets)):
            subset = subsets[j]

            S = NaYX.union(subset)
            # ----------------------------------------
            S_ = np.array(list(S) + [INIT_ELEM_NUMBA], dtype=np.int64)
            # ----------------------------------------
            if check_clique_numba(S_, GRAPH_NUMBA):
                S = S.union(parents(GRAPH_NUMBA, y))
                data_y = np.ascontiguousarray(DATA_NUMBA[:, y]).reshape(
                    DATA_NUMBA.shape[0], -1)
                # -----Numba setup---------
                array_S = np.array(list(S), dtype=np.int64)
                array_S_X = np.array(list(S) + [x], dtype=np.int64)
                # --------------------------
                X_0 = np.ascontiguousarray(DATA_NUMBA[:, array_S]).reshape(
                    DATA_NUMBA.shape[0], -1)
                X = np.ascontiguousarray(DATA_NUMBA[:, array_S_X]).reshape(
                    DATA_NUMBA.shape[0], -1)

                bic_X = BIC_numba(X, data_y, PENALTY)
                bic_X_0 = BIC_numba(X_0, data_y, PENALTY)
                b = bic_X - bic_X_0

                if b > 0:
                    # First value is whether it's inserted or not
                    new_arrs_chunk[i][j] = (True, ((x, y), NaYX, subset, b))

    return new_arrs_chunk


@numba.jit(nopython=True, parallel=True)
def calculate_arrows_forward_for_numba(node_set, GRAPH_NUMBA, BICS_NUMBA,
                                       DATA_NUMBA):
    all_new_arrs = []
    max_size = 5
    init_tuple_numba = (False, ((0, 0), set({np.int64(0)}), set({np.int64(0)}),
                                0.0))  # First value is whether it's inserted or not

    # ---Preallocation of possible new arrows (for Numba to fill them in the p
    for node_i in range(len(node_set)):
        y = node_set[node_i]
        num_nodes = np.arange(len(GRAPH_NUMBA))
        positive_bics = [j for j in num_nodes if BICS_NUMBA[j, y] > 0]
        n = len(positive_bics)

        num_inserted = 0
        new_arrs = []
        for xi in range(n):
            x = positive_bics[xi]
            inside_arr = []

            T = undirecteds(GRAPH_NUMBA, y) - adjacencies(GRAPH_NUMBA, x)
            T.add(INIT_ELEM_NUMBA)
            num_subsets = min(2 ** len(T), num_comb_leq_than_numba(len(T),
                                                                   min(len(T),
                                                                       max_size)))
            for xj in range(num_subsets):
                inside_arr.append(init_tuple_numba)

            new_arrs.append(inside_arr)
        # ----------------------------------------------------------------------------------------------------

        for i in numba.prange(n):
            x = positive_bics[i]
            # This checks all possible edge additions to y from x depending on the adjacent nodes.
            # Then adds them to the list of arrows if they are valid and with positive score.
            # Not sure why we do it this way

            T = undirecteds(GRAPH_NUMBA, y) - adjacencies(GRAPH_NUMBA, x)
            NaYX = undirecteds(GRAPH_NUMBA, y) & adjacencies(GRAPH_NUMBA, x)

            T.add(INIT_ELEM_NUMBA)
            subsets = parts_of_numba(T, max_size)
            for j in numba.prange(len(subsets)):
                subset = subsets[j]
                S = NaYX.union(subset)
                # ----------------------------------------
                S_ = np.array(list(S) + [INIT_ELEM_NUMBA], dtype=np.int64)
                # ----------------------------------------
                if check_clique_numba(S_, GRAPH_NUMBA):
                    S = S.union(parents(GRAPH_NUMBA, y))
                    data_y = np.ascontiguousarray(DATA_NUMBA[:, y]).reshape(
                        DATA_NUMBA.shape[0], -1)
                    # -----Numba setup---------
                    array_S = np.array(list(S), dtype=np.int64)
                    array_S_X = np.array(list(S) + [x], dtype=np.int64)
                    # --------------------------
                    X_0 = np.ascontiguousarray(DATA_NUMBA[:, array_S]).reshape(
                        DATA_NUMBA.shape[0], -1)
                    X = np.ascontiguousarray(DATA_NUMBA[:, array_S_X]).reshape(
                        DATA_NUMBA.shape[0], -1)

                    bic_X = BIC_numba(X, data_y, PENALTY)
                    bic_X_0 = BIC_numba(X_0, data_y, PENALTY)
                    b = bic_X - bic_X_0

                    if b > 0:
                        num_inserted += 1
                        # First value is whether it's inserted or not
                        new_arrs[i][j] = (True, ((x, y), NaYX, subset, b))

        for arr_i in range(len(new_arrs)):
            for arr_j in range(len(new_arrs[arr_i])):
                if new_arrs[arr_i][arr_j][0]:
                    ar = new_arrs[arr_i][arr_j][1]
                    all_new_arrs.append(ar)

    return all_new_arrs


def calculate_arrows_forward_for(positive_bics, y):
    new_arrows_inserted = []
    for x in positive_bics:
        result = calculate_arrows_forward(x, y)
        new_arrows_inserted.extend(result)

    return new_arrows_inserted


def create_graph():
    # matrix = get_matrix()
    BN = nx.from_numpy_matrix(GRAPH, create_using=nx.DiGraph)

    return BN


def draw_net(title="", node_set1=("", []), delay=0):
    global NUM_FIGS

    fig, ax = plt.subplots()
    fig.set_tight_layout(False)

    # Draw to check
    labels = {}
    for i, node in enumerate(NODES_NAMES):
        labels[i] = r'$' + str(node) + '$'

    BN = create_graph()

    # apt-get install -y python-pygraphviz graphviz libgraphviz-dev
    pos = nx.drawing.nx_agraph.graphviz_layout(BN, prog='dot')
    # pos = nx.spring_layout(BN)

    # Edges style:
    edge_color = []
    for u, v in BN.edges():
        if u in BN._adj[v]:  # If self cycle
            edge_color.append('b')
        else:
            edge_color.append('r')

    # Nodes style:
    node_color = []
    for node in BN.nodes():
        if node in node_set1[1]:
            node_color.append('silver')
        else:
            node_color.append('r')

    nx.draw(BN, pos, node_color=node_color, edge_color=edge_color)
    nx.draw_networkx_labels(BN, pos, labels, font_size=16)

    # Matplotlib options:
    if len(list(NODES)) > 10:
        plt.rcParams['figure.figsize'] = 9, 9
    else:
        plt.rcParams['figure.figsize'] = 6.5, 4
    plt.title(title)
    plt.legend(
        ["Silver: {} \nRed: directed\nBlue: undirected".format(node_set1[0])],
        loc=9, bbox_to_anchor=(0.5, -0.02))
    plt.axis('off')

    plots_path = "plots/"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plt.savefig(plots_path + "{}.png".format(NUM_FIGS), bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    if delay > 0:
        time.sleep(delay)

    NUM_FIGS += 1


# These variables are global to not copy them into every spawned process in the parallelized code:
# Idea taken from: https://stackoverflow.com/questions/37068981/minimize-overhead-in-python-multiprocessing-pool-with-numpy-scipy/37072511#37072511
# Another solution:
# https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing
DATA = {}
CORRCOEFS = {}
BICS = {}
BIC_TOTAL = {}
CACHE_BICS = {}
ARROWS = {}
NODES = {}
NODES_NAMES = {}
GRAPH = {}
PENALTY = {}
INIT_ELEM_NUMBA = {}
INIT_TUPLE_NUMBA = {}
VISUAL_DEBUG = {}
NUM_FIGS = {}
NUM_CACHED = 0
NUM_NOT_CACHED = 0
MEM_TRACKER = {}
USE_MPI = {}
PATH_SAVE_GRAPHS = {}
LOOKUP_TABLE_FACTORIAL = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


class FGESAlgorithm:
    def __init__(self, data, penalty=1.0, path_save_graphs='./local_graphs',
                 use_mpi=True, visual_debug=False):
        global DATA
        global CORRCOEFS
        global BICS
        global BIC_TOTAL
        global ARROWS
        global NODES
        global NODES_NAMES
        global GRAPH
        global PENALTY
        global INIT_ELEM_NUMBA
        global INIT_TUPLE_NUMBA
        global VISUAL_DEBUG
        global NUM_FIGS
        global CACHE_BICS
        global MEM_TRACKER
        global PATH_SAVE_GRAPHS

        USE_MPI = use_mpi
        PATH_SAVE_GRAPHS = path_save_graphs

        # MEM_TRACKER = tracker.SummaryTracker()

        INIT_ELEM_NUMBA = -11111
        INIT_TUPLE_NUMBA = (
            False, ((0, 0), set({np.int64(0)}), set({np.int64(0)}), 0.0))

        NUM_FIGS = 0
        DATA = data.values
        n = len(data.columns)
        NODES = list(range(n))
        NODES_NAMES = data.columns.values

        # Keep dictionaries with all children, parents and undirected edges for
        # each node
        GRAPH = np.zeros((n, n), dtype=np.int64)
        # init_MATRIX(GRAPH)

        # Each column is, given the parents of this node, add the node in the
        # row.
        BICS = np.zeros((n, n))
        PENALTY = penalty
        CACHE_BICS = {}

        if USE_MPI:
            BICS = init_bics_mpi(DATA, NODES, BICS, PENALTY, save_csv=False)
        else:
            start_time = time.time()

            # ----compile--
            # To precompile Numba functions see:
            # https://numba.pydata.org/numba-doc/dev/user/pycc.html
            bics_test = np.zeros((n, n))
            compile1 = init_bics_numba(DATA, NODES, bics_test, PENALTY,
                                       compile=True)
            BICS = init_bics_numba(DATA, NODES, BICS, PENALTY, compile=False)
            print("Numba threading layer: ", numba.threading_layer())

            end_time = time.time()
            time_func = end_time - start_time
            print(
                "........................................................................")
            print(
                "........................................................................")
            print(
                "............ init_BICS NUMBA finished in: {} ............".format(
                    time_func))
            print(
                "........................................................................")
            print(
                "........................................................................")

        if rank == 0 or not USE_MPI:
            BIC_TOTAL = 0.0

            # Add set of arrows
            ARROWS = [
                create_arrow_str(((i, j), set(), set(), BICS[i, j]))
                for i in range(n) for j in range(n) if BICS[i, j] > 0
            ]

            VISUAL_DEBUG = visual_debug
            print("ARROWS is of size: ",
                  str(aof.asizeof(ARROWS) / 1024 / 1024 / 1024), "GB")
            sys.stdout.flush()
            # MEM_TRACKER.print_diff()

    def BIC(self, X, y):
        return bic(X, y, PENALTY)

    # @profile_each_line
    def apply_meek_rules(self, node_set=None):
        # Apply Meek's rules to the graph to direct some of the edges.
        # Check if any changes happen
        changes = set()

        if not (node_set):
            node_set = NODES
        for node in node_set:
            if len(adjacencies(GRAPH, node)) < 2:
                continue
            prev_parents = parents(GRAPH, node).copy()
            for parent in prev_parents:
                # Rule 1: Away from collider
                prev_undirected = undirecteds(GRAPH, node).copy()
                for undir in prev_undirected:
                    if not (undir in adjacencies(GRAPH, parent)):
                        remove_undirected(GRAPH, node, undir)
                        add_children(GRAPH, node, undir)
                        changes.add(undir)
                        changes.add(node)

                # Rule 2: Away from cycle
                prev_children = children(GRAPH, node).copy()
                for child in prev_children:
                    if child in undirecteds(GRAPH, parent):
                        remove_undirected(GRAPH, child, parent)
                        add_children(GRAPH, parent, child)
                        changes.add(child)
                        changes.add(parent)

                # Rule 3: Double triangle
            prev_undirected = undirecteds(GRAPH, node).copy()
            kite_changes = set()
            if len(prev_undirected) < 3:
                continue
            kite_permutations = permutations(prev_undirected, 3)
            kite_permutations = (perm for perm in kite_permutations if
                                 perm[0] > perm[2])
            for node_b, node_c, node_d in kite_permutations:
                if node_b not in adjacencies(GRAPH, node_d):
                    if node_c in (
                            children(GRAPH, node_b) & children(GRAPH, node_d)):
                        try:
                            remove_undirected(GRAPH, node, node_c)
                        except KeyError:
                            if node_c in kite_changes:
                                continue
                            else:
                                raise KeyError

                        add_children(GRAPH, node, node_c)
                        kite_changes.add(node)
                        kite_changes.add(node_c)
            changes.update(kite_changes)
        return changes

    def CPDAG(self, node_set=None):
        if not (node_set):
            node_set = NODES
        edges = self.find_non_v_structures(node_set)
        changes = self.remove_orientation(edges)
        return changes, edges

    def find_non_v_structures(self, node_set=None):
        # Find the set of all edges not involved in v-structures in the graph
        edges = set()

        if not (node_set):
            node_set = NODES

        for node in node_set:
            node_set = node_set.union(parents(GRAPH, node))

        for node in node_set:
            children_node = children(GRAPH, node)
            for adj in children_node:
                v_struct = False
                parents_adj = parents(GRAPH, adj)
                parents_adj_not_node = (parents_adj - set({node}))
                for double_adj in parents_adj_not_node:
                    if not (double_adj in adjacencies(GRAPH, node)):
                        v_struct = True
                        break
                    else:
                        continue
                if not v_struct:
                    edges.add((node, adj))

        return edges

    def remove_orientation(self, edge_set):
        # Remove the orientation of a set of edges, return the nodes involved.
        changes = set()
        for node, adj in edge_set:
            add_undirected(GRAPH, node, adj)
            changes.add(adj)
            changes.add(node)
        return changes

    def local_meek(self, node_set):
        # Transform to CPDAG and correct possible cycles between the
        # V-structures of the CPDAG

        # Do-while
        cpdag_changes, not_v_struct_edges = self.CPDAG(node_set)
        changes = node_set.union(cpdag_changes)
        node_set = changes
        if VISUAL_DEBUG:
            draw_net("local_meek - CPDAG",
                     ("CPDAG changes", list(cpdag_changes)))
            # draw_net("local_meek - Meek rules", ("Meek rules changes", list(changes)))

        i = 0
        while (changes != set()):
            changes = self.apply_meek_rules(changes)
            node_set = node_set.union(changes)
            if VISUAL_DEBUG:
                draw_net("local_meek while (i={})".format(i),
                         ("Meek rules changes", list(changes)))
            i += 1

        new_non_v_structure_edges = self.find_non_v_structures(node_set)

        if new_non_v_structure_edges == not_v_struct_edges:
            return node_set

        else:
            return self.local_meek(node_set)

    def reevaluate_forward(self, node_set):
        global ARROWS

        if rank == 0 or not USE_MPI:
            # Recalculate all possible edge additions towards every node in the
            # node set.
            arrows_indexes = set([i for i, arrow in enumerate(ARROWS) if
                                  get_j(arrow) not in node_set])
            ARROWS = [arrow for i, arrow in enumerate(ARROWS) if
                      i in arrows_indexes]
            """
            if VISUAL_DEBUG:
                draw_net("reevaluate_forward - Before loop",("node_set", list(node_set)))
            """

        self.reevaluate_potential_edges(node_set)

    # @profile_each_line
    # @timecall(immediate=True)
    # @profile(immediate=True)
    def reevaluate_potential_edges(self, node_set, use_numba=True):
        global BICS
        global GRAPH
        # print("Length node_set: ",  len(node_set))

        if use_numba:
            node_set = list(node_set)

            if USE_MPI:
                all_new_arrs = calculate_arrows_forward_for_mpi(node_set,
                                                                GRAPH, BICS)
            else:
                start_time = time.time()

                all_new_arrs = calculate_arrows_forward_for_numba(node_set,
                                                                  GRAPH, BICS,
                                                                  DATA)
                for i, ar in enumerate(all_new_arrs):
                    all_new_arrs[i] = create_arrow_str(ar)

                end_time = time.time()
                time_func = end_time - start_time
                print(
                    "............ calculate_arrows_forward_for_numba RANK {} finished in: {} ............".format(
                        rank, time_func))

            # calculate_arrows_forward_for_numba.inspect_types()
        else:
            all_new_arrs = []
            for node in node_set:
                positive_bics = [j for j in range(len(NODES)) if
                                 BICS[j, node] > 0]
                """
                if VISUAL_DEBUG:
                    draw_net("reevaluate_forward - Loop: node y ({})".format(node), ("positive_bics", positive_bics))
                """
                new_arrs = calculate_arrows_forward_for(list(positive_bics),
                                                        node)
                all_new_arrs.extend(new_arrs)

        if rank == 0 or not USE_MPI:
            ARROWS.extend(all_new_arrs)

        if USE_MPI:
            COMM.barrier()

        return 0

    def check_arrow(self, arrow):
        x, y, NaYX, T, bic = get_all_vals(arrow)

        if not (y in adjacencies(GRAPH, x)):
            if NaYX == (undirecteds(GRAPH, y) & adjacencies(GRAPH, x)):
                if T <= (undirecteds(GRAPH, y) - adjacencies(GRAPH, x)):
                    return True

        return False

    def check_semi_directed_cycle(self, y, x, T, NaYX):
        union = set(T)
        if NaYX != set([]):
            union.update(NaYX)

        valid = not exists_unblocked_semi_directed_path(GRAPH, y, x, union, -1)

        return valid

    # @profile_each_line
    # @timecall(immediate=True)
    # @profile
    def forward_equivalence_search(self, stochastic=True, temperature=None,
                                   delta=None):
        global BIC_TOTAL
        global ARROWS

        if rank == 0 or not USE_MPI:
            start_time = time.time()
            num_arrows = len(ARROWS)
        else:
            num_arrows = 0

        if USE_MPI:
            num_arrows = COMM.bcast(len(ARROWS), root=0)

        i_loop = 0
        if stochastic and temperature is None:
            temperature = GRAPH.shape[0] ** 2
        while num_arrows > 0:
            print(
                "========== Arrows {}; Rank {}; i {}".format(num_arrows, rank,
                                                             i_loop))
            if USE_MPI:
                COMM.barrier()
            sys.stdout.flush()

            if VISUAL_DEBUG:
                draw_net("FES (while); Arrows: {}".format(num_arrows))

            if rank == 0 or not USE_MPI:
                arrows_keys = np.array([get_bic(ar) for ar in ARROWS],
                                       dtype=np.float64)
                max_index = np.argmax(arrows_keys)
                if stochastic:
                    # Added probability of failure
                    best_bic = np.max(arrows_keys)
                    rand_bic = np.random.choice(arrows_keys)
                    index_rand = np.where(arrows_keys == rand_bic)[0][0]
                    if temperature > 0 and np.random.random(1) <= np.exp(
                            (rand_bic - best_bic) / temperature):
                        best_edge = ARROWS.pop(index_rand)

                    else:
                        best_edge = ARROWS.pop(max_index)
                else:
                    best_edge = ARROWS.pop(max_index)
            else:
                best_edge = None

            if USE_MPI:
                best_edge = COMM.bcast(best_edge, root=0)
            x, y, NaYX, T, bic = get_all_vals(best_edge)
            print("Rank: ", rank, "best_edge: ", best_edge)
            sys.stdout.flush()

            NaYX_T = NaYX.union(T)

            if self.check_arrow(best_edge) and check_clique(
                    NaYX_T) and self.check_semi_directed_cycle(y, x, T, NaYX):
                if rank == 0 or not USE_MPI:
                    BICS[x, y] = 0
                    BICS[y, x] = 0
                    BIC_TOTAL += bic

                # Add edge
                add_children(GRAPH, x, y)
                if stochastic:
                    temperature = temperature - GRAPH.shape[
                        0] if delta is None else temperature - delta
                if VISUAL_DEBUG:
                    draw_net(
                        "FES - Add max arrow edge: x ({}), y ({})".format(x,
                                                                          y),
                        ("NaYx (Adjs to Y and X)", list(NaYX)))
                    draw_net(
                        "FES - Add max arrow edge: x ({}), y ({})".format(x,
                                                                          y),
                        ("T (Adjs to Y but not to X)", T))

                # Orient every node on T into y #Why they do this?
                self.orient_T_into_y(T, y)

                if VISUAL_DEBUG:
                    draw_net(
                        "FES arrow({}, {})- Orient every node on T into y ({})".format(
                            x, y, y), ("T (Adjs to Y but not to X)", T))

                # Undirect all nodes except unshielded colliders
                node_set = self.local_meek({x, y}.union(T))

                self.reevaluate_forward(node_set)

            if USE_MPI:
                num_arrows = COMM.bcast(len(ARROWS), root=0)
            else:
                num_arrows = len(ARROWS)

            print("BIC_TOTAL: {}; Rank {}".format(BIC_TOTAL, rank))
            sys.stdout.flush()
            i_loop += 1

            # MEM_TRACKER.print_diff()

        if rank == 0 or not USE_MPI:
            print("NUM_CACHED: ", NUM_CACHED)
            print("NUM_NOT_CACHED: ", NUM_NOT_CACHED)
            # draw_net("FES final BN graph (CPDAG)")

            end_time = time.time()
            time_func = end_time - start_time
            print(
                "....................................................................................")
            print(
                "....................................................................................")
            print(
                "............ forward_equivalence_search finished in: {} ............".format(
                    time_func))
            print(
                "....................................................................................")
            print(
                "....................................................................................")
            sys.stdout.flush()

    def orient_T_into_y(self, T, y):
        # Single process version. The overhead of the parallel version would be
        # worse than a single process in this case:

        for node in T:
            if is_undirected(GRAPH, node, y):
                remove_undirected(GRAPH, node, y)
            elif is_parent(GRAPH, node, y):
                remove_parent(GRAPH, node, y)

            add_children(GRAPH, node, y)

        return 0

    def reevaluate_backward(self, node_set):
        global ARROWS
        global ARROW_KEYS
        for node in node_set:
            parents_node = parents(GRAPH, node)
            for other in parents_node:
                arrows_indexes = set([i for i, arrow in enumerate(ARROWS) if
                                      get_i_j(arrow) != (
                                          other, node) and get_i_j(arrow) != (
                                          node, other)])
                ARROWS = [arrow for i, arrow in enumerate(ARROWS) if
                          i in arrows_indexes]
                ARROWS.extend(calculate_arrows_backward(other, node))

            undirecteds_node = undirecteds(GRAPH, node)
            for other in undirecteds_node:
                if other < node:
                    continue
                arrows_indexes = set([i for i, arrow in enumerate(ARROWS) if
                                      get_i_j(arrow) != (
                                          other, node) and get_i_j(arrow) != (
                                          node, other)])
                ARROWS = [arrow for i, arrow in enumerate(ARROWS) if
                          i in arrows_indexes]
                ARROWS.extend(calculate_arrows_backward(other, node))
                ARROWS.extend(calculate_arrows_backward(node, other))

    def backward_equivalence_search(self):
        global BIC_TOTAL

        while ARROWS != []:
            if VISUAL_DEBUG:
                draw_net("FES (while); Arrows: {}".format(len(ARROWS)))
            print("Arrows: ", len(ARROWS))
            # -------------------------
            arrows_keys = np.array([get_bic(ar) for ar in ARROWS],
                                   dtype=np.float64)
            max_index = np.argmax(arrows_keys)
            best_edge = ARROWS.pop(max_index)

            # best_edge = ARROWS.pop(0)
            # ARROW_KEYS.pop(0)
            # ⨪-----------------------
            x, y, NaYX, S, bic = get_all_vals(best_edge)

            NaYX_S = NaYX - S
            if x in children(GRAPH, y):
                continue

            if x in adjacencies(GRAPH, y) and NaYX == (
                    undirecteds(GRAPH, y) & adjacencies(GRAPH, x)):
                if check_clique(NaYX_S):
                    BIC_TOTAL += bic
                    if is_undirected(GRAPH, x, y):
                        remove_undirected(GRAPH, x, y)
                    elif is_children(GRAPH, x, y):
                        remove_children(GRAPH, x, y)

                    else:
                        continue

                    if VISUAL_DEBUG:
                        draw_net(
                            "BES - Remove max arrow edge: x ({}), y ({})".format(
                                x, y), ("NaYX (Adjs to Y and X)", list(NaYX)))
                        draw_net(
                            "BES - Remove max arrow edge: x ({}), y ({})".format(
                                x, y),
                            ("S (Subset of NaYX + Pa(y) - x)", list(S)))

                    for node in S:
                        # Orient x and y into every node
                        self.orient_T_into_y({x, y}, node)

                    if VISUAL_DEBUG:
                        draw_net(
                            "BES arrow({}, {})- Orient {}, {}, into every node NaYX-S".format(
                                x, y, x, y), ("NaYX - S", list(NaYX_S)))

                    node_set = self.local_meek({x, y}.union(NaYX_S))
                    self.reevaluate_backward(node_set)

            print("BIC_TOTAL: ", BIC_TOTAL)

        if VISUAL_DEBUG:
            draw_net("BES final BN graph (CPDAG)")

    def orient_graph(self):
        # Orients undirected edges to go from CPDAG to DAG
        for node in NODES:
            for adj in undirecteds(GRAPH, node):
                remove_undirected(GRAPH, node, adj)
                if exists_unblocked_semi_directed_path(GRAPH, node, adj, set(),
                                                       GRAPH.shape[0]):
                    add_children(GRAPH, node, adj)
                else:
                    add_children(GRAPH, adj, node)

        if VISUAL_DEBUG:
            draw_net("Oriented graph (DAG)")

    def final_bics_of_edges(self):
        # Calculates final bic of each edge in the graph
        final_bics = np.zeros(GRAPH.shape)
        for node in NODES:
            all_parents = parents(GRAPH, node)
            data_y = DATA[:, node].reshape(DATA.shape[0], -1)
            X = DATA[:, list(all_parents)].reshape(DATA.shape[0], -1)
            BIC_X = BIC_numba(X, data_y, PENALTY)
            for parent in all_parents:
                S = list(all_parents - {parent})
                X_0 = DATA[:, S].reshape(DATA.shape[0], -1)
                BIC_X0 = BIC_numba(X_0, data_y, PENALTY)
                final_bics[node, parent] = BIC_X - BIC_X0

        return final_bics
