"""Utilities to calculate prediction scores for structure learning methods."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

import graph_utils


def compare_kl_divergences(dists_computed):
    results = {}

    for net0, vals0 in dists_computed.items():
        mu0 = vals0[0]
        sigma0 = vals0[1]
        results[net0] = {}
        for net1, vals1 in dists_computed.items():
            if net0 != net1:
                mu1 = vals1[0]
                sigma1 = vals1[1]
                score_kl_div = kl_div(mu0, sigma0, mu1, sigma1)
                results[net0][net1] = score_kl_div

    results_pd = pd.DataFrame(results)

    return results_pd


def kl_univariate_div(mu0, sigma0, mu1, sigma1):
    kl_divergence = np.log(sigma1 / sigma0) + (
            sigma0 ** 2 + (mu0 - mu1) ** 2) / (2 * sigma1 ** 2) - 0.5
    return kl_divergence


def kl_div(mu0, sigma0, mu1, sigma1):
    # Return the Kullback Liebler divergence of two multivariate gaussians
    # Do cholesky decomposition to ensure numerical stability. Calculate necessary inverse.
    # Ensure the matrices are lower triangular.
    dim = mu0.shape[0]
    dim_idx = list(range(dim))

    chol0 = np.linalg.cholesky(sigma0)
    if not np.allclose(chol0, np.tril(chol0)):
        chol0 = chol0.T

    chol1 = np.linalg.cholesky(sigma1)
    if not np.allclose(chol1, np.tril(chol1)):
        chol1 = chol1.T

    inv_chol1 = np.linalg.inv(chol1)

    trace = np.sum(np.square(np.dot(inv_chol1, chol0)))
    half_density = np.dot(inv_chol1, (mu1 - mu0))
    density = np.dot(half_density.T, half_density)

    determinant = 2 * np.sum(np.log(chol1[dim_idx, dim_idx])) - 2 * np.sum(
        np.log(chol0[dim_idx, dim_idx]))

    return np.round(1 / 2 * (trace + density - dim + determinant), 5)


def hellinger(mu0, sigma0, mu1, sigma1):
    # Compute Hellinger distance between two multivariate distributions
    dim = mu0.shape[0]
    dim_idx = list(range(dim))

    chol0 = np.linalg.cholesky(sigma0)
    chol1 = np.linalg.cholesky(sigma1)
    chol_sum = np.linalg.cholesky(sigma0 + sigma1)
    chol_sum_inv = np.linalg.inv(chol_sum)

    density = -1 / 8 * np.sum(np.square(np.dot(chol_sum_inv, mu0 - mu1)))

    # Weird way of computing the determinant to avoid underflow since dim is huge
    det0 = np.sqrt(chol0)[dim_idx, dim_idx].tolist()
    det1 = np.sqrt(chol1)[dim_idx, dim_idx].tolist()
    det_sum = chol_sum[dim_idx, dim_idx].tolist()
    determinant = np.prod(
        np.array([i * j / k for i, j, k in zip(det0, det1, det_sum)]))

    # Final result
    hellinger_distance = 1 - determinant * np.exp(density)

    return hellinger_distance


def conf_matrix_undirected(pred_matrix, true_matrix, threshold):
    graph_utils.undirect_all_edges(pred_matrix)

    # Calculates tp, tn, fp, fn
    pred_matrix = pred_matrix > threshold
    n = pred_matrix.shape[0]
    score_matrix = np.equal(pred_matrix, true_matrix)
    # tp
    all_p = int(np.sum(pred_matrix == 1) / 2)
    all_n = np.sum(pred_matrix == 0)

    tp = np.sum(score_matrix[pred_matrix == 1])
    tn = np.sum(score_matrix[pred_matrix == 0])
    fp = all_p - tp
    fn = all_n - tn

    confusion = np.array([[tp, fn], [fp, tn]])
    shd = 0  # np.sum(np.abs(pred_matrix - true_matrix))
    return confusion, shd


def conf_matrix(pred_matrix, true_matrix, threshold):
    # Calculates tp, tn, fp, fn
    n = pred_matrix.shape[0]
    # pred_matrix = pred_matrix > threshold
    pred_matrix = graph_utils.filter_edges_by_threshold(pred_matrix,
                                                        threshold=0)
    pred_matrix = pred_matrix > 0

    score_matrix = np.equal(pred_matrix, true_matrix)

    # tp
    tp = np.sum(score_matrix[pred_matrix == 1])
    tn = np.sum(score_matrix[pred_matrix == 0])
    fp = np.sum(~score_matrix[pred_matrix == 1])
    fn = n * n - tp - fp - tn

    confusion = np.array([[tp, fn], [fp, tn]])
    shd = np.sum(
        np.abs(pred_matrix.astype(np.int64) - true_matrix.astype(np.int64)))
    return confusion, shd


def combined_scores(pred_matrix, true_matrix, undirected=False, threshold=0,
                    list_hubs_true=[]):
    if undirected:
        confusion, shd = conf_matrix_undirected(pred_matrix, true_matrix,
                                                threshold)
    else:
        confusion, shd = conf_matrix(pred_matrix, true_matrix, threshold)

    n = pred_matrix.shape[0] ** 2
    tp, fn, fp, tn = confusion.flatten()
    accuracy = np.trace(confusion) / np.sum(confusion)
    precision = confusion[0, 0] / np.sum(confusion[:, 0])
    recall = confusion[0, 0] / np.sum(confusion[0, :])  # True positive rate
    fpr = confusion[1, 0] / np.sum(confusion[1, :])
    f1 = 2 * recall * precision / (
            recall + precision) if recall + precision > 0 else 0
    true_matrix = np.asarray(true_matrix)
    pred_matrix = np.asarray(pred_matrix)
    roc = roc_auc_score(true_matrix.flatten(), pred_matrix.flatten())
    prc = average_precision_score(true_matrix.flatten(), pred_matrix.flatten())
    p = (tp + fn) / n
    s = (tp + fp) / n
    mcc = (tp / n - s * p) / np.sqrt(p * s * (1 - s) * (1 - p))

    list_hubs_t_false = set(range(true_matrix.shape[0])) - set(list_hubs_true)
    list_hubs_graph = graph_utils.get_list_hubs(pred_matrix, percentile=None,
                                                method="out_degree",
                                                threshold_out_degree=2)
    list_hubs_graph_false = set(range(pred_matrix.shape[0])) - set(
        list_hubs_graph)
    tp_hubs = len(set(list_hubs_true) & set(list_hubs_graph))
    fp_hubs = len(list_hubs_graph) - tp_hubs
    tn_hubs = len(set(list_hubs_t_false) & set(list_hubs_graph_false))
    fn_hubs = len(list_hubs_graph_false) - tn_hubs

    return tp, fn, fp, tn, accuracy, precision, recall, f1, shd, mcc, roc, prc, tp_hubs, fp_hubs, tn_hubs, fn_hubs


def autoreshape_graph(graph, nodes_names):
    full_graph = pd.DataFrame(np.zeros((len(nodes_names), len(nodes_names))),
                              columns=nodes_names)

    cols = graph.columns.values.tolist()
    indices = [nodes_names.index(gene) for gene in cols]
    full_graph.iloc[indices, indices] = graph.values

    return full_graph


def run_scores_structures(graphs):
    pred_graphs = []
    for i, graph in enumerate(graphs):
        numpy_matrix = nx.to_numpy_matrix(graph)
        if i == 0:
            numpy_matrix = np.asarray(numpy_matrix > 0).astype(np.int64)
            real_graph = numpy_matrix
        pred_graphs.append(numpy_matrix)

    scores = np.zeros((len(pred_graphs), 12))
    threshold = 0
    for i, graph in enumerate(pred_graphs):
        score = combined_scores(graph, real_graph, False, threshold)
        scores[i, :] = score[0:12]

    results = pd.DataFrame(data=np.round(scores, 10),
                           columns=["TP", "FN", "FP", "TN", "Accuracy",
                                    "Precision", "Recall", "F-score", "SHD",
                                    "MCC", "AUROC", "AUPRC"])
    results.sort_values(by=["MCC"], inplace=True, ascending=False)
    return results


def run_scores():
    import os
    from graph_utils import adj_list_to_matrix

    path = "local_graphs/DREAM tests/network_4__s_cerevisiae/"
    # path = "local_graphs/DREAM tests/Net 3 global/"
    pred_graphs = []
    transformed = []
    names = []
    names_t = []
    undirected = False
    # nodes_names = set()
    real_graph_is_tsv = False
    nodes_names = None
    for filename in os.listdir(path):
        name, extension = os.path.splitext(filename)
        if extension == ".tsv":
            if name == "True":
                real_graph = pd.read_csv(path + filename, header=None,
                                         sep='\t')
                real_graph_is_tsv = True
            elif name == "nodes_names":
                nodes_names_pd = pd.read_csv(path + filename, header=None,
                                             sep='\t')
                nodes_names = list(nodes_names_pd.iloc[1:, 0].values)
            else:
                transformed.append(
                    pd.read_csv(path + filename, header=None, sep='\t'))
                names_t.append(filename)
        else:
            if extension == ".csv":
                graph = pd.read_csv(path + filename, na_filter=False,
                                    dtype=np.float64, low_memory=False).iloc[:,
                        1:]
                if name == "True":
                    real_graph = graph
            elif extension == ".gzip":
                graph = pd.read_parquet(path + filename,
                                        engine="fastparquet").astype(
                    np.float64).iloc[:, 1:]
            else:
                continue
            pred_graphs.append(graph)
            # node_names.update(set(graph.columns.values))
            names.append(filename)
    if nodes_names is None:
        nodes_names = pred_graphs[0].columns.values.tolist()
    if real_graph_is_tsv:
        real_graph, _ = adj_list_to_matrix(real_graph, nodes_names)
    real_graph_df = autoreshape_graph(real_graph, nodes_names)
    real_graph = real_graph_df.values
    # real_graph_df.to_csv(path+"./True.csv")

    transformed = [adj_list_to_matrix(trans, nodes_names)[0] for trans in
                   transformed]
    list_hubs_true = graph_utils.get_list_hubs(real_graph, percentile=None,
                                               method="out_degree",
                                               threshold_out_degree=2)

    scores = np.zeros((len(pred_graphs) + 1, 16))

    for i, graph in enumerate(pred_graphs):
        try:
            print("Scoring net: ", i)

            threshold = 0  # 1/(len(nodes_names)) #0.5
            graph = autoreshape_graph(graph, nodes_names)
            scores[i, :] = combined_scores(graph.values, real_graph > 0,
                                           undirected, threshold,
                                           list_hubs_true)
        except Exception as e:
            print("exception in {}: {}. Graph shape: {}".format(names[i], e,
                                                                graph.shape))
    scores[-1, :] = combined_scores(
        np.ones((len(nodes_names), len(nodes_names))) * 1 / len(nodes_names),
        real_graph > 0, undirected, 0)
    names.append("Random")
    """
    for j, graph in enumerate(transformed):
        print("Scoring net: ", i+j)
        if "local_global" in names_t[j]:
            threshold = 0
        else:
            threshold = 1/len(nodes_names)
        indices = [nodes_names.index(gene) for gene in graph.columns]
        graph = graph.iloc[:, indices]
        graph = graph.iloc[indices, :]
        scores[i, :] = combined_scores(graph.values, real_graph > 0, undirected, threshold)
    """
    results = pd.DataFrame(data=np.round(scores, 10), index=names,
                           columns=["TP", "FN", "FP", "TN", "Accuracy",
                                    "Precision", "Recall", "F-score", "SHD",
                                    "MCC", "AUROC", "AUPRC", "TP_hubs",
                                    "FP_hubs",
                                    "TN_hubs", "FN_hubs"])
    results.sort_values(by=["MCC"], inplace=True, ascending=False)

    print(results.to_string())

    file_name = "../results.csv"
    if undirected:
        file_name = "../results_undirected.csv"
    results.to_csv(path + file_name)


if __name__ == "__main__":
    run_scores()
