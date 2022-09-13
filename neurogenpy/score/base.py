"""
This module provides some functions to calculate prediction scores for
structure learning methods.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np

from ..util.adjacency import undirect, get_hubs


# TODO: Decide what to do with the threshold.
def _conf_matrix_undirected(m_pred, m_true, threshold):
    """Returns the confusion matrix in the undirected case. It first makes
    the prediction matrix undirected."""

    m_pred = undirect(m_pred)
    m_true = undirect(m_true)

    m_pred = m_pred > threshold
    score_matrix = np.equal(m_pred, m_true)

    pos = int(np.sum(m_pred == 1) / 2)
    neg = int(np.sum(m_pred == 0) / 2)

    tp = int(np.sum(score_matrix[m_pred == 1] / 2))
    tn = int(np.sum(score_matrix[m_pred == 0] / 2))
    fp = pos - tp
    fn = neg - tn

    confusion = np.array([[tp, fn], [fp, tn]])
    return confusion


def _conf_matrix_directed(m_pred, m_true):
    """Returns the confusion matrix in the directed case."""

    n = m_pred.shape[0]
    m_pred = m_pred > 0

    not_diagonal = ~np.eye(n, dtype=bool)
    score_matrix = np.equal(m_pred, m_true)

    tp = np.sum(score_matrix[np.logical_and(m_true == 1, not_diagonal)])
    tn = np.sum(score_matrix[np.logical_and(m_pred == 0, not_diagonal)])
    fp = np.sum(~score_matrix[np.logical_and(m_pred == 1, not_diagonal)])
    fn = n * (n - 1) - tp - fp - tn

    confusion = np.array([[tp, fn], [fp, tn]])
    return confusion


def confusion_matrix(m_pred, m_true, undirected=False, threshold=0):
    """
    Calculates the confusion matrix for the prediction.

    Parameters
    ----------
    m_pred : numpy.array
        Prediction adjacency matrix of the graph.

    m_true : numpy.array
        Actual adjacency matrix of the graph.

    undirected : bool, default=False
        Whether the prediction has to be transformed to an undirected matrix.

    threshold : float, default=0
        In the undirected case, it represents the minimum value for an edge
        to be considered in the confusion matrix.

    Returns
    -------
    numpy.array
        The confusion matrix for the prediction.
    """

    if undirected:
        return _conf_matrix_undirected(m_pred, m_true, threshold)
    else:
        return _conf_matrix_directed(m_pred, m_true)


def confusion_hubs(m_pred, m_true, method='out_degree', hubs_threshold=0):
    """
    Calculates the confusion matrix for the set of hubs obtained with the
    prediction matrix compared to the set of hubs obtained with the actual
    matrix. Both sets are calculated using the same method.

    Parameters
    ----------
    m_pred : numpy.array
        Prediction adjacency matrix of the graph.

    m_true : numpy.array
        Actual adjacency matrix of the graph.

    method : str, default='out_degree'

    hubs_threshold : int, default = 0
        If the selected method is 'out_degree', it represents the minimum
        amount of children needed for the hubs set.

    Returns
    -------
    numpy.array
        The confusion matrix for the sets of hubs.
    """

    hubs_true = get_hubs(m_true, method=method, threshold=hubs_threshold)
    non_hubs_true = set(range(m_true.shape[0])) - set(hubs_true)
    hubs_pred = get_hubs(m_pred, method=method, threshold=hubs_threshold)
    non_hubs_pred = set(range(m_pred.shape[0])) - set(hubs_pred)
    tp_hubs = len(set(hubs_true) & set(hubs_pred))
    fp_hubs = len(hubs_pred) - tp_hubs
    tn_hubs = len(set(non_hubs_true) & set(non_hubs_pred))
    fn_hubs = len(non_hubs_pred) - tn_hubs

    return np.array([[tp_hubs, fn_hubs], [fp_hubs, tn_hubs]])


def accuracy(m_pred, m_true, undirected, threshold=0, confusion=None):
    """
    Calculates the accuracy of the prediction matrix according to the
    actual matrix.

    Parameters
    ----------
    m_pred : numpy.array
        Prediction matrix.

    m_true : numpy.array
        Actual matrix.

    undirected : bool
        Whether to make `m_pred` undirected or keep it as a directed graph
        before calculating the accuracy.

    threshold : float, default=0
        In the undirected case, it represents the minimum value for an edge
        to be considered in the confusion matrix.

    confusion : numpy.array, optional
        If it is provided, the function returns the accuracy according to the
        introduced confusion matrix ignoring the rest of the arguments.

    Returns
    -------
    float
        The accuracy of the prediction.
    """

    if confusion is None:
        confusion = confusion_matrix(m_pred, m_true, undirected=undirected,
                                     threshold=threshold)
    return np.trace(confusion) / np.sum(confusion)


def f1_score(m_pred, m_true, undirected, threshold, confusion=None):
    """
    Calculates the F1 score of the prediction matrix according to the actual
    matrix.

    Parameters
    ----------
    m_pred : numpy.array
        Prediction matrix.

    m_true : numpy.array
        Actual matrix.

    undirected : bool
        Whether to make `m_pred` undirected or keep it as a directed graph
        before calculating the score.

    threshold : float, default=0
        In the undirected case, it represents the minimum value for an edge
        to be considered in the confusion matrix.

    confusion : numpy.array, optional
        If it is provided, the function returns the F1 score according to the
        introduced confusion matrix ignoring the rest of the arguments.

    Returns
    -------
    float
        The F1 score of the prediction.
    """

    if confusion is None:
        confusion = confusion_matrix(m_pred, m_true, undirected=undirected,
                                     threshold=threshold)
    precision = confusion[0, 0] / np.sum(confusion[:, 0])
    recall = confusion[0, 0] / np.sum(confusion[0, :])
    return 2 * recall * precision / (
            recall + precision) if recall + precision > 0 else 0


def mcc_score(m_pred, m_true, undirected, threshold, confusion=None):
    """
    Calculates the Matthews correlation coefficient of the prediction matrix
    according to the actual matrix.

    Parameters
    ----------
    m_pred : numpy.array
        Prediction matrix.

    m_true : numpy.array
        Actual matrix.

    undirected : bool
        Whether to make `m_pred` undirected or keep it as a directed graph
        before calculating the coefficient.

    threshold : float, default=0
        In the undirected case, it represents the minimum value for an edge
        to be considered in the confusion matrix.

    confusion : numpy.array, optional
        If it is provided, the function returns the Matthews correlation
        coefficient according to the introduced confusion matrix ignoring the
        rest of the arguments.

    Returns
    -------
    float
        The Matthews correlation coefficient of the prediction.
    """

    if confusion is None:
        confusion = confusion_matrix(m_pred, m_true, undirected=undirected,
                                     threshold=threshold)
    n = m_pred.shape[0] ** 2
    tp, fn, fp, tn = confusion.flatten()
    p = (tp + fn) / n
    s = (tp + fp) / n
    return (tp / n - s * p) / np.sqrt(p * s * (1 - s) * (1 - p))
