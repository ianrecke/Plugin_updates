"""
Naive Bayes structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class NB(LearnStructure):
    """
    Naive Bayes structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous'}
        Type of the data introduced.

    class_variable : str
        Name of the class variable.

    Raises
    ------
    ValueError
        If the data is not discrete or `class_variable` is not set.
    """

    def __init__(self, df, data_type, *, class_variable):
        if data_type != 'discrete':
            raise ValueError(
                'This algorithm is only available for discrete data.')
        if not class_variable:
            raise ValueError(
                'To run this classifier, you must supply a class variable.')

        super().__init__(df, data_type)
        self.class_variable = class_variable

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`naive_bayes`

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == 'neurogenpy':
            return self._run_neurogenpy()
        elif env == 'bnlearn':
            return self._run_nb_bnlearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_nb_bnlearn(self):
        explanatory = list(self.df.columns.values)

        explanatory.remove(self.class_variable)
        explanatory = np.array(explanatory)

        return self._run_bnlearn(importr('bnlearn').naive_bayes,
                                 training=self.class_variable,
                                 explanatory=explanatory)

    def _run_neurogenpy(self):

        return None
