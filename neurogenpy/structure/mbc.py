"""
Multidimensional Bayesian network classifier structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class MBC(LearnStructure):
    """
    Multidimensional Bayesian network classifier class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    class_variables : list

    Raises
    ------
    ValueError
        If `class_variables` is empty.
    """

    def __init__(self, df, data_type, *, class_variables=None):
        if not class_variables:
            raise ValueError(
                'To run this classifier, you must supply at least one class '
                'feature.')

        super().__init__(df, data_type)
        self.class_variables = class_variables

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`hc`

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
            return self._run_mbc_bnlearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_mbc_bnlearn(self):
        features = list(
            set(self.df.columns.values) - set(self.class_variables))

        # Black list of arcs from features to classes
        blacklist = pd.DataFrame(columns=['from', 'to'])
        blacklist['from'] = features * len(self.class_variables)
        blacklist['to'] = np.repeat(self.class_variables, [len(features)],
                                    axis=0)

        # Learn MBC structure
        return self._run_bnlearn(importr('bnlearn').hc, blacklist=blacklist)

    def _run_neurogenpy(self):

        return None
