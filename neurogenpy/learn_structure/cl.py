from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class CL(LearnStructure):
    """Chow Liu structure learning class.
    """

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'bnlearn', 'neurogenpy'}, default='bnlearn'
            Environment used to run the algorithm.

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
            return self._run_bnlearn(importr('bnlearn').chow_liu)
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):
        return None
