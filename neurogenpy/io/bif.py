"""
BIF input/output module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from pgmpy.readwrite import BIFReader, BIFWriter

from .bnio import BNIO
from ..utils.data_structures import pgmpy2nx, nx2pgmpy


class BIF(BNIO):
    """
    BIF input/output class.
    """

    def read_file(self, file_path):
        """
        Returns the graph and parameters stored in a BIF file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored.

        Returns
        -------
        (networkx.DiGraph, dict)
            The structure of the network and its parameters.
        """

        bif_reader = BIFReader(path=file_path)
        bn_pgmpy = bif_reader.get_model()

        return pgmpy2nx(bn_pgmpy)

    def write_file(self, file_path, bn):
        """
        Writes a Bayesian network in a BIF file.

        Parameters
        ----------
        file_path : str
            Path of the file to store the Bayesian network in.

        bn : BayesianNetwork
            Bayesian network to be stored.
        """

        bn_pgmpy = nx2pgmpy(bn.graph, bn.parameters)

        bif_writer = BIFWriter(model=bn_pgmpy)
        for param in bn_pgmpy.cpds:
            bif_writer.variable_states[param.variable] = param.state_names[
                param.variable]

        bif_writer.write_bif(file_path)
