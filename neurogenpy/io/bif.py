"""
BIF input/output module. It uses `pgmpy` functionality.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from pgmpy.readwrite import BIFReader, BIFWriter

from .bnio import BNIO
from ..util.data_structures import pgmpy2nx, nx2pgmpy


# TODO: Look for BIF object representation.
class BIF(BNIO):
    """
    BIF (Bayesian Interchange Format) input/output class.
    It uses `pgmpy` BIF reading and writing capabilities :cite:`bif`.
    """

    def convert(self, io_object):
        """
        Creates the graph structure object and parameters dictionary from the
        BIF object received.

        Parameters
        ----------
        io_object :

        Returns
        -------
        (networkx.DiGraph, dict)
            The graph structure loaded and the parameters.
        """
        pass

    def generate(self):
        """
        Generates the BIF object that represents the network.

        Returns
        -------
            The object that represents the network.
        """
        pass

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

    def write_file(self, file_path):
        """
        Writes a Bayesian network in a BIF file.

        Parameters
        ----------
        file_path : str
            Path of the file to store the Bayesian network in..
        """

        bn_pgmpy = nx2pgmpy(self.bn.graph, self.bn.parameters)

        bif_writer = BIFWriter(model=bn_pgmpy)
        for param in bn_pgmpy.cpds:
            bif_writer.variable_states[param.variable] = param.state_names[
                param.variable]

        bif_writer.write_bif(file_path)
