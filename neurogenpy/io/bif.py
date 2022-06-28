"""
BIF input/output module. It uses `pgmpy`.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

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
        bif_reader = BIFReader(string=io_object)
        bn_pgmpy = bif_reader.get_model()

        return pgmpy2nx(bn_pgmpy)

    def generate(self):
        """
        Generates the BIF string that represents the network.

        Returns
        -------
            The BIF string representation.
        """

        return self._get_writer().__str__()

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

        writer = self._get_writer()

        writer.write_bif(file_path)

    def _get_writer(self):
        """Returns the BIFWriter for the network."""

        bn_pgmpy = nx2pgmpy(self.bn.graph, self.bn.parameters)

        bif_writer = BIFWriter(model=bn_pgmpy)
        for param in bn_pgmpy.cpds:
            bif_writer.variable_states[param.variable] = param.state_names[
                param.variable]
        return bif_writer
