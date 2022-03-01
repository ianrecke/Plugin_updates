import os
import tempfile
from pgmpy import readwrite as pgmpy_io

from .IOBN import IOBN, pgmpy_model_to_nx_graph_parameters, \
    nx_graph_parameters_to_pgmpy_model


class BIF(IOBN):
    def __init__(self):
        super(BIF, self).__init__()
        self.extension = "bif"

    def import_file(self, file_path):
        bn_pgmpy = pgmpy_io.BIFReader(path=file_path)
        bn_model_pgmpy = bn_pgmpy.get_model()
        parameter_states = bn_pgmpy.variable_states
        bn_graph, bn_parameters = pgmpy_model_to_nx_graph_parameters(
            bn_model_pgmpy)
        # bn_model_pybn = pyBN.read_bn(path_file)

        for parameter in bn_model_pgmpy.cpds:
            parameter.state_names = {
                parameter.variable: parameter_states[parameter.variable]}

        bn_model_original_import = bn_model_pgmpy

        return bn_graph, bn_parameters, bn_model_original_import

    def export_file(self, file_path, bn):
        file_path = file_path + "." + self.extension
        file_path = os.path.join(tempfile.gettempdir(), file_path)

        bn_model_pgmpy = nx_graph_parameters_to_pgmpy_model(bn.graph,
                                                            bn.parameters)

        bn_pgmpy = pgmpy_io.BIFWriter(model=bn_model_pgmpy)
        for parameter in bn_model_pgmpy.cpds:
            bn_pgmpy.variable_states[parameter.variable] = \
                parameter.state_names[parameter.variable]

        bn_pgmpy.write_bif(file_path)

        file_exported_memory = {}
        with open(file_path) as file_exported:
            file_exported_memory = file_exported.read()

        os.remove(file_path)  # Remove output file because now is in memory

        return file_exported_memory
