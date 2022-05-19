import unittest

import networkx as nx
import pandas as pd

from neurogenpy.parameters.gaussian_mle import GaussianMLE


class TestGaussianMLE(unittest.TestCase):
    def test_run(self):
        # TODO: Find an example with data, learnt graph and parameters.
        expected_estimation = {}
        df = pd.DataFrame
        graph = nx.DiGraph()
        mle = GaussianMLE(df, graph)

        estimation = mle.run()
        self.assertDictEqual(estimation, expected_estimation)


if __name__ == '__main__':
    unittest.main()
