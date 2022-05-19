import unittest

import networkx as nx
import pandas as pd
from networkx.utils.misc import graphs_equal

from neurogenpy.structure.hill_climbing import HillClimbing


# The purpose of this test case is to check the correct use of bnlearn. As
# LearnStructure is an abstract class, we did it with one of its subclasses.
class TestLearnStructure(unittest.TestCase):
    def test_run_bnlearn(self):
        # TODO: Find a Hill Climbing example and compare
        actual_graph = nx.DiGraph
        df = pd.DataFrame
        hc_learn = HillClimbing(df, "continuous")

        learnt_graph = hc_learn.run(env='bnlearn')
        assert graphs_equal(learnt_graph, actual_graph)


if __name__ == '__main__':
    unittest.main()
