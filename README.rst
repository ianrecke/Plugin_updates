NeurogenPy
==========

.. include:: imgs/images.rst

..
 TODO: add badges

Description
-----------
.. description-start

NeurogenPy is a Python package for working with Bayesian networks. It is focused on the analysis of gene expression data and learning of gene
regulatory networks, modeled as Bayesian networks. It has been developed from the already existing web framework  `NeuroSuites <https://neurosuites.com/>`_.

.. description-end

Installation
------------

Usage
-----
.. usage-start

The use of the package is focused on the :class:`~neurogenpy.models.bayesian_network.BayesianNetwork` class.

#. If you already have a graph structure and the network parameters (or joint probability distribution) in the right formats, it is posible to use the constructor for building the network. See :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.fit` and :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.load` methods for other ways of creating Bayesian networks.

   .. code-block:: py

    from neurogenpy.models.bayesian_network import BayesianNetwork
    from networkx import DiGraph
    graph = DiGraph([1, 2])
    ps = {1: GaussianNode(0, 1, [], []), 2: GaussianNode(0, 1, [1], [0.8])}
    bn = BayesianNetwork(graph=graph, parameters=ps)

#. Learning the structure and parameters of a Bayesian network from the data in a CSV file.

    - Set the structure and parameter learning methods with arguments:

      .. code-block:: py

        import pandas as pd
        from neurogenpy.models.bayesian_network import BayesianNetwork
        df = pd.read_csv('file.csv')
        bn = BayesianNetwork().fit(df, estimation='mle', algorithm='PC')

    - Additional parameters for the structure learning or parameters estimation algorithm can be provided too:

      .. code-block:: py

        bn = BayesianNetwork()
        bn = bn.fit(df, algorithm='FGESMerge', penalty=45)

    - Instance a particular :class:`~neurogenpy.structure.learn_structure.LearnStructure` or :class:`~neurogenpy.parameters.learn_parameters.LearnParameters` subclass:

      .. code-block:: py

        from neurogenpy.models.bayesian_network import BayesianNetwork
        from neurogenpy.structure.fges_merge import FGESMerge
        from neurogenpy.parameters.gaussian_mle import GaussianMLE

#. Loading an already saved Bayesian network:

    - BIF file (pgmpy): it loads the graph structure and the parameters of the network.

      .. code-block:: py

        from neurogenpy.models.bayesian_network import BayesianNetwork
        bn = BayesianNetwork().load('bn.bif')

    - GEXF (graph stored with .gexf extension), CSV (adjacency matrix stored with '.csv') or parquet (adjacency matrix stored with '.gzip' extension) file, it only loads the graph structure of the network. The parameters can be learnt according to this graph and the initial data.

      .. code-block:: py

        import pandas as pd
        from neurogenpy.models.bayesian_network import BayesianNetwork
        bn = BayesianNetwork.load('bn.gexf')
        df = pd.read_csv('file.csv')
        bn = bn.fit(df, estimation='mle', skip_structure=True)

.. usage-end

Support
-------

Contributing
------------

Authors
-------
.. authors-start

This project has been developed by the `Computer Intelligence Group (CIG) <http://cig.fi.upm.es/>`_ of `Universidad Politécnica de Madrid (UPM) <https://www.upm.es/>`_.

|cig| |upm|

.. authors-end

Acknowledgements
---------------
.. acknowledgements-start

This project has received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

|hbp|

.. acknowledgements-end

License
-------

.. |cig| image:: imgs/cig.png
    :target: http://cig.fi.upm.es/

.. |upm| image:: imgs/upm.jpg
    :target: https://www.upm.es/

.. |hbp| image:: imgs/logo_hbp.png
    :target: https://www.humanbrainproject.eu/en/
