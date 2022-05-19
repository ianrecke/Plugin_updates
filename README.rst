NeurogenPy
==========

.. include:: imgs/images.rst

..
 TODO: add badges

Description
-----------
.. description-start

**NeurogenPy** is a Python package for working with Bayesian networks. It is focused on the analysis of gene expression data and learning of gene regulatory networks, modeled as Bayesian networks. For that reason, at the moment, the continuous case is the only one that is fully provided. However, discrete inference will be implemented in the near future.

The package provides different structure learning algorithms, parameters estimation and input/output formats. For some of them, already existing implementations have been used, being `bnlearn <https://www.bnlearn.com/>`_, `pgmpy <https://pgmpy.org/>`_, `networkx <https://networkx.org/>`_ and `igraph <https://igraph.org/python/>`_ the most relevant used packages. Particularly, we provide an implementation of the **FGES-Merge** algorithm :cite:`fges_merge`.

This project has been conceived to be included as a plugin in the `EBRAINS interactive atlas viewer <https://interactive-viewer.apps.hbp.eu/>`_, but it may be used for other purposes.

NeurogenPy has been developed from **BayeSuites** :cite:`bayesuites`, which is included in the already existing web framework `NeuroSuites <https://neurosuites.com/>`_.

.. description-end

`The documentation <https://neurogenpy.readthedocs.io/en/latest/>`_ is available in Read the Docs.

Installation
------------
``neurogenpy`` can be installed with ``pip`` using the command:

.. code-block:: bash

 pip install git+https://github.com/javiegal/neurogenpy.git@main
 
As it makes use of R's package ``bnlearn`` via `rpy2 <https://rpy2.github.io/>`_, you should have installed an R compatible version. For any installation issues related to this, we recommend to check `rpy2 documentation <https://rpy2.github.io/doc.html>`_.
If ``bnlearn`` is not installed, the package does it via ``rpy2``.

Usage
-----
.. usage-start

The use of the package is focused on the :class:`~neurogenpy.models.bayesian_network.BayesianNetwork` class.

#. If you already have a graph structure and the network parameters (or joint probability distribution) in the right formats, it is posible to use the constructor for building the network. See :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.fit` and :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.load` methods for other ways of creating Bayesian networks.

   .. code-block:: py

    from neurogenpy import BayesianNetwork, GaussianNode
    from networkx import DiGraph
    graph = DiGraph()
    graph.add_nodes_from([1, 2])
    graph.add_edges_from([(1, 2)])
    ps = {1: GaussianNode(0, 1, [], []), 2: GaussianNode(0, 1, [1], [0.8])}
    bn = BayesianNetwork(graph=graph, parameters=ps)

#. Learning the structure and parameters of a Bayesian network from the data in a CSV file.

    - Set the structure and parameter learning methods with arguments:

      .. code-block:: py

        import pandas as pd
        from neurogenpy import BayesianNetwork
        df = pd.read_csv('file.csv')
        bn = BayesianNetwork().fit(df, estimation='mle', algorithm='PC')

    - Additional parameters for the structure learning or parameters estimation algorithm can be provided too:

      .. code-block:: py

        bn = BayesianNetwork()
        bn = bn.fit(df, algorithm='FGESMerge', penalty=45)

    - Instance a particular :class:`~neurogenpy.structure.learn_structure.LearnStructure` or :class:`~neurogenpy.parameters.learn_parameters.LearnParameters` subclass:

      .. code-block:: py

        from neurogenpy import BayesianNetwork, FGESMerge, GaussianMLE

#. Loading an already saved Bayesian network:

    - BIF file (pgmpy): it loads the graph structure and the parameters of the network.

      .. code-block:: py

        from neurogenpy import BayesianNetwork
        bn = BayesianNetwork().load('bn.bif')

    - GEXF (graph stored with .gexf extension), CSV (adjacency matrix stored with '.csv') or parquet (adjacency matrix stored with '.gzip' extension) file, it only loads the graph structure of the network. The parameters can be learnt according to this graph and the initial data.

      .. code-block:: py

        import pandas as pd
        from neurogenpy import BayesianNetwork
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

This project has been developed by the `Computational Intelligence Group (CIG) <http://cig.fi.upm.es/>`_ of `Universidad Politécnica de Madrid (UPM) <https://www.upm.es/>`_. `Javier Gallego Gutiérrez <https://github.com/javiegal/>`_ has developed the package based on `NeuroSuites <https://neurosuites.com/>`_, done by `Mario Michiels Toquero <https://www.linkedin.com/in/mario-michiels-toquero-02ab9a137/>`_ and `Hugo Nugra <https://www.linkedin.com/in/hugonugramadero/>`_.

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
