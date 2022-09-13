NeurogenPy
==========

.. include:: imgs/images.rst

..
 TODO: add badges

.. description-start

.. image:: https://readthedocs.org/projects/neurogenpy/badge/?version=latest
  :target: https://neurogenpy.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

**NeurogenPy** is a Python package for working with Bayesian networks. It is focused on the analysis of gene expression data and learning of gene regulatory networks, modeled as Bayesian networks. For that reason, at the moment, the Gaussian and fully discrete cases are the only supported.

The package provides different structure learning algorithms, parameters estimation and input/output formats. For some of them, already existing implementations have been used, being `bnlearn <https://www.bnlearn.com/>`_, `pgmpy <https://pgmpy.org/>`_, `networkx <https://networkx.org/>`_ and `igraph <https://igraph.org/python/>`_ the most relevant used packages. Particularly, we provide an implementation of the **FGES-Merge** algorithm :cite:`fges_merge`.

This project has been conceived to be included as a plugin in the `EBRAINS interactive atlas viewer <https://interactive-viewer.apps.hbp.eu/>`_, but it may be used for other purposes.

NeurogenPy has been developed from **BayeSuites** :cite:`bayesuites`, which is included in the already existing web framework `NeuroSuites <https://neurosuites.com/>`_.

.. description-end


Installation
------------

.. installation-start

``neurogenpy`` can be installed with ``pip`` using the command:

.. code-block:: bash

 pip install git+https://github.com/javiegal/neurogenpy.git@master

As it makes use of R's packages ``bnlearn`` and ``sparsebn`` via `rpy2 <https://rpy2.github.io/>`_, you should have installed an R compatible version. For any installation issues related to this, we recommend to check `rpy2 documentation <https://rpy2.github.io/doc.html>`_.
If ``bnlearn`` or ``sparsebn`` are not installed, the package does it via ``rpy2``.

.. installation-end

Usage
-----
`The documentation <https://neurogenpy.readthedocs.io/en/latest/>`_ is available in Read the Docs and includes a user guide.


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
