pyjuice.nodes
=============

.. currentmodule:: pyjuice.nodes

Nodes
-----

.. autosummary::
    :toctree: generated
    :nosignatures:

    InputNodes
    ProdNodes
    SumNodes
    SparseProdNodes
    SparseSumNodes

The two ``Sparse*`` subclasses are dispatch markers for the sparse-emission
fast path (see :doc:`backend`). They are produced automatically by
:func:`pyjuice.multiply` / :func:`pyjuice.summate` whenever the structural
pattern is matched, and explicitly by
:func:`pyjuice.sparse_multiply` / :func:`pyjuice.sparse_summate`.

Methods
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    foreach
    foldup_aggregate

Input Distributions
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    distributions.Bernoulli
    distributions.Categorical
    distributions.DenseCategorical
    distributions.SparseCategorical
    distributions.DiscreteLogistic
    distributions.Gaussian
    distributions.MaskedCategorical

:class:`~pyjuice.distributions.DenseCategorical` and
:class:`~pyjuice.distributions.SparseCategorical` are Categorical-family
distributions that opt the input layer into a faster compile-time path. See
:doc:`backend` for what they trade off and when to use them.
