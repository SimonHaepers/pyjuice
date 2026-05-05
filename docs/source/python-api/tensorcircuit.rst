pyjuice.TensorCircuit
=====================

.. autoclass:: pyjuice.TensorCircuit

    .. automethod:: pyjuice.TensorCircuit.forward
    .. automethod:: pyjuice.TensorCircuit.backward
    .. automethod:: pyjuice.TensorCircuit.mini_batch_em
    .. automethod:: pyjuice.TensorCircuit.init_param_flows
    .. automethod:: pyjuice.TensorCircuit.update_parameters
    .. automethod:: pyjuice.TensorCircuit.update_param_flows

Performance flags
-----------------

A few constructor kwargs control which compiled-layer fast paths are
considered. See :doc:`backend` for the dispatch rules and when each path
fires.

``use_dense_sum_layer`` (``bool``, default ``False``)
    Opt in to :class:`pyjuice.layer.DenseSumLayer` for any sum node block
    that is fully connected at the block level and has a single child group.
    Inference-only — parameter-flow accumulation goes through the regular
    :class:`pyjuice.layer.SumLayer` path. Recommended for inference-heavy
    workloads on dense topologies (e.g. HMMs whose transition matrix is
    block-dense).

``param_dtype`` (``torch.dtype``, default ``torch.float32``)
    Storage dtype for the flat parameter buffer. Reduce-precision dtypes
    (e.g. ``torch.bfloat16``) trade a little numerical headroom for memory
    and bandwidth.
