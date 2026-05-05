Backend: dense and sparse fast paths
=====================================

PyJuice compiles a circuit DAG into one tensor layer per (depth, kind, mode)
group. Most users only see :class:`pyjuice.layer.InputLayer`,
:class:`pyjuice.layer.ProdLayer`, and :class:`pyjuice.layer.SumLayer` — but
when certain structural patterns hold, the compiler quietly substitutes a
faster, specialized layer class. This page describes the patterns the
compiler looks for and what each fast path costs and provides, so you can
shape your DAG (or pick the right distribution / construction helper) to
land on them.

The fast paths are *opt-in by structure*: building a node with
:class:`~pyjuice.distributions.Categorical` and a plain
:func:`~pyjuice.summate` always works and always produces a correct
result; choosing :class:`~pyjuice.distributions.SparseCategorical` or
:func:`~pyjuice.sparse_summate` instead simply gives the compiler enough
information to pick a specialized layer.

Dense sum fast path
-------------------

**Trigger.** Pass ``use_dense_sum_layer=True`` to
:class:`pyjuice.TensorCircuit`. For each :class:`~pyjuice.nodes.SumNodes`
the compiler then checks:

* ``ns.is_block_dense`` — every block in the sum is connected to every
  block of its child group;
* ``len(ns.chs) == 1`` — exactly one child group.

Eligible sums compile to :class:`pyjuice.layer.DenseSumLayer`; the rest
fall back to :class:`pyjuice.layer.SumLayer` automatically. Tied sums are
fine — the dense layer reuses the source's ``_param_range``, which is what
makes homogeneous HMMs at realistic sizes feasible (one shared H×H
transition instead of ``T-1`` copies on the CPU at DAG construction).

**Constraint.** ``DenseSumLayer`` is **inference-only**; parameter-flow
accumulation is not implemented, so the kwarg is safe to leave off when
training. It is a forward/backward (gradient) substitution, not an EM
substitution.

**Why it's faster.** ``SumLayer`` walks block-sparse partition tables. The
dense layer skips that bookkeeping entirely and addresses parameters by
direct pointer arithmetic in Triton, which is the right shape when the
block topology is genuinely dense.

Sparse-categorical input fast path
----------------------------------

**Trigger.** Wrap an input with
:class:`pyjuice.distributions.SparseCategorical` and attach a CSC sparsity
pattern via :meth:`~pyjuice.nodes.InputNodes.set_meta_params` (or kwargs
on :func:`pyjuice.inputs`):

.. code-block:: python

    dist = juice.distributions.SparseCategorical(num_cats = V)
    ns_emit = juice.inputs(
        var = t, num_node_blocks = H,
        dist = dist,
        num_nodes = H,
        csc_indptr  = csc_indptr,   # [V + 1] long
        csc_indices = csc_indices,  # [nnz]   long
    )
    ns_emit.set_params(csc_values)  # CSC-ordered probabilities

The ``num_nodes × num_cats`` emission matrix is stored in CSC form
(column pointers + row ids + values). Forward and backward kernels are
CSC-native: they touch exactly the active column for each observed token
instead of densifying ``num_nodes × num_cats``. Positions outside the
sparsity pattern are treated as ~``1e-10`` (matching
:class:`~pyjuice.distributions.MaskedCategorical`).

**Constraints.**

* Sampling is not supported.
* The downstream sparsity-propagating layers
  (:class:`~pyjuice.layer.SparseProdLayer`,
  :class:`~pyjuice.layer.SparseInputSumLayer`,
  :class:`~pyjuice.layer.SparseIOSumLayer`,
  :class:`~pyjuice.layer.CoSparseProdLayer`) are inference-only and
  ``batch_size == 1`` only.

Sparse propagation through prod / sum
-------------------------------------

A standalone ``SparseCategorical`` input already saves work at the input
layer. The bigger win is *propagating* sparsity through the next prod
layer and into the next sum, so the inner layers also operate on the
active CSC column instead of the full ``H``-vector.

The two node classes that mark this propagation are
:class:`~pyjuice.nodes.SparseProdNodes` and
:class:`~pyjuice.nodes.SparseSumNodes`. They are produced automatically by
:func:`pyjuice.multiply` / :func:`pyjuice.summate` when the DAG matches
the pattern below; you can also force the check (and get a clear error
otherwise) with :func:`pyjuice.sparse_multiply` /
:func:`pyjuice.sparse_summate`.

**SparseProdNodes pattern** — exactly one
:class:`~pyjuice.distributions.SparseCategorical` input child with
identity block-sparse edges on that slot, plus zero or more non-input
(sum) siblings, all with matching ``block_size`` and ``num_node_blocks``.
Two acceptable shapes:

* ``len(chs) >= 2`` — one sparse input + one or more dense (sum) siblings
  (the usual HMM emission × transition product).
* ``len(chs) == 1`` — a single sparse input. The compiler inserts this
  shape as an identity pass-through wrapper bridging an ``InputNodes`` at
  one depth to a sum at a deeper depth; it compiles to a
  :class:`~pyjuice.layer.SparseProdLayer` whose consumer sum can pick
  :class:`~pyjuice.layer.SparseInputSumLayer`.

**SparseSumNodes pattern** — a single :class:`SparseProdNodes` child and
block-dense edges. The subclass is purely a dispatch marker; sum
semantics are inherited from :class:`~pyjuice.nodes.SumNodes`.

The compiler also has a *structural fallback*: even when the DAG builder
hands it a plain ``SumNodes`` (typical for HMM transitions duplicated via
``ns.duplicate``), it promotes the node to the
:class:`~pyjuice.layer.SparseInputSumLayer` fast path whenever the
structural pattern holds (block-dense single-child sum whose child was
compiled as a :class:`~pyjuice.layer.SparseProdLayer`). Pin a sum to the
plain path with ``_force_plain=True`` if you need param-flow accumulation
through it (the sparse fast path is inference-only).

Co-sparse chain
---------------

The deepest fast path fuses two adjacent sparse layers along an HMM-style
chain. When a :class:`SparseProdNodes` sees both its inputs arrive as
:class:`~pyjuice.layer.SparseNodeValues` packets with **identical CSC
indices** (the sparse emission's column at ``var_id`` and the upstream
:class:`~pyjuice.layer.SparseIOSumLayer` output for the same ``var_id``),
the compiler emits a :class:`~pyjuice.layer.CoSparseProdLayer`. The
log-space product collapses to an element-wise add over the packed
``.values`` tensor; ``indices`` passes through untouched and
``element_mars`` is never written.

This requires ``num_dense_chs == 1`` and is what unlocks the per-timestep
O(nnz) cost on the sparse HMM forward.

Dense-categorical input fast path
---------------------------------

**Trigger.** Use :class:`pyjuice.distributions.DenseCategorical` instead
of :class:`~pyjuice.distributions.Categorical`. Math, parameters, and
on-the-wire layout are identical; only the ``get_signature()`` string
changes, which causes the input-layer dispatcher to pick
:class:`pyjuice.layer.DenseCategoricalInputLayer`. The class asserts a
``[V, K, C]`` input layout (every group has scope size 1, the same
``num_nodes = K``, and the same ``num_cats = C``) at construction.

**Why it's faster.** Forward and the regular training backward are
inherited unchanged. The override is on the *conditional-query* backward,
where the standard atomic-add scatter kernel is replaced by a single
``torch.bmm`` / ``torch.matmul``. On large HMMs this is 30–55× over the
plain path. EM training is unaffected.

Dispatcher summary
------------------

A simplified view of what the compiler emits at each (depth, kind):

============================  =============================================
Pattern                       Compiled layer
============================  =============================================
``Categorical`` input         :class:`pyjuice.layer.InputLayer`
``DenseCategorical`` input    :class:`pyjuice.layer.DenseCategoricalInputLayer`
``SparseCategorical`` input   :class:`pyjuice.layer.InputLayer` (custom
                              CSC kernels)
plain ``ProdNodes``           :class:`pyjuice.layer.ProdLayer`
``SparseProdNodes``           :class:`pyjuice.layer.SparseProdLayer`
``SparseProdNodes`` in chain  :class:`pyjuice.layer.CoSparseProdLayer`
plain ``SumNodes``            :class:`pyjuice.layer.SumLayer`
``is_block_dense``,           :class:`pyjuice.layer.DenseSumLayer`
single child,
``use_dense_sum_layer=True``
``SparseSumNodes``            :class:`pyjuice.layer.SparseInputSumLayer`
``SparseSumNodes``            :class:`pyjuice.layer.SparseIOSumLayer`
feeding co-sparse chain
============================  =============================================

Anything that doesn't match a fast-path pattern silently falls back to the
plain :class:`~pyjuice.layer.InputLayer` /
:class:`~pyjuice.layer.ProdLayer` / :class:`~pyjuice.layer.SumLayer` path.
