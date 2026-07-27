"""
Train an HMM with sparse-IO propagation
=======================================

This tutorial demonstrates how to train an HMM whose emission matrix is **pruned**
(each latent state can only emit a small subset of the vocabulary) using the
sparse-IO fast path: a :code:`SparseCategorical` input distribution whose sparsity
propagates through the chain interior, so every timestep only computes over the
emission-active subset of the latent states instead of all of them.

The training pipeline is *identical* to the standard one from the
"Train a PC" tutorial — same :code:`TensorCircuit`, same
:code:`CircuitOptimizer`, same mini-batch / full-batch EM loops, any batch
size — only the DAG construction changes. During EM, the emission
probabilities learn within the fixed pruned support (which never regrows), and
the full dense transition matrix is trained with only its emission-active
tiles touched per step.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

# %%
# Let's start by importing the necessary packages.

import torch
import time
import pyjuice as juice
import pyjuice.nodes.distributions as dists

# %%
# Specify the structural parameters of the HMM. These match the
# "Construct an HMM" tutorial, plus one new knob: the **density** of the
# pruned emission matrix (the fraction of (latent, token) pairs kept).

seq_length = 32
num_latents = 256
num_emits = 4096
density = 0.02

block_size = min(juice.utils.util.max_cdf_power_of_2(num_latents), 1024)
num_node_blocks = num_latents // block_size

device = torch.device("cuda:0")

# %%
# Build the CSC sparsity pattern of the ``num_latents x num_emits`` emission
# matrix. The pattern is a *meta-parameter*: it fixes which emissions exist at
# all, and EM will only ever learn the probabilities of these surviving
# entries. In practice the pattern comes from pruning a trained dense model or
# from domain knowledge; here we draw a random one with two coverage
# guarantees: every latent keeps at least one token (an empty row could never
# emit anything), and every token stays emittable by at least one latent — a
# token outside the support has probability *exactly zero* under the sparse
# model, so a dataset containing it would evaluate to ``LL = -inf``.

def random_csc_pattern(H, V, density, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    rows = torch.randint(0, H, (int(density * H * V * 1.05) + H,), generator = g)
    cols = torch.randint(0, V, (rows.numel(),), generator = g)
    # Coverage: every latent gets >= 1 token, every token gets >= 1 latent.
    rows = torch.cat([rows, torch.arange(H), torch.randint(0, H, (V,), generator = g)])
    cols = torch.cat([cols, torch.randint(0, V, (H,), generator = g), torch.arange(V)])
    linear = torch.unique(rows.long() * V + cols.long())
    rows_d, cols_d = linear // V, linear % V
    order = torch.argsort(cols_d * H + rows_d)          # sort by (col, row) = CSC
    csc_indices = rows_d[order].contiguous()
    csc_indptr = torch.zeros(V + 1, dtype = torch.long)
    csc_indptr[1:] = torch.cumsum(torch.bincount(cols_d[order], minlength = V), dim = 0)
    # Row-normalised random probabilities for the surviving entries.
    raw = torch.rand(csc_indices.numel(), generator = g)
    row_sums = torch.zeros(H).scatter_add_(0, csc_indices, raw)
    csc_values = (raw / row_sums[csc_indices]).to(torch.float32)
    return csc_indptr, csc_indices, csc_values

csc_indptr, csc_indices, csc_values = random_csc_pattern(
    num_latents, num_emits, density, seed = 7,
)

# %%
# Construct the HMM. Compared to the dense version there are exactly three
# changes: the input distribution is :code:`SparseCategorical` (with the CSC
# pattern passed as meta-parameters and the values set in CSC order), and the
# chain uses :code:`sparse_multiply` / :code:`sparse_summate` — the markers
# that let the compiler's DAG pre-pass upgrade the chain interior to the
# sparsity-propagating layers.

with juice.set_block_size(block_size):
    ns_input = juice.inputs(
        seq_length - 1, num_node_blocks = num_node_blocks,
        dist = dists.SparseCategorical(num_cats = num_emits),
        csc_indptr = csc_indptr, csc_indices = csc_indices,
    )
    ns_input.set_params(csc_values, normalize = False)

    ns_sum = None
    curr_zs = ns_input
    for var in range(seq_length - 2, -1, -1):
        curr_xs = ns_input.duplicate(var, tie_params = True)
        if ns_sum is None:
            ns = juice.summate(curr_zs, num_node_blocks = num_node_blocks)
            ns_sum = ns
        else:
            ns = ns_sum.duplicate(curr_zs, tie_params = True)
        curr_zs = juice.sparse_multiply(curr_xs, ns)
    root_ns = juice.sparse_summate(curr_zs, num_node_blocks = 1, block_size = 1)

# %%
# Compile and move to the GPU. The chain interior compiles to
# :code:`SparseProdLayer` / :code:`SparseIOSumLayer` / :code:`CoSparseProdLayer`
# instead of the plain layers.

pc = juice.compile(root_ns)
pc.to(device)

from pyjuice.layer import SparseIOSumLayer, CoSparseProdLayer
assert any(isinstance(l, SparseIOSumLayer)
           for lg in pc.inner_layer_groups for l in lg)
assert any(isinstance(l, CoSparseProdLayer)
           for lg in pc.inner_layer_groups for l in lg)

# %%
# Create a synthetic token dataset. Substitute your own ``[N, seq_length]``
# long tensor of token ids here.

torch.manual_seed(0)
train_data = torch.randint(0, num_emits, (8192, seq_length), device = device)
valid_data = torch.randint(0, num_emits, (1024, seq_length), device = device)

batch_size = 512

# %%
# Train with mini-batch EM. This loop is copied verbatim from the
# "Train a PC" tutorial — the sparse chain supports any batch size, so a
# whole mini-batch of sequences runs in one forward/backward. The
# ``pseudocount`` must be positive: a latent whose surviving tokens were
# never observed in a batch would otherwise have a zero normaliser.

optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.01, method = "EM")

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_ll = 0.0
    for batch_start in range(0, train_data.size(0), batch_size):
        x = train_data[batch_start:batch_start + batch_size]

        # Zero out the parameter flows, like zeroing gradients in PyTorch
        optimizer.zero_grad()

        # Forward pass
        lls = pc(x)

        # Backward pass (E-step: accumulate expected counts)
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

        # Mini-batch EM step (M-step)
        optimizer.step()

    train_ll /= train_data.size(0) // batch_size

    t1 = time.time()
    valid_ll = pc(valid_data).mean().detach().cpu().numpy().item()
    t2 = time.time()

    print(f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; val LL: {valid_ll:.2f}]"
          f".....[train {t1-t0:.2f}s; val {t2-t1:.2f}s]")

# %%
# Full-batch EM works the same way as in the dense tutorial: accumulate the
# flows over the whole dataset, then apply one M-step with step size 1.

pc.init_param_flows(flows_memory = 0.0)

train_ll = 0.0
for batch_start in range(0, train_data.size(0), batch_size):
    x = train_data[batch_start:batch_start + batch_size]
    lls = pc(x)
    lls.mean().backward()
    train_ll += lls.mean().detach().cpu().numpy().item()

pc.mini_batch_em(step_size = 1.0, pseudocount = 0.01)

train_ll /= train_data.size(0) // batch_size
print(f"[Full-batch EM][train LL: {train_ll:.2f}]")

# %%
# Notes and current constraints of the sparse-IO fast path:
#
# * Training uses ``propagation_alg = "LL"`` on complete data. Conditional /
#   generative queries and ``missing_mask`` are currently only supported at
#   ``batch_size = 1`` on the sparse chain.
# * The emission support is fixed: pruned (latent, token) pairs stay at
#   probability zero through every EM step. The transition matrix is a full
#   dense parameter block — only its emission-active tiles are computed and
#   receive flows at each timestep.
# * The speedup over the dense path grows with ``num_latents`` and with the
#   pruning strength: at ``num_latents = 32768`` and 1% density, the sparse
#   chain trains at ~0.34 ms/sequence (batch 64+) where the dense fast path
#   cannot train at all. See ``tests/layer/sparse_io_sum_perf_test.py`` for
#   the measurement harnesses.
