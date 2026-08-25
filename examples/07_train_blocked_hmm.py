"""
Blocked-emission HMMs (Chiu & Rush, 2020) — compute and model quality
=====================================================================

Compares three HMM parameterisations of the same latent size ``H`` on
word-level Penn Treebank (Mikolov preprocessing, 10k vocab):

  * ``dense``    — unconstrained emissions, dense transition
                   (``use_dense_sum_layer=True``).
  * ``blocked``  — blocked emissions: ``M`` groups of ``k = H // M`` states,
                   every token owned by one group
                   (:class:`pyjuice.nodes.distributions.BlockedCategorical`,
                   blocked chain layers). Token→group assignment:
                   ``ctx`` (k-means over PPMI bigram-context vectors, a
                   Brown-cluster stand-in), ``freq`` (frequency-balanced
                   greedy) or ``rand``.
  * ``sparse``   — the *same* block pattern expressed as a CSC
                   :class:`SparseCategorical` on the sparse-IO chain
                   (identical model, generic kernels — compute baseline).

All models train with mini-batch EM. Reports per-token validation
perplexity and ms per training iteration.

Usage::

    python examples/07_train_blocked_hmm.py --H 1024 --M 4 8 --epochs 5
    python examples/07_train_blocked_hmm.py --H 4096 --M 16 --models blocked sparse

Data: ``examples/data/ptb/ptb.{train,valid,test}.txt`` (raw Mikolov files,
e.g. from github.com/wojzaremba/lstm/tree/master/data). Falls back to a
synthetic token stream if the files are missing.
"""

import argparse
import os
import time

import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size, sparse_multiply, sparse_summate
from pyjuice.structures import BlockedHMM, frequency_balanced_partition, context_cluster_partition
from pyjuice.utils.util import max_cdf_power_of_2


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_ptb(data_dir, seq_length):
    paths = {s: os.path.join(data_dir, f"ptb.{s}.txt") for s in ("train", "valid", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    vocab = {}

    def encode(path):
        ids = []
        with open(path) as f:
            for line in f:
                for tok in line.split() + ["<eos>"]:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
                    ids.append(vocab[tok])
        return torch.tensor(ids, dtype=torch.long)

    streams = {s: encode(p) for s, p in paths.items()}

    def chunk(x):
        n = x.numel() // seq_length * seq_length
        return x[:n].reshape(-1, seq_length)

    return {s: chunk(x) for s, x in streams.items()}, len(vocab)


def synthetic(seq_length, V=2000, n_train=20000, n_valid=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Zipfian-ish unigram stream so a frequency partition is meaningful.
    probs = 1.0 / torch.arange(1, V + 1, dtype=torch.float)
    probs /= probs.sum()
    tr = torch.multinomial(probs, n_train * seq_length, replacement=True, generator=g)
    va = torch.multinomial(probs, n_valid * seq_length, replacement=True, generator=g)
    return {"train": tr.reshape(n_train, seq_length), "valid": va.reshape(n_valid, seq_length),
            "test": va.reshape(n_valid, seq_length)}, V


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def build_dense(T, H, V, block_size):
    return juice.structures.HMM(seq_length=T, num_latents=H, num_emits=V,
                                homogeneous=True, block_size=block_size)


def build_blocked(T, H, V, M, token_block, block_size):
    return BlockedHMM(seq_length=T, num_latents=H, num_emits=V, num_blocks=M,
                      token_block=token_block, block_size=block_size)


def build_sparse_equiv(T, H, V, M, token_block, block_size):
    k = H // M
    # Same pattern + a random row-normalised init (SparseCategorical's
    # set_meta_parameters returns zeros, which InputNodes would keep as the
    # initial params).
    proto = dists.BlockedCategorical(num_cats=V)
    init = proto.set_meta_parameters(H, token_block=token_block, num_blocks=M)
    indptr, indices, values = proto.to_csc(init)
    num_node_blocks = H // block_size
    with set_block_size(block_size=block_size):
        ns_input = inputs(T - 1, num_node_blocks=num_node_blocks,
                          dist=dists.SparseCategorical(num_cats=V),
                          csc_indptr=indptr, csc_indices=indices)
        ns_input.set_params(values, normalize=False)
        ns_sum = None
        curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = sparse_multiply(curr_xs, ns)
        return sparse_summate(curr_zs, num_node_blocks=1, block_size=1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def evaluate(pc, data, batch_size, T):
    total = 0.0
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            total += pc(data[i:i + batch_size]).sum().item()
    return total / (data.size(0) * T)   # per-token LL


def train(pc, data, batch_size, epochs, lr, pseudocount, T, log):
    optimizer = juice.optim.CircuitOptimizer(pc, lr=lr, pseudocount=pseudocount, method="EM")
    train_data, valid_data = data["train"], data["valid"]
    iters = 0
    t_iter = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_data.size(0), device=train_data.device)
        t0 = time.time()
        tr_ll = 0.0
        for i in range(0, train_data.size(0) - batch_size + 1, batch_size):
            x = train_data[perm[i:i + batch_size]]
            torch.cuda.synchronize(); ts = time.time()
            optimizer.zero_grad()
            lls = pc(x)
            lls.mean().backward()
            optimizer.step()
            torch.cuda.synchronize(); t_iter += time.time() - ts
            iters += 1
            tr_ll += lls.sum().item()
        tr_ll /= (train_data.size(0) // batch_size * batch_size * T)
        va_ll = evaluate(pc, valid_data, batch_size, T)
        history.append((epoch, tr_ll, va_ll))
        log(f"  epoch {epoch}: train ppl {torch.exp(torch.tensor(-tr_ll)).item():8.1f} "
            f"| valid ppl {torch.exp(torch.tensor(-va_ll)).item():8.1f} "
            f"| {1000 * t_iter / iters:.1f} ms/iter | {time.time() - t0:.1f}s")
    return history, 1000 * t_iter / max(iters, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "ptb"))
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--M", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--pseudocount", type=float, default=0.01)
    ap.add_argument("--models", nargs="+", default=["dense", "blocked", "sparse"])
    ap.add_argument("--partition", nargs="+", default=["ctx"], choices=["ctx", "freq", "rand"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    loaded = load_ptb(args.data, args.T)
    if loaded is None:
        print(f"PTB files not found in {args.data}; using a synthetic Zipfian stream.")
        data, V = synthetic(args.T)
    else:
        data, V = loaded
    data = {k: v.to(dev) for k, v in data.items()}
    counts = torch.bincount(data["train"].reshape(-1).cpu(), minlength=V)
    print(f"V={V}  train={tuple(data['train'].shape)}  valid={tuple(data['valid'].shape)}  H={args.H}")

    results = []

    def run(name, root, **compile_kw):
        print(f"[{name}]")
        pc = juice.TensorCircuit(root, verbose=False, **compile_kw).to(dev)
        hist, ms = train(pc, data, args.batch_size, args.epochs, args.lr, args.pseudocount, args.T, print)
        results.append((name, hist[-1][2], ms))
        del pc
        torch.cuda.empty_cache()

    if "dense" in args.models:
        bs_dense = min(max_cdf_power_of_2(args.H), 1024)
        run(f"dense H={args.H}", build_dense(args.T, args.H, V, bs_dense), use_dense_sum_layer=True)

    for M in args.M:
        k = args.H // M
        block_size = min(max_cdf_power_of_2(k), 256)
        for part in args.partition:
            if part == "ctx":
                token_block = context_cluster_partition(data["train"], V, M, seed=args.seed)
            elif part == "freq":
                token_block = frequency_balanced_partition(counts, M)
            else:
                token_block = torch.randint(0, M, (V,), generator=torch.Generator().manual_seed(args.seed))
                token_block[:M] = torch.arange(M)     # every block owns >= 1 token
            tag = f"H={args.H} M={M} k={k} part={part}"
            if "blocked" in args.models:
                run(f"blocked {tag}", build_blocked(args.T, args.H, V, M, token_block, block_size))
            if "sparse" in args.models:
                run(f"sparse-io {tag}", build_sparse_equiv(args.T, args.H, V, M, token_block, block_size))

    print("\nSummary (final validation perplexity, ms per training iteration):")
    for name, va_ll, ms in results:
        print(f"  {name:<40} ppl {torch.exp(torch.tensor(-va_ll)).item():8.1f}   {ms:8.1f} ms/iter")


if __name__ == "__main__":
    main()
