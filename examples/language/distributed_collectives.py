# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""L3 multi-card collectives (``pld.tensor`` InCore rail), P=2.

One entry per collective, each fixed at two ranks, each with its own host
orchestrator, window buffers and torch golden:

    c_allreduce        mesh allreduce, twice in a row on the same signal
                       (rebind + signal-reuse idiom)
    c_ring             ring allreduce with the [2*(NR-1), NR] signal shape
    c_broadcast        root=0 broadcast
    c_allgather        every rank ends up with all rank rows
    c_reduce_scatter   each rank ends up with its own reduced chunk
    c_all_to_all       symmetric personalized exchange
    c_barrier          barrier with the rebind idiom

Run:  python examples/language/distributed_collectives.py -p a2a3 -d 0,1
      python examples/language/distributed_collectives.py -p a2a3 -d 0,1 -k c_ring
"""

import pypto.language as pl
import pypto.language.distributed as pld

SIZE = 256
N_RANKS = 2  # every entry runs P=2; the window shapes need it statically


# ------------------------------------------------------------------ allreduce


@pl.jit.incore
def c_allreduce_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    # The rebind idiom, and the self-clearing signal that makes a second
    # back-to-back call on the same signal legal.
    pub = pld.tensor.allreduce(data, sig, op=pld.ReduceOp.Sum)
    pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
    acc = pl.load(pub, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_allreduce(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_allreduce_step(inp, out, data, sig, my_rank)


@pl.jit.host
def c_allreduce_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_allreduce(inputs[r], outputs[r], data, sig, r, device=r)


# ------------------------------------------------------------------ ring


@pl.jit.incore
def c_ring_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[2 * (N_RANKS - 1), N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    pub = pld.tensor.allreduce(data, sig, op=pld.ReduceOp.Sum, mode="ring")
    acc = pl.load(pub, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_ring(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[2 * (N_RANKS - 1), N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_ring_step(inp, out, data, sig, my_rank)


@pl.jit.host
def c_ring_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([2 * (N_RANKS - 1), N_RANKS], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [2 * (N_RANKS - 1), N_RANKS], dtype=pl.INT32)
        c_ring(inputs[r], outputs[r], data, sig, r, device=r)


# ------------------------------------------------------------------ broadcast


@pl.jit.incore
def c_broadcast_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    if my_rank == 0:
        local = pl.load(inp, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)
    pub = pld.tensor.broadcast(data, sig, root=0)
    acc = pl.load(pub, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_broadcast(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_broadcast_step(inp, out, data, sig, my_rank)


@pl.jit.host
def c_broadcast_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_broadcast(inputs[r], outputs[r], data, sig, r, device=r)


# ------------------------------------------------------------------ allgather


@pl.jit.incore
def c_allgather_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    target: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    gathered = pld.tensor.allgather(inp, target, sig)
    # Row src of the gathered window is local only on rank src. Storing the
    # NEXT rank's row therefore exercises the remote read-back path; the
    # local read path is covered by every other entry.
    nxt = my_rank + 1
    if nxt >= N_RANKS:
        nxt = nxt - N_RANKS
    row = pld.tile.remote_load(gathered, peer=nxt, offsets=[nxt, 0], shape=[1, SIZE])
    return pl.store(row, [0, 0], out)


@pl.jit
def c_allgather(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, N_RANKS * SIZE], pl.FP32]],
    target: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_allgather_step(inp, out, target, sig, my_rank)


@pl.jit.host
def c_allgather_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32]],
):
    tgt_buf = pld.alloc_window_buffer([N_RANKS, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        target = pld.window(tgt_buf, [N_RANKS, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_allgather(inputs[r], outputs[r], target, sig, r, device=r)


# ------------------------------------------------------------------ reduce_scatter


@pl.jit.incore
def c_reduce_scatter_step(
    inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    # Stage all NR chunks, one per row.
    for j in pl.range(N_RANKS):
        chunk = pl.load(inp, [j, 0], [1, SIZE])
        data = pl.store(chunk, [j, 0], data)
    reduced = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
    acc = pl.load(reduced, [my_rank, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_reduce_scatter(
    inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_reduce_scatter_step(inp, out, data, sig, my_rank)


@pl.jit.host
def c_reduce_scatter_host(
    inputs: pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([N_RANKS, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [N_RANKS, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_reduce_scatter(inputs[r], outputs[r], data, sig, r, device=r)


# ------------------------------------------------------------------ all_to_all


@pl.jit.incore
def c_all_to_all_step(
    inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[N_RANKS, SIZE], pl.FP32]],
    target: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    # input[dest, :] is this rank's chunk destined for rank dest. The InCore
    # rail takes the plain [NR, SIZE] Tensor parameter directly.
    received = pld.tensor.all_to_all(inp, target, sig)
    acc = pl.load(received, [0, 0], [N_RANKS, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_all_to_all(
    inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[N_RANKS, SIZE], pl.FP32]],
    target: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_all_to_all_step(inp, out, target, sig, my_rank)


@pl.jit.host
def c_all_to_all_host(
    inputs: pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32]],
):
    tgt_buf = pld.alloc_window_buffer([N_RANKS, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        target = pld.window(tgt_buf, [N_RANKS, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_all_to_all(inputs[r], outputs[r], target, sig, r, device=r)


# ------------------------------------------------------------------ barrier


@pl.jit.incore
def c_barrier_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    sig = pld.tensor.barrier(sig)
    acc = pl.load(data, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def c_barrier(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return c_barrier_step(inp, out, data, sig, my_rank)


@pl.jit.host
def c_barrier_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        c_barrier(inputs[r], outputs[r], data, sig, r, device=r)


# ------------------------------------------------------------------ harness


def _inputs_1d():
    import torch

    def init():
        rows = [torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE) for r in range(N_RANKS)]
        return torch.stack(rows)

    return init


def _inputs_2d():
    import torch

    def init():
        blocks = []
        for r in range(N_RANKS):
            rows = [
                torch.arange((r * 10 + j) * 1000.0, (r * 10 + j) * 1000.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
                for j in range(N_RANKS)
            ]
            blocks.append(torch.cat(rows, dim=0))
        return torch.stack(blocks)

    return init


def _spec_sets():


    one = dict(
        inputs=[N_RANKS, 1, SIZE],
        outputs=[N_RANKS, 1, SIZE],
    )
    two = dict(
        inputs=[N_RANKS, N_RANKS, SIZE],
        outputs=[N_RANKS, N_RANKS, SIZE],
    )
    mixed = dict(
        inputs=[N_RANKS, N_RANKS, SIZE],
        outputs=[N_RANKS, 1, SIZE],
    )
    gather = dict(
        inputs=[N_RANKS, 1, SIZE],
        outputs=[N_RANKS, 1, SIZE],
    )
    return {
        "one": (one, _inputs_1d),
        "two": (two, _inputs_2d),
        "mixed": (mixed, _inputs_2d),
        "gather": (gather, _inputs_1d),
    }


def _specs(kind):
    import torch

    from golden import TensorSpec

    shapes, init = _spec_sets()[kind]
    return [
        TensorSpec("inputs", shapes["inputs"], torch.float32, init_value=init()),
        TensorSpec("outputs", shapes["outputs"], torch.float32),
    ]


def golden_allreduce(tensors):

    s = tensors["inputs"].sum(dim=0, keepdim=True)
    tensors["outputs"][:] = (2.0 * s).expand_as(tensors["outputs"])


def golden_ring(tensors):
    s = tensors["inputs"].sum(dim=0, keepdim=True)
    tensors["outputs"][:] = s.expand_as(tensors["outputs"])


def golden_broadcast(tensors):
    tensors["outputs"][:] = tensors["inputs"][0:1].expand_as(tensors["outputs"])


def golden_allgather(tensors):
    import torch

    # Each rank stored the NEXT rank's row, read back through the window.
    tensors["outputs"][:] = torch.roll(tensors["inputs"], 1, dims=0)


def golden_reduce_scatter(tensors):
    # Chunk r of every rank holds inputs[r][r]; after the reduce, rank r's
    # row r holds sum over ranks r' of inputs[r'][r].
    summed = tensors["inputs"].sum(dim=0)  # [N_RANKS, SIZE]
    tensors["outputs"][:] = summed.reshape(N_RANKS, 1, SIZE)


def golden_all_to_all(tensors):
    # target[src] on rank my_rank = inputs[src][my_rank]
    tensors["outputs"][:] = tensors["inputs"].permute(1, 0, 2).contiguous()


def golden_barrier(tensors):
    tensors["outputs"][:] = tensors["inputs"]


ENTRIES = [
    ("c_allreduce", c_allreduce_host, "one", golden_allreduce),
    ("c_ring", c_ring_host, "one", golden_ring),
    ("c_broadcast", c_broadcast_host, "one", golden_broadcast),
    ("c_allgather", c_allgather_host, "gather", golden_allgather),
    ("c_reduce_scatter", c_reduce_scatter_host, "mixed", golden_reduce_scatter),
    ("c_all_to_all", c_all_to_all_host, "two", golden_all_to_all),
    ("c_barrier", c_barrier_host, "one", golden_barrier),
]


if __name__ == "__main__":
    import argparse

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1")
    parser.add_argument("-k", "--only", type=str, default="", help="comma-separated entry names")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    selected = [e for e in ENTRIES if not args.only or e[0] in args.only.split(",")]
    failures = []
    for name, host_fn, kind, golden_fn in selected:
        print(f"\n===== {name} =====", flush=True)
        try:
            result = run(
                fn=host_fn,
                specs=_specs(kind),
                golden_fn=golden_fn,
                compile_only=args.compile_only,
                config=dict(
                    platform=args.platform,
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                ),
                rtol=1e-5,
                atol=1e-5,
            )
            status = "PASS" if result.passed else "FAIL"
            if not result.passed:
                failures.append(name)
                if result.error:
                    print(result.error)
            print(f"[ENTRY] {name} {status}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"[ENTRY] {name} ERROR {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print(f"[RUN] FAILED entries: {failures}")
        raise SystemExit(1)
    print("[RUN] PASS - all selected collectives validated")
