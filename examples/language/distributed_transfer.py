# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""L3 multi-card point-to-point transfers (``pld``), P=2.

    xfer_put           pld.tensor.put: GM-to-GM TPUT of the local window into
                       the peer's window, then a barrier, then read back
    xfer_remote_store  pld.tensor.remote_store of a computed tile value
                       straight into the peer's window (on-core, no GM round trip)
    xfer_get           pld.tensor.get: pull the peer's window into the local one

Run:  python examples/language/distributed_transfer.py -p a2a3 -d 0,1
      python examples/language/distributed_transfer.py -p a2a3 -d 0,1 -k xfer_get
"""

import pypto.language as pl
import pypto.language.distributed as pld

SIZE = 256
N_RANKS = 2


# ------------------------------------------------------------------ put


@pl.jit.incore
def xfer_put_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    local = pl.load(inp, [0, 0], [1, SIZE])
    pub = pl.store(local, [0, 0], pub)
    peer = my_rank + 1
    if peer >= N_RANKS:
        peer = peer - N_RANKS
    # Write my window into the peer's result window (peer resolves the window).
    pld.tensor.put(res, peer, pub)
    sig = pld.tensor.barrier(sig)
    acc = pl.load(res, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def xfer_put(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return xfer_put_step(inp, out, pub, res, sig, my_rank)


@pl.jit.host
def xfer_put_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    pub_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    res_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        pub = pld.window(pub_buf, [1, SIZE], dtype=pl.FP32)
        res = pld.window(res_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        xfer_put(inputs[r], outputs[r], pub, res, sig, r, device=r)


# ------------------------------------------------------------------ remote_store


@pl.jit.incore
def xfer_remote_store_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    peer = my_rank + 1
    if peer >= N_RANKS:
        peer = peer - N_RANKS
    # An InCore body is already device code: no pl.at region here (a nested
    # CORE_GROUP scope is rejected by SplitIncoreOrch). The value is a Tile,
    # so the tile-level remote_store applies.
    t = pl.load(inp, [0, 0], [1, SIZE])
    scaled = pl.mul(t, 2.0)
    pld.tile.remote_store(scaled, res, peer, [0, 0])
    sig = pld.tensor.barrier(sig)
    acc = pl.load(res, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def xfer_remote_store(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return xfer_remote_store_step(inp, out, pub, res, sig, my_rank)


@pl.jit.host
def xfer_remote_store_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    pub_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    res_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        pub = pld.window(pub_buf, [1, SIZE], dtype=pl.FP32)
        res = pld.window(res_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        xfer_remote_store(inputs[r], outputs[r], pub, res, sig, r, device=r)


# ------------------------------------------------------------------ get


@pl.jit.incore
def xfer_get_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    local = pl.load(inp, [0, 0], [1, SIZE])
    pub = pl.store(local, [0, 0], pub)
    peer = my_rank + 1
    if peer >= N_RANKS:
        peer = peer - N_RANKS
    sig = pld.tensor.barrier(sig)
    # Pull the peer's published window into my result window.
    pld.tensor.get(res, peer, pub)
    acc = pl.load(res, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], out)


@pl.jit
def xfer_get(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    pub: pld.DistributedTensor[[1, SIZE], pl.FP32],
    res: pld.DistributedTensor[[1, SIZE], pl.FP32],
    sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return xfer_get_step(inp, out, pub, res, sig, my_rank)


@pl.jit.host
def xfer_get_host(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    pub_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    res_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    sig_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        pub = pld.window(pub_buf, [1, SIZE], dtype=pl.FP32)
        res = pld.window(res_buf, [1, SIZE], dtype=pl.FP32)
        sig = pld.window(sig_buf, [N_RANKS, 1], dtype=pl.INT32)
        xfer_get(inputs[r], outputs[r], pub, res, sig, r, device=r)


# ------------------------------------------------------------------ harness


def _specs():
    import torch

    from golden import TensorSpec

    def init_inputs():
        rows = [torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE) for r in range(N_RANKS)]
        return torch.stack(rows)

    return [
        TensorSpec("inputs", [N_RANKS, 1, SIZE], torch.float32, init_value=init_inputs),
        TensorSpec("outputs", [N_RANKS, 1, SIZE], torch.float32),
    ]


def golden_peer(tensors):
    import torch

    tensors["outputs"][:] = torch.roll(tensors["inputs"], 1, dims=0)


def golden_peer_x2(tensors):
    import torch

    tensors["outputs"][:] = 2.0 * torch.roll(tensors["inputs"], 1, dims=0)


ENTRIES = [
    ("xfer_put", xfer_put_host, golden_peer),
    ("xfer_remote_store", xfer_remote_store_host, golden_peer_x2),
    ("xfer_get", xfer_get_host, golden_peer),
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
    for name, host_fn, golden_fn in selected:
        print(f"\n===== {name} =====", flush=True)
        try:
            result = run(
                fn=host_fn,
                specs=_specs(),
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
    print("[RUN] PASS - all selected transfers validated")
