# Direct-Cabled Ring RDMA (`NCCL` / RoCEv2)

This note covers 2-node to 4-node clusters wired with direct 200 Gb/s copper cables and no switch, where stock NCCL 2.30.7 either dies in the first collective or falls back to slow `NET/Socket`. For four Sparks on one switch, see [SWITCH.md](SWITCH.md).

## Symptoms in the log

1. **`ibv_modify_qp` timeout (`errno 110`)**:
   ```
   ncclIbReceiverQpsCreateToRts -> ncclIbQpRtr: ibv_modify_qp failed with 110 Connection timed out
   ```
   Seen on `rocep1s0f1:1` (or `rocep1s0f0:1`) during the first collective.
2. **Silent socket fallback**:
   NCCL logs `NET/Socket` instead of `Using network IB` / `Connected all rings`. On a 4-node ring, 156 decode gathers take 619.4055 ms (component; spread: 489.608 to 749.203 ms across run medians) on Socket vs 13.6225 ms (component; spread: 10.613 to 16.632 ms across run medians) on IB.

## Why it happens on direct cables

- Each direct cable connects port `f1` on one node to port `f0` on its clockwise neighbor on a separate `/24` subnet (four disjoint `/24` subnets on a 4-node ring).
- Default `NET/IB` merges `f0+f1` or pairs `f0 <-> f0` and `f1 <-> f1` by port index, so QPs target an unreachable subnet (`errno 110`).
- Even with subnet-aware routing enabled, stock NCCL 2.30.7 still opens a Tree transport edge between opposite ranks (`0 <-> 2`) on a 4-node ring, which has no direct cable and times out with `errno 110`.

## Settings and what each one does

All of these are set in `launch/ring.env.example`:

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_NET` | `IB` | Force RDMA IB/RoCE; refuse silent socket fallback |
| `NCCL_IB_DISABLE` | `0` | Keep IB/RoCE enabled |
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1` | Pin both ConnectX-7 HCAs |
| `NCCL_IB_GID_INDEX` | `3` | Pin RoCEv2 GID index 3 |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | Route each QP through the HCA on the matching `/24` subnet |
| `NCCL_IB_SUBNET_PREFIX_LEN` | `24` | Match `/24` per-cable subnets |
| `NCCL_IB_MERGE_NICS` | `0` | Keep `f0` and `f1` unmerged so `/24` routing applies per port |
| `NCCL_CROSS_NIC` | `1` | Allow `f1 -> f0` cross-NIC connections across neighbors |
| `NCCL_ALGO` | `Ring` | Use Ring all-gather (no Tree) |
| `NCCL_PROTO` | `^LL128` | Disable LL128 protocol on RoCE |
| `NCCL_CUMEM_ENABLE` | `1` | Enable cuMem allocation |
| `NCCL_RUNTIME_CONNECT` | `1` | Defer QP setup to runtime so opposite-rank Tree edges are not opened |
| `NCCL_SHM_DISABLE` | `1` | Disable shared-memory transport across containers |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `4` | Pin ring channels to 4 (faster than 8 at every decode width on stock IB with identical bytes) |
| `NCCL_BUFFSIZE` | `1048576` | 1 MiB NCCL buffer |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `enp1s0f0np0` | Bootstrap control plane on the cable interface |

## Quick probe (`< 1 min`)

Run the probe via `preflight.sh` to verify `Using network IB` from the actual NCCL debug log, measure `p50`/`p95`/`max` all-gather latency, check bit-exactness, and enforce the memory floor:

```bash
bash launch/preflight.sh launch/ring.env --probe --ranks 4
```

To test a single directly cabled pair (the first two hosts in `RING_HOSTS`, which must be direct ring neighbors):

```bash
bash launch/preflight.sh launch/ring.env --probe --ranks 2
```

- **Fail-closed on Sockets**: `--probe` refuses `NCCL_NET=Socket` and fails closed if any rank's NCCL debug log does not report `Using network IB` (even if the bytes match).
- **2-node note**: On a 2-node-only cluster with a single `f1 <-> f0` cable and no `f0` cable on `node1`, set `MASTER_ADDR` to `node1`'s active cabled interface IP and set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to `enp1s0f0np0,enp1s0f1np1` (or the active interface) so bootstrap reaches `node1`.
- **Four-rank path (measured)**: 156-gather decode burst 13.6225 ms (component; spread: 10.613 to 16.632 ms across run medians); 2,048-row prefill chunk 20.038 ms (component; spread: 17.773 to 22.303 ms across run medians).
- **Two-rank path (`--ranks 2`)**: Verified over `Using network IB` on a directly cabled pair (156 gathers in 8.227 ms (component; spread: p95 8.473 ms, max 8.527 ms), one gather in 2.369 ms (component; spread: p95 2.419 ms, max 4.551 ms), bytes exact).

## Script run status

- `launch/preflight.sh` and `launch/ring-rdma-probe.py`: run on `2026-11-20` on 4-node and 2-node Spark cluster (`IB` PASS); missing/malformed `ring.env` checks run on `2026-11-20` (`rc=2`/`rc=3`). Cable-order subnet/route check run against the four Sparks on `2026-11-20` (PASS).
- `launch/launch.sh`: missing/malformed `ring.env` checks run on `2026-11-20` (`rc=2`); four-Spark serve launch run on 16 real layers on `2026-11-20`; 78-layer launches of both builds run on `2026-11-21` (`receipts/systems/exl3/`, `receipts/systems/lossless-32k/`).
