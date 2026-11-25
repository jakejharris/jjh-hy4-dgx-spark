# Four Sparks on one switch (`SWITCH_FABRIC`)

The kit was made and measured on a cabled ring. This note covers the same build on four Sparks that share one switch, with stock NCCL and no relay. It was run on one such cluster on 2026-11-24. The ring stays the default, and nothing on this page changes it.

## What differs from the ring

| | Ring (the kit) | Switch |
| --- | --- | --- |
| Interfaces on each Spark | `f0` and `f1`, one cable to each neighbour | `enp1s0f0np0` and `enP2p1s0f0np0`, the two interfaces of port `f0`, one cable to the switch |
| Subnets | one `/24` per cable | one `/24` per rail, the same on all four Sparks |
| Who reaches whom | neighbours only | every Spark, on both rails |
| `RING_HOSTS` | physical cable order | rank order |
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1` | `rocep1s0f0,roceP2p1s0f0` |
| `NCCL_CROSS_NIC` | `1` | `0` |
| Relay overlay | used | not used |

All other `NCCL_*` lines stay as `launch/ring.env.example` has them.

**Why no relay.** The relay plan accepts four ranks with exactly one link to each neighbour and none between opposite ranks (`zig/src/fabric/tp/roce_relay_topology.zig`). On a switch every pair has two. This is read from the code; the relay was not started on the switch. `launch/preflight.sh` refuses the combination.

## Steps

1. Give the two rails of each Spark fixed addresses, for example `10.50.1.1` to `10.50.1.4` on the first and `10.50.2.1` to `10.50.2.4` on the second, `/24` each, with the same MTU on the switch and on all Sparks.
2. Check that GID index 3 is RoCE v2 with the IPv4 address on both devices (`show_gids`).
3. Copy `launch/ring.switch.env.example` to `launch/ring.switch.env` and set `MASTER_ADDR` and, if yours differ, `SWITCH_RAILS` and `NCCL_IB_HCA`.
4. Run the fabric probe. It takes one settings file, so join them first:

   ```bash
   cat launch/ring.env launch/ring.switch.env > /tmp/hy4-switch.env
   bash launch/preflight.sh /tmp/hy4-switch.env --probe --ranks 4
   ```

5. Launch as usual, with the switch overlay in place of the relay overlay.

With `SWITCH_FABRIC="1"`, preflight checks on each Spark that both rails have an address, that each rail is one `/24` on all four Sparks, that the two rails are separate `/24`s, and that the route to every other Spark leaves through that rail's own interface with no gateway. It refuses the launch otherwise (exit 4).

## What was run

Four DGX Sparks on one MikroTik CRS812, one 200 Gb/s link per Spark, MTU 9000 on the Sparks, stock clocks, host driver 580.173.02. The kit at `356028c`, the engine at `2cee7d9` with this kit's patch (tree `9a5fcb117f9e6c3937591c41ffabc357bd5f3ee6`), cross-built on an x86-64 PC (`-Dtarget=aarch64-linux-gnu.2.39 -Dcpu=cortex_a725 -Dsm=121`). Full quant `4b6b` from per-Spark parts, 4-bit drafter, drafts on, 131,072-token window, built-in row cost table.

The runs used a local, shorter form of this check in `launch/preflight.sh`. The form here was then run against the same cluster on 2026-11-25, read-only, with the server up and the memory minimum lowered for that: it passed, and it refused a rail without an address, the ring setting and the relay. No launch has used it yet.

| | Result |
| --- | --- |
| Transport | `Using network IB` on all four ranks, bytes exact |
| 156-gather decode burst | p50 6.362 ms, p95 9.171 ms, max 9.650 ms on rank 0 (one run, 20 bursts; `receipts/switch/probe.json`) |
| Load to ready | 49 to 52 s |
| Lowest free memory logged | 24.62 GiB, text only; with vision on 20.87 GiB on rank 0 and 24.59 GiB on the others |
| Decode, greedy, thinking off | prose 19.9, code 29.9, JSON 45.0 and 45.4 tok/s (one prompt each, two requests) |
| Cold prompt to first token | 5,127 tokens 9.9 s, 23,457 tokens 47.1 s, 46,857 tokens 97.9 s; the needle was found in each |
| Tool call, `tool_choice` `auto` | correct (one request) |
| `bench/vision_check.py` | known answers 3 of 3, JPEG and WebP right, exit 0 (`receipts/switch/vision_check.json`) |
| GPU temperature | up to 82 °C in one generation of more than 17,000 tokens; no Spark lost power |

The decode and prompt numbers come from a short script of our own with German prompts, not from this kit's `bench/`, so they do not compare with the kit's tables line by line. The burst is the kit's own probe.

## What was not run

- No long run under load. The longest boot so far was up 10 hours, mostly idle.
- No RigMark, `bench/bench.py`, needle at 128K or accuracy check on the switch. The parts were compared with the full pack (1,181 files, 598,517 tensors, 0 differences); the served tokens were not compared with a ring.
- No row cost table measured on the switch (`tools/row_cost_table.sh`).
- The overlays as separate files. The run had the same settings in one `ring.env`.
- A launch from the full pack, and the lossless trunk.
- `launch/stage_layers.py` and `launch/stage_trunk.py` were not used; the parts were copied with `rsync`.
- `NCCL_CROSS_NIC="1"`, other channel counts, one rail only, and more than one request at a time.
- The switch's own flow control settings (PFC, ECN) were not recorded.
