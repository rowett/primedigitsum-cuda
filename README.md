# ds(n) — GPU-Accelerated Prime Digit-Sum Search

A high-performance CUDA program that searches for **ds(n)** numbers: the smallest prime whose digit sum in every base from 2 through n+1 is also prime.

Searches approximately **1.63 trillion numbers per second** on an RTX 4070, and **2.81 trillion** on an RTX 5070 Ti.

---

## The Problem

Define ds(n) as the smallest prime p such that for every base b in {2, 3, 4, ..., n+1}, the sum of the digits of p written in base b is also prime.

Known values:

| n | ds(n) | | n | ds(n) |
|---|-------|---|---|-------|
| 1 | 3 | | 17 | 68433401 |
| 2 | 5 | | 18 | 68433401 |
| 3 | 5 | | 19 | 68433401 |
| 4 | 11 | | 20 | 89757221 |
| 5 | 17 | | 21 | 89757221 |
| 6 | 17 | | 22 | 28941023651 |
| 7 | 17 | | 23 | 92343621959 |
| 8 | 131 | | 24 | 141648275861 |
| 9 | 131 | | 25 | 533958144803 |
| 10 | 131 | | 26 | 533958144803 |
| 11 | 2663 | | 27 | 533958144803 |
| 12 | 2663 | | 28 | 533958144803 |
| 13 | 892747 | | 29 | 533958144803 |
| 14 | 892747 | | 30 | 267716684796523 |
| 15 | 3748207 | | 31 | 628379388522107 |
| 16 | 10110857 | | 32 | 42532353830718823 |

Each additional base multiplies the search space by roughly 4, since each contributes an approximately independent 1-in-4 chance that its digit sum is prime. Empirically log₁₀ ds(n) grows at about 0.54 decades per base, against a heuristic prediction of log₁₀(4) = 0.60. Finding ds(n) for large n therefore requires testing enormous numbers, making brute-force CPU search infeasible.

> **64-bit range limit.** Extrapolating that growth, ds(n) crosses 2⁶⁴ = 1.845e19 somewhere between n = 37 and n = 39. Everything in this program — the candidate type, the radix-chunk boundaries, the wheel arithmetic, the Miller-Rabin witness set — assumes 64-bit candidates. Searches beyond roughly ds(38) will need 128-bit support.

---

## Architecture

The search is split into two pipeline stages running concurrently:

**Stage 1 — GPU Filtering**

The GPU kernel applies a fast multi-base digit-sum primality filter to reject the overwhelming majority of candidates. Two kernel instances run on alternating CUDA streams (A/B double-buffering) so that while one stream is transferring results back over PCIe, the other is processing the next block.

**Stage 2 — CPU Verification**

A dedicated CPU worker thread receives GPU survivors and performs full Miller-Rabin primality testing plus any remaining base checks above the GPU's filter range. Confirmed primes that set new ds(n) records are logged immediately. Verification takes under 2 ms per sub-block, roughly 20× faster than the GPU produces work, so the CPU is never the bottleneck.

---

## GPU Kernel Design

### Wheel-510510 Sieve

Every thread maps to a fixed offset in the mod-510510 wheel, which eliminates all multiples of 2, 3, 5, 7, 11, 13, and 17 before any digit-sum work begins. This rejects **418,350 of every 510,510 integers — 81.947%** — at zero runtime cost, since the rejected numbers are never generated in the first place.

| prime | additional kills per 510,510 |
|---|---|
| 2 | 255,255 |
| 3 | 85,085 |
| 5 | 34,034 |
| 7 | 19,448 |
| 11 | 10,608 |
| 13 | 8,160 |
| 17 | 5,760 |

The 92,160 surviving offsets reach 510,509, so they are stored as `uint32_t`. At 360 KB the table is far past the 64 KB constant bank and lives in global memory as a `__device__` array instead. This costs nothing in the inner loop: each thread reads exactly one offset before its walk begins, and consecutive lanes read consecutive entries, so a warp takes a single coalesced 128-byte transaction per launch.

Extending to mod-9699690 (adding 19) would reject a further 5.3%, at the cost of a 6.6 MB offset table. Beyond that the returns collapse — adding 23 buys 4.3% for a 152 MB table.

### Grid Sizing

Two constraints have to be satisfied at once.

**Wheel alignment.** Each thread owns exactly one wheel offset for the life of the kernel, so `gridSize × blockSize` must be a whole number of wheel revolutions. At 768 threads per block the quantum is 92160 / 768 = **120 blocks**.

**Wave alignment.** `__launch_bounds__(768, 2)` keeps two blocks resident per SM, so the GPU runs waves of `2 × numSMs` blocks. A grid that leaves a partial final wave idles most of the machine for the duration of that wave.

The launcher queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than assuming the occupancy, then picks the smallest multiple of the quantum that wastes least of the final wave, comparing candidates with exact integer cross-multiplication:

| GPU | SMs | wave | grid chosen | waves | fill | stride |
|---|---|---|---|---|---|---|
| RTX 4070 | 46 | 92 | 2760 | 30 | 100% | 6,126,120 |
| RTX 5070 Ti | 70 | 140 | 840 | 6 | 100% | 3,573,570 |

This matters more than it looks. With the older mod-30030 wheel the quantum was 15, fine enough that naive rounding-down cost nothing. At 120 it cost 2% of the machine until the wave-fill logic was added.

### High-Word Hoisting

The dominant cost in this kernel is `POPC`, which on Ada runs at **16 results per SM per clock** against 64 for ordinary INT32 — quarter rate, or 8 cycles per warp instruction against 2. Reducing the number of popcount instructions is worth more than almost anything else.

The candidate walk is therefore a two-level loop. The outer loop pins the top 32 bits of the candidate; the inner loop varies only the low 32 bits. Because the stride is about 6.1 million, `candidate >> 32` changes only once every ~700 inner iterations, so the high half's contribution to all five power-of-two digit sums is loop-invariant and gets computed once in the outer loop.

Every mask involved is a compile-time 64-bit constant, so the split needs no re-phasing: mask bit 32+j is simply bit j of `mask >> 32`, which already carries the correct mod-3 or mod-5 phase. `0x2492492492492492` becomes `0x24924924` / `0x92492492`, and so on.

| base | POPC before | POPC after |
|---|---|---|
| 2 | 2 | 1 |
| 4 | 2 | 1 |
| 8 | 4 | 2 |
| 32 | 8 | 4 |

Base 16 likewise keeps one `__dp4a` and one nibble fold instead of two of each, accumulating straight onto the cached high-word byte sum. `__popc(lo)` is computed once and shared by bases 2, 8, and 32.

This change alone was worth **1.36×**.

### Base Filter Order

Bases are checked in empirically determined order of filtering efficiency per unit cost at large candidate values:

```
Powers of two:   2, 4, 16, 8, 32
Remaining even:  12, 6, 10, 14
Remaining odd:   5, 9, 3, 11, 7, 13, 15
```

The do-while/break structure provides early exit as soon as any base check fails. Divergence is inherent and structural: a warp pays the full instruction latency for any filter that even one of its 32 lanes still needs, so the effective cost of each filter is its instruction count weighted by the probability that at least one lane survives to reach it. The power-of-two prefix is cheap enough and selective enough that the later bases are reached by only a few percent of warp-iterations.

### Digit-Sum Computation Techniques

Each base uses the cheapest arithmetic for that base's structure. `lo` is the low 32 bits of the candidate; `DS*_HI` are the cached high-word partial sums.

**Base 2** — one `__popc` plus a cached constant:
```cuda
p = DS2_HI + __popc(lo)
```

**Base 4** — reuses `__popc(lo)`; the masked popcount gives Σ bit1 directly, and popcount(all bits) = Σ bit0 + Σ bit1 across all 2-bit digits:
```cuda
ds4 = DS4_HI + __popc(lo) + __popc(lo & 0xAAAAAAAAu)
```

**Base 8** — bit-plane decomposition across the 3-bit digit grid, with the bit-0 plane derived from the shared popcount:
```cuda
ds8 = DS8_HI + __popc(lo) + __popc(lo & 0x92492492u)
                          + 3u * __popc(lo & 0x24924924u)
```

**Base 16** — nibble extraction into packed bytes followed by `__dp4a` horizontal summation. Dotting packed bytes with `0x01010101` is a horizontal byte-sum in one instruction, and it accumulates the cached high-word sum for free via the addend operand:
```cuda
n_low = (lo & 0x0F0F0F0F) + ((lo >> 4) & 0x0F0F0F0F)
ds16  = __dp4a(n_low, 0x01010101U, DS16_HI)
```

**Base 32** — five-plane bit decomposition, same principle:
```cuda
ds32 = DS32_HI + __popc(lo) + __popc(lo & 0x84210842u)
                            + 3u  * __popc(lo & 0x08421084u)
                            + 7u  * __popc(lo & 0x10842108u)
                            + 15u * __popc(lo & 0x21084210u)
```

**Non-power-of-2 bases (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)** — radix-chunked lookup tables. Each 64-bit candidate is split at a power-of-the-base boundary into a high and low part, each small enough to index a precomputed byte table. This avoids all 64-bit division in the kernel; the compiler replaces the constant divisors with multiply-shift sequences, confirmed by zero division instructions in SASS output.

### Combined Base 2+4 Check

The first primality check tests bases 2 and 4 simultaneously with a single branch:
```cuda
if (!(local_sp[p] & local_sp[ds4])) break;
```
Fusing them is deliberate rather than incidental. Splitting the two checks would save nothing — with 32 lanes and a base-2 pass rate near 25%, essentially every warp has at least one survivor, so the base-4 instructions would be issued anyway. Fusing at least buys instruction-level parallelism between the two independent popcounts.

### Shared Memory

The small-prime lookup table (`local_sp`) and the three most-reached non-power-of-2 tables (base 6, base 10, base 12) are staged into shared memory via `__pipeline_memcpy_async`. The remaining tables (bases 3, 5, 7, 9, 11, 13, 14, 15) are read with `__ldg`.

`local_sp` is small enough that the ~45 entries actually indexed span about 12 distinct 32-bit words. Sub-word accesses to the same word broadcast rather than conflict, so 32 lanes land in 12 distinct banks with zero replays.

### Result Output

Survivors are written with a plain per-thread `atomicAdd` on the result counter. Warp aggregation via `__ballot_sync` was considered and rejected: after sixteen digit-sum filters the probability of two lanes in the same warp surviving simultaneously is negligible, so the aggregation logic would cost more than the contention it removes.

### Kernel Templating

The kernel is templated on `MIN_BASE` and instantiated at compile time for values 9 through 16 plus 32. `if constexpr` blocks for higher bases compile away entirely in lower-base instantiations. Register usage ranges from 26 to 32 across all variants with zero spill, comfortably inside the 42-register budget that two resident blocks of 768 threads allows.

---

## CPU Verification

The Miller-Rabin primality test uses deterministic witness sets sufficient for all 64-bit integers. A dynamic checklist handles the gap between what the GPU kernel checked (fixed at launch time) and the current `global_minbase` target, which may have advanced while the block was in flight.

When a candidate passes all base checks and primality testing, it claims any newly validated ds(n) records via a compare-exchange loop on `global_minbase`, then continues checking higher bases to potentially claim additional consecutive records from the same candidate.

---

## Performance

| stage | RTX 4070 | step |
|---|---|---|
| baseline | 1138.30 G/s | — |
| high-word hoisting | 1553.17 G/s | ×1.364 |
| wheel-510510 | 1602.71 G/s | ×1.032 |
| wave-aligned grid | 1631.25 G/s | ×1.018 |
| | | **×1.433 cumulative** |

RTX 5070 Ti (70 SMs, sm_120) reaches **2809.33 G/s**, or 1.722× the 4070. Of that, 1.522× is SM count alone; the per-SM architectural gain is only about 1.13× despite Blackwell's unified INT32/FP32 cores doubling plain integer throughput, which is consistent with `POPC` not having scaled with them.

Under load on the RTX 4070 the core clock holds a steady 2760 MHz at 173–178 W against a 200 W board limit and 74–75 °C, with SM activity at 100% and **memory bandwidth utilisation at 0–1%**. The kernel is instruction-issue bound, not memory bound, and is neither power- nor thermally throttled.

### Verified dead ends

Documented so they aren't re-attempted:

- **Register-resident prime bitmask** replacing the `local_sp` shared lookup. Measured slower. A 64-bit immediate cannot be an immediate operand, so the mask needs registers or a constant-bank load, and the variable 64-bit shift lowers to a funnel-shift pair plus an AND — three or four INT32 ops against one fully-hidden shared load.
- **Larger shared-memory carveout** for the `__ldg` tables. With memory utilisation at 1% there is no bandwidth pressure to relieve.

---

## Output Files

`ds_records.txt` — appended whenever a new ds(n) record is found:
```
>>> NEW RECORD FOUND: ds(22) = 28,941,023,651 <<<
```

`ds_state.txt` — overwritten every 60 seconds with the exact command needed to resume from the last completed sub-block:
```
--- ENGINE HEARTBEAT ---
Last Saved: 2026-05-01 14:32:11
Restart Command Args:
.\ds 12501 <end_subblock> <subblock_size> 13 <maxbase>
```

> Record values quickly exceed 15 digits. Any tool that keeps only 15 significant figures — Excel, Google Sheets, `%.15g` — will silently truncate them into composite-looking numbers. Read results from the program's own output.

---

## Requirements

- CUDA Toolkit 12.x or later (developed against 13.3.1)
- C++17 or later for `if constexpr`; built with C++20
- GPU with compute capability 7.0 or higher (Volta+) for `__dp4a`. Tested on sm_89 (Ada) and sm_120 (Blackwell)
- Windows (uses `_umul128` / `_udiv128` for 128-bit arithmetic in Miller-Rabin)

---

## Build

```bash
nvcc -O3 -std=c++20 -lineinfo ^
     -gencode arch=compute_89,code=sm_89 ^
     -gencode arch=compute_120,code=sm_120 ^
     -gencode arch=compute_120,code=compute_120 ^
     -Xptxas "-v" -Xcompiler "/O2 /Ob2 /Ot" ds-cuda.cu -o ds
```

Adjust `-gencode` for your GPU (sm_70 for Volta, sm_75 for Turing, sm_86 for Ampere, sm_89 for Ada, sm_120 for Blackwell). The trailing `compute_120` entry embeds PTX for forward compatibility. `-Xptxas -v` reports register usage; watch that it stays at or below 42, since crossing that drops the second resident block per SM and costs roughly half the throughput.

---

## Usage

```
ds.exe start_subblock end_subblock subblock_size minbase maxbase
```

| Argument | Description |
|---|---|
| `start_subblock` | First sub-block index to process (inclusive) |
| `end_subblock` | Last sub-block index to process (exclusive) |
| `subblock_size` | Number of integers per sub-block (typically 1000000000) |
| `minbase` | Lowest ds(n) base to search for (resumes from here) |
| `maxbase` | Highest ds(n) base; program exits when this is found |

### Example — search from the beginning up to ds(20), noting base is n + 1:

```
ds.exe 0 1000000 1000000000 2 21
```

### Example — resume from a heartbeat:

```
ds.exe 12501 1000000 1000000000 13 21
```
