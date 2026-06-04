# ds(n) — GPU-Accelerated Prime Digit-Sum Search

A high-performance CUDA program that searches for **ds(n)** numbers: the smallest prime whose digit sum in every base from 2 through n+1 is also prime.

Can search approximately **1.1 trillion numbers per second** on an RTX 4070.

---

## The Problem

Define ds(n) as the smallest prime p such that for every base b in {2, 3, 4, ..., n+1}, the sum of the digits of p written in base b is also prime.

The first few known values are:

| n | ds(n) |
|---|-------|
| 1 | 3 |
| 2 | 5 |
| 3 | 5 |
| 4 | 11 |
| 5 | 17 |
| 6 | 17 |
| 7 | 17 |
| 8 | 131 |
| 9 | 131 |
| 10 | 131 |
| 11 | 2663 |
| 12 | 2663 |
| 13 | 892747 |
| 14 | 892747 |
| 15 | 3748207 |
| 16 | 10110857 |
| 17 | 68443401 |
| 18 | 68443401 |
| 19 | 68443401 |
| 20 | 89757221 |
| 21 | 89757221 |
| 22 | 28941023651 |

Finding ds(n) for large n requires testing enormous numbers — the candidate values grow rapidly, making brute-force CPU search infeasible.

---

## Architecture

The search is split into two pipeline stages running concurrently:

**Stage 1 — GPU Filtering**

The GPU kernel applies a fast multi-base digit-sum primality filter to reject the overwhelming majority of candidates. Two kernel instances run on alternating CUDA streams (A/B double-buffering) so that while one stream is transferring results back over PCIe, the other is processing the next block.

**Stage 2 — CPU Verification**

A dedicated CPU worker thread receives GPU survivors and performs full Miller-Rabin primality testing plus any remaining base checks above the GPU's filter range. Confirmed primes that set new ds(n) records are logged immediately.

---

## GPU Kernel Design

### Wheel-30030 Sieve

Every thread maps to a fixed offset in the mod-30030 wheel, which eliminates all multiples of 2, 3, 5, 7, 11, and 13 before any digit-sum work begins. This rejects 80.82% of the integer range for free. The wheel offsets (5,760 entries) are stored in constant memory for broadcast-cache efficiency.

### Base Filter Order

Bases are checked in empirically determined order of filtering efficiency at large candidate values:

```
Powers of two:  2, 4, 16, 8, 32
Even bases:     12, 6, 10
Odd bases:      5, 9, 3, 11, 7
```

The do-while/break structure provides early exit as soon as any base check fails. The vast majority of candidates are eliminated by the first few checks, so later bases are reached by only a tiny fraction of the candidate population.

### Digit-Sum Computation Techniques

Each base uses the cheapest possible arithmetic for that base's structure:

**Base 2** — single `__popcll` instruction.

**Base 4** — two popcounts; the base-2 result `p` is reused:
```cuda
ds4 = p + __popcll(candidate & 0xAAAAAAAAAAAAAAAAULL)
```
This works because popcount(all bits) = Σ bit0 + Σ bit1 across all 2-bit digits, and the masked popcount gives Σ bit1 directly.

**Base 8** — bit-plane decomposition across the 3-bit digit grid, deriving the bit-0 plane from `p` to save one popcount:
```cuda
b1 = __popcll(candidate & 0x2492492492492492ULL)
b2 = __popcll(candidate & 0x4924924924924924ULL)
ds8 = (p - b1 - b2) + (b1 << 1) + (b2 << 2)
```

**Base 16** — nibble extraction into packed bytes followed by `__dp4a` horizontal summation across two 32-bit halves:
```cuda
n_low  = (c_low  & 0x0F0F0F0F) + ((c_low  >> 4) & 0x0F0F0F0F)
n_high = (c_high & 0x0F0F0F0F) + ((c_high >> 4) & 0x0F0F0F0F)
ds16   = __dp4a(n_high, 0x01010101U, __dp4a(n_low, 0x01010101U, 0U))
```
Dotting packed bytes with `0x01010101` is a horizontal byte-sum in two instructions. Faster than the equivalent popcount-based approach because the nibble structure maps directly onto `__dp4a`'s packed arithmetic.

**Base 32** — five-plane bit decomposition with the bit-0 plane derived from `p`:
```cuda
ds32 = (p - b1 - b2 - b3 - b4) + (b1<<1) + (b2<<2) + (b3<<3) + (b4<<4)
```

**Non-power-of-2 bases (3, 5, 6, 7, 9, 10, 11, 12)** — radix-chunked lookup tables. Each 64-bit candidate is split at a power-of-the-base boundary into a high and low part, each small enough to index a precomputed byte table. This avoids all 64-bit division in the kernel; the compiler replaces the constant divisors with multiply-shift sequences, confirmed by zero `IDIV`/`REMAINDER` instructions in SASS output.

### Combined Base 2+4 Check

The first primality check tests bases 2 and 4 simultaneously with a single branch:
```cuda
if (!(local_sp[p] & local_sp[ds4])) break;
```
Both digit sums are computed at near-zero cost (one extra popcount), and ANDing the lookup results eliminates a branch.

### Shared Memory

The small-prime lookup table (`local_sp`) and the three most-used non-power-of-2 base tables (base 6, base 10, base 12) are loaded into shared memory via `__pipeline_memcpy_async` for ~20-cycle vs ~32-cycle latency. Less-frequently-reached tables (base 3, 5, 7, 9, 11) use `__ldg` for L2 cache access.

### Warp-Coalesced Output

Surviving candidates are written to the output buffer using `__ballot_sync` + `__ffs` + `__popc` for a single `atomicAdd` per warp rather than per thread, minimising atomic contention on the result counter.

### Kernel Templating

The kernel is templated on `MIN_BASE` and instantiated at compile time for each relevant value (9 through 32+). `if constexpr` blocks for higher bases compile away entirely in lower-base instantiations, keeping register usage at 24 (on sm_89) with zero spill across all variants.

---

## CPU Verification

The Miller-Rabin primality test uses deterministic witness sets sufficient for all 64-bit integers. A dynamic checklist handles the gap between what the GPU kernel checked (fixed at launch time) and the current `global_minbase` target, which may have advanced while the block was in flight.

When a candidate passes all base checks and primality testing, it claims any newly validated ds(n) records via a compare-exchange loop on `global_minbase`, then continues checking higher bases to potentially claim additional consecutive records from the same candidate.

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

---

## Requirements

- CUDA Toolkit 12.x or later
- GPU with compute capability 7.0 or higher (Volta+) for `__dp4a` (note only tested on 8.9)
- Windows (uses `_umul128` / `_udiv128` for 128-bit arithmetic in Miller-Rabin)
- C++17 (`if constexpr`)

---

## Build

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/O2 /Ob2 /Ot" ds.cu -o ds
```

Adjust `-arch` for your GPU (sm_70 for Volta, sm_75 for Turing, sm_86/sm_89 for Ampere/Ada, sm_120 for Blackwell).

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

### Example — search from the beginning up to ds(20) - note base is n + 1:

```
ds.exe 0 1000000 1000000000 2 21
```

### Example — resume from a heartbeat:

```
ds.exe 12501 1000000 1000000000 13 21
```

---

## Performance

Measured on an RTX 4070 (Ada Lovelace, 46 SMs) with AMD Ryzen 9 3950X host:

The GPU can search a block 1.1 trillion numbers per second.

The GPU runs at 100% SM utilisation (~156W). The CPU has approximately 20× spare headroom and is not the bottleneck.
