# ds(n) — GPU-Accelerated Prime Digit-Sum Search

A high-performance CUDA program that searches for **ds(n)** numbers: the smallest prime whose digit sum in every base from 2 through n+1 is also prime.

Searches approximately **2.18 trillion numbers per second** on an RTX 4070.

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

A dedicated CPU worker thread receives GPU survivors and performs full Miller-Rabin primality testing plus any remaining base checks above the GPU's filter range. Confirmed primes that set new ds(n) records are logged immediately. Verification takes 1–5 ms per dispatch against roughly 1000 ms of GPU work, so the CPU has about three orders of magnitude of headroom and is never remotely the bottleneck.

That gap is the design working, not capacity going to waste. A CPU-only version of the same search on a Ryzen 9 3950X measured the RTX 4070 at **860× a single CPU thread**; allowing for all 16 cores and SMT, the whole CPU is worth about 2% of GPU throughput. Nor can the CPU usefully pre-compute anything: the GPU consumes 3.5e11 wheel candidates per second, so even a one-bit hint per candidate would need 43.7 GB/s against ~25 GB/s of PCIe 4.0 — and producing that bit would require the CPU to do GPU-scale work at 2% of the rate.

---

## GPU Kernel Design

### Wheel-9699690 Sieve

Every thread maps to a fixed offset in the mod-9699690 wheel, which eliminates all multiples of 2, 3, 5, 7, 11, 13, 17, and 19 before any digit-sum work begins. This rejects **8,040,810 of every 9,699,690 integers — 82.898%** — at zero runtime cost, since the rejected numbers are never generated in the first place.

| prime | additional kills per revolution | running total |
|---|---|---|
| 2 | 4,849,845 | 4,849,845 |
| 3 | 1,616,615 | 6,466,460 |
| 5 | 646,646 | 7,113,106 |
| 7 | 369,512 | 7,482,618 |
| 11 | 201,552 | 7,684,170 |
| 13 | 155,040 | 7,839,210 |
| 17 | 109,440 | 7,948,650 |
| 19 | 92,160 | 8,040,810 |

There's a tidy structural identity in that last row: the marginal kills of a newly added prime p over modulus M·p are exactly φ(M) — the offset count of the previous wheel. 19 contributes 92,160 kills, which is precisely how many offsets the old mod-510510 wheel had. It also makes the diminishing returns explicit: each new prime kills a fixed count while the modulus it has to cover grows by a factor of p.

The 1,658,880 surviving offsets reach 9,699,689, so they are stored as `uint32_t`. At 6.33 MB the table lives in global memory as a `__device__` array. This still costs nothing in the inner loop: each thread reads exactly one offset before its walk begins, and consecutive lanes read consecutive entries, so a warp takes a single coalesced 128-byte transaction per launch.

**This is the last profitable wheel.** Extending to mod-223092870 (adding 23) would reject a further 4.35%, but φ grows to 36,495,360, forcing the grid quantum to 47,520 blocks and the minimum stride to 223 million. That collapses the high-word hoist window from 221 inner iterations to 19, costing roughly 9.8% in outer-loop overhead to buy 4.35% of filtering. No grid choice escapes it — 47,520 blocks is the floor, not a tuning knob. The wheel and the hoist are complementary at 17 and 19, and in direct opposition at 23.

### Grid Sizing

Two constraints have to be satisfied at once, and a third consideration decides between the survivors.

**Wheel alignment.** Each thread owns exactly one wheel offset for the life of the kernel, so `gridSize × blockSize` must be a whole number of wheel revolutions. At 768 threads per block the quantum is 1658880 / 768 = **2160 blocks**.

**Wave alignment.** `__launch_bounds__(768, 2)` keeps two blocks resident per SM, so the GPU runs waves of `2 × numSMs` blocks. A grid that leaves a partial final wave idles most of the machine for the duration of that wave.

**Hoist window.** A larger grid means a larger stride, which shortens the high-word hoist window and raises the per-candidate share of the outer loop. The launcher therefore picks the *smallest* grid clearing a 99.5% fill threshold, rather than the fullest grid available. On 70 SMs the exact-fill grid is 15,120 blocks — 100% fill, but a 67.9 M stride and only 63 inner iterations per outer, which costs 3.3% in outer-loop overhead against 0.9% at 4320. Chasing the last half-percent of fill loses several times what it returns.

Resident block count is queried with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than assumed, and grid candidates are compared with exact integer cross-multiplication.

| GPU | SMs | wave | grid | fill | stride | inner iters / outer |
|---|---|---|---|---|---|---|
| RTX 4070 | 46 | 92 | 4320 | 99.907% | 19,399,380 | 221 |
| RTX 5070 Ti | 70 | 140 | 4320 | 99.539% | 19,399,380 | 221 |

### Instruction Economics: POPC is Quarter-Rate

Almost every design decision below follows from one hardware fact. On sm_89 the SM has 128 FP32 lanes but only 64 that execute INT32, and a handful of instructions run at a quarter of *that*:

| class | instructions | results / clk / SM |
|---|---|---|
| FP32 | add, mul, FMA | 128 |
| **INT32 baseline** | IADD3, IMAD, LOP3, SHF, ISETP, **IDP4A** | **64** |
| **quarter rate** | **POPC**, FLO, BREV | **16** |
| SFU | rcp, rsqrt, log2, exp2, sin, cos | 16 |

A warp-wide `POPC` occupies its scheduler partition for 8 cycles against 2 for an `IADD3`. In the current kernel popcount is **9.7% of the instruction count but 30.1% of the ALU issue cost**, so a naive instruction count understates the real cost by 1.29×.

This is why base 16 runs *before* base 8 despite being the weaker filter, why `__dp4a` is used wherever a horizontal sum will do, and why hoisting popcounts out of the inner loop was the single largest early win. The kernel issues no SFU instructions at all — there is no floating point in it.

### High-Word Hoisting

The candidate walk is therefore a two-level loop. The outer loop pins the top 32 bits of the candidate; the inner loop varies only the low 32 bits. Because the stride is about 19.4 million, `candidate >> 32` changes only once every ~221 inner iterations, so the high half's contribution to all five power-of-two digit sums is loop-invariant and gets computed once in the outer loop.

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

Bases are checked in order of filtering efficiency per unit cost:

```
Powers of two:   2, 4, 16, 8, 32
Remaining even:  12, 6, 10, 14
Remaining odd:   5, 9, 3, 11, 7, 13, 15
```

The do-while/break structure provides early exit as soon as any base check fails. Divergence is inherent and structural: a warp pays the full instruction latency for any filter that even one of its 32 lanes still needs, so the effective cost of each filter is its instruction count weighted by the probability that at least one lane survives to reach it.

**Base 16 runs before base 8, and this is counter-intuitive.** Base 8 is the stronger filter — marginal pass rate 0.2875 against base 16's 0.4167 — so filter-strength reasoning says it should go first. Running it first was measured and lost 1.3%. The reason is the POPC table above: base 8 is 2 LOP3 + **2 POPC** + IMAD + 2 IADD ≈ 13 ALU-equivalents, while base 16 is 2 LOP3 + SHF + IADD + **DP4A with no POPC at all** ≈ 5. Base 8 costs 2.6× as much, which swamps its extra filtering power. Ordering by strength alone is wrong here; ordering by strength *per unit of weighted cost* gives the order shown.

Measured over 10.26 M real post-wheel candidates at the current search depth:

| filter | candidates reaching | **warps** reaching |
|---|---|---|
| 2 | 100% | 100% |
| 4 | 29.9% | 100.0% |
| 16 | 15.0% | 99.4% |
| 8 | 6.79% | 89.5% |
| 32 | 1.91% | 46.0% |
| 12 | 0.366% | 11.1% |
| 6 | 0.092% | 2.90% |
| 10 | 0.025% | 0.80% |
| 14 | 0.0089% | 0.28% |
| 5, 9, 3, 11, 7, 13, 15 | — | all under 0.08% |

Four of the 10.26 M cleared all sixteen filters, a survival rate of 3.9e-7.

> **These rates are depth-dependent, and locally very noisy.** Within a 2³² window the high bits are fixed, so `ds2 = popc(hi) + popc(lo)` carries a constant offset that shifts the whole popcount distribution relative to the primes near it. Measured across fourteen adjacent windows at the current depth, the base-2 pass rate ranges from 18.5% to 31.2% and the post-base-6 survival rate varies by **8×** — driven purely by `popcount(hi)`. Each dispatch spans ~465 windows, which averages that down to the ~1.65× spread visible in per-dispatch candidate counts. Any A/B comparison must therefore use a fixed block range, or the window effect will swamp the change being measured.

Base 32 is the strongest single filter in the cascade: **4.5×** on the final output, stable from 10¹⁴ to 10¹⁷. Standalone it culls 5.6×, so it loses about 20% of its power to correlation with the filters around it — all sixteen digit sums are functions of the same bit pattern. There is an exact reason for that correlation: casting out nines generalises to ds_b(n) ≡ n (mod b−1), so every digit sum is pinned to n's residues. A corollary is that any candidate divisible by 31 can only pass base 32 if its digit sum is exactly 31 — verified over 642,148 samples, none passed.

Two things follow. The power-of-two prefix is effectively the entire kernel — bases 2/4, 8, and 32 dominate the cycle budget, and all three sit at their instruction floor. And the ordering beyond the powers of two is settled: base 12 leads the remaining group because it is simultaneously the cheapest (5 chunk lookups against base 6's 7) and the strongest (19.2% pass rate against base 6's 25.1%), and everything after base 10 is reached by fewer than one warp in three hundred. Reordering there is unmeasurable.

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
Fusing them is deliberate rather than incidental. Splitting the two checks would save nothing — with a measured base-2 pass rate of 29.9%, the probability that at least one of 32 lanes survives is 99.99%, so the base-4 instructions would be issued anyway. Fusing at least buys instruction-level parallelism between the two independent popcounts.

### Shared Memory and the Carveout Boundary

The small-prime lookup table (`local_sp`) and the two most-reached non-power-of-2 tables (base 6 and base 12) are staged into shared memory via `__pipeline_memcpy_async`. Everything else — bases 3, 5, 7, 9, 10, 11, 13, 14, 15 — is read with `__ldg`.

**Base 10 is deliberately excluded, and the reason is the carveout boundary rather than the table itself.** With base 10 resident the block needs 34,800 bytes, so two resident blocks need 69,600 — over the 64 KB line, which forces Ada's 100 KB shared carveout and leaves only 28 KB of L1 out of the 128 KB unified cache. Dropping base 10 to global brings the block to **24,688 bytes / 49,376 for two**, which fits the 64 KB carveout and leaves ~62 KB of L1 for every other table plus the 6.33 MB wheel. Base 10 is reached by ~0.6% of warp-iterations, so the residency given up costs almost nothing. Measured +1.3%, and confirmed in the profile as `Shared Memory Configuration Size 65.54 KB`.

`local_sp` is small enough that the entries actually indexed span about 12 distinct 32-bit words. Sub-word accesses to the same word broadcast rather than conflict, so 32 lanes land in distinct banks with essentially zero replays — measured at **1.000-way mean conflict degree**. Padding the table to `uint32_t` would make this *worse* (1.267-way), since spreading indices over 4× the address range puts more distinct words in the same bank.

The `extern __shared__` block is declared as `uint4` rather than `uint8_t` so its 16-byte alignment comes from the type system: the loaders issue 16-byte `__pipeline_memcpy_async` stores into it, and a misaligned 16-byte shared store is an illegal-address abort rather than a slow path.

### Pinning the Loop Invariants

Hoisting the `DS*_HI` partial sums into the outer loop is only half the job — ptxas will quietly undo it. Profiling showed `__popc(hi)` executing **214,411,441 times, or 2.01 per *inner* iteration**, despite being written as an outer-loop invariant. It was rematerialising the value inside the loop, and half of those instructions were quarter-rate POPC.

The cause is that `hi` is the high half of the candidate register pair, so re-reading it is free and recomputing from it looks cheap to a cost model that does not know POPC is quarter-rate. The fix is an empty inline asm that makes the value opaque:

```cuda
__device__ __forceinline__ void dsPinRegister(uint32_t &v)
{
    asm("" : "+r"(v));   // no instructions; just blocks rematerialisation
}
```

**The whole chain has to be pinned as a unit.** Pinning `DS2_HI` alone worked exactly as intended — that line fell to 1.00 per outer iteration — but ptxas simply relocated the identical work onto `DS4_HI` for zero net change. Pinning both moved it to `DS8_HI`. Only pinning all five stopped it. The three rounds measured +0.17%, +1.14% and +2.43%.

Two results from this are worth keeping. The +1.14% round came at an **identical instruction count** — 3.8731e9 against 3.8725e9 — because the gain was entirely in moving the recompute off the critical path feeding the first branch. And pinning five extra values *reduced* register usage from 31 to 27: the rematerialisation was not saving a register, it was costing four, since `hi` and the intermediate masks had to stay live to recompute from.

### Inner Loop Unrolling

`#pragma unroll 3` on the inner `while` is worth **+6.7%**. The trip count is not known at that point, so ptxas duplicates the body with an exit check between copies rather than producing a clean 3× body — but that still cuts the back-edge branch and the 64-bit `candidate < hi_limit` compare, which costs 6.02 instructions per iteration on its own, to a third. The larger effect is scheduling: three fully independent filter cascades give the scheduler something to interleave, and the kernel is issue-limited (0.87 warps/cycle against a ceiling of 1.0) rather than ALU-limited, so shortening dependency chains is what actually buys throughput.

The factor was swept rather than guessed, against a 2040 G/s unrolled-1 baseline and a 0.27% variance floor:

| unroll | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| G/s | 2040 | 2142 | **2176** | 2144 | 2125 | 2132 |

2 and 4 differ by two, and 5 and 6 by seven — both inside noise. This is a genuine peak at 3 with everything else on a plateau about 1.5% below, not a curve to round up.

This is **not** the "two candidates per thread" dead end. That tested two candidates inside a single cascade and lost early exit to intra-thread divergence. Here each unrolled copy keeps its own cascade and its own `break`s, so per-candidate divergence is unchanged.

### Direct Emission

Survivors are emitted from inside the filter cascade rather than via a flag tested afterwards. The earlier form —

```cuda
bool pass = false;
do { ... pass = true; } while (0);
if (pass) { /* emit */ }
```

— measured **4.00 instructions per inner iteration and 12.98% of all warp stall samples**, third-largest in the kernel, for what is nominally a boolean test. Every `break` in the cascade already skips the tail of the `do {} while (0)`, so emitting there is identical in behaviour and the flag's init, set, test, branch and reconvergence bookkeeping all disappear.

**Worth +12.4%** — the largest single optimisation in the kernel since high-word hoisting.

### Result Output

Survivors are written with a plain per-thread `atomicAdd` on the result counter. Warp aggregation via `__ballot_sync` was considered and rejected: at a measured survival rate of 3.9e-7 the probability of two lanes in the same warp surviving simultaneously is negligible, so the aggregation logic would cost more than the contention it removes.

The counter is deliberately allowed to run past the buffer capacity so the host can see the true hit count and abort, rather than saturating and reporting a full buffer as a normal result.

### Kernel Templating

The kernel is templated on `MIN_BASE` and instantiated at compile time for values 9 through 16 plus 32. `if constexpr` blocks for higher bases compile away entirely in lower-base instantiations. Register usage ranges from **24 to 27** across all variants with zero spill, comfortably inside the 40-register budget that two resident blocks of 768 threads allows (65536 / 1536 = 42.67, rounded down to the 8-register allocation granularity).

Because the kernel is only instantiated at a few `MIN_BASE` values, the value actually compiled in is not always the campaign's current minbase — a campaign at minbase 31 runs the `MIN_BASE=16` instantiation. The launch dispatcher and the CPU verifier both route through a single `instantiatedMinBase()` function so they cannot disagree about which bases the GPU checked. If they ever did, the verifier would skip a base the kernel never looked at and the run would emit a **false record** with nothing to detect it.

---

## CPU Verification

The Miller-Rabin primality test uses deterministic witness sets sufficient for all 64-bit integers. A dynamic checklist handles the gap between what the GPU kernel checked (fixed at launch time) and the current `global_minbase` target, which may have advanced while the block was in flight.

When a candidate passes all base checks and primality testing, it claims any newly validated ds(n) records via a compare-exchange loop on `global_minbase`, then continues checking higher bases to potentially claim additional consecutive records from the same candidate.

> `global_minbase` holds a **base**, not an n. The convention throughout is base = n + 1, which is why records are logged as `claim_base - 1`.

Ordering matters for correctness, not just tidiness: ds(n) is defined as the *smallest* prime with the property, so survivors are sorted within each payload and payloads are processed strictly FIFO by a single worker. Dispatches that produce no candidates still queue an empty payload, which keeps the heartbeat advancing through the long barren stretches at high minbase.

---

## Robustness

Failure modes that produce wrong answers silently are treated as more dangerous than crashes, since a record search that quietly skips a range is unrecoverable and undetectable.

- **Result-buffer overflow is fatal, not truncating.** If the GPU emits more than `MAX_GPU_RESULTS`, the run aborts naming the sub-block rather than clamping and reporting success on a range it never fully examined.
- **`cudaGetLastError()` after every launch.** A failed launch would otherwise leave the count at its memset zero, queue no payload, and let the dispatch loop advance as though the range simply held no candidates — then checkpoint it as done.
- **Range validation against 64-bit wrap.** `end_block × subblock_size` is bounded by `UINT64_MAX − stride` before the loop starts, which is what lets the inner walk use a bare `candidate += stride` with no wrap test on the hot path.
- **Strict argument parsing.** An unchecked `strtoull` turns a typo into a silent zero, and a zero `subblock_size` makes the whole campaign a no-op that still reports success.
- **Deterministic Miller-Rabin.** Witnesses 2 through 37 are deterministic to ~3.19e24, well past 2⁶⁴ — there is no probabilistic gap anywhere in the searched range.

### Verification

The kernel's arithmetic has been differentially tested against naive digit sums: 600k values across the five power-of-two SWAR decompositions, 122k values × 11 bases across the radix-chunk paths including every `uint32_t` narrowing, and 1.08 billion candidates through the real `MIN_BASE=32` kernel at live search depth. A cold-start run reproduced ds(1) through ds(31) from zero, and ds(1) through ds(21) have been confirmed minimal by exhaustive brute force.

---

## Performance

| stage | RTX 4070 | step |
|---|---|---|
| baseline | 1138.30 G/s | — |
| high-word hoisting | 1553.17 G/s | ×1.364 |
| wheel-510510 | 1602.71 G/s | ×1.032 |
| wave-aligned grid | 1631.25 G/s | ×1.018 |
| wheel-9699690 | 1721.26 G/s | ×1.050 |
| base 10 → global (64 KB carveout) | 1733 G/s | ×1.013 |
| direct emission | 1947 G/s | ×1.124 |
| DS\*_HI chain pinned | 2042.08 G/s | ×1.049 |
| inner loop unrolled 3× | 2176 G/s | ×1.066 |
| | | **×1.912 cumulative** |

Final figure verified over a 312.60 s sustained run, not a short benchmark. Run-to-run variance on a fixed block range is **0.27%**.

RTX 5070 Ti (70 SMs, sm_120) reached **3042.43 G/s** on the mod-9699690 build, before any of the last three optimisations above. Normalised per SM per clock the two cards differ by only 1.15× despite Blackwell unifying INT32 with FP32 and doubling plain integer throughput — almost all of the observed gap is SM count. That is consistent with the kernel being **instruction-issue bound rather than ALU bound**: issue is 4 warp-instructions/clk/SM on both architectures, and widening a pipe that is only 66% utilised buys very little.

Under load on the RTX 4070 the core clock holds a steady 2760 MHz at 173–178 W against a 200 W board limit and 74–75 °C, with SM activity at 100% and **memory bandwidth utilisation at 0–1%**. The kernel is instruction-issue bound, not memory bound, and is neither power- nor thermally throttled.

### Profile summary (final build)

| metric | value |
|---|---|
| SM throughput | 86.84% |
| top pipeline | ALU, 66.1% |
| memory pipes busy | 58.16% |
| DRAM throughput | 0.52% |
| executed IPC | 3.24 / 4 |
| issued warps per scheduler | 0.87 / 1.0 |
| theoretical / achieved occupancy | 100% / 96.38% |
| registers, spills | 27, zero |
| instructions per inner iteration | 35.00 |
| active threads per warp | 19.30 / 32 |

The remaining headroom is warp divergence. At 19.3 of 32 lanes active the profiler estimates 36.8% available in principle, but it is structural to the early-exit cascade and has resisted five separate attacks.

### Validating a change

Per-sub-block candidate counts are deterministic and reproduce exactly across runs, across grid sizes, and across architectures. That makes them a free checksum over roughly 10¹² integers of arithmetic, evaluated in about 12 seconds. Any change meant to preserve behaviour — an instruction rewrite, a grid change, a clock offset — should leave them bit-identical. Changes that legitimately alter the candidate set should move them by exactly the predicted ratio: adding 19 to the wheel predicted 18/19 = 0.9474 and measured 0.9472.

### Verified dead ends

Documented so they aren't re-attempted:

- **Register-resident prime bitmask** replacing the `local_sp` shared lookup. Measured slower. A 64-bit immediate cannot be an immediate operand, so the mask needs registers or a constant-bank load, and the variable 64-bit shift lowers to a funnel-shift pair plus an AND — three or four INT32 ops against one fully-hidden shared load.
- **32-bit inner walk.** Carrying the candidate as two `uint32_t` halves and terminating the inner loop on the carry out of a 32-bit add instead of a 64-bit add plus 64-bit compare. Measured 1% slower: the instruction saving was real, but the 64-bit version has one exit condition feeding one branch where the split version has two, and the deep filters then need the 64-bit value reconstructed.
- **Larger shared-memory carveout** for the `__ldg` tables. With memory utilisation at 1% there is no bandwidth pressure to relieve.
- **Reordering the non-power-of-two filters.** Reached by too few warps to measure — see the cascade table above.
- **Running base 8 before base 16.** Base 8 is the stronger filter, so this looks like a clear win. Measured 1685 against 1711, a 1.3% loss. Base 8's two quarter-rate popcounts make it 2.6× the weighted cost of base 16. Ordering by filter strength alone is the wrong model.
- **Fusing base 8 and base 16 into one branch,** the way bases 2 and 4 are fused. Fusion forces the second filter up to the first one's warp-reach, so it is only cheap when adjacent filters have similar reach. The gaps here are 0.101 (16→8) and 0.420 (8→32); there is no cheap pair. Bases 2 and 4 fuse for free precisely because both sit at reach 1.000.
- **32-bit trip count** replacing the 64-bit `while (candidate < hi_limit)`. Removes ~3 instructions per iteration for one divide per outer iteration, amortised 171.8:1. Measured twice — 1687 against 1732 (−2.6%) and 1978 against 2038.71 (−2.98%) — the second time under conditions that should have favoured it far more.
- **Precomputing anything on the CPU.** Bounded by PCIe, not cleverness: a one-bit-per-candidate hint needs 43.7 GB/s against ~25 GB/s available, and the CPU runs the same cascade at 2% of the GPU's rate.

### The rule these produced

Three of the last six optimisations attempted made the kernel *slower*, and all three failures came from reasoning about instruction counts. The discriminator that actually works:

> **Stall samples on a computation line mean the dependency chain is stalling, and are worth acting on. Stall samples on a loop tail or back-edge mean warps are queueing there, and they will simply queue somewhere else if you remove the instructions.**

`if (pass)` carried 12.98% of stalls on a computation path and removing it gained 12.4%. The loop condition carried 7.36% on a back-edge and removing it lost 2.6% — and after `if (pass)` was deleted, most of its stall share reappeared on `candidate += stride` at an identical instruction count.

---

## Output Files

`ds_records.txt` — appended whenever a new ds(n) record is found:
```
>>> NEW RECORD FOUND: ds(22) = 28,941,023,651 <<<
```

`ds_state.txt` — overwritten every 60 seconds with the exact command needed to resume from the last completed sub-block (the program name is taken from `argv[0]`, so the line is directly runnable):
```
--- ENGINE HEARTBEAT ---
Last Saved: 2026-05-01 14:32:11
Restart Command Args:
<program> 12501 <end_subblock> <subblock_size> 13 <maxbase>
```

> Record values now exceed 15 digits. Any tool that keeps only 15 significant figures — Excel, Google Sheets, `%.15g` — will silently truncate them into composite-looking numbers. Read results from the program's own output.

---

## Requirements

- CUDA Toolkit 12.x or later (developed against 13.3.1)
- C++17 or later for `if constexpr`; built with C++20
- GPU with compute capability 7.0 or higher (Volta+) for `__dp4a`. `__pipeline_memcpy_async` is available from 7.0 and hardware-accelerated from 8.0. Tested on sm_89 (Ada) and sm_120 (Blackwell)
- Windows or Linux. Miller-Rabin uses `_umul128` / `_udiv128` on MSVC and `unsigned __int128` on GCC/Clang
- ~7 MB of free device memory for the wheel table, and the same again in host `.bss`

---

## Build

```bash
nvcc -O3 -std=c++20 -lineinfo ^
     -gencode arch=compute_89,code=sm_89 ^
     -gencode arch=compute_120,code=sm_120 ^
     -gencode arch=compute_120,code=compute_120 ^
     -Xptxas "-v" -Xcompiler "/O2 /Ob2 /Ot" ds-cuda.cu -o ds
```

Adjust `-gencode` for your GPU (sm_70 for Volta, sm_75 for Turing, sm_86 for Ampere, sm_89 for Ada, sm_120 for Blackwell). The trailing `compute_120` entry embeds PTX for forward compatibility; a single `-arch=sm_89` builds faster if you only ever run locally. `-Xptxas -v` reports register usage; watch that it stays at or below **40**, since crossing that drops the second resident block per SM and costs roughly half the throughput. The current build uses 24–27 with zero spill, so there is headroom for changes that need it.

The one piece of inline assembly in the file is hidden from Microsoft's IntelliSense parser with `#if !defined(__INTELLISENSE__)`, since it cannot parse GCC-style extended asm and flags the colon as a syntax error. This changes nothing about what nvcc or clang compile.

Startup builds the 9.7 M-entry wheel on the host, adding 50–100 ms before the first launch.

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
