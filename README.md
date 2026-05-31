# CUDA `ds(n)` Search 🚀

A hyper-optimized, dual-kernel GPU search engine for finding `ds(n)` numbers. 

**Definition:** Let `ds(n)` be the smallest prime number for which the digit sums, when written in bases 2 to `n+1`, are all prime. 

This engine offloads the initial heavy filtering to the GPU using advanced warp-level operations and radix chunking, leaving only a tiny fraction of candidates (approx. 1 in 2,000,000) for the CPU to strictly verify. This specific engine architecture has been successfully used to push the known boundaries of the sequence, verifying extreme ranges and identifying records up to `ds(31)` and `ds(32)`.

## ⚡ Performance

* **GPU:** NVIDIA RTX 4070
* **Throughput:** **~755 Billion candidates / second**
* **Verification:** CPU worker thread processes 1 million GPU survivors in ~50ms (benchmarked on AMD 3950X).
* **Speed:** Can find `ds(2)` to `ds(25)` in roughly **8 seconds** depending on batch sizes.

## 🧠 Architecture & Optimizations

The pipeline is split between an aggressive GPU filtering phase and an asynchronous CPU verification worker.

### GPU Filtering
1. **Mod-210 Prime Wheel:** The kernel natively evaluates 48 pre-calculated offsets to reject any number that is a multiple of 2, 3, 5, or 7 immediately upon generation. This reduces the mathematical search space by over 77% before a single digit sum is even calculated.
2. **Double-Buffered Execution:** The GPU runs two asynchronous kernels. While Stream A is transferring its survivors back to the host via PCIe DMA, Stream B is already crunching the next block of candidates.
3. **Optimized Base Order:** Bases are checked in a statistically optimal order to fail composites as early as possible.
4. **Popcount Optimization:** Powers of two (Bases 2, 4, 8, 16, 32) bypass standard division and instead use extremely fast CUDA hardware population count (`__popcll`) instructions interleaved with bitwise masks.
5. **Radix Chunking:** Even and odd bases (12, 6, 10, 5, 9, 3, 11, 7) utilize compile-time radix chunking narrowing. This minimizes 64-bit calculations by leveraging shared memory lookup tables. 

### CPU Verification
Surviving candidates are processed via a thread-safe queue. The CPU performs dynamic fallback checks (catching any bases up to `maxbase` that the GPU skipped) before executing a rigorous Baillie-PSW / Miller-Rabin style primality test on the number itself.

## 🛠️ Prerequisites

* **NVIDIA GPU** (Compute Capability 8.9+ recommended for Ada Lovelace, but supports older architectures).
* **CUDA Toolkit** * **C++ Compiler** with standard `<thread>` and C++17 support (due to `if constexpr` usage).

## 🚀 Building

The code is meant to be compiled on Windows.
Compile the project directly using `nvcc`. Ensure you optimize for your specific GPU architecture (e.g., `-arch=sm_89` for RTX 40-series).

```
nvcc -O3 -std=c++20 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 -Xptxas "-v" -Xcompiler "/O2 /Ob2 /Ot" ds-cuda.cu -o ds
```

## 📖 Usage

The engine is executed from the command line by defining the block ranges and target bases.

```
./ds <start_subblock> <end_subblock> <subblock_size> <minbase> <maxbase>
```

**Example Run:**
To search for `ds(2)` up to `ds(25)` using an optimal subblock size of 1 billion (1,000,000,000):
```
./ds 0 5000 1000000000 2 26
```

## 💾 State Management & Output

* **`ds_records.txt`**: All successful discoveries are appended here dynamically as comma-formatted integers.
* **`ds_state.txt`**: The engine writes a heartbeat to this file every 60 seconds. If your run is interrupted, this file contains the exact command-line arguments needed to resume the search from the last completed sub-block.
