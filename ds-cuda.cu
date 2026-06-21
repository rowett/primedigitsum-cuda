// Let ds(n) be the smallest prime number for which the digit sums
// when written in bases 2 to n+1 are all prime.
//
// This program searches for ds(n) numbers.
//
// The GPU is used identify candidate numbers by:
// 1. Using a mod-30030 prime wheel to reject any number that is a
//    multiple of 2, 3, 5, 7, 11, or 13 (this rejects 80.82% of candidates)
// 2. Checking if the digit sums are prime in the following
//    bases in this order: (powers of two) 2, 4, 16, 8, 32,
//                         (even bases)    12, 6, 10, 14,
//                         (odd bases)     5, 9, 3, 11, 7, 13, 15
// The powers of two use popcount instructions with a small
// prime lookup array.
// The even and odd bases use radix chunking narrowing to
// minimize 64bit calculations and the constant divisions
// are replaced by the compiler.
// Empircal testing showed that even bases filter more candidates
// than odd bases and also provided the optimal base order.
//
// The GPU has two kernels so while one is sending back results
// the other is processing the next block of candidates in
// parallel.
//
// Results are appended to 'ds-records.txt'.
// A progress heartbeat is written to 'ds-state.txt' every minute
// containing the command to restart from the last heartbeat.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>
#include <chrono>
#include <time.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

// Helper function to format large integers with commas
std::string formatCommas(uint64_t num)
{
	std::string s = std::to_string(num);
	int insertPosition = s.length() - 3;
	while (insertPosition > 0)
	{
		s.insert(insertPosition, ",");
		insertPosition -= 3;
	}
	return s;
}

// Block size on the GPU
// Kept at 768 for optimal 2-block SM occupancy and register mapping
#define BLOCK_SIZE 768

// Maximum number of candidates returned by the GPU
#define MAX_GPU_RESULTS 20000000ULL

#define CUDA_CHECK(call)                                                           \
	do                                                                           \
	{                                                                            \
		cudaError_t err = (call);                                              \
		if (err != cudaSuccess)                                                \
		{                                                                      \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
				  cudaGetErrorString(err));                                \
			exit(EXIT_FAILURE);                                              \
		}                                                                      \
	} while (0)

// constants for digit sum chunking
// base 3 widened
const uint32_t base3Size = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3;

// base 5 widened
const uint32_t base5Size = 5 * 5 * 5 * 5 * 5 * 5;

// base 6
const uint32_t base6Size = 6 * 6 * 6 * 6;

// base 7
const uint32_t base7Size = 7 * 7 * 7 * 7;

// base 9
const uint32_t base9Size = 9 * 9 * 9 * 9;

// base 10
const uint32_t base10Size = 10 * 10 * 10 * 10;

// base 11
const uint32_t base11Size = 11 * 11 * 11 * 11;

// base 12
const uint32_t base12Size = 12 * 12 * 12 * 12;

// base 13
const uint32_t base13Size = 13 * 13 * 13 * 13;

// base 14
const uint32_t base14Size = 14 * 14 * 14 * 14;

// base 15
const uint32_t base15Size = 15 * 15 * 15 * 15;

// memory alignment in bytes -1
const uint32_t ALIGN = 127;

// bases 16 byte aligned
const uint32_t base3Size16 = (base3Size + ALIGN) & ~ALIGN;
const uint32_t base5Size16 = (base5Size + ALIGN) & ~ALIGN;
const uint32_t base6Size16 = (base6Size + ALIGN) & ~ALIGN;
const uint32_t base7Size16 = (base7Size + ALIGN) & ~ALIGN;
const uint32_t base9Size16 = (base9Size + ALIGN) & ~ALIGN;
const uint32_t base10Size16 = (base10Size + ALIGN) & ~ALIGN;
const uint32_t base11Size16 = (base11Size + ALIGN) & ~ALIGN;
const uint32_t base12Size16 = (base12Size + ALIGN) & ~ALIGN;
const uint32_t base13Size16 = (base13Size + ALIGN) & ~ALIGN;
const uint32_t base14Size16 = (base14Size + ALIGN) & ~ALIGN;
const uint32_t base15Size16 = (base15Size + ALIGN) & ~ALIGN;

// --- Host Verification ---
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t n)
{
	uint64_t high, low, remainder;
	low = _umul128(a, b, &high);
	_udiv128(high, low, n, &remainder);
	return remainder;
}

static uint64_t powmod(uint64_t a, uint64_t r, uint64_t n)
{
	uint64_t x = 1;
	while (r != 0)
	{
		if (r & 1)
			x = mulmod(a, x, n);
		a = mulmod(a, a, n);
		r >>= 1;
	}
	return x;
}

static int spsp(uint64_t n, uint64_t p)
{
	uint64_t x;
	uint64_t r = n - 1;
	int k = 0;
	while ((r & 1) == 0)
	{
		k++;
		r >>= 1;
	}
	x = powmod(p, r, n);
	if (x == 1)
		return 1;
	while (k > 0)
	{
		if (x == n - 1)
			return 1;
		x = mulmod(x, x, n);
		k--;
	}
	return 0;
}

int cpu_isPrime(uint64_t n)
{
	if (n < 2ULL)
		return 0;

	// Quick-pass known small primes
	if (n == 2ULL || n == 3ULL || n == 5ULL || n == 7ULL ||
	    n == 11ULL || n == 13ULL || n == 17ULL || n == 19ULL ||
	    n == 23ULL || n == 29ULL || n == 31ULL || n == 37ULL)
		return 1;

	// Fast hardware modulus to weed out 50%+ of composites
	if ((n & 1) == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0 ||
	    n % 11 == 0 || n % 13 == 0 || n % 17 == 0 || n % 19 == 0 ||
	    n % 23 == 0 || n % 29 == 0 || n % 31 == 0 || n % 37 == 0)
		return 0;

	if (!spsp(n, 2))
		return 0;
	if (n < 2047ULL)
		return 1;
	if (!spsp(n, 3))
		return 0;
	if (n < 1373653ULL)
		return 1;
	if (!spsp(n, 5))
		return 0;
	if (n < 25326001ULL)
		return 1;
	if (!spsp(n, 7))
		return 0;
	if (n < 3215031751ULL)
		return 1;
	if (!spsp(n, 11))
		return 0;
	if (n < 2152302898747ULL)
		return 1;
	if (!spsp(n, 13))
		return 0;
	if (n < 3474749660383ULL)
		return 1;
	if (!spsp(n, 17))
		return 0;
	if (n < 341550071728321ULL)
		return 1;
	if (!spsp(n, 19))
		return 0;
	if (!spsp(n, 23))
		return 0;
	if (n < 3825123056546413051ULL)
		return 1;
	if (!spsp(n, 29) || !spsp(n, 31) || !spsp(n, 37))
		return 0;
	return 1;
}

uint64_t sumDigits(uint64_t value, const uint32_t radix)
{
	uint64_t sum = 0;
	while (value > 0)
	{
		sum += value % radix;
		value /= radix;
	}
	return sum;
}

// 1. The Templated (Compile-Time) version of sumDigits
template <uint32_t RADIX>
inline uint64_t sumDigitsConst(uint64_t value)
{
	uint64_t sum = 0;
	while (value > 0)
	{
		sum += value % RADIX;
		value /= RADIX;
	}
	return sum;
}

// 2. The Router to force compile-time evaluation
inline uint64_t sumDigitsFast(uint64_t value, uint32_t radix)
{
	switch (radix)
	{
	case 2:
		return sumDigitsConst<2>(value);
	case 3:
		return sumDigitsConst<3>(value);
	case 4:
		return sumDigitsConst<4>(value);
	case 5:
		return sumDigitsConst<5>(value);
	case 6:
		return sumDigitsConst<6>(value);
	case 7:
		return sumDigitsConst<7>(value);
	case 8:
		return sumDigitsConst<8>(value);
	case 9:
		return sumDigitsConst<9>(value);
	case 10:
		return sumDigitsConst<10>(value);
	case 11:
		return sumDigitsConst<11>(value);
	case 12:
		return sumDigitsConst<12>(value);
	case 13:
		return sumDigitsConst<13>(value);
	case 14:
		return sumDigitsConst<14>(value);
	case 15:
		return sumDigitsConst<15>(value);
	case 16:
		return sumDigitsConst<16>(value);
	case 17:
		return sumDigitsConst<17>(value);
	case 18:
		return sumDigitsConst<18>(value);
	case 19:
		return sumDigitsConst<19>(value);
	case 20:
		return sumDigitsConst<20>(value);
	case 21:
		return sumDigitsConst<21>(value);
	case 22:
		return sumDigitsConst<22>(value);
	case 23:
		return sumDigitsConst<23>(value);
	case 24:
		return sumDigitsConst<24>(value);
	case 25:
		return sumDigitsConst<25>(value);
	case 26:
		return sumDigitsConst<26>(value);
	case 27:
		return sumDigitsConst<27>(value);
	case 28:
		return sumDigitsConst<28>(value);
	case 29:
		return sumDigitsConst<29>(value);
	case 30:
		return sumDigitsConst<30>(value);
	case 31:
		return sumDigitsConst<31>(value);
	case 32:
		return sumDigitsConst<32>(value);
	case 33:
		return sumDigitsConst<33>(value);
	case 34:
		return sumDigitsConst<34>(value);
	case 35:
		return sumDigitsConst<35>(value);
	case 36:
		return sumDigitsConst<36>(value);
	case 37:
		return sumDigitsConst<37>(value);
	case 38:
		return sumDigitsConst<38>(value);
	case 39:
		return sumDigitsConst<39>(value);
	case 40:
		return sumDigitsConst<40>(value);
	default:
		return sumDigits(value, radix); // Fallback to slow division if base > 40
	}
}

struct BufferBundle
{
	uint64_t *memory;
	cudaEvent_t ready_event;
};

struct ComputePayload
{
	BufferBundle bundle;
	uint32_t candidate_count;
	uint64_t raw_start_range;
	uint64_t subblock_id;
	uint32_t kernel_minbase;
};

std::queue<ComputePayload> verification_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool engine_running = true;

std::atomic<uint32_t> global_minbase{0};
std::atomic<bool> global_target_achieved{false};
uint8_t *global_smallprimes;

constexpr int POOL_SIZE = 4;
uint64_t *h_buffer_pool[POOL_SIZE];

std::queue<BufferBundle> free_buffers;
std::mutex pool_mutex;
std::condition_variable pool_cv;

std::mutex file_mutex; // Protects file I/O

void logRecord(uint32_t base, uint64_t candidate)
{
	std::lock_guard<std::mutex> lock(file_mutex);

	// Convert the candidate number into a comma-separated string
	std::string formatted_candidate = formatCommas(candidate);

	// Output to screen
	printf(">>> NEW RECORD FOUND: ds(%u) = %s <<<\n", base, formatted_candidate.c_str());
	fflush(stdout);

	// Append to file
	FILE *f = fopen("ds_records.txt", "a");
	if (f != NULL)
	{
		fprintf(f, ">>> NEW RECORD FOUND: ds(%u) = %s <<<\n", base, formatted_candidate.c_str());
		fclose(f);
	}
}

void saveHeartbeatState(uint64_t completed_block, uint64_t end_block, uint64_t subblock_size, uint32_t current_base, uint32_t max_base)
{
	std::lock_guard<std::mutex> lock(file_mutex);

	// Open in overwrite mode ("w") so it only stores the latest state
	FILE *f = fopen("ds_state.txt", "w");
	if (f != NULL)
	{
		time_t rawtime;
		struct tm *timeinfo;
		char buffer[80];
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

		fprintf(f, "--- ENGINE HEARTBEAT ---\n");
		fprintf(f, "Last Saved: %s\n", buffer);
		fprintf(f, "Restart Command Args:\n");
		// Print the exact arguments you need to paste into the terminal to resume
		fprintf(f, ".\\ds %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %u\n",
			  completed_block + 1, end_block, subblock_size, current_base, max_base);
		fclose(f);
	}
}

// ------------------------------------------------------------------------------------
// SILICON WALL CONSTANTS: Wheel 30030 and Warp Masks explicitly placed in Constant Bank
// ------------------------------------------------------------------------------------

__constant__ uint16_t c_wheel_30030[5760]; // 5760 offsets coprime to 2, 3, 5, 7, 11, and 13 (Takes 11.5 KB)
__constant__ uint32_t c_lane_mask[32];	 // Pre-calculated offset masks for warp counting

// This is the GPU kernel that filters candidate numbers
template <uint32_t MIN_BASE>
__global__ void __launch_bounds__(BLOCK_SIZE, 2) unifiedSearchKernel(
    const uint64_t start_range, const uint64_t total_numbers, const uint64_t end_range,
    const uint8_t *__restrict__ g_sp, const uint32_t sp_size,
    uint64_t *__restrict__ d_results, uint32_t *__restrict__ d_count,
    const uint8_t *__restrict__ b3, const uint8_t *__restrict__ b5,
    const uint8_t *__restrict__ b6, const uint8_t *__restrict__ b7,
    const uint8_t *__restrict__ b9, const uint8_t *__restrict__ b10,
    const uint8_t *__restrict__ b11, const uint8_t *__restrict__ b12,
    const uint8_t *__restrict__ b13, const uint8_t *__restrict__ b14,
    const uint8_t *__restrict__ b15,
    const uint64_t stride)
{
	// --- SHARED MEMORY ALLOCATION BLOCK ---
	extern __shared__ uint8_t s_mem[];

	// Constant pointers to mutable shared memory during initialization
	uint8_t *const s_sp = s_mem;
	uint8_t *const s_b12 = s_sp + sp_size;
	uint8_t *const s_b6 = s_b12 + base12Size16;
	uint8_t *const s_b10 = s_b6 + base6Size16;

	// Collaboratively load tables into ultra-fast L1 Shared Memory
	// Scoped heavily to allow const parameters on iterators and pointers
	{
		uint4 *const shared = (uint4 *)s_sp;
		const uint4 *const global = (const uint4 *)g_sp;
		const uint32_t words = sp_size >> 4;
		for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
			__pipeline_memcpy_async(&shared[i], &global[i], 16);
	}

	{
		uint4 *const shared = (uint4 *)s_b6;
		const uint4 *const global = (const uint4 *)b6;
		const uint32_t words = base6Size16 >> 4;
		for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
			__pipeline_memcpy_async(&shared[i], &global[i], 16);
	}

	{
		uint4 *const shared = (uint4 *)s_b10;
		const uint4 *const global = (const uint4 *)b10;
		const uint32_t words = base10Size16 >> 4;
		for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
			__pipeline_memcpy_async(&shared[i], &global[i], 16);
	}

	{
		uint4 *const shared = (uint4 *)s_b12;
		const uint4 *const global = (const uint4 *)b12;
		const uint32_t words = base12Size16 >> 4;
		for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
			__pipeline_memcpy_async(&shared[i], &global[i], 16);
	}

	// Re-alias to strictly const data pointers for the compute phase
	const uint8_t *__restrict__ const local_sp = s_sp;
	const uint8_t *__restrict__ const local_b12 = s_b12;
	const uint8_t *__restrict__ const local_b6 = s_b6;
	const uint8_t *__restrict__ const local_b10 = s_b10;

	// Calculate global thread mapping for Wheel 30030
	const uint64_t global_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

	// Fast exact math
	const uint32_t wheel_idx = global_id % 5760ULL;
	const uint64_t cycle = global_id / 5760ULL;

	// Initial candidate via the Constant Cache broadcast
	uint64_t candidate = start_range + (cycle * 30030ULL) + c_wheel_30030[wheel_idx];

	// wait for the tables to finish copying
	__pipeline_commit();
	__pipeline_wait_prior(0);
	__syncthreads();

	while (candidate < end_range)
	{
		bool pass = false;

		do
		{
			// Base 2
			const uint32_t p = __popcll(candidate);

			// Base 4
			const uint32_t ds4 = p + __popcll(candidate & 0xAAAAAAAAAAAAAAAAULL);

			// check base 2 and base 4 simultaneously
			if (!(local_sp[p] & local_sp[ds4]))
				break;

			// Base 16
			if constexpr (MIN_BASE >= 16)
			{
				const uint32_t c_low = (uint32_t)candidate;
				const uint32_t c_high = (uint32_t)(candidate >> 32);

				const uint32_t n_low = (c_low & 0x0F0F0F0F) + ((c_low >> 4) & 0x0F0F0F0F);
				const uint32_t n_high = (c_high & 0x0F0F0F0F) + ((c_high >> 4) & 0x0F0F0F0F);

				// Sum the 8 bytes across two single-cycle instructions
				const uint32_t ds16 = __dp4a(n_high, 0x01010101U, __dp4a(n_low, 0x01010101U, 0U));

				if (!local_sp[ds16])
					break;
			}

			// Base 8
			const uint32_t b1 = __popcll(candidate & 0x2492492492492492ULL);
			const uint32_t b2 = __popcll(candidate & 0x4924924924924924ULL);
			const uint32_t ds8 = p + b1 + b2 + (b2 << 1);

			if (!local_sp[ds8])
				break;

			// Base 32
			if constexpr (MIN_BASE >= 32)
			{
				const uint32_t b1 = __popcll(candidate & 0x2108421084210842ULL);
				const uint32_t b2 = __popcll(candidate & 0x4210842108421084ULL);
				const uint32_t b3 = __popcll(candidate & 0x8421084210842108ULL);
				const uint32_t b4 = __popcll(candidate & 0x0842108421084210ULL);
				const uint32_t ds32 = p + b1 + ((b2 << 1) + b2) + ((b3 << 3) - b3) + ((b4 << 4) - b4);

				if (!local_sp[ds32])
					break;
			}

			// --- Base 12 ---
			if constexpr (MIN_BASE >= 12)
			{
				const uint64_t v_hi = candidate / 429981696ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 429981696ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 20736ULL;
				sum += local_b12[r64 - rq64 * 20736ULL];

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 20736U;
				sum += local_b12[r - rq * 20736U];

				r = rq;
				sum += local_b12[r];

				r = v_low;
				rq = r / 20736U;
				sum += local_b12[r - rq * 20736U];

				r = rq;
				sum += local_b12[r];

				if (!local_sp[sum])
					break;
			}

			// --- Base 6 ---
			{
				const uint64_t v_hi = candidate / 2176782336ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 2176782336ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 1296ULL;
				sum += local_b6[r64 - rq64 * 1296ULL];

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 1296U;
				sum += local_b6[r - rq * 1296U];

				r = rq;
				rq = r / 1296U;
				sum += local_b6[r - rq * 1296U];

				r = rq;
				sum += local_b6[r];

				r = v_low;
				rq = r / 1296U;
				sum += local_b6[r - rq * 1296U];

				r = rq;
				rq = r / 1296U;
				sum += local_b6[r - rq * 1296U];

				r = rq;
				sum += local_b6[r];

				if (!local_sp[sum])
					break;
			}

			// --- Base 10 ---
			if constexpr (MIN_BASE >= 10)
			{
				const uint64_t v_hi = candidate / 1000000000ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 1000000000ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 10000ULL;
				sum += local_b10[r64 - rq64 * 10000ULL];

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 10000U;
				sum += local_b10[r - rq * 10000U];

				r = rq;
				sum += local_b10[r];

				r = v_low;
				rq = r / 10000U;
				sum += local_b10[r - rq * 10000U];

				r = rq;
				rq = r / 10000U;
				sum += local_b10[r - rq * 10000U];

				r = rq;
				sum += local_b10[r];

				if (!local_sp[sum])
					break;
			}

			// --- Base 14 ---
			if constexpr (MIN_BASE >= 14)
			{
				// 14^8 = 1,475,789,056
				const uint64_t v_hi = candidate / 1475789056ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 1475789056ULL);

				uint32_t sum = 0;

				// Chunking factor 14^4 = 38416
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 38416ULL;
				sum += __ldg(&b14[r64 - rq64 * 38416ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 38416U;
				sum += __ldg(&b14[r - rq * 38416U]);

				r = rq;
				sum += __ldg(&b14[r]);

				r = v_low;
				rq = r / 38416U;
				sum += __ldg(&b14[r - rq * 38416U]);

				r = rq;
				rq = r / 38416U;
				sum += __ldg(&b14[r - rq * 38416U]);

				r = rq;
				sum += __ldg(&b14[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 5 ---
			{
				const uint64_t v_hi = candidate / 244140625ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 244140625ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 15625ULL;
				sum += __ldg(&b5[r64 - rq64 * 15625ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 15625U;
				sum += __ldg(&b5[r - rq * 15625U]);

				r = rq;
				sum += __ldg(&b5[r]);

				r = v_low;
				rq = r / 15625U;
				sum += __ldg(&b5[r - rq * 15625U]);

				r = rq;
				sum += __ldg(&b5[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 9 ---
			{
				const uint64_t v_hi = candidate / 3486784401ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 3486784401ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 6561ULL;
				sum += __ldg(&b9[r64 - rq64 * 6561ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 6561U;
				sum += __ldg(&b9[r - rq * 6561U]);

				r = rq;
				sum += __ldg(&b9[r]);

				r = v_low;
				rq = r / 6561U;
				sum += __ldg(&b9[r - rq * 6561U]);

				r = rq;
				rq = r / 6561U;
				sum += __ldg(&b9[r - rq * 6561U]);

				r = rq;
				sum += __ldg(&b9[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 3 ---
			{
				const uint64_t v_hi = candidate / 43046721ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 43046721ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 6561ULL;
				sum += __ldg(&b3[r64 - rq64 * 6561ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 6561U;
				sum += __ldg(&b3[r - rq * 6561U]);

				r = rq;
				rq = r / 6561U;
				sum += __ldg(&b3[r - rq * 6561U]);

				r = rq;
				sum += __ldg(&b3[r]);

				r = v_low;
				rq = r / 6561U;
				sum += __ldg(&b3[r - rq * 6561U]);

				r = rq;
				sum += __ldg(&b3[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 11 ---
			if constexpr (MIN_BASE >= 11)
			{
				const uint64_t v_hi = candidate / 2357947691ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 2357947691ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 14641ULL;
				sum += __ldg(&b11[r64 - rq64 * 14641ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 14641U;
				sum += __ldg(&b11[r - rq * 14641U]);

				r = rq;
				sum += __ldg(&b11[r]);

				r = v_low;
				rq = r / 14641U;
				sum += __ldg(&b11[r - rq * 14641U]);

				r = rq;
				rq = r / 14641U;
				sum += __ldg(&b11[r - rq * 14641U]);

				r = rq;
				sum += __ldg(&b11[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 7 ---
			{
				const uint64_t v_hi = candidate / 1977326743ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 1977326743ULL);

				uint32_t sum = 0;
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 2401ULL;
				sum += __ldg(&b7[r64 - rq64 * 2401ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 2401U;
				sum += __ldg(&b7[r - rq * 2401U]);

				r = rq;
				sum += __ldg(&b7[r]);

				r = v_low;
				rq = r / 2401U;
				sum += __ldg(&b7[r - rq * 2401U]);

				r = rq;
				rq = r / 2401U;
				sum += __ldg(&b7[r - rq * 2401U]);

				r = rq;
				sum += __ldg(&b7[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 13 ---
			if constexpr (MIN_BASE >= 13)
			{
				// 13^8 = 815,730,721
				const uint64_t v_hi = candidate / 815730721ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 815730721ULL);

				uint32_t sum = 0;

				// Chunking factor 13^4 = 28561
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 28561ULL;
				sum += __ldg(&b13[r64 - rq64 * 28561ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 28561U;
				sum += __ldg(&b13[r - rq * 28561U]);

				r = rq;
				sum += __ldg(&b13[r]);

				r = v_low;
				rq = r / 28561U;
				sum += __ldg(&b13[r - rq * 28561U]);

				r = rq;
				rq = r / 28561U;
				sum += __ldg(&b13[r - rq * 28561U]);

				r = rq;
				sum += __ldg(&b13[r]);

				if (!local_sp[sum])
					break;
			}

			// --- Base 15 ---
			if constexpr (MIN_BASE >= 15)
			{
				// 15^8 = 2,562,890,625
				const uint64_t v_hi = candidate / 2562890625ULL;
				const uint32_t v_low = (uint32_t)(candidate - v_hi * 2562890625ULL);

				uint32_t sum = 0;

				// Chunking factor 15^4 = 50625
				uint64_t r64 = v_hi;
				uint64_t rq64 = r64 / 50625ULL;
				sum += __ldg(&b15[r64 - rq64 * 50625ULL]);

				r64 = rq64;
				uint32_t r = (uint32_t)r64;
				uint32_t rq = r / 50625U;
				sum += __ldg(&b15[r - rq * 50625U]);

				r = rq;
				sum += __ldg(&b15[r]);

				r = v_low;
				rq = r / 50625U;
				sum += __ldg(&b15[r - rq * 50625U]);

				r = rq;
				rq = r / 50625U;
				sum += __ldg(&b15[r - rq * 50625U]);

				r = rq;
				sum += __ldg(&b15[r]);

				if (!local_sp[sum])
					break;
			}

			// If we successfully run the gauntlet without breaking:
			pass = true;
		} while (0);

		const uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, pass);
		if (warp_mask)
		{
			const uint32_t warp_count = __popc(warp_mask);
			const uint32_t elect_lane = __ffs(warp_mask) - 1; // lowest set bit = first active lane
			const uint32_t lane_id = threadIdx.x & 31;
			uint32_t base_idx = 0;

			if (lane_id == elect_lane)
			{
				base_idx = atomicAdd(d_count, warp_count);
			}
			base_idx = __shfl_sync(0xFFFFFFFF, base_idx, elect_lane);

			const uint32_t local_offset = __popc(warp_mask & c_lane_mask[lane_id]);
			if (pass && base_idx + local_offset < MAX_GPU_RESULTS)
			{
				d_results[base_idx + local_offset] = candidate;
			}
		}

		candidate += stride;
	}
}

// CPU verification worker that checks surviving candidates from the GPU filtering
void verificationWorker(uint64_t end_block, uint64_t subblock_size, uint32_t max_target_base)
{
	auto last_heartbeat_time = std::chrono::steady_clock::now();
	const std::chrono::seconds HEARTBEAT_INTERVAL(60); // Heartbeat interval is 60 seconds

	while (true)
	{
		ComputePayload task;
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			queue_cv.wait(lock, []
					  { return !verification_queue.empty() || !engine_running; });

			if (!engine_running && verification_queue.empty())
				break;

			task = verification_queue.front();
			verification_queue.pop();
		}

		// THE BARRIER: Wait for the DMA transfer to physically finish across the PCIe bus
		CUDA_CHECK(cudaEventSynchronize(task.bundle.ready_event));

		auto start_time = std::chrono::high_resolution_clock::now();
		size_t survivor_count = 0;

		if (task.candidate_count > 0)
		{
			std::vector<uint64_t> survivors;
			uint32_t current_needed = global_minbase.load();
			uint32_t max_fast_check = std::min(current_needed, max_target_base);

			// ---------------------------------------------------------
			// BUILD THE DYNAMIC CHECKLIST (Handles non-sequential gaps)
			// ---------------------------------------------------------
			uint8_t bases_to_check[64];
			uint32_t check_count = 0;

			for (uint32_t b = 2; b <= max_fast_check; ++b)
			{
				bool checked_by_gpu = false;

				// Only assume the GPU checked it if it was within the kernel's target scope
				if (b <= task.kernel_minbase)
				{
					// SWAR Bases handled by GPU
					if (b == 2 || b == 4 || b == 8 || b == 16 || b == 32)
						checked_by_gpu = true;

					// Modulo Bases handled by GPU
					if (b == 3 || b == 5 || b == 6 || b == 7 || b == 9 || b == 10 || b == 11 || b == 12 || b == 13 || b == 14 || b == 15)
						checked_by_gpu = true;
				}

				if (!checked_by_gpu)
				{
					bases_to_check[check_count++] = b;
				}
			}

			// 1. Initial array pre-filter based on the block's launch state
			for (uint32_t i = 0; i < task.candidate_count; i++)
			{
				uint64_t candidate = task.bundle.memory[i];
				if (candidate < task.raw_start_range)
					continue;

				bool survived = true;

				// Un-branched, sequentially accessed L1 cache loop
				for (uint32_t c = 0; c < check_count; ++c)
				{
					uint32_t b = bases_to_check[c];
					if (!global_smallprimes[sumDigitsFast(candidate, b)])
					{
						survived = false;
						break;
					}
				}

				if (survived)
					survivors.push_back(candidate);
			}

			survivor_count = survivors.size();
			std::sort(survivors.begin(), survivors.end());

			// 2. The Dynamic Record Hunt
			for (uint64_t candidate : survivors)
			{
				uint32_t current_expected = global_minbase.load();
				if (current_expected > max_target_base)
					break;

				bool ds_valid = true;

				// Because of our dynamic checklist in Loop 1, we know with 100% certainty
				// that everything up to max_fast_check is fully validated.
				uint32_t highest_checked = max_fast_check;

				// If the engine has advanced since this block was launched,
				// catch up on the missing fast digit sum checks FIRST.
				if (current_expected > highest_checked)
				{
					for (uint32_t r = highest_checked + 1; r <= current_expected; r++)
					{
						if (!global_smallprimes[sumDigitsFast(candidate, r)])
						{
							ds_valid = false;
							break;
						}
					}
					if (!ds_valid)
						continue; // Fast Rejection! CPU primality test bypassed.
					highest_checked = current_expected;
				}
				// ------------------------

				// Candidate satisfies all bases up to current_expected. Now do the heavy math.
				if (!cpu_isPrime(candidate))
					continue;

				// It's prime! Claim any records we just validated.
				uint32_t claim_base = global_minbase.load();
				while (claim_base <= highest_checked && claim_base <= max_target_base)
				{
					if (global_minbase.compare_exchange_weak(claim_base, claim_base + 1))
					{
						logRecord(claim_base - 1, candidate);
						if (claim_base >= max_target_base)
						{
							global_target_achieved.store(true);
							break;
						}
						claim_base++;
					}
				}
				if (global_target_achieved.load())
					break;

				// Continue checking higher bases for this specific candidate
				uint32_t next_r = highest_checked + 1;
				while (next_r <= max_target_base)
				{
					if (!global_smallprimes[sumDigitsFast(candidate, next_r)])
						break;

					claim_base = global_minbase.load();
					while (claim_base <= next_r && claim_base <= max_target_base)
					{
						if (global_minbase.compare_exchange_weak(claim_base, claim_base + 1))
						{
							logRecord(claim_base - 1, candidate);
							if (claim_base >= max_target_base)
							{
								global_target_achieved.store(true);
								break;
							}
							claim_base++;
						}
					}
					if (global_target_achieved.load())
						break;
					next_r++;
				}
			}
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

		std::string formatted_subblock = formatCommas(task.subblock_id + 1);
		std::string formatted_candidate_count = formatCommas(task.candidate_count);
		std::string formatted_survivor_count = formatCommas(survivor_count);

		printf("Next is ds(%u): Sub-Block %s had %s candidates, %s survivors (processing time: %.2f ms)\n",
			 global_minbase.load() - 1, formatted_subblock.c_str(), formatted_candidate_count.c_str(), formatted_survivor_count.c_str(), elapsed.count());
		fflush(stdout);

		// Heartbeat to save state at regular intervals
		auto current_time = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_heartbeat_time) >= HEARTBEAT_INTERVAL)
		{
			saveHeartbeatState(task.subblock_id, end_block, subblock_size, global_minbase.load(), max_target_base);
			last_heartbeat_time = current_time;
		}

		// Return the buffer and event bundle to the pool for the GPU to reuse
		{
			std::lock_guard<std::mutex> lock(pool_mutex);
			free_buffers.push(task.bundle);
		}
		pool_cv.notify_one();
	}
}

// Main entry point
int main(int argc, char **argv)
{
	if (argc != 6)
	{
		fprintf(stderr, "Usage: %s start_subblock end_subblock subblock_size minbase maxbase\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	uint64_t start_block = strtoull(argv[1], NULL, 10);
	uint64_t end_block = strtoull(argv[2], NULL, 10);
	uint64_t subblock_size = strtoull(argv[3], NULL, 10);

	global_minbase.store(strtoul(argv[4], NULL, 10));
	uint32_t max_target_base = strtoul(argv[5], NULL, 10);

	uint32_t largestds = 65 * (max_target_base - 1);
	uint32_t sp_bytes = largestds + 1;
	if (sp_bytes < 155)
		sp_bytes = 155;

	// 16 byte align
	sp_bytes = (sp_bytes + 15) & ~15;

	global_smallprimes = (uint8_t *)calloc(sp_bytes, sizeof(uint8_t));
	for (uint32_t i = 0; i <= largestds; i++)
		global_smallprimes[i] = cpu_isPrime(i) ? 1 : 0;

	// The widened Base 3 and Base 5 CPU generation
	uint8_t h_base3[base3Size];
	for (int i = 0; i < base3Size; i++)
		h_base3[i] = sumDigits(i, 3);

	uint8_t h_base5[base5Size];
	for (int i = 0; i < base5Size; i++)
		h_base5[i] = sumDigits(i, 5);

	uint8_t h_base6[base6Size];
	for (int i = 0; i < base6Size; i++)
		h_base6[i] = sumDigits(i, 6);

	uint8_t h_base7[base7Size];
	for (int i = 0; i < base7Size; i++)
		h_base7[i] = sumDigits(i, 7);

	uint8_t h_base9[base9Size];
	for (int i = 0; i < base9Size; i++)
		h_base9[i] = sumDigits(i, 9);

	uint8_t h_base10[base10Size];
	for (int i = 0; i < base10Size; i++)
		h_base10[i] = sumDigits(i, 10);

	uint8_t h_base11[base11Size];
	for (int i = 0; i < base11Size; i++)
		h_base11[i] = sumDigits(i, 11);

	uint8_t h_base12[base12Size];
	for (int i = 0; i < base12Size; i++)
		h_base12[i] = sumDigits(i, 12);

	uint8_t h_base13[base13Size];
	for (int i = 0; i < base13Size; i++)
		h_base13[i] = sumDigits(i, 13);

	uint8_t h_base14[base14Size];
	for (int i = 0; i < base14Size; i++)
		h_base14[i] = sumDigits(i, 14);

	uint8_t h_base15[base15Size];
	for (int i = 0; i < base15Size; i++)
		h_base15[i] = sumDigits(i, 15);

	// Setup Wheel 30030 and masks in Constant Memory ---
	uint16_t h_wheel_30030[5760];
	int w_idx = 0;
	for (int i = 1; i < 30030; i++)
	{
		if (i % 2 != 0 && i % 3 != 0 && i % 5 != 0 && i % 7 != 0 && i % 11 != 0 && i % 13 != 0)
		{
			h_wheel_30030[w_idx++] = (uint16_t)i;
		}
	}
	CUDA_CHECK(cudaMemcpyToSymbol(c_wheel_30030, h_wheel_30030, sizeof(h_wheel_30030)));

	uint32_t h_lane_mask[32];
	for (int i = 0; i < 32; i++)
	{
		h_lane_mask[i] = (1u << i) - 1;
	}
	CUDA_CHECK(cudaMemcpyToSymbol(c_lane_mask, h_lane_mask, sizeof(h_lane_mask)));

	uint8_t *d_sp, *d_b3, *d_b5, *d_b6, *d_b7, *d_b9, *d_b10, *d_b11, *d_b12, *d_b13, *d_b14, *d_b15;
	CUDA_CHECK(cudaMalloc((void **)&d_sp, sp_bytes));
	CUDA_CHECK(cudaMalloc((void **)&d_b3, base3Size16)); // Expanded allocation
	CUDA_CHECK(cudaMalloc((void **)&d_b5, base5Size16)); // Expanded allocation
	CUDA_CHECK(cudaMalloc((void **)&d_b6, base6Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b7, base7Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b9, base9Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b10, base10Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b11, base11Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b12, base12Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b13, base13Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b14, base14Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b15, base15Size16));

	CUDA_CHECK(cudaMemcpy(d_sp, global_smallprimes, sp_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b3, h_base3, base3Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b5, h_base5, base5Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b6, h_base6, base6Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b7, h_base7, base7Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b9, h_base9, base9Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b10, h_base10, base10Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b11, h_base11, base11Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b12, h_base12, base12Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b13, h_base13, base13Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b14, h_base14, base14Size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b15, h_base15, base15Size, cudaMemcpyHostToDevice));

	uint64_t *d_results_A, *d_results_B;
	uint32_t *d_count_A, *d_count_B;
	CUDA_CHECK(cudaMalloc((void **)&d_results_A, MAX_GPU_RESULTS * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_results_B, MAX_GPU_RESULTS * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_count_A, sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_count_B, sizeof(uint32_t)));

	uint32_t *h_count_A, *h_count_B;
	CUDA_CHECK(cudaHostAlloc((void **)&h_count_A, sizeof(uint32_t), cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc((void **)&h_count_B, sizeof(uint32_t), cudaHostAllocDefault));

	cudaStream_t stream_A, stream_B;
	CUDA_CHECK(cudaStreamCreate(&stream_A));
	CUDA_CHECK(cudaStreamCreate(&stream_B));

	cudaStream_t stream_transfer;
	CUDA_CHECK(cudaStreamCreate(&stream_transfer));

	for (int i = 0; i < POOL_SIZE; i++)
	{
		CUDA_CHECK(cudaHostAlloc((void **)&h_buffer_pool[i], MAX_GPU_RESULTS * sizeof(uint64_t), cudaHostAllocDefault));

		cudaEvent_t event;
		CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync));
		free_buffers.push({h_buffer_pool[i], event});
	}

	int numSMs;
	CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
	int blockSize = BLOCK_SIZE; // Fixed 768
	int gridSize = numSMs * 32;

	// Perfect Grid Divisibility (Wheel 30030)
	// To use Wheel 30030 efficiently, we map exactly 1 physical thread to 1 logical wheel offset.
	// This means (gridSize * 768) MUST be perfectly divisible by 5760.
	// 5760 / 768 = 7.5. Therefore, gridSize must be a multiple of 15.
	// E.g., for an RTX 4070 (46 SMs), original grid is 1472. (1472 / 15) * 15 = 1470 blocks.
	gridSize = (gridSize / 15) * 15;

	// Calculate stride based on perfect alignment
	// stride = exactly how many full wheels the grid processes per inner loop iteration
	const uint64_t stride = ((uint64_t)gridSize * (uint64_t)blockSize / 5760ULL) * 30030ULL;

	// get the GPU name
	int deviceId;
	CUDA_CHECK(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

	printf("====================================================\n");
	printf(" Active GPU: %s Compute %d.%d\n", prop.name, prop.major, prop.minor);
	printf(" Execution Grid Topology: %d blocks x %d threads\n", gridSize, blockSize);
	printf(" Campaign Scope Targets: Bases %u to %u\n", global_minbase.load(), max_target_base);
	printf("====================================================\n");

	if (global_minbase.load() <= 2 && 2 <= max_target_base)
	{
		logRecord(1, 3);
		global_minbase.store(3);
	}
	if (global_minbase.load() <= 3 && 3 <= max_target_base)
	{
		logRecord(2, 5);
		global_minbase.store(4);
	}
	if (global_minbase.load() <= 4 && 4 <= max_target_base)
	{
		logRecord(3, 5);
		global_minbase.store(5);
	}
	if (global_minbase.load() <= 5 && 5 <= max_target_base)
	{
		logRecord(4, 11);
		global_minbase.store(6);
	}
	if (global_minbase.load() <= 6 && 6 <= max_target_base)
	{
		logRecord(5, 17);
		global_minbase.store(7);
	}
	if (global_minbase.load() <= 7 && 7 <= max_target_base)
	{
		logRecord(6, 17);
		global_minbase.store(8);
	}
	if (global_minbase.load() <= 8 && 8 <= max_target_base)
	{
		logRecord(7, 17);
		global_minbase.store(9);
	}
	if (global_minbase.load() > max_target_base)
		global_target_achieved.store(true);

	std::thread worker(verificationWorker, end_block, subblock_size, max_target_base);

	auto t_start = std::chrono::high_resolution_clock::now();

	uint64_t current_block = start_block;
	bool buffer_A_active = true;
	bool stream_A_inflight = false, stream_B_inflight = false;
	uint64_t start_range_A = 0, start_range_B = 0;
	uint64_t dispatched_block_id_A = 0, dispatched_block_id_B = 0;
	uint32_t dispatched_minbase_A = 0, dispatched_minbase_B = 0;

	// bases 12, 6 and 10 fit in shared memory
	size_t shared_mem_bytes = sp_bytes + base12Size16 + base6Size16 + base10Size16;

	while (current_block < end_block && !global_target_achieved.load())
	{
		uint32_t active_minbase = global_minbase.load();

		uint64_t dynamic_batch;
		if (active_minbase <= 6)
			dynamic_batch = 2;
		else if (active_minbase <= 8)
			dynamic_batch = 25;
		else if (active_minbase <= 10)
			dynamic_batch = 50;
		else if (active_minbase <= 12)
			dynamic_batch = 500;
		else if (active_minbase <= 14)
			dynamic_batch = 1000;
		else
			dynamic_batch = 2000;

		uint64_t dispatch_blocks = std::min(dynamic_batch, end_block - current_block);

		uint64_t raw_start_range = current_block * subblock_size;

		// Update bounds logic to explicitly align with the 30030 jump space
		uint64_t range_start = (raw_start_range / 30030ULL) * 30030ULL;

		uint64_t total_numbers_in_launch = dispatch_blocks * subblock_size + (raw_start_range - range_start);

		auto launch_search = [&](cudaStream_t stream, uint64_t start, uint64_t total, uint64_t *results, uint32_t *count, uint32_t mb)
		{
			if (mb >= 32)
			{
				unifiedSearchKernel<32><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
			}
			else if (mb >= 16)
			{
				unifiedSearchKernel<16><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
			}
			else
			{
				switch (mb)
				{
				case 9:
					unifiedSearchKernel<9><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 10:
					unifiedSearchKernel<10><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 11:
					unifiedSearchKernel<11><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 12:
					unifiedSearchKernel<12><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 13:
					unifiedSearchKernel<13><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 14:
					unifiedSearchKernel<14><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				case 15:
					unifiedSearchKernel<15><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				default:
					unifiedSearchKernel<15><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, total, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
					break;
				}
			}
		};

		if (buffer_A_active)
		{
			if (stream_A_inflight)
			{
				CUDA_CHECK(cudaStreamSynchronize(stream_A));

				if (*h_count_A > 0)
				{
					uint32_t fetch_count = (*h_count_A > MAX_GPU_RESULTS) ? MAX_GPU_RESULTS : *h_count_A;
					BufferBundle bundle;
					{
						std::unique_lock<std::mutex> lock(pool_mutex);
						pool_cv.wait(lock, []
								 { return !free_buffers.empty(); });
						bundle = free_buffers.front();
						free_buffers.pop();
					}

					CUDA_CHECK(cudaMemcpyAsync(bundle.memory, d_results_A, fetch_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream_transfer));
					CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));

					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						verification_queue.push({bundle, fetch_count, start_range_A, dispatched_block_id_A, dispatched_minbase_A});
					}
					queue_cv.notify_one();
				}
			}

			CUDA_CHECK(cudaMemsetAsync(d_count_A, 0, sizeof(uint32_t), stream_A));
			launch_search(stream_A, range_start, total_numbers_in_launch, d_results_A, d_count_A, active_minbase);
			CUDA_CHECK(cudaMemcpyAsync(h_count_A, d_count_A, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_A));

			start_range_A = raw_start_range;
			dispatched_block_id_A = current_block + dispatch_blocks - 1;
			dispatched_minbase_A = active_minbase;
			stream_A_inflight = true;
			buffer_A_active = false;
		}
		else
		{
			if (stream_B_inflight)
			{
				CUDA_CHECK(cudaStreamSynchronize(stream_B));

				if (*h_count_B > 0)
				{
					uint32_t fetch_count = (*h_count_B > MAX_GPU_RESULTS) ? MAX_GPU_RESULTS : *h_count_B;
					BufferBundle bundle;
					{
						std::unique_lock<std::mutex> lock(pool_mutex);
						pool_cv.wait(lock, []
								 { return !free_buffers.empty(); });
						bundle = free_buffers.front();
						free_buffers.pop();
					}

					CUDA_CHECK(cudaMemcpyAsync(bundle.memory, d_results_B, fetch_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream_transfer));
					CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						verification_queue.push({bundle, fetch_count, start_range_B, dispatched_block_id_B, dispatched_minbase_B});
					}
					queue_cv.notify_one();
				}
			}

			CUDA_CHECK(cudaMemsetAsync(d_count_B, 0, sizeof(uint32_t), stream_B));
			launch_search(stream_B, range_start, total_numbers_in_launch, d_results_B, d_count_B, active_minbase);
			CUDA_CHECK(cudaMemcpyAsync(h_count_B, d_count_B, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_B));

			start_range_B = raw_start_range;
			dispatched_block_id_B = current_block + dispatch_blocks - 1;
			dispatched_minbase_B = active_minbase;
			stream_B_inflight = true;
			buffer_A_active = true;
		}
		current_block += dispatch_blocks;
	}

	// --- CHRONOLOGICAL TEARDOWN ---
	auto push_A = [&]()
	{
		if (stream_A_inflight)
		{
			CUDA_CHECK(cudaStreamSynchronize(stream_A));
			if (*h_count_A > 0)
			{
				uint32_t fetch_count = (*h_count_A > MAX_GPU_RESULTS) ? MAX_GPU_RESULTS : *h_count_A;
				BufferBundle bundle;

				{
					std::unique_lock<std::mutex> lock(pool_mutex);
					pool_cv.wait(lock, []
							 { return !free_buffers.empty(); });
					bundle = free_buffers.front();
					free_buffers.pop();
				}

				CUDA_CHECK(cudaMemcpy(bundle.memory, d_results_A, fetch_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
				{
					std::lock_guard<std::mutex> lock(queue_mutex);
					verification_queue.push({bundle, fetch_count, start_range_A, dispatched_block_id_A, dispatched_minbase_A});
				}
				queue_cv.notify_one();
			}
		}
	};

	auto push_B = [&]()
	{
		if (stream_B_inflight)
		{
			CUDA_CHECK(cudaStreamSynchronize(stream_B));
			if (*h_count_B > 0)
			{
				uint32_t fetch_count = (*h_count_B > MAX_GPU_RESULTS) ? MAX_GPU_RESULTS : *h_count_B;
				BufferBundle bundle;
				{
					std::unique_lock<std::mutex> lock(pool_mutex);
					pool_cv.wait(lock, []
							 { return !free_buffers.empty(); });
					bundle = free_buffers.front();
					free_buffers.pop();
				}
				CUDA_CHECK(cudaMemcpy(bundle.memory, d_results_B, fetch_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
				{
					std::lock_guard<std::mutex> lock(queue_mutex);
					verification_queue.push({bundle, fetch_count, start_range_B, dispatched_block_id_B, dispatched_minbase_B});
				}
				queue_cv.notify_one();
			}
		}
	};

	if (!buffer_A_active)
	{
		push_B();
		push_A();
	}
	else
	{
		push_A();
		push_B();
	}

	{
		std::lock_guard<std::mutex> lock(queue_mutex);
		engine_running = false;
	}
	queue_cv.notify_one();
	worker.join();

	auto t_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = t_end - t_start;
	double total_time = diff.count();
	uint64_t total_numbers_checked = (current_block - start_block) * subblock_size;

	printf("\n===================================================\n");
	printf(" Execution Terminated Successfully.\n");
	printf(" Elapsed Time: %.2f seconds\n", total_time);
	printf(" Effective Throughput: %.2f Billion numbers/sec\n", (total_numbers_checked / total_time) / 1e9);
	printf("===================================================\n");

	CUDA_CHECK(cudaStreamDestroy(stream_A));
	CUDA_CHECK(cudaStreamDestroy(stream_B));
	CUDA_CHECK(cudaStreamDestroy(stream_transfer));

	CUDA_CHECK(cudaFree(d_sp));
	CUDA_CHECK(cudaFree(d_b3));
	CUDA_CHECK(cudaFree(d_b5));
	CUDA_CHECK(cudaFree(d_b6));
	CUDA_CHECK(cudaFree(d_b7));
	CUDA_CHECK(cudaFree(d_b9));
	CUDA_CHECK(cudaFree(d_b10));
	CUDA_CHECK(cudaFree(d_b11));
	CUDA_CHECK(cudaFree(d_b12));
	CUDA_CHECK(cudaFree(d_b13));
	CUDA_CHECK(cudaFree(d_b14));
	CUDA_CHECK(cudaFree(d_b15));
	CUDA_CHECK(cudaFree(d_results_A));
	CUDA_CHECK(cudaFree(d_results_B));
	CUDA_CHECK(cudaFree(d_count_A));
	CUDA_CHECK(cudaFree(d_count_B));
	CUDA_CHECK(cudaFreeHost(h_count_A));
	CUDA_CHECK(cudaFreeHost(h_count_B));

	for (int i = 0; i < POOL_SIZE; i++)
	{
		CUDA_CHECK(cudaFreeHost(h_buffer_pool[i]));
	}
	free(global_smallprimes);

	return 0;
}
