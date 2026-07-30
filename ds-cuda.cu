// Let ds(n) be the smallest prime number for which the digit sums
// when written in bases 2 to n+1 are all prime.
//
// This program searches for ds(n) numbers.
//
// The GPU is used identify candidate numbers by:
// 1. Using a mod-9699690 prime wheel to reject any number that is a
//    multiple of 2, 3, 5, 7, 11, 13, 17, or 19 (this rejects 82.90% of candidates)
// 2. Checking if the digit sums are prime in the following
//    bases in this order: (powers of two) 2, 4, 8, 16, 32,
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
// Results are appended to 'ds_records.txt'.
// A progress heartbeat is written to 'ds_state.txt' every minute
// containing the command to restart from the last heartbeat.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <errno.h>
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

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// ---------------------------------------------------------------------------
// Design decisions in the inner loop, and what each one measured.
//
// These were all A/B toggles at one point. The winning arm is now the only arm;
// the numbers are kept because they are the reason the code looks like this, and
// because three of the six things tried here made it SLOWER. All figures come
// from the same pinned benchmark range (see speed.bat) so they are comparable,
// against a run-to-run variance floor of 0.27%.
//
// 1. BASE 16 RUNS BEFORE BASE 8.  Counter-intuitive: base 8 is the stronger
//    filter (marginal pass 0.2875 against base 16's 0.4167), so filter-strength
//    reasoning says run it first. Measured 1685 against 1711 -- a 1.3% LOSS.
//    The reason is POPC. On sm_89 population count runs at 16 results/clk/SM
//    against 64 for add, logic and shift, so it costs 4x an ordinary op. Base 8
//    is 2 LOP3 + 2 POPC + IMAD + 2 IADD ~= 13 ALU-equivalents; base 16 is 2 LOP3
//    + SHF + IADD + DP4A with no POPC at all ~= 5. Weighted properly base 8 is
//    2.6x the cost, which swamps its extra filtering. Do not reorder these.
//
// 2. BASE 10 LIVES IN GLOBAL MEMORY, NOT SHARED.  With base 10 resident the
//    block needs 34,800 bytes, so two resident blocks need 69,600 and force the
//    100 KB shared carveout on Ada, leaving only 28 KB of L1 out of the 128 KB
//    unified cache. Dropping it to __ldg brings the block to 24,688 / 49,376 for
//    two, which fits the 64 KB carveout and leaves ~62 KB of L1 for every other
//    table plus the 6.33 MB wheel. Base 10 is reached by ~0.6% of warp
//    iterations, so the traded-away residency costs almost nothing. Measured
//    1733 against 1711, +1.3%. Confirmed in the profile: Shared Memory
//    Configuration Size 65.54 KB.
//
// 3. THE DS*_HI CHAIN IS PINNED (see dsPinRegister).  ptxas insists on
//    rematerialising one masked-popcount pair inside the inner loop, because
//    `hi` is the high half of the candidate register pair and so is free to
//    re-read. Pinning DS2_HI alone moved the work to DS4_HI for no net change
//    (+0.17%); pinning both moved it to DS8_HI (+1.14%, at IDENTICAL instruction
//    count -- the gain was purely getting it off the critical path feeding the
//    first branch); pinning all five stopped it (+2.43%). Registers fell 31 -> 27
//    in the process: the recompute was not saving a register, it was costing four.
//
// 4. SURVIVORS ARE EMITTED INSIDE THE CASCADE, NOT VIA A FLAG.  The old
//    `bool pass` plus `if (pass)` after the do/while(0) measured 4.00 instructions
//    per inner iteration and 12.98% of all warp stall samples. Emitting directly
//    is identical in behaviour because every break above already skips it.
//    Measured 1947 against 1732: +12.4%, the largest single win in the kernel.
//
// 5. DEAD END -- 32-BIT TRIP COUNT for the inner loop, replacing
//    `while (candidate < hi_limit)`. Removes ~3 instructions per iteration at the
//    cost of one divide per outer iteration, amortised 171.8:1. Measured twice:
//    1687 against 1732 (-2.6%) and 1978 against 2038.71 (-2.98%). The second run
//    came after ~750M instructions and four registers had been freed and after
//    the loop condition had itself grown 50% -- conditions that should have
//    favoured it far more. The compare is genuinely free: it issues in slots the
//    warp is already stalled on, so its 7-9% stall attribution is warps queueing
//    at the loop back-edge, not the compare costing anything.
//
// The rule all of this produced: stall samples on a COMPUTATION line mean the
// dependency chain is stalling and are worth acting on; stall samples on a loop
// tail or back-edge mean warps are queueing there and will simply queue somewhere
// else if you remove the instructions.
// ---------------------------------------------------------------------------

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

// Prime wheel geometry.
// WHEEL_MOD   = 2*3*5*7*11*13*17*19
// WHEEL_COUNT = phi(WHEEL_MOD) = 1*2*4*6*10*12*16*18, i.e. the residues coprime
//               to all eight primes. 1658880/9699690 = 17.10% survive the
//               wheel, against 92160/510510 = 18.05% for mod-510510 and
//               5760/30030 = 19.18% for the original mod-30030 wheel.
#define WHEEL_MOD 9699690ULL
#define WHEEL_COUNT 1658880U

// WHEEL_MOD and WHEEL_COUNT must stay in lockstep: the build loop in main() writes
// exactly one h_wheel[] entry per residue coprime to the eight primes, into an
// array sized WHEEL_COUNT. If WHEEL_COUNT were ever smaller than phi(WHEEL_MOD) --
// the obvious way to get that wrong is editing one constant and not the other --
// the loop would write past the end of a 6.33 MB array long before any check could
// notice. Deriving both from a single prime list makes that a compile error.
constexpr uint32_t DS_WHEEL_PRIMES[] = {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u};

constexpr uint64_t dsWheelMod()
{
	uint64_t m = 1;
	for (uint32_t p : DS_WHEEL_PRIMES)
		m *= p;
	return m;
}

constexpr uint32_t dsWheelCount()
{
	uint32_t c = 1;
	for (uint32_t p : DS_WHEEL_PRIMES)
		c *= (p - 1u);
	return c;
}

static_assert(dsWheelMod() == WHEEL_MOD, "WHEEL_MOD does not match DS_WHEEL_PRIMES");
static_assert(dsWheelCount() == WHEEL_COUNT, "WHEEL_COUNT is not phi(WHEEL_MOD)");

// Block size on the GPU
// Kept at 768 for optimal 2-block SM occupancy and register mapping
#define BLOCK_SIZE 768

// Maximum number of candidates returned by the GPU
#define MAX_GPU_RESULTS 20000000ULL

// Maximum base to check
#define MAX_BASE 64

// Wrap Cuda calls in this to display failures
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

// Constants for digit sum chunking
// Base 3 (8 digits)
const uint32_t base3Size = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3;

// Base 5 (6 digits)
const uint32_t base5Size = 5 * 5 * 5 * 5 * 5 * 5;

// Base 6 (4 digits)
const uint32_t base6Size = 6 * 6 * 6 * 6;

// Base 7 (4 digits)
const uint32_t base7Size = 7 * 7 * 7 * 7;

// Base 9 (4 digits)
const uint32_t base9Size = 9 * 9 * 9 * 9;

// Base 10 (4 digits)
const uint32_t base10Size = 10 * 10 * 10 * 10;

// Base 11 (4 digits)
const uint32_t base11Size = 11 * 11 * 11 * 11;

// Base 12 (4 digits)
const uint32_t base12Size = 12 * 12 * 12 * 12;

// Base 13 (4 digits)
const uint32_t base13Size = 13 * 13 * 13 * 13;

// Base 14 (4 digits)
const uint32_t base14Size = 14 * 14 * 14 * 14;

// Base 15 (4 digits)
const uint32_t base15Size = 15 * 15 * 15 * 15;

// Memory alignment in bytes -1
const uint32_t ALIGN = 127;

// Bases byte aligned
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

// Host Verification
//
// The MSVC path uses the 128-bit multiply/divide intrinsics. _udiv128 raises
// #DE if the quotient will not fit in 64 bits, which here requires high >= n:
// every caller reduces both operands mod n first, so a*b < n^2 and
// high = (a*b)>>64 < n^2/2^64 < n. The precondition therefore holds for every
// reachable call, and spsp() is only entered for n >= 41 with bases <= 37.
#if defined(_MSC_VER)
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t n)
{
	uint64_t high, low, remainder;
	low = _umul128(a, b, &high);
	_udiv128(high, low, n, &remainder);
	return remainder;
}
#else
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t n)
{
	return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % (unsigned __int128)n);
}
#endif

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

// The Templated (Compile-Time) version of sumDigits
// Compiler will replace constant division for better performance
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

// The Router to force compile-time evaluation
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
		// Fallback to slow division if base > 40
		return sumDigits(value, radix);
	}
}

// ---------------------------------------------------------------------------
// GPU / CPU shared contract.
//
// The kernel is only instantiated at a small set of MIN_BASE values, so the
// MIN_BASE actually compiled into a launch is NOT always the campaign's current
// minbase -- a campaign at minbase 31 runs the MIN_BASE=16 instantiation, and
// anything at or above 32 runs MIN_BASE=32.
//
// The CPU verifier decides which bases it can skip by assuming the GPU already
// checked them. If the launch dispatcher and the verifier ever disagree about
// which MIN_BASE was compiled in, the verifier will skip a base the kernel never
// looked at and the run will emit a FALSE RECORD with nothing to detect it.
// Both sides therefore route through this one function, and the value stored in
// the payload is the instantiated MIN_BASE, never the raw campaign minbase.
// ---------------------------------------------------------------------------
constexpr uint32_t instantiatedMinBase(uint32_t mb)
{
	return (mb >= 32u) ? 32u : ((mb >= 16u) ? 16u : mb);
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

// argv[0], so the restart line in the heartbeat names the binary that wrote it.
//
// This value is written verbatim into ds_state.txt on a line whose stated purpose
// is to be run, and argv[0] is entirely under the caller's control -- execve() lets
// it be set to any string at all, independent of the actual binary. A name
// containing shell metacharacters would turn the checkpoint file into a command
// injection aimed at whoever (or whatever script) resumes the run. It is filtered
// on the way in rather than trusted.
const char *global_program_name = "ds";

static bool isSafeProgramName(const char *s)
{
	if (s == NULL || *s == '\0')
		return false;
	for (const char *c = s; *c != '\0'; ++c)
	{
		const unsigned char u = (unsigned char)*c;
		const bool ok = (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
				    (u >= '0' && u <= '9') ||
				    u == '.' || u == '_' || u == '-' ||
				    u == '/' || u == '\\' || u == ':';
		if (!ok)
			return false;
	}
	return true;
}

constexpr int POOL_SIZE = 4;
uint64_t *h_buffer_pool[POOL_SIZE];

std::queue<BufferBundle> free_buffers;
std::mutex pool_mutex;
std::condition_variable pool_cv;

std::mutex file_mutex; // Protects file I/O

// Log a new record
void logRecord(uint32_t base, uint64_t candidate)
{
	std::lock_guard<std::mutex> lock(file_mutex);

	// Convert the candidate number into a comma-separated string
	std::string formatted_candidate = formatCommas(candidate);

	// Output to screen
	printf(">>> NEW RECORD FOUND: ds(%u) = %s <<<\n", base, formatted_candidate.c_str());
	fflush(stdout);

	// Append to file. A failure here is loud: the record only otherwise exists
	// in the scrollback of whatever terminal is running the campaign.
	FILE *f = fopen("ds_records.txt", "a");
	if (f != NULL)
	{
		fprintf(f, ">>> NEW RECORD FOUND: ds(%u) = %s <<<\n", base, formatted_candidate.c_str());
		fclose(f);
	}
	else
	{
		fprintf(stderr,
			  "WARNING: could not open ds_records.txt -- ds(%u) = %s exists only on stdout\n",
			  base, formatted_candidate.c_str());
		fflush(stderr);
	}
}

// Save the progress
void saveHeartbeatState(uint64_t completed_block, uint64_t end_block, uint64_t subblock_size, uint32_t current_base, uint32_t max_base)
{
	std::lock_guard<std::mutex> lock(file_mutex);

	// Written to a temporary file and moved into place, never truncated in situ.
	// fopen("w") truncates immediately, so a crash or power loss anywhere between
	// the truncate and the final fprintf would leave an empty or half-written
	// checkpoint and lose the campaign's position. With a temp file the reader
	// only ever sees the previous complete state or the new complete state.
	FILE *f = fopen("ds_state.tmp", "w");
	if (f == NULL)
	{
		fprintf(stderr, "WARNING: could not write ds_state.tmp -- progress is not being checkpointed\n");
		fflush(stderr);
		return;
	}

	time_t rawtime;
	struct tm *timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

	fprintf(f, "--- ENGINE HEARTBEAT ---\n");
	fprintf(f, "Last Saved: %s\n", buffer);
	fprintf(f, "Restart Command Args:\n");
	fprintf(f, "%s %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %u\n",
		  global_program_name, completed_block + 1, end_block, subblock_size, current_base, max_base);
	fflush(f);
	fclose(f);

	// POSIX rename() replaces atomically. Windows rename() fails if the target
	// exists, so remove first -- that leaves a window where no state file exists,
	// but never one where a partial state file exists, which is the failure that
	// actually loses work.
	remove("ds_state.txt");
	if (rename("ds_state.tmp", "ds_state.txt") != 0)
	{
		fprintf(stderr, "WARNING: could not move ds_state.tmp into place -- checkpoint not updated\n");
		fflush(stderr);
	}
}

// Strict command line integer parsing. An unchecked strtoull turns a typo into
// a silent zero, and a zero subblock_size makes the whole campaign a no-op that
// still reports success.
static uint64_t parseU64(const char *text, const char *name)
{
	char *endp = NULL;
	errno = 0;
	unsigned long long v = strtoull(text, &endp, 10);
	if (endp == text || *endp != '\0' || errno == ERANGE)
	{
		fprintf(stderr, "Error: could not parse %s from '%s'.\n", name, text);
		exit(EXIT_FAILURE);
	}
	return (uint64_t)v;
}

// The GPU result buffer is a hard capacity, not a soft one. Silently truncating
// a sub-block means the range was never fully searched while the run still
// reports success, and for a "smallest prime" hunt a skipped candidate is
// unrecoverable and undetectable. Fail the run instead.
static uint32_t checkResultCount(uint32_t count, uint64_t block_id)
{
	if ((uint64_t)count > MAX_GPU_RESULTS)
	{
		fprintf(stderr,
			  "FATAL: GPU result buffer overflowed at sub-block %" PRIu64 " -- %u hits against a "
			  "capacity of %" PRIu64 ".\n"
			  "       This range was NOT fully searched. Reduce subblock_size, reduce the\n"
			  "       dynamic batch for this minbase, or raise MAX_GPU_RESULTS and rerun it.\n",
			  block_id, count, (uint64_t)MAX_GPU_RESULTS);
		exit(EXIT_FAILURE);
	}
	return count;
}

// Prime wheel offsets. Offsets reach 9699689, so they need uint32 and the
// table is 6.33 MB -- far past the 64 KB constant bank, so it lives in global
// memory. This still costs nothing in the loop: each thread reads exactly one
// offset before the walk begins, and consecutive lanes read consecutive
// entries, so a warp takes a single coalesced 128-byte transaction per launch.
// The 6.33 MB working set sits comfortably in L2 on any target part.
__device__ uint32_t d_wheel[WHEEL_COUNT];

// Pin a loop invariant into a register for the life of the inner loop.
//
// Emits no instructions -- the empty asm has no PTX in it at all. The "+r"
// constraint simply declares the value as both read and written by an opaque
// operation, which stops ptxas deciding to recompute it inside the inner loop
// instead of keeping it live. See note 3 at the top of the file for why that
// matters here and what it measured.
//
// Encapsulated for three reasons: the kernel body stays free of inline assembly,
// the pinned expressions are written once instead of once per preprocessor arm
// (they previously had to be kept in sync by hand), and there is a single place
// to change if a future toolkit stops needing this.
__device__ __forceinline__ void dsPinRegister(uint32_t &v)
{
	// __INTELLISENSE__ is defined only by the Microsoft IntelliSense engine, never by
	// a real compiler. Its parser does not understand GCC-style extended asm (it wants
	// the MSVC __asm block form) and flags the colon as "expected a ')'". Hiding the
	// statement from it removes the squiggle in VS Code and Visual Studio without
	// changing a single byte of what nvcc or clang actually compile.
#if !defined(__INTELLISENSE__)
	asm("" : "+r"(v));
#else
	(void)v;
#endif
}

// This is the GPU kernel that filters candidate numbers
// It's templated on minimum base to remove rendundant base checks at lower bases
//
// PRECONDITION: end_range <= UINT64_MAX - stride. The host enforces this when it
// validates the campaign range, which is what lets the inner walk do a bare
// `candidate += stride` with no wrap test on the hot path -- without the
// guarantee, a walk that reached the top of the uint64 range would wrap to a
// small value, satisfy `candidate < end_range` again and spin.
template <uint32_t MIN_BASE>
__global__ void __launch_bounds__(BLOCK_SIZE, 2) unifiedSearchKernel(
    const uint64_t start_range, const uint64_t end_range,
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
	// Shared memory.
	//
	// Declared as uint4 rather than uint8_t so the 16-byte alignment comes from
	// the type itself. The loaders below issue 16-byte __pipeline_memcpy_async
	// stores into this block, and a uint8_t declaration only guarantees byte
	// alignment -- dynamic shared memory happens to be 16-byte aligned in every
	// current runtime, but a misaligned 16-byte shared store is an
	// illegal-address abort rather than a slow path, so the guarantee is worth
	// taking from the type system instead of from the runtime's habits.
	extern __shared__ uint4 s_mem_aligned[];
	uint8_t *const s_mem = (uint8_t *)s_mem_aligned;

	// Constant pointers to mutable shared memory during initialization
	uint8_t *const s_sp = s_mem;
	uint8_t *const s_b12 = s_sp + sp_size;
	uint8_t *const s_b6 = s_b12 + base12Size16;

	// Collaboratively load tables into ultra-fast L1 Shared Memory
	// Only bases 12 and 6 are promoted to shared memory -- the highest-traffic
	// filters that fit the budget, plus the small-primes table since every base
	// check indexes into it. Everything else, base 10 included, stays in global
	// memory and is read through __ldg(). See note 2 at the top of the file for
	// why base 10 is deliberately excluded.
	// Scoped to allow const parameters on iterators and pointers
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

	// Calculate global thread mapping for the prime wheel
	const uint64_t global_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

	// Constant division will be replaced by the compiler for speed
	const uint32_t wheel_idx = global_id % (uint64_t)WHEEL_COUNT;
	const uint64_t cycle = global_id / (uint64_t)WHEEL_COUNT;

	// Initial candidate; one coalesced load per warp, hoisted out of the walk
	uint64_t candidate = start_range + (cycle * WHEEL_MOD) + __ldg(&d_wheel[wheel_idx]);

	// Wait for the tables to finish copying
	__pipeline_commit();
	__pipeline_wait_prior(0);
	__syncthreads();

	// ---------------------------------------------------------------------
	// OUTER LOOP: pins the top 32 bits of the candidate.
	// Runs once per 2^32 of range covered, i.e. roughly once per 730 inner
	// iterations at the standard stride.
	// ---------------------------------------------------------------------
	while (candidate < end_range)
	{
		const uint32_t hi = (uint32_t)(candidate >> 32);

		// Partial digit sums contributed by bits 32..63. Loop-invariant for
		// the whole inner loop. Each mask below is the top half of the
		// corresponding 64-bit mask in the original kernel.

		// Base 2: popcount of the high word.
		// Without the barrier ptxas sinks this back into the inner loop and re-runs
		// a quarter-rate POPC on every candidate. See note 3 at the top of the file.
		uint32_t DS2_HI = __popc(hi);
		dsPinRegister(DS2_HI);

		// Base 4: ds4 = ds2 + (count of set bits in odd positions).
		// Pinned as well: with only DS2_HI opaque, ptxas simply moved the
		// rematerialisation here instead, for no net change.
		uint32_t DS4_HI = DS2_HI + __popc(hi & 0xAAAAAAAAu);
		dsPinRegister(DS4_HI);

		// Base 8: ds8 = ds2 + P1 + 3*P2, Pj = set bits at positions == j (mod 3)
		uint32_t DS8_HI = DS2_HI + __popc(hi & 0x24924924u) + 3u * __popc(hi & 0x49249249u);
		dsPinRegister(DS8_HI);

		// Base 16: nibble fold, then horizontal byte sum
		uint32_t DS16_HI = 0u;
		if constexpr (MIN_BASE >= 16)
		{
			const uint32_t n_high = (hi & 0x0F0F0F0F) + ((hi >> 4) & 0x0F0F0F0F);
			DS16_HI = __dp4a(n_high, 0x01010101U, 0U);
			dsPinRegister(DS16_HI);
		}

		// Base 32: ds32 = ds2 + P1 + 3*P2 + 7*P3 + 15*P4, Pj mod 5
		uint32_t DS32_HI = 0u;
		if constexpr (MIN_BASE >= 32)
		{
			DS32_HI = DS2_HI + __popc(hi & 0x21084210u) + 3u * __popc(hi & 0x42108421u) + 7u * __popc(hi & 0x84210842u) + 15u * __popc(hi & 0x08421084u);
			dsPinRegister(DS32_HI);
		}

		// Upper bound of this high-word window, clamped to the launch range.
		// The == 0 test catches the hi == 0xFFFFFFFF wrap.
		uint64_t hi_limit = ((uint64_t)hi + 1ULL) << 32;
		if (hi_limit == 0ULL || hi_limit > end_range)
			hi_limit = end_range;

		// -----------------------------------------------------------------
		// INNER LOOP: only the low 32 bits vary.
		// -----------------------------------------------------------------
		// Unrolled 3x. The trip count is not known here, so ptxas duplicates the body
		// with an exit check between copies rather than producing a clean 3x body --
		// but that still cuts the back-edge branch and the 64-bit `candidate < hi_limit`
		// compare (6.02 instructions per iteration on its own) to a third, and more
		// importantly hands the scheduler three fully independent filter cascades to
		// interleave. The kernel is issue-limited rather than ALU-limited, so shortening
		// dependency chains is what buys throughput.
		//
		// Swept against a 2040 unrolled-1 baseline, variance floor 0.27%:
		//   2x 2142   3x 2176   4x 2144   5x 2125   6x 2132
		// 2 and 4 are two apart and 5 and 6 are seven apart -- both inside noise. So
		// this is not a gentle curve but a genuine peak at 3 with everything else on a
		// plateau ~1.5% below. Do not "round up" to 4.
		//
		// Note this is NOT the documented "two candidates per thread" dead end: that
		// tested several candidates inside ONE cascade and lost early exit. Here each
		// copy keeps its own cascade and its own breaks, so per-candidate divergence is
		// unchanged -- which is why the checksum is identical at every unroll factor.
#pragma unroll 3
		while (candidate < hi_limit)
		{
			const uint32_t lo = (uint32_t)candidate;
			const uint32_t pc_lo = __popc(lo);

			// Check digit sum primality
			// Base order is optimal for filter strength and check cost
			do
			{
				// Power of 2 bases have very low cost
				// Base 2
				const uint32_t p = DS2_HI + pc_lo;

				// Base 4
				const uint32_t ds4 = DS4_HI + pc_lo + __popc(lo & 0xAAAAAAAAu);

				// Check base 2 and base 4 simultaneously (good for ILP)
				if (!(local_sp[p] & local_sp[ds4]))
					break;

				// Base 16 -- nibble fold, then accumulate straight onto the cached
				// high-word byte sum with a single dp4a. No POPC anywhere in this
				// chain, which is why it runs before base 8 (see note 1 at the top).
				if constexpr (MIN_BASE >= 16)
				{
					const uint32_t n_low = (lo & 0x0F0F0F0F) + ((lo >> 4) & 0x0F0F0F0F);
					const uint32_t ds16 = __dp4a(n_low, 0x01010101U, DS16_HI);
					if (!local_sp[ds16])
						break;
				}

				// Base 8 -- two masked popcounts, so 4x the ALU cost of base 16
				// despite being the stronger filter.
				{
					const uint32_t ds8 = DS8_HI + pc_lo + __popc(lo & 0x92492492u) +
								 3u * __popc(lo & 0x24924924u);
					if (!local_sp[ds8])
						break;
				}

				// Base 32
				if constexpr (MIN_BASE >= 32)
				{
					const uint32_t ds32 = DS32_HI + pc_lo + __popc(lo & 0x84210842u) + 3u * __popc(lo & 0x08421084u) + 7u * __popc(lo & 0x10842108u) + 15u * __popc(lo & 0x21084210u);

					if (!local_sp[ds32])
						break;
				}

				// Remaining even bases are stronger filters than odd bases
				// Each filter from here uses division by constants which are replaced by the compiler
				// and narrow to 32 bit as soon as possible
				//
				// These operate on the full 64-bit candidate and are unchanged.
				// A warp only reaches this point when at least one of its lanes
				// survived every power-of-two filter, so they are off the hot path.

				// NOTE on gating: bases 2, 4, 8, 3, 6, 5, 9, 7 are checked
				// UNCONDITIONALLY regardless of MIN_BASE (they're cheap and always in
				// scope, because the campaign never enters the kernel below minbase 9).
				// Bases 16, 32, 12, 10, 14, 11, 13, 15 are gated by
				// `if constexpr (MIN_BASE >= N)` and compiled out entirely for kernel
				// instantiations targeting a lower MIN_BASE.

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
					sum += __ldg(&b10[r64 - rq64 * 10000ULL]);

					r64 = rq64;
					uint32_t r = (uint32_t)r64;
					uint32_t rq = r / 10000U;
					sum += __ldg(&b10[r - rq * 10000U]);

					r = rq;
					sum += __ldg(&b10[r]);

					r = v_low;
					rq = r / 10000U;
					sum += __ldg(&b10[r - rq * 10000U]);

					r = rq;
					rq = r / 10000U;
					sum += __ldg(&b10[r - rq * 10000U]);

					r = rq;
					sum += __ldg(&b10[r]);

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

					// v_low only needs two chunks
					r = v_low;
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

					// v_low only needs two chunks
					r = v_low;
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

					// v_low only needs two chunks
					r = v_low;
					rq = r / 50625U;
					sum += __ldg(&b15[r - rq * 50625U]);

					r = rq;
					sum += __ldg(&b15[r]);

					if (!local_sp[sum])
						break;
				}

				// Ran the whole gauntlet without breaking.
				// (no point using __ballot_sync() since chance of multiple threads passing is tiny)
				// Emitted here rather than via a flag: every break above already
				// skips this, so behaviour is identical and the per-iteration flag
				// init / test / branch disappears. Worth 12.4% -- see note 4.
				{
					// The counter is deliberately allowed to run past the buffer so
					// the host can see the true hit count and abort, rather than
					// saturating and reporting a full buffer as a normal result.
					uint32_t idx = atomicAdd(d_count, 1);

					// Ensure result buffer doesn't overflow
					if (idx < MAX_GPU_RESULTS)
					{
						d_results[idx] = candidate;
					}
				}
			} while (0);

			// Safe without a wrap test: the host guarantees
			// end_range <= UINT64_MAX - stride, so candidate < hi_limit <= end_range
			// implies candidate + stride cannot overflow.
			candidate += stride;
		}
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

		// A dispatch that produced no candidates still gets a payload, with a
		// null bundle. It carries no data but it does carry the completed
		// sub-block id, which is what keeps the heartbeat advancing through
		// long empty stretches at high minbase.
		const bool has_buffer = (task.bundle.memory != nullptr);

		// Wait for the DMA transfer to physically finish across the PCIe bus
		if (has_buffer)
			CUDA_CHECK(cudaEventSynchronize(task.bundle.ready_event));

		auto start_time = std::chrono::high_resolution_clock::now();
		size_t survivor_count = 0;

		if (has_buffer && task.candidate_count > 0)
		{
			std::vector<uint64_t> survivors;
			uint32_t current_needed = global_minbase.load();
			uint32_t max_fast_check = std::min(current_needed, max_target_base);

			// Build the dynamic checklist to handles non-sequential gaps
			uint8_t bases_to_check[MAX_BASE];
			uint32_t check_count = 0;

			for (uint32_t b = 2; b <= max_fast_check; ++b)
			{
				bool checked_by_gpu = false;

				// Only assume the GPU checked it if it was within the scope of the
				// MIN_BASE that was actually instantiated for that launch (see
				// instantiatedMinBase) -- not the campaign minbase, which can be
				// higher than the kernel that ran.
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

			// Initial array pre-filter based on the block's launch state
			for (uint32_t i = 0; i < task.candidate_count; i++)
			{
				uint64_t candidate = task.bundle.memory[i];
				if (candidate < task.raw_start_range)
					continue;

				bool survived = true;

				for (uint32_t c = 0; c < check_count; ++c)
				{
					uint32_t b = bases_to_check[c];
					if (!global_smallprimes[sumDigitsFast(candidate, b)])
					{
						survived = false;
						break;
					}
				}

				// Add any survivors to the survivor list
				if (survived)
					survivors.push_back(candidate);
			}

			// Sort the survivors to ensure "ds(n) is the smallest prime" is enforced
			survivor_count = survivors.size();
			std::sort(survivors.begin(), survivors.end());

			// The Dynamic Record Hunt
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

		// Heartbeat to save state at regular intervals. Every dispatch reaches
		// this point, empty or not, so the checkpoint keeps moving even when
		// the GPU is returning nothing at all.
		auto current_time = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_heartbeat_time) >= HEARTBEAT_INTERVAL)
		{
			saveHeartbeatState(task.subblock_id, end_block, subblock_size, global_minbase.load(), max_target_base);
			last_heartbeat_time = current_time;
		}

		// Return the buffer and event bundle to the pool for the GPU to reuse
		if (has_buffer)
		{
			{
				std::lock_guard<std::mutex> lock(pool_mutex);
				free_buffers.push(task.bundle);
			}
			pool_cv.notify_one();
		}
	}
}

// Main entry point
int main(int argc, char **argv)
{
	if (argc != 6)
	{
		fprintf(stderr, "Usage: %s start_subblock end_subblock subblock_size minbase maxbase\n",
			  (argc > 0 && argv[0] != NULL) ? argv[0] : "ds");
		exit(EXIT_FAILURE);
	}

	// Anything with whitespace, quotes or shell metacharacters is dropped in favour
	// of the safe default rather than echoed into the restart line.
	if (isSafeProgramName(argv[0]))
		global_program_name = argv[0];

	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	uint64_t start_block = parseU64(argv[1], "start_subblock");
	uint64_t end_block = parseU64(argv[2], "end_subblock");
	uint64_t subblock_size = parseU64(argv[3], "subblock_size");

	uint64_t minbase_arg = parseU64(argv[4], "minbase");
	uint64_t maxbase_arg = parseU64(argv[5], "maxbase");

	// verificationWorker builds a checklist bases_to_check[MAX_BASE] and fills it
	// for bases 2..max_fast_check (bounded by max_target_base). Without this guard, a
	// max_target_base >= MAX_BASE silently overflows that stack array
	if (maxbase_arg < 2 || maxbase_arg >= MAX_BASE)
	{
		fprintf(stderr, "Error: maxbase must be in range [2, %u). Got %" PRIu64 ".\n", MAX_BASE, maxbase_arg);
		exit(EXIT_FAILURE);
	}

	// minbase is allowed to sit one past maxbase: that is the "campaign already
	// finished" state a resume can legitimately be handed.
	if (minbase_arg < 2 || minbase_arg > maxbase_arg + 1)
	{
		fprintf(stderr, "Error: minbase must be in range [2, %" PRIu64 "]. Got %" PRIu64 ".\n",
			  maxbase_arg + 1, minbase_arg);
		exit(EXIT_FAILURE);
	}

	if (subblock_size == 0)
	{
		fprintf(stderr, "Error: subblock_size must be non-zero.\n");
		exit(EXIT_FAILURE);
	}

	if (start_block > end_block)
	{
		fprintf(stderr, "Error: start_subblock (%" PRIu64 ") is past end_subblock (%" PRIu64 ").\n",
			  start_block, end_block);
		exit(EXIT_FAILURE);
	}

	global_minbase.store((uint32_t)minbase_arg);
	uint32_t max_target_base = (uint32_t)maxbase_arg;

	uint32_t largestds = 65 * (max_target_base - 1);
	uint32_t sp_bytes = largestds + 1;
	if (sp_bytes < 155)
		sp_bytes = 155;

	// 16 byte align
	sp_bytes = (sp_bytes + 15) & ~15;

	// Compute the small primes array
	global_smallprimes = (uint8_t *)calloc(sp_bytes, sizeof(uint8_t));
	if (global_smallprimes == NULL)
	{
		fprintf(stderr, "Error: could not allocate the small primes table.\n");
		exit(EXIT_FAILURE);
	}
	for (uint32_t i = 0; i <= largestds; i++)
		global_smallprimes[i] = cpu_isPrime(i) ? 1 : 0;

	// Compute the digit sum arrays for each non-power-of-2 base.
	// 'static' keeps roughly 191 KB of tables off the stack; the default MSVC
	// stack is only 1 MB and these were most of it.
	static uint8_t h_base3[base3Size];
	for (uint32_t i = 0; i < base3Size; i++)
		h_base3[i] = sumDigits(i, 3);

	static uint8_t h_base5[base5Size];
	for (uint32_t i = 0; i < base5Size; i++)
		h_base5[i] = sumDigits(i, 5);

	static uint8_t h_base6[base6Size];
	for (uint32_t i = 0; i < base6Size; i++)
		h_base6[i] = sumDigits(i, 6);

	static uint8_t h_base7[base7Size];
	for (uint32_t i = 0; i < base7Size; i++)
		h_base7[i] = sumDigits(i, 7);

	static uint8_t h_base9[base9Size];
	for (uint32_t i = 0; i < base9Size; i++)
		h_base9[i] = sumDigits(i, 9);

	static uint8_t h_base10[base10Size];
	for (uint32_t i = 0; i < base10Size; i++)
		h_base10[i] = sumDigits(i, 10);

	static uint8_t h_base11[base11Size];
	for (uint32_t i = 0; i < base11Size; i++)
		h_base11[i] = sumDigits(i, 11);

	static uint8_t h_base12[base12Size];
	for (uint32_t i = 0; i < base12Size; i++)
		h_base12[i] = sumDigits(i, 12);

	static uint8_t h_base13[base13Size];
	for (uint32_t i = 0; i < base13Size; i++)
		h_base13[i] = sumDigits(i, 13);

	static uint8_t h_base14[base14Size];
	for (uint32_t i = 0; i < base14Size; i++)
		h_base14[i] = sumDigits(i, 14);

	static uint8_t h_base15[base15Size];
	for (uint32_t i = 0; i < base15Size; i++)
		h_base15[i] = sumDigits(i, 15);

	// Setup the prime wheel in Global Memory.
	// 'static' keeps this 6.33 MB table out of the stack.
	static uint32_t h_wheel[WHEEL_COUNT];
	uint32_t w_idx = 0;
	for (uint32_t i = 1; i < (uint32_t)WHEEL_MOD; i++)
	{
		if (i % 2 != 0 && i % 3 != 0 && i % 5 != 0 && i % 7 != 0 && i % 11 != 0 && i % 13 != 0 && i % 17 != 0 && i % 19 != 0)
		{
			// Bounded write. The static_asserts above make a mismatch a compile
			// error, but this is the only unbounded write in the program and it
			// targets a 6.33 MB array, so it is checked at runtime as well.
			if (w_idx >= WHEEL_COUNT)
			{
				fprintf(stderr, "FATAL: wheel build overran %u offsets -- WHEEL_COUNT is wrong.\n", WHEEL_COUNT);
				return 1;
			}
			h_wheel[w_idx++] = i;
		}
	}
	if (w_idx != WHEEL_COUNT)
	{
		fprintf(stderr, "Wheel build failed: got %u offsets, expected %u\n", w_idx, WHEEL_COUNT);
		return 1;
	}
	CUDA_CHECK(cudaMemcpyToSymbol(d_wheel, h_wheel, sizeof(h_wheel)));

	// Allocate the small primes array and digit sum arrays on the device
	uint8_t *d_sp, *d_b3, *d_b5, *d_b6, *d_b7, *d_b9, *d_b10, *d_b11, *d_b12, *d_b13, *d_b14, *d_b15;
	CUDA_CHECK(cudaMalloc((void **)&d_sp, sp_bytes));
	CUDA_CHECK(cudaMalloc((void **)&d_b3, base3Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b5, base5Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b6, base6Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b7, base7Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b9, base9Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b10, base10Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b11, base11Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b12, base12Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b13, base13Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b14, base14Size16));
	CUDA_CHECK(cudaMalloc((void **)&d_b15, base15Size16));

	// Zero the alignment padding. The tables are rounded up to 128 bytes and the
	// shared-memory loaders copy the full padded length, so the tails would
	// otherwise be uninitialised device memory. They are never indexed, but this
	// keeps compute-sanitizer --tool initcheck clean.
	CUDA_CHECK(cudaMemset(d_b3, 0, base3Size16));
	CUDA_CHECK(cudaMemset(d_b5, 0, base5Size16));
	CUDA_CHECK(cudaMemset(d_b6, 0, base6Size16));
	CUDA_CHECK(cudaMemset(d_b7, 0, base7Size16));
	CUDA_CHECK(cudaMemset(d_b9, 0, base9Size16));
	CUDA_CHECK(cudaMemset(d_b10, 0, base10Size16));
	CUDA_CHECK(cudaMemset(d_b11, 0, base11Size16));
	CUDA_CHECK(cudaMemset(d_b12, 0, base12Size16));
	CUDA_CHECK(cudaMemset(d_b13, 0, base13Size16));
	CUDA_CHECK(cudaMemset(d_b14, 0, base14Size16));
	CUDA_CHECK(cudaMemset(d_b15, 0, base15Size16));

	// Copy the small primes array and digit sum arrays from the host to the device
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

	// Allocate the results arrays an counts for the double buffered streams (A and B)
	uint64_t *d_results_A, *d_results_B;
	uint32_t *d_count_A, *d_count_B;
	CUDA_CHECK(cudaMalloc((void **)&d_results_A, MAX_GPU_RESULTS * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_results_B, MAX_GPU_RESULTS * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_count_A, sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_count_B, sizeof(uint32_t)));

	// volatile: these are pinned completion flags written by the copy engine.
	// cudaStreamSynchronize is opaque enough that the compiler must reload them
	// today, but volatile is the correct idiom and costs nothing.
	volatile uint32_t *h_count_A, *h_count_B;
	CUDA_CHECK(cudaHostAlloc((void **)&h_count_A, sizeof(uint32_t), cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc((void **)&h_count_B, sizeof(uint32_t), cudaHostAllocDefault));

	cudaStream_t stream_A, stream_B;
	CUDA_CHECK(cudaStreamCreate(&stream_A));
	CUDA_CHECK(cudaStreamCreate(&stream_B));

	// Create the transfer stream
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
	int gridSize = 0;

	// Shared memory footprint. Declared here because the occupancy query below
	// needs it; it is used again at every kernel launch. Excluding base 10 is what
	// keeps two resident blocks inside the 64 KB carveout rather than forcing the
	// 100 KB one -- see note 2 at the top of the file.
	size_t shared_mem_bytes = sp_bytes + base12Size16 + base6Size16;

	// ---- Grid sizing --------------------------------------------------
	// Two constraints have to be satisfied at once.
	//
	// Wheel alignment: each thread owns exactly one wheel offset for the life
	// of the kernel, so (gridSize * blockSize) must be a whole number of wheel
	// revolutions. At 768 threads the quantum is WHEEL_COUNT / 768, which is
	// 2160 blocks for the mod-9699690 wheel (it was 120 for mod-510510).
	//
	// Wave alignment: __launch_bounds__ keeps 2 blocks resident per SM, so the
	// GPU runs waves of (2 * numSMs) blocks. A grid leaving a partial final
	// wave idles most of the machine for that wave's duration.
	const int gridQuantum = (int)(WHEEL_COUNT / (uint32_t)blockSize);

	// Query the resident block count rather than trusting __launch_bounds__.
	// MIN_BASE=32 is the heaviest instantiation, so it is the conservative one
	// to size against: lower instantiations use fewer registers and can only be
	// at least as resident, never less.
	int blocksPerSM = 0;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	    &blocksPerSM, unifiedSearchKernel<32>, blockSize, shared_mem_bytes));
	if (blocksPerSM < 1)
		blocksPerSM = 1;
	const int wave = numSMs * blocksPerSM;

	// Smallest grid whose final wave is at least 99.5% full. Deliberately NOT
	// the fullest grid: a larger grid means a larger stride, which shortens the
	// high-word hoist window and raises the per-candidate share of the outer
	// loop. That outer loop costs roughly 114 cycles (eight POPCs plus the mask
	// work for DS32_HI) amortised over 2^32 / stride inner iterations. At the
	// mod-510510 quantum of 120 the effect was invisible; at 2160 chasing the
	// last half-percent of fill costs several times what it returns. On 70 SMs
	// the exact-fill grid is 15120 blocks -- 100% fill, but a 67.9M stride and
	// only 63 inner iterations, which is 3.3% of outer-loop overhead against
	// 0.9% at 4320 blocks.
	for (int g = gridQuantum; g <= gridQuantum * 16; g += gridQuantum)
	{
		const int waves = (g + wave - 1) / wave;
		if ((long long)g * 1000 >= (long long)waves * wave * 995)
		{
			gridSize = g;
			break;
		}
	}

	// Nothing cleared the threshold: fall back to whichever grid fills best.
	// Integer cross-multiplication, so no floating point and no tie-break
	// epsilon; the strict > keeps the smallest grid among equal fills.
	if (gridSize == 0)
	{
		gridSize = gridQuantum;
		int bestWaves = (gridQuantum + wave - 1) / wave;
		for (int g = gridQuantum * 2; g <= gridQuantum * 16; g += gridQuantum)
		{
			const int waves = (g + wave - 1) / wave;
			if ((long long)g * bestWaves > (long long)gridSize * waves)
			{
				gridSize = g;
				bestWaves = waves;
			}
		}
	}

	// Calculate stride based on perfect alignment
	// stride = exactly how many full wheels the grid processes per inner loop
	// iteration. 46 SMs: (4320 * 768 / 1658880) * 9699690 = 2 * 9699690
	// = 19,399,380, leaving the high-word hoist ~221 inner iterations per
	// outer iteration.
	const uint64_t stride = ((uint64_t)gridSize * (uint64_t)blockSize / (uint64_t)WHEEL_COUNT) * WHEEL_MOD;

	// A zero stride would make the inner loop advance nowhere and spin forever, which
	// is the one failure mode here that produces no output and no error. The grid
	// sizing above cannot produce it (gridSize >= gridQuantum forces stride >=
	// WHEEL_MOD), but that is a property of code far from this line, so check it.
	if (stride == 0ULL)
	{
		fprintf(stderr, "FATAL: stride computed as zero -- grid sizing is broken.\n");
		exit(EXIT_FAILURE);
	}

	// ---- Range validation ---------------------------------------------
	// The kernel walks with a bare `candidate += stride` and no wrap test, which
	// is only sound while end_range <= UINT64_MAX - stride. Everything the loop
	// below can produce is bounded by end_block * subblock_size, so bounding
	// that one product covers raw_start_range, range_start and end_range at once
	// -- and it simultaneously rules out the silent overflow of
	// current_block * subblock_size in the dispatch loop.
	const uint64_t range_ceiling = UINT64_MAX - stride;
	if (end_block > range_ceiling / subblock_size)
	{
		fprintf(stderr,
			  "Error: end_subblock * subblock_size (%" PRIu64 " * %" PRIu64 ") exceeds the safe\n"
			  "       64-bit search ceiling of %" PRIu64 ".\n",
			  end_block, subblock_size, range_ceiling);
		exit(EXIT_FAILURE);
	}

	// Get the GPU name
	int deviceId;
	CUDA_CHECK(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

	printf("====================================================\n");
	printf(" Active GPU: %s Compute %d.%d\n", prop.name, prop.major, prop.minor);
	printf(" Execution Grid Topology: %d blocks x %d threads\n", gridSize, blockSize);
	printf(" Shared Memory: %zu bytes/block (%zu for %d resident)\n",
		 shared_mem_bytes, shared_mem_bytes * (size_t)blocksPerSM, blocksPerSM);
	printf(" Inner Loop: base 16 before base 8, base 10 in global, DS*_HI pinned, direct emit\n");
	printf(" Campaign Scope Targets: Bases %u to %u\n", global_minbase.load(), max_target_base);
	printf("====================================================\n");

	// Output the trivial entries which are ds(1) through ds(7).
	// This also guarantees that minbase is at least 9 by the time the dispatch
	// loop runs, which is the precondition for the kernel's unconditional
	// bases 3, 5, 6, 7, 8 and 9 to be in scope.
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

	// Create the CPU verification worker thread
	std::thread worker(verificationWorker, end_block, subblock_size, max_target_base);

	auto t_start = std::chrono::high_resolution_clock::now();

	uint64_t current_block = start_block;
	bool buffer_A_active = true;
	bool stream_A_inflight = false, stream_B_inflight = false;
	uint64_t start_range_A = 0, start_range_B = 0;
	uint64_t dispatched_block_id_A = 0, dispatched_block_id_B = 0;
	uint32_t dispatched_minbase_A = 0, dispatched_minbase_B = 0;

	// Push a payload for a dispatch that returned no candidates at all. It has
	// no buffer and no work, but it carries the completed sub-block id so the
	// heartbeat keeps advancing through empty stretches.
	auto queue_empty_payload = [&](uint64_t raw_start, uint64_t block_id, uint32_t mb)
	{
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			verification_queue.push({{nullptr, nullptr}, 0u, raw_start, block_id, mb});
		}
		queue_cv.notify_one();
	};

	// Main loop: launch the next block of candidate numbers to the GPU kernel
	// Two compute streams are used (A and B) so the results from the last stream can be
	// asynchronously fetched while the new block is running
	while (current_block < end_block && !global_target_achieved.load())
	{
		// Dynamically size the batch depending on the current base since
		// smaller bases have much larger result sets and we have a
		// fixed-size results buffer
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

		// Update bounds logic to explicitly align with the wheel jump space
		uint64_t range_start = (raw_start_range / WHEEL_MOD) * WHEEL_MOD;

		uint64_t total_numbers_in_launch = dispatch_blocks * subblock_size + (raw_start_range - range_start);

		// Templated Cuda kernel launch so the kernel only contains the checks needed for the current base
		auto launch_search = [&](cudaStream_t stream, uint64_t start, uint64_t total, uint64_t *results, uint32_t *count, uint32_t mb)
		{
			switch (instantiatedMinBase(mb))
			{
			case 32:
				unifiedSearchKernel<32><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 16:
				unifiedSearchKernel<16><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 15:
				unifiedSearchKernel<15><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 14:
				unifiedSearchKernel<14><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 13:
				unifiedSearchKernel<13><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 12:
				unifiedSearchKernel<12><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 11:
				unifiedSearchKernel<11><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 10:
				unifiedSearchKernel<10><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			case 9:
				unifiedSearchKernel<9><<<gridSize, blockSize, shared_mem_bytes, stream>>>(start, start + total, d_sp, sp_bytes, results, count, d_b3, d_b5, d_b6, d_b7, d_b9, d_b10, d_b11, d_b12, d_b13, d_b14, d_b15, stride);
				break;
			default:
				// Unreachable: the trivial ds(1)..ds(7) seeding above lifts
				// minbase to at least 9 before the dispatch loop runs. Loud
				// rather than silently substituting a wrong instantiation,
				// because a kernel that over-filters produces wrong terms.
				fprintf(stderr, "FATAL: no kernel instantiation for minbase %u.\n", mb);
				exit(EXIT_FAILURE);
			}

			// Launch failures are asynchronous and otherwise invisible here: the
			// count would stay at its memset zero, no payload would be queued,
			// and the dispatch loop would advance as though the range had simply
			// held no candidates. A skipped range then gets checkpointed as done.
			CUDA_CHECK(cudaGetLastError());
		};

		// Ping-pong across A and B
		if (buffer_A_active)
		{
			// Tracks the D2H drain of d_results_A on stream_transfer, if one is issued below.
			// stream_A must wait on this before its relaunch writes into d_results_A again.
			cudaEvent_t drain_event_A = nullptr;

			if (stream_A_inflight)
			{
				CUDA_CHECK(cudaStreamSynchronize(stream_A));

				const uint32_t hits_A = checkResultCount(*h_count_A, dispatched_block_id_A);

				if (hits_A > 0)
				{
					BufferBundle bundle;
					{
						std::unique_lock<std::mutex> lock(pool_mutex);
						pool_cv.wait(lock, []
								 { return !free_buffers.empty(); });
						bundle = free_buffers.front();
						free_buffers.pop();
					}

					CUDA_CHECK(cudaMemcpyAsync(bundle.memory, d_results_A, (size_t)hits_A * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream_transfer));
					CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
					drain_event_A = bundle.ready_event;

					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						verification_queue.push({bundle, hits_A, start_range_A, dispatched_block_id_A, dispatched_minbase_A});
					}
					queue_cv.notify_one();
				}
				else
				{
					queue_empty_payload(start_range_A, dispatched_block_id_A, dispatched_minbase_A);
				}
			}

			// d_results_A is shared between stream_transfer (reading it out above) and
			// stream_A (about to write fresh candidates into it below). These are
			// independent streams with no implicit ordering, so without this wait the
			// relaunch could start overwriting d_results_A before the copy engine has
			// finished draining it -- a silent data race, not a crash.
			if (drain_event_A != nullptr)
				CUDA_CHECK(cudaStreamWaitEvent(stream_A, drain_event_A, 0));

			CUDA_CHECK(cudaMemsetAsync(d_count_A, 0, sizeof(uint32_t), stream_A));
			launch_search(stream_A, range_start, total_numbers_in_launch, d_results_A, d_count_A, active_minbase);
			CUDA_CHECK(cudaMemcpyAsync((void *)h_count_A, d_count_A, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_A));

			start_range_A = raw_start_range;
			dispatched_block_id_A = current_block + dispatch_blocks - 1;
			dispatched_minbase_A = instantiatedMinBase(active_minbase);
			stream_A_inflight = true;
			buffer_A_active = false;
		}
		else
		{
			// Tracks the D2H drain of d_results_B on stream_transfer, if one is issued below.
			// stream_B must wait on this before its relaunch writes into d_results_B again.
			cudaEvent_t drain_event_B = nullptr;

			if (stream_B_inflight)
			{
				CUDA_CHECK(cudaStreamSynchronize(stream_B));

				const uint32_t hits_B = checkResultCount(*h_count_B, dispatched_block_id_B);

				if (hits_B > 0)
				{
					BufferBundle bundle;
					{
						std::unique_lock<std::mutex> lock(pool_mutex);
						pool_cv.wait(lock, []
								 { return !free_buffers.empty(); });
						bundle = free_buffers.front();
						free_buffers.pop();
					}

					CUDA_CHECK(cudaMemcpyAsync(bundle.memory, d_results_B, (size_t)hits_B * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream_transfer));
					CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
					drain_event_B = bundle.ready_event;

					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						verification_queue.push({bundle, hits_B, start_range_B, dispatched_block_id_B, dispatched_minbase_B});
					}
					queue_cv.notify_one();
				}
				else
				{
					queue_empty_payload(start_range_B, dispatched_block_id_B, dispatched_minbase_B);
				}
			}

			// Same race as above, mirrored for the B buffer/stream pair.
			if (drain_event_B != nullptr)
				CUDA_CHECK(cudaStreamWaitEvent(stream_B, drain_event_B, 0));

			CUDA_CHECK(cudaMemsetAsync(d_count_B, 0, sizeof(uint32_t), stream_B));
			launch_search(stream_B, range_start, total_numbers_in_launch, d_results_B, d_count_B, active_minbase);
			CUDA_CHECK(cudaMemcpyAsync((void *)h_count_B, d_count_B, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_B));

			start_range_B = raw_start_range;
			dispatched_block_id_B = current_block + dispatch_blocks - 1;
			dispatched_minbase_B = instantiatedMinBase(active_minbase);
			stream_B_inflight = true;
			buffer_A_active = true;
		}
		current_block += dispatch_blocks;
	}

	// Finished so tear down the streams
	auto push_A = [&]()
	{
		if (stream_A_inflight)
		{
			CUDA_CHECK(cudaStreamSynchronize(stream_A));

			const uint32_t hits_A = checkResultCount(*h_count_A, dispatched_block_id_A);

			if (hits_A > 0)
			{
				BufferBundle bundle;

				{
					std::unique_lock<std::mutex> lock(pool_mutex);
					pool_cv.wait(lock, []
							 { return !free_buffers.empty(); });
					bundle = free_buffers.front();
					free_buffers.pop();
				}

				CUDA_CHECK(cudaMemcpy(bundle.memory, d_results_A, (size_t)hits_A * sizeof(uint64_t), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
				{
					std::lock_guard<std::mutex> lock(queue_mutex);
					verification_queue.push({bundle, hits_A, start_range_A, dispatched_block_id_A, dispatched_minbase_A});
				}
				queue_cv.notify_one();
			}
			else
			{
				queue_empty_payload(start_range_A, dispatched_block_id_A, dispatched_minbase_A);
			}
		}
	};

	auto push_B = [&]()
	{
		if (stream_B_inflight)
		{
			CUDA_CHECK(cudaStreamSynchronize(stream_B));

			const uint32_t hits_B = checkResultCount(*h_count_B, dispatched_block_id_B);

			if (hits_B > 0)
			{
				BufferBundle bundle;
				{
					std::unique_lock<std::mutex> lock(pool_mutex);
					pool_cv.wait(lock, []
							 { return !free_buffers.empty(); });
					bundle = free_buffers.front();
					free_buffers.pop();
				}
				CUDA_CHECK(cudaMemcpy(bundle.memory, d_results_B, (size_t)hits_B * sizeof(uint64_t), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaEventRecord(bundle.ready_event, stream_transfer));
				{
					std::lock_guard<std::mutex> lock(queue_mutex);
					verification_queue.push({bundle, hits_B, start_range_B, dispatched_block_id_B, dispatched_minbase_B});
				}
				queue_cv.notify_one();
			}
			else
			{
				queue_empty_payload(start_range_B, dispatched_block_id_B, dispatched_minbase_B);
			}
		}
	};

	// Drain oldest first so the "smallest prime" ordering survives teardown.
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

	// Output the throughput attained
	printf("\n===================================================\n");
	printf(" Execution Terminated Successfully.\n");
	printf(" Elapsed Time: %.2f seconds\n", total_time);
	printf(" Effective Throughput: %.2f Billion numbers/sec\n", (total_numbers_checked / total_time) / 1e9);
	printf("===================================================\n");

	// Destroy the streams
	CUDA_CHECK(cudaStreamDestroy(stream_A));
	CUDA_CHECK(cudaStreamDestroy(stream_B));
	CUDA_CHECK(cudaStreamDestroy(stream_transfer));

	// Free memory
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
	CUDA_CHECK(cudaFreeHost((void *)h_count_A));
	CUDA_CHECK(cudaFreeHost((void *)h_count_B));

	for (int i = 0; i < POOL_SIZE; i++)
	{
		CUDA_CHECK(cudaFreeHost(h_buffer_pool[i]));
	}

	// Drain the buffer pool and destroy the events created alongside each buffer.
	// By this point every bundle is guaranteed to be back in free_buffers: the
	// worker thread has joined, and the push_A()/push_B() calls above already
	// returned any in-flight bundles before engine_running was cleared.
	while (!free_buffers.empty())
	{
		CUDA_CHECK(cudaEventDestroy(free_buffers.front().ready_event));
		free_buffers.pop();
	}

	free(global_smallprimes);

	return 0;
}
