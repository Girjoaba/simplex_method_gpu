#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
// #include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using real = float;

constexpr int BS_1D = 256;
constexpr int BS_2D = 16;
constexpr real EPS = 1E-4f;
constexpr real ALPHA_TOL = 1E-6f;  // Tolerance for alpha values in ratio test
constexpr real PIVOT_TOL = 1E-7f;  // Minimum acceptable pivot magnitude
constexpr int MAX_ITER = 50000;

// Periodic refactorization to fix numerical instability in B_inv
constexpr int REFACTOR_INTERVAL = 50;

// #define PRINT
// TODO: CudaGetDeviceProperties()

#define cuda_malloc_host(ptr, n) cuda_malloc_host_impl(&ptr, n, #ptr)
#define cuda_malloc(d_ptr, n) cuda_malloc_impl(&d_ptr, n, #d_ptr)
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
#define load_matrix(file, ptr, m, n) load_matrix_impl(file, ptr, m, n, #ptr)
#define print_matrix(ptr, m, n) print_matrix_impl(ptr, m, n, #ptr)
#define print_int(ptr) print_int_impl(ptr, #ptr)

template<typename T>
struct PtrAlloc {
	T*& ptr;
	int size;
};

struct TimeStruct {
	TimePoint start, host_alloc_start, file_read_start, solve_start;
	TimePoint alloc_start, init_start, init_end;
	TimePoint y_start, p_start, B_inv_start, x_b_start;
	TimePoint dealloc_start, dealloc_end;
	TimePoint print_result_start, host_free_start, end;

	double y_duration = 0.0;
  double p_duration = 0.0;
  double B_inv_duration = 0.0;
  double x_b_duration = 0.0;
};

enum class SolveStatus {
	MaxIter,
	OptimumFound,
	Unbounded,
	ThetaOverflow,
	Infeasible
};

__host__ __device__ __forceinline__
constexpr int AT(int i, int j, int s) { return i * s + j; }

__host__ __device__ __forceinline__
constexpr int R2C(int i, int j, int m) { return i + j * m; }

__host__ __device__ __forceinline__
constexpr int num_blocks_1D(int n) { return (n + BS_1D - 1) / BS_1D; }

/* ===================== HELPERS ===================== */

template<typename T>
void cuda_malloc_host_impl(T** ptr, int n, const char* name) {
	cudaError_t err = cudaMallocHost((void**)ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		std::cerr << "cudaMallocHost failed for " << name << ": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t err = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		// std::cerr << std::format("cudaMalloc failed for {}: {}\n", name, cudaGetErrorString(err));
		std::cerr << "cudaMalloc failed for " << name <<": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (err != cudaSuccess) {
		// std::cerr << std::format("cudaMemcpy failed for {}: {}\n", name, cudaGetErrorString(err));
		std::cerr << "cudaMemcpy failed for " << name << ": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template<typename T>
void load_matrix_impl(std::ifstream& file, T* a, int m, int n, const char* name) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!(file >> a[R2C(i, j, m)])) {
				// std::cerr <<std::format("Failed to read ({},{}) for {}\n", i, j, name);
				std::cerr << "Failed to read (" << i << "," << j << ") for " << name << "\n";
				std::exit(EXIT_FAILURE);
			}
		}
	}
}

template<typename T>
void print_matrix_impl(T* d_a, int m, int n, const char* msg) {
#ifndef PRINT
	return;
#endif
	T* a;

	cuda_malloc_host(a, m * n);
	cuda_memcpy(a, d_a, m * n, cudaMemcpyDeviceToHost);

	std::cout << (msg + 2) << ":\n";	// to skip "d_"
	std::cout << std::fixed << std::setprecision(2);

	for (int i = 0; i < m; ++i) {
		std::cout << '\t';
		for (int j = 0; j < n - 1; ++j)
			std::cout << std::setw(6) << a[R2C(i,j,m)] << ' ';
		std::cout << std::setw(6) << a[R2C(i,n-1,m)] << '\n';
	}

	cudaFreeHost(a);
}

void print_int_impl(int i, const char* msg) {
#ifndef PRINT
	return;
#endif
	std::cout << msg << ":\n\t" << i << '\n';
}

void print_iteration(int i) {
#ifdef PRINT
	std::cout << "# Iteration " << ++i << '\n';
#endif
}

void print_endline() {
#ifndef PRINT
	return;
#endif
	std::cout << std::endl;
}

void print_elapsed_time(const char *msg, double dur) {
	auto label = std::string(msg) + ": ";

	std::cout << std::setw(19) << label;
	std::cout << std::fixed << std::setprecision(2);
	std::cout << std::setw(6) << dur << '\n';
}

inline double duration(const TimePoint& start, const TimePoint& end) {
	return std::chrono::duration<double>(end - start).count();
}

inline void print_elapsed_time(const char *msg, const TimePoint& start, const TimePoint& end) {
	print_elapsed_time(msg, duration(start, end));
}

/* ===================== KERNELS ===================== */

__global__ void init_D_from_c(real* c, real* D, int m, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < n) D[R2C(0,j,m+1)] = -c[j];  // Fixed: D has leading dimension m+1, not m
}

__global__ void init_D_from_A(real* A, real* D, int m, int n) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m && j < n)
		D[R2C(i+1,j,m+1)] = A[R2C(i,j,m)];
}

__global__ void init_I(real* I, int m) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m && j < m)
		I[R2C(i,j,m)] = (real)(i == j);
}

__global__ void init_b_ixs(int* b_ixs, int m, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < m) b_ixs[j] = n - m + j;
}

// Phase I: Create objective to minimize artificial variables
// Since we're in maximization form, minimizing means using NEGATIVE costs
__global__ void init_phase1_costs(real* c_phase1, int n, int n_artificial_start) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < n) {
		c_phase1[j] = (j >= n_artificial_start) ? -1.0f : 0.0f;
	}
}

// Update c_b from basis indices (for Phase II initialization)
__global__ void update_c_b_from_basis(real* c_b, real* c, int* b_ixs, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m) {
		c_b[i] = c[b_ixs[i]];
	}
}

// Extract current basis columns from full constraint matrix to form B
// Note: Handles both original variables and artificial variables (identity columns)
__global__ void extract_basis_columns(real* A, int* b_ixs, real* B, int m, int n_total) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
	int j = blockIdx.x * blockDim.x + threadIdx.x;  // column

	if (i < m && j < m) {
		int col_idx = b_ixs[j];  // which column to use
		int first_artificial_idx = n_total - m;  // Index where identity columns start

		if (col_idx < first_artificial_idx) {
			// Original variable or slack/surplus - in matrix A
			B[R2C(i, j, m)] = A[R2C(i, col_idx, m)];
		} else {
			// Artificial variable - from identity matrix (last m columns)
			int identity_col = col_idx - first_artificial_idx;  // which identity column
			B[R2C(i, j, m)] = (i == identity_col) ? 1.0f : 0.0f;
		}
	}
}

__global__ void reduce_min(real* vec, int n, real* mins) {
	int tid = threadIdx.x;
	int j = blockIdx.x * blockDim.x + tid;

	__shared__ real sf[BS_1D];
	sf[tid] = j < n ? vec[j] : INFINITY;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			sf[tid] = sf[tid] > sf[tid + s] ? sf[tid + s] : sf[tid];
		__syncthreads();
	}
	
	if (tid == 0) mins[blockIdx.x] = sf[0];
}

__global__ void get_ix(real* vec, int n, int* ix, real* val) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	real v = *val;

	if (j < n && vec[j] == v)
		atomicCAS(ix, -1, j);
}

__global__ void compute_theta(real* x_b, real* alpha, real* theta, int* flags, int m, int* num_non_pos) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < m) {
		// Only count as "non-positive" if significantly negative (< -ALPHA_TOL)
		// This prevents false unbounded detection from near-zero numerical noise
		int flag = alpha[j] > ALPHA_TOL;  // Positive enough for ratio test
		int is_non_pos = (alpha[j] < -ALPHA_TOL);  // Significantly negative for unbounded check
		flags[j] = flag;
		theta[j] = flag ? (x_b[j] / alpha[j]) : INFINITY;
		atomicAdd(num_non_pos, is_non_pos);
	}
}

__global__ void compute_new_E(real* E, real* alpha, int m, int q, real alpha_q) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m)
		E[R2C(i, q, m)] = (i != q) ? (-alpha[i] / alpha_q) : (1 / alpha_q);	// explosion?
}

/* ===================== WRAPPERS ===================== */

int get_min_ix(real* d_vec, int n, int blocks_for_n, real* val, real* d_curr, real* d_next, int* d_ix) {
	cudaMemcpy(d_curr, d_vec, n * sizeof(real), cudaMemcpyDeviceToDevice);

	int size = n;
	while (size > 1) {
		reduce_min<<<blocks_for_n, BS_1D>>>(d_curr, size, d_next);
		size = blocks_for_n;
		blocks_for_n = num_blocks_1D(blocks_for_n);
		std::swap(d_curr, d_next);
	}
	
	blocks_for_n = num_blocks_1D(n);
	cudaMemset(d_ix, -1, sizeof(int));	// works only for 0 and -1
	get_ix<<<blocks_for_n, BS_1D>>>(d_vec, n, d_ix, d_curr);

	int min_ix;
	cudaMemcpy(&min_ix, d_ix, sizeof(int), cudaMemcpyDeviceToHost);
	if (val != nullptr)
		cudaMemcpy(val, d_curr, sizeof(real), cudaMemcpyDeviceToHost);
		
	return min_ix;
}

inline int entering_var(real* d_e, int n, int blocks_for_n, real* d_curr, real* d_next, int* d_ix) {
	real min_val;
	int min_ix = get_min_ix(d_e, n, blocks_for_n, &min_val, d_curr, d_next, d_ix);
	return (min_val >= -EPS) ? -1 : min_ix;	// squeezing zeros is pointless
}

int compute_E(real* d_E, real* d_alpha, int m, int blocks_for_m, int q, real* d_alpha_q) {
	real alpha_q;
	cudaMemcpy(&alpha_q, d_alpha + q, sizeof(real), cudaMemcpyDeviceToHost);

	/*
		If alpha_q <= 0, then min_theta = INF. Moreover, since we are here,
		there were positive alpha_t, thus the division exploded for all of them.
		Luckily, we can still find min_theta using log-tricks, if we want to.
	*/

	if (fabs(alpha_q) < PIVOT_TOL) return 1;  // Reject pivots that are too small

	init_I<<<dim3(blocks_for_m, blocks_for_m), dim3(BS_2D, BS_2D)>>>(d_E, m);
	compute_new_E<<<blocks_for_m, BS_1D>>>(d_E, d_alpha, m, q, alpha_q);

	return 0;
}

// Recompute B_inv from scratch to eliminate accumulated numerical errors
int refactorize_basis_inverse(
	cusolverDnHandle_t solver_handle,
	real* d_A,        // Full constraint matrix (m x n_total)
	int* d_b_ixs,     // Current basis indices (m)
	real* d_B,        // Workspace for basis matrix (m x m)
	real* d_B_inv,    // Output: fresh B_inv (m x m)
	int* d_pivot,     // Workspace for pivoting (m)
	int* d_info,      // Error info (1)
	real* d_work,     // Workspace for cuSOLVER
	int lwork,        // Size of workspace
	int m,
	int n_total       // Total number of columns (including artificials)
) {
	// Step 1: Extract basis columns from full matrix to form B
	dim3 block_2d(BS_2D, BS_2D);
	dim3 grid_2d((m + BS_2D - 1) / BS_2D, (m + BS_2D - 1) / BS_2D);
	extract_basis_columns<<<grid_2d, block_2d>>>(d_A, d_b_ixs, d_B, m, n_total);
	cudaDeviceSynchronize();

	// Step 2: LU factorization of B
	cusolverDnSgetrf(solver_handle, m, m, d_B, m, d_work, d_pivot, d_info);

	// Check if factorization succeeded
	int info_host;
	cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if (info_host != 0) {
		std::cerr << "Warning: LU factorization failed with info=" << info_host << std::endl;
		return 1;
	}

	// Step 3: Set B_inv to identity
	dim3 grid_I((m + BS_2D - 1) / BS_2D, (m + BS_2D - 1) / BS_2D);
	init_I<<<grid_I, block_2d>>>(d_B_inv, m);
	cudaDeviceSynchronize();

	// Step 4: Solve B * B_inv = I using LU factors
	cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, m, m, d_B, m, d_pivot, d_B_inv, m, d_info);

	return 0;
}

/* ===================== SOLVER ===================== */

// Core revised simplex iteration logic (reusable for Phase I and Phase II)
SolveStatus revised_simplex_core(
	// cuBLAS/cuSOLVER handles
	cublasHandle_t handle,
	cusolverDnHandle_t solver_handle,

	// Problem data (input)
	real* d_A,           // Constraint matrix (m x n)
	real* d_b,           // RHS vector (m)
	real* d_c,           // Objective coefficients (n) - PHASE-SPECIFIC
	int m, int n,

	// Basis state (input/output)
	real* d_B_inv,       // Basis inverse (m x m)
	real* d_c_b,         // Basis costs (m)
	real* d_x_b,         // Basic solution (m)
	int* d_b_ixs,        // Basis indices (m)

	// Workspace
	real* d_y_aug,       // Dual variables augmented (m+1)
	real* d_D,           // Augmented tableau row (m+1 x n)
	real* d_e,           // Reduced costs (n)
	real* d_alpha,       // Direction vector (m)
	real* d_theta,       // Ratio test (m)
	real* d_E,           // Eta matrix (m x m)
	real* d_new_B_inv,   // New basis inverse (m x m)
	real* d_curr,        // Reduction workspace (n)
	real* d_next,        // Reduction workspace (blocks_for_n)
	int* d_theta_flags,
	int* d_alpha_num_non_pos,
	int* d_ix,

	// Refactorization workspace
	real* d_B_scratch,
	real* d_work_refactor,
	int* d_pivot,
	int* d_info,
	int lwork_refactor,

	// Parameters
	int max_iter,
	int* iter_count,     // Output: actual iterations
	real* d_alpha_q,     // Workspace for alpha_q
	TimeStruct& t        // Timing
) {
	real one = 1.0f, zero = 0.0f;
	real neg_one = -1.0f;
	int blocks_for_n = (n + BS_1D - 1) / BS_1D;
	int blocks_for_m = (m + BS_1D - 1) / BS_1D;

	int i = 0;
	int p, q, alpha_num_non_pos;
	auto status = SolveStatus::MaxIter;
	int iter_since_refactor = 0;  // Counter for periodic refactorization

	// ============== Main loop ==============
	do {
		print_iteration(i);
		// ========= Entering variable =========

		t.y_start = Clock::now();
		// y_aug[1..m] = c_b * B_inv
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, m, m, &one, d_c_b, 1, d_B_inv, m, &zero, d_y_aug + 1, 1);
		t.y_duration += duration(t.y_start, Clock::now());
		print_matrix(d_y_aug, 1, m + 1);
		// e = [1 y] * [-c; A]
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, n, m+1, &one, d_y_aug, 1, d_D, m+1, &zero, d_e, 1);
		print_matrix(d_e, 1, n);

		// Debug: Print reduced cost diagnostics at iteration 0
		if (i == 0) {
			real* e_host = new real[n];
			real* y_aug_host = new real[m+1];
			cudaMemcpy(e_host, d_e, n * sizeof(real), cudaMemcpyDeviceToHost);
			cudaMemcpy(y_aug_host, d_y_aug, (m+1) * sizeof(real), cudaMemcpyDeviceToHost);

			std::cout << "Iteration 0 diagnostics:\n";
			std::cout << "  y_aug[0] = " << y_aug_host[0] << "\n";
			std::cout << "  y (first 5): ";
			for (int j = 0; j < std::min(5, m); j++) std::cout << y_aug_host[j+1] << " ";
			std::cout << "\n  Reduced costs (first 10): ";
			for (int j = 0; j < std::min(10, n); j++) std::cout << e_host[j] << " ";
			std::cout << "\n  Reduced costs (last 10): ";
			for (int j = std::max(0, n-10); j < n; j++) std::cout << e_host[j] << " ";

			real min_e = e_host[0], max_e = e_host[0];
			int min_idx = 0, max_idx = 0;
			for (int j = 0; j < n; j++) {
				if (e_host[j] < min_e) { min_e = e_host[j]; min_idx = j; }
				if (e_host[j] > max_e) { max_e = e_host[j]; max_idx = j; }
			}
			std::cout << "\n  Min reduced cost: " << min_e << " at index " << min_idx;
			std::cout << "\n  Max reduced cost: " << max_e << " at index " << max_idx;
			std::cout << "\n  EPS = " << EPS << "\n";
			std::cout << "  Min < -EPS? " << (min_e < -EPS ? "YES (should enter)" : "NO (optimal)") << "\n\n";

			delete[] e_host;
			delete[] y_aug_host;
		}

		t.p_start = Clock::now();
		p = entering_var(d_e, n, blocks_for_n, d_curr, d_next, d_ix);
		t.p_duration += duration(t.p_start, Clock::now());
		print_int(p);
		if (p < 0) {
			status = SolveStatus::OptimumFound;
			break;
		}

		// ============ Leaving variable ============

		// alpha = B_inv * A_p
		cublasSgemv(handle, CUBLAS_OP_N,
			m, m, &one, d_B_inv, m, d_A + p * m, 1, &zero, d_alpha, 1);
		print_matrix(d_alpha, m, 1);
		// reset the number of non-positive elements in alpha
		cudaMemset(d_alpha_num_non_pos, 0, sizeof(int));

		compute_theta<<<blocks_for_m, BS_1D>>>
			(d_x_b, d_alpha, d_theta, d_theta_flags, m, d_alpha_num_non_pos);
		print_matrix(d_theta, m, 1);
		cudaMemcpy(&alpha_num_non_pos, d_alpha_num_non_pos, sizeof(int), cudaMemcpyDeviceToHost);
		print_int(alpha_num_non_pos);

		if (alpha_num_non_pos == m) {
			status = SolveStatus::Unbounded;
			break;
		}
		q = get_min_ix(d_theta, m, blocks_for_m, nullptr, d_curr, d_next, d_ix);
		print_int(q);

		// ============ Update the basis ============

		t.B_inv_start = Clock::now();
		if (compute_E(d_E, d_alpha, m, blocks_for_m, q, d_alpha_q)) {
			status = SolveStatus::ThetaOverflow;
			break;
		}
		// new_B_inv = E * B_inv
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			m, m, m, &one, d_E, m, d_B_inv, m, &zero, d_new_B_inv, m);
		t.B_inv_duration += duration(t.B_inv_start, Clock::now());

		std::swap(d_B_inv, d_new_B_inv);
		print_matrix(d_E, m, m);
		print_matrix(d_B_inv, m, m);

		// ======= Periodic refactorization to fix numerical errors =======
		iter_since_refactor++;
		if (iter_since_refactor >= REFACTOR_INTERVAL) {
			refactorize_basis_inverse(solver_handle, d_A, d_b_ixs, d_B_scratch, d_B_inv,
			                         d_pivot, d_info, d_work_refactor, lwork_refactor, m, n);
			iter_since_refactor = 0;
		}

		// ======= Update the cost, indices and solution =======

		cudaMemcpy(d_c_b + q, d_c + p, sizeof(real), cudaMemcpyDeviceToDevice);
		print_matrix(d_c_b, 1, m);
		cudaMemcpy(d_b_ixs + q, &p, sizeof(int), cudaMemcpyHostToDevice);
		print_matrix(d_b_ixs, 1, m);

		t.x_b_start = Clock::now();
		// x_b = B_inv * b
		cublasSgemv(handle, CUBLAS_OP_N,
			m, m, &one, d_B_inv, m, d_b, 1, &zero, d_x_b, 1);
		t.x_b_duration += duration(t.x_b_start, Clock::now());
		print_matrix(d_x_b, m, 1);

	} while (++i < max_iter);
	print_endline();

	*iter_count = i;
	return status;
}

std::pair<real, SolveStatus> solve(real* A, real* b, real* c, real* x_b, int* b_ixs, int m, int n, TimeStruct& t) {
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasCreate failed.\n";
		std::exit(EXIT_FAILURE);
	}

	// Create cuSOLVER handle for periodic refactorization
	cusolverDnHandle_t solver_handle;
	if (cusolverDnCreate(&solver_handle) != CUSOLVER_STATUS_SUCCESS) {
		std::cerr << "cusolverDnCreate failed.\n";
		std::exit(EXIT_FAILURE);
	}

	real *d_A, *d_b, *d_c, *d_c_phase1;
	real *d_B_inv, *d_c_b, *d_x_b;
	real *d_y_aug, *d_D, *d_e;
	real *d_alpha, *d_theta;
	real *d_alpha_q, *d_E, *d_new_B_inv;
	real *d_next, *d_curr;
	// Refactorization workspace
	real *d_B_scratch, *d_work_refactor;
	int *d_b_ixs;
	int *d_theta_flags, *d_alpha_num_non_pos;
	int *d_ix;
	int *d_pivot, *d_info;

	int blocks_for_n = num_blocks_1D(n);
	int blocks_for_m = num_blocks_1D(m);

	const real one = 1.0f, zero = 0.0f;

	// Query cuSOLVER for workspace size needed for LU factorization
	int lwork_refactor = 0;
	cusolverDnSgetrf_bufferSize(solver_handle, m, m, nullptr, m, &lwork_refactor);

	PtrAlloc<real> real_allocs[] = {
		{d_A, m * n}, {d_b, m}, {d_c, n}, {d_c_phase1, n}, {d_B_inv, m * m},
		{d_c_b, m}, {d_x_b, n}, {d_y_aug, m + 1},
		{d_D, (m + 1) * n}, {d_e, n}, {d_alpha, m},
		{d_theta, m}, {d_alpha_q, 1}, {d_E, m * m}, {d_new_B_inv, m * m},
		{d_next, blocks_for_n}, {d_curr, n},
		// Refactorization workspace
		{d_B_scratch, m * m}, {d_work_refactor, lwork_refactor}
	};

	PtrAlloc<int> int_allocs[] = {
		{d_b_ixs, m}, {d_theta_flags, m}, {d_alpha_num_non_pos, 1}, {d_ix, 1},
		// Refactorization workspace
		{d_pivot, m}, {d_info, 1}
	};
	
	// ============== Allocation ==============

	t.alloc_start = Clock::now();
	for (auto &[ptr, size] : real_allocs)
		cuda_malloc(ptr, size);
	for (auto &[ptr, size] : int_allocs)
		cuda_malloc(ptr, size);
	
	// ============ Common Initialization ============

	t.init_start = Clock::now();
	cuda_memcpy(d_A, A, m * n, cudaMemcpyHostToDevice);
	cuda_memcpy(d_b, b, m, cudaMemcpyHostToDevice);
	cuda_memcpy(d_c, c, n, cudaMemcpyHostToDevice);
	cuda_memcpy(d_y_aug, &one, 1, cudaMemcpyHostToDevice);

	// For 2D kernel, calculate grid using BS_2D not BS_1D
	int blocks_2d_n = (n + BS_2D - 1) / BS_2D;
	int blocks_2d_m = (m + BS_2D - 1) / BS_2D;
	init_D_from_A<<<dim3(blocks_2d_n, blocks_2d_m), dim3(BS_2D, BS_2D)>>>(d_A, d_D, m, n);
	cudaDeviceSynchronize();  // Ensure D is fully initialized
	t.init_end = Clock::now();

	// ============ PHASE I: Find Basic Feasible Solution ============

	std::cout << "\n========== PHASE I: Finding feasible solution ==========\n";

	// Phase I objective: minimize sum of artificial variables
	int n_artificial_start = n - m;  // Artificial variables are last m columns
	init_phase1_costs<<<blocks_for_n, BS_1D>>>(d_c_phase1, n, n_artificial_start);
	init_D_from_c<<<blocks_for_n, BS_1D>>>(d_c_phase1, d_D, m, n);  // Pass m, kernel adds 1 for leading dim
	cudaDeviceSynchronize();  // Ensure D is initialized before proceeding

	// Initialize basis to artificial variables (identity matrix, last m columns)
	init_I<<<dim3(blocks_for_m, blocks_for_m), dim3(BS_2D, BS_2D)>>>(d_B_inv, m);
	init_b_ixs<<<blocks_for_m, BS_1D>>>(d_b_ixs, m, n);

	// Initialize Phase I basis costs (all artificials have cost 1.0)
	cudaMemset(d_c_b, 0, m * sizeof(real));
	update_c_b_from_basis<<<blocks_for_m, BS_1D>>>(d_c_b, d_c_phase1, d_b_ixs, m);

	// Initial solution: x_b = B^{-1} * b = I * b = b
	cuda_memcpy(d_x_b, d_b, m, cudaMemcpyDeviceToDevice);

	// Debug: Print Phase I setup
	real* c_phase1_host = new real[n];
	int* b_ixs_host = new int[m];
	real* c_b_host = new real[m];
	cudaMemcpy(c_phase1_host, d_c_phase1, n * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_ixs_host, d_b_ixs, m * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_b_host, d_c_b, m * sizeof(real), cudaMemcpyDeviceToHost);

	std::cout << "Phase I setup:\n";
	std::cout << "  n_artificial_start = " << n_artificial_start << "\n";
	std::cout << "  First 5 c_phase1: ";
	for (int i = 0; i < std::min(5, n); i++) std::cout << c_phase1_host[i] << " ";
	std::cout << "\n  Last 5 c_phase1: ";
	for (int i = std::max(0, n-5); i < n; i++) std::cout << c_phase1_host[i] << " ";
	std::cout << "\n  Initial basis indices (first 5): ";
	for (int i = 0; i < std::min(5, m); i++) std::cout << b_ixs_host[i] << " ";
	std::cout << "\n  Initial c_b (first 5): ";
	for (int i = 0; i < std::min(5, m); i++) std::cout << c_b_host[i] << " ";
	std::cout << "\n  Initial c_b (last 5): ";
	for (int i = std::max(0, m-5); i < m; i++) std::cout << c_b_host[i] << " ";

	// Debug: Check if last m columns of A form identity
	real* A_host = new real[m * n];
	cudaMemcpy(A_host, d_A, m * n * sizeof(real), cudaMemcpyDeviceToHost);
	std::cout << "\n  Checking last m columns of A (should be identity):";
	std::cout << "\n    A[0, n-m] to A[0, n-m+4]: ";
	for (int j = n-m; j < std::min(n-m+5, n); j++)
		std::cout << A_host[R2C(0, j, m)] << " ";
	std::cout << "\n    A[m-5, n-5] to A[m-1, n-1] (should be diagonal): ";
	for (int k = std::max(0, m-5); k < m; k++)
		std::cout << "A[" << k << "," << (n-m+k) << "]=" << A_host[R2C(k, n-m+k, m)] << " ";
	delete[] A_host;

	// Debug: Check what D contains for the identity columns
	real* D_host = new real[(m+1) * n];
	cudaMemcpy(D_host, d_D, (m+1) * n * sizeof(real), cudaMemcpyDeviceToHost);
	std::cout << "\n  Checking D matrix for column " << (n-m) << " (first artificial):";
	std::cout << "\n    D[0," << (n-m) << "] = " << D_host[R2C(0, n-m, m+1)] << " (should be 1 from -c_phase1)";
	std::cout << "\n    D[1," << (n-m) << "] = " << D_host[R2C(1, n-m, m+1)] << " (should be 1 from A[0,n-m])";
	std::cout << "\n    D[2," << (n-m) << "] = " << D_host[R2C(2, n-m, m+1)] << " (should be 0)";
	std::cout << "\n  Checking D matrix for column " << (n-1) << " (last artificial):";
	std::cout << "\n    D[0," << (n-1) << "] = " << D_host[R2C(0, n-1, m+1)] << " (should be 1)";
	std::cout << "\n    D[" << m << "," << (n-1) << "] = " << D_host[R2C(m, n-1, m+1)] << " (should be 1 from A[m-1,n-1])";
	delete[] D_host;

	std::cout << "\n\n";

	delete[] c_phase1_host;
	delete[] b_ixs_host;
	delete[] c_b_host;

	// Run Phase I
	int phase1_iter = 0;
	SolveStatus phase1_status = revised_simplex_core(
		handle, solver_handle,
		d_A, d_b, d_c_phase1, m, n,
		d_B_inv, d_c_b, d_x_b, d_b_ixs,
		d_y_aug, d_D, d_e, d_alpha, d_theta, d_E, d_new_B_inv,
		d_curr, d_next, d_theta_flags, d_alpha_num_non_pos, d_ix,
		d_B_scratch, d_work_refactor, d_pivot, d_info, lwork_refactor,
		MAX_ITER, &phase1_iter, d_alpha_q, t
	);

	std::cout << "Phase I completed in " << phase1_iter << " iterations with status: ";
	if (phase1_status == SolveStatus::OptimumFound) std::cout << "OptimumFound\n";
	else if (phase1_status == SolveStatus::Unbounded) std::cout << "Unbounded\n";
	else if (phase1_status == SolveStatus::MaxIter) std::cout << "MaxIter\n";
	else std::cout << "ThetaOverflow\n";

	// ============ Check Feasibility ============

	if (phase1_status != SolveStatus::OptimumFound) {
		std::cerr << "Phase I failed to find optimal solution\n";
		auto status = SolveStatus::Infeasible;
		real z = 0.0f;

		t.dealloc_start = Clock::now();
		cublasDestroy(handle);
		cusolverDnDestroy(solver_handle);
		for (auto &[ptr,_] : real_allocs)
			if (ptr) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)
			if (ptr) cudaFree(ptr);
		t.dealloc_end = Clock::now();

		return std::make_pair(z, status);
	}

	// Compute Phase I objective value (should be ~0 for feasible problem)
	// Since Phase I costs are negative for artificials, objective should be >= -EPS if feasible
	real phase1_obj;
	cublasSdot(handle, m, d_c_b, 1, d_x_b, 1, &phase1_obj);
	std::cout << "Phase I objective value: " << phase1_obj << "\n";

	if (phase1_obj < -EPS) {
		std::cerr << "Problem is infeasible (Phase I objective = " << phase1_obj << " < " << -EPS << ")\n";
		auto status = SolveStatus::Infeasible;
		real z = 0.0f;

		t.dealloc_start = Clock::now();
		cublasDestroy(handle);
		cusolverDnDestroy(solver_handle);
		for (auto &[ptr,_] : real_allocs)
			if (ptr) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)
			if (ptr) cudaFree(ptr);
		t.dealloc_end = Clock::now();

		return std::make_pair(z, status);
	}

	// ============ PHASE II: Optimize Original Objective ============

	std::cout << "\n========== PHASE II: Optimizing original objective ==========\n";

	// Phase II uses the basis found in Phase I (b_ixs, B_inv, x_b unchanged)
	// Update objective to original problem
	init_D_from_c<<<blocks_for_n, BS_1D>>>(d_c, d_D, m, n);  // Pass m, kernel adds 1 for leading dim

	// Recompute c_b from original objective for current basis
	update_c_b_from_basis<<<blocks_for_m, BS_1D>>>(d_c_b, d_c, d_b_ixs, m);

	// Run Phase II
	int phase2_iter = 0;
	SolveStatus phase2_status = revised_simplex_core(
		handle, solver_handle,
		d_A, d_b, d_c, m, n,
		d_B_inv, d_c_b, d_x_b, d_b_ixs,
		d_y_aug, d_D, d_e, d_alpha, d_theta, d_E, d_new_B_inv,
		d_curr, d_next, d_theta_flags, d_alpha_num_non_pos, d_ix,
		d_B_scratch, d_work_refactor, d_pivot, d_info, lwork_refactor,
		MAX_ITER - phase1_iter, &phase2_iter, d_alpha_q, t
	);

	std::cout << "Phase II completed in " << phase2_iter << " iterations with status: ";
	if (phase2_status == SolveStatus::OptimumFound) std::cout << "OptimumFound\n";
	else if (phase2_status == SolveStatus::Unbounded) std::cout << "Unbounded\n";
	else if (phase2_status == SolveStatus::MaxIter) std::cout << "MaxIter\n";
	else std::cout << "ThetaOverflow\n";

	std::cout << "Total iterations (Phase I + Phase II): " << (phase1_iter + phase2_iter) << "\n\n";

	auto status = phase2_status;

	real z;
	if (status == SolveStatus::OptimumFound) {
		cublasSdot(handle, m, d_c_b, 1, d_x_b, 1, &z);
		cudaMemcpy(x_b, d_x_b, m * sizeof(real), cudaMemcpyDeviceToHost);
		cudaMemcpy(b_ixs, d_b_ixs, m * sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	t.dealloc_start = Clock::now();
	cublasDestroy(handle);
	cusolverDnDestroy(solver_handle);
	for (auto &[ptr,_] : real_allocs)
		if (ptr) cudaFree(ptr);
	for (auto &[ptr,_] : int_allocs)
		if (ptr) cudaFree(ptr);
	t.dealloc_end = Clock::now();
	
	return std::make_pair(z, status);
}

/* ===================== MAIN ===================== */

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);
	
	if (argc < 2) {
		std::cerr << "Please, specify an input file.\n";
		return 1;
	}

	TimeStruct t;
	t.start = Clock::now();
	
	std::ifstream file(argv[1]);
	if (!file.is_open()) {
		std::cerr << "Could not open " << argv[1] << ".\n";
		return 1;
	}

	int m, n;
	if (!(file >> m >> n) || m > n) {
		std::cerr << "Either failed to read m and n, or m > n.\n";
		return 1;
	}

	t.host_alloc_start = Clock::now();
	real *A, *b, *c, *x_b;
	int *b_ixs;
	cuda_malloc_host(A, m * n);
	cuda_malloc_host(b, m);
	cuda_malloc_host(c, n);
	cuda_malloc_host(x_b, m);
	cuda_malloc_host(b_ixs, m);

	t.file_read_start = Clock::now();
	load_matrix(file, A, m, n);
	load_matrix(file, b, m, 1);
	load_matrix(file, c, 1, n);
	file.close();

	t.solve_start = Clock::now();
	auto [z, status] = solve(A, b, c, x_b, b_ixs, m, n, t);

	t.print_result_start = Clock::now();
	switch (status) {
		case SolveStatus::OptimumFound:
			std::cout << "Optimum found: " << z << '\n';
			for (int i = 0; i < m; ++i)
				std::cout << "\tx_" << b_ixs[i] << " = " << x_b[i] << "\n";
			break;

		case SolveStatus::Unbounded:
			std::cout << "Problem unbounded.\n";
			break;

		case SolveStatus::ThetaOverflow:
			std::cout << "Theta overflow.\n";
			break;

		case SolveStatus::MaxIter:
			std::cout << "MAX_ITER exceeded.\n";
			break;

		case SolveStatus::Infeasible:
			std::cout << "Problem is infeasible.\n";
			break;
	}
	std::cout << '\n';

	t.host_free_start = Clock::now();
	cudaFreeHost(A);
	cudaFreeHost(b);
	cudaFreeHost(c);
	cudaFreeHost(x_b);
	cudaFreeHost(b_ixs);

	t.end = Clock::now();

	print_elapsed_time("Total", t.start, t.end);
	std::cout << '\n';
	print_elapsed_time("y", t.y_duration);
	print_elapsed_time("p", t.p_duration);
	print_elapsed_time("B_inv", t.B_inv_duration);
	print_elapsed_time("x_b", t.x_b_duration);
	std::cout << '\n';
	print_elapsed_time("Alloc", t.alloc_start, t.init_start);
	print_elapsed_time("Init", t.init_start, t.init_end);
	print_elapsed_time("Dealloc", t.dealloc_start, t.dealloc_end);
	std::cout << '\n';
	print_elapsed_time("Host alloc", t.host_alloc_start, t.file_read_start);
	print_elapsed_time("Read file", t.file_read_start, t.solve_start);
	print_elapsed_time("Solve call", t.solve_start, t.print_result_start);
	print_elapsed_time("Print result", t.print_result_start, t.host_free_start);
	print_elapsed_time("Host free", t.host_free_start, t.end);

	return 0;
}