/*
=================== FUNCTIONALITY ===================
- input and correctness
- division by a small number (two spots)
- what if m,n < BS_1D
- what if x_b_t < 0 for some t (compute_theta)
- branch CUBLAS for "real"

=================== OPTIMIZATIONS ===================
- steepest edge (including the recurrence)
- CUB for reduction
- faster update of x_b
----------------------------------------------------
- explore different data storage
- move the logic around between CPU and GPU
- optimize kernels (warps, cache, sync, atomic, ...)
----------------------------------------------------
- tune BS differently for distinct tasks
- combine kernels to avoid restarts
----------------------------------------------------
- dereferencing the same pointer in get_ix
*/

#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
// #include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>
#include <utility>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using real = float;

constexpr int BS_1D = 256;
constexpr int BS_2D = 16;
constexpr real EPS = 1E-4f;
constexpr int MAX_ITER = 1000;

#define cuda_malloc_host(ptr, n) cuda_malloc_host_impl(&ptr, n, #ptr)
#define cuda_malloc(d_ptr, n) cuda_malloc_impl(&d_ptr, n, #d_ptr)
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
#define load_array(file, ptr, m, n) load_array_impl(file, ptr, m, n, #ptr)

template<typename T>
struct PtrAlloc {
	T *&ptr;
	int size;
};

struct TimePoints {
	TimePoint start, end;
	TimePoint file_read_end, file_read_start;
	TimePoint host_alloc_start, host_alloc_end;
	TimePoint ev_start, ev_end;
	TimePoint lv_start, lv_end;
	TimePoint B_start, B_end;
	TimePoint alloc_start, alloc_end;
	TimePoint dealloc_start, dealloc_end;
	TimePoint init_start, init_end;
	TimePoint blas_end;
};

enum class SolveStatus {
	MaxIter,
	OptimumFound,
	Unbounded,
	ThetaOverflow
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
		// std::cerr << std::format("cudaMallocHost failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t err = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		// std::cerr << std::format("cudaMalloc failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (err != cudaSuccess) {
		// std::cerr << std::format("cudaMemcpy failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template<typename T>
void load_array_impl(std::ifstream& file, T* a, int m, int n, const char* name) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!(file >> a[R2C(i, j, m)])) {
				// std::cerr <<std::format("Failed to read ({},{}) for {}\n", i, j, name);
				std::exit(EXIT_FAILURE);
			}
		}
	}
}

inline void print_elapsed_time(const char *label, const TimePoint& start, const TimePoint& end) {
	std::cout << label << ": " << std::chrono::duration<double>(end - start).count() << '\n';
}

/* ===================== KERNELS ===================== */

__global__ void init_D_from_c(real *c, real *D, int m, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < n) D[R2C(0,j,m)] = -c[j];
}

__global__ void init_D_from_A(real *A, real *D, int m, int n) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m && j < n)
		D[R2C(i+1,j,m+1)] = A[R2C(i,j,m)];
}

__global__ void init_I(real *I, int m) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m && j < m)
		I[R2C(i,j,m)] = (real)(i == j);
}

__global__ void init_b_ixs(int *b_ixs, int m, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < m) b_ixs[j] = n - m + j;
}

__global__ void reduce_min(real *vec, int n, real *mins) {
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

__global__ void get_ix(real *vec, int n, int *ix, real *val) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	real v = *val;

	if (j < n && vec[j] == v)
		atomicCAS(ix, -1, j);
}

__global__ void compute_theta(real *x_b, real *alpha, real *theta, int *flags, int m, int *num_non_pos) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < m) {
		int flag = alpha[j] > 0;
		int not_flag = alpha[j] <= 0;
		flags[j] = flag;
		theta[j] = x_b[j] / alpha[j] * flag + INFINITY * not_flag;	// explosion?
		atomicAdd(num_non_pos, not_flag);
	}
}

__global__ void compute_new_E(real *E, real *alpha, int m, int q, real alpha_q) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m)
		E[R2C(i, q, m)] = (i != q) ? (-alpha[i] / alpha_q) : (1 / alpha_q);	// explosion?
}

/* ===================== WRAPPERS ===================== */

int get_min_ix(real *d_vec, int n, int blocks_for_n, real *val, real* d_curr, real* d_next, int* d_ix) {
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

inline int entering_var(real *d_e, int n, int blocks_for_n, real* d_curr, real* d_next, int* d_ix) {
	real min_val;
	int min_ix = get_min_ix(d_e, n, blocks_for_n, &min_val, d_curr, d_next, d_ix);
	return (min_val >= -EPS) ? -1 : min_ix;	// squeezing zeros is pointless
}

int compute_E(real *d_E, real *d_alpha, int m, int blocks_for_m, int q, real *d_alpha_q) {
	real alpha_q;	
	cudaMemcpy(&alpha_q, d_alpha + q, sizeof(real), cudaMemcpyDeviceToHost);

	/*
		If alpha_q <= 0, then min_theta = INF. Moreover, since we are here,
		there were positive alpha_t, thus the division exploded for all of them.
		Luckily, we can still find min_theta using log-tricks, if we want to.
	*/

	if (alpha_q <= 0) return 1;

	init_I<<<dim3(blocks_for_m, blocks_for_m), dim3(BS_2D, BS_2D)>>>(d_E, m);
	compute_new_E<<<blocks_for_m, BS_1D>>>(d_E, d_alpha, m, q, alpha_q);

	return 0;
}

/* ===================== SOLVER ===================== */

std::pair<real, SolveStatus> solve(real *A, real *b, real *c, real *x_b, int *b_ixs, int m, int n, TimePoints &t) {
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasCreate failed.\n";
		std::exit(EXIT_FAILURE);
	}

	real *d_A, *d_b, *d_c;
	real *d_B_inv, *d_c_b, *d_x_b;
	real *d_y, *d_y_aug, *d_D, *d_e;
	real *d_A_p, *d_alpha, *d_theta;
	real *d_alpha_q, *d_E, *d_new_B_inv;
	real *d_next, *d_curr;
	int *d_b_ixs;
	int *d_theta_flags, *d_alpha_num_non_pos;
	int *d_ix;

	int blocks_for_n = num_blocks_1D(n);
	int blocks_for_m = num_blocks_1D(m);

	const real alpha = 1.0f, beta = 0.0f;

	PtrAlloc<real> real_allocs[] = {
		{d_A, m * n}, {d_b, m}, {d_c, n}, {d_B_inv, m * m},
		{d_c_b, m}, {d_x_b, n}, {d_y, m}, {d_y_aug, m + 1},
		{d_D, (m + 1) * n}, {d_e, n}, {d_A_p, m}, {d_alpha, m},
		{d_theta, m}, {d_alpha_q, 1}, {d_E, m * m}, {d_new_B_inv, m * m},  
		{d_next, blocks_for_n}, {d_curr, n}
	};

	PtrAlloc<int> int_allocs[] = {
		{d_b_ixs, m}, {d_theta_flags, m}, {d_alpha_num_non_pos, 1}, {d_ix, 1}
	};
	
	// ============== Allocation ==============

	t.alloc_start = Clock::now();
	for (auto &[ptr, size] : real_allocs)
		cuda_malloc(ptr, size);
	for (auto &[ptr, size] : int_allocs)
		cuda_malloc(ptr, size);
	t.alloc_end = Clock::now();
	
	// ============ Initialization ============

	t.init_start = Clock::now();
	cuda_memcpy(d_A, A, m * n, cudaMemcpyHostToDevice);
	cuda_memcpy(d_b, b, m, cudaMemcpyHostToDevice);
	cuda_memcpy(d_c, c, n, cudaMemcpyHostToDevice);
	init_I<<<dim3(blocks_for_m, blocks_for_m), dim3(BS_2D, BS_2D)>>>(d_B_inv, m);
	cuda_memcpy(d_c_b, d_c + n - m, m, cudaMemcpyDeviceToDevice);
	cuda_memcpy(d_x_b, d_b, m, cudaMemcpyDeviceToDevice);
	init_b_ixs<<<blocks_for_m, BS_1D>>>(d_b_ixs, m, n);
	cuda_memcpy(d_y_aug, &alpha, 1, cudaMemcpyHostToDevice);
	init_D_from_c<<<blocks_for_n, BS_1D>>>(d_c, d_D, m+1, n);
	init_D_from_A<<<dim3(blocks_for_n, blocks_for_m + 1), dim3(BS_2D, BS_2D)>>>(d_A, d_D, m, n);
	t.init_end = Clock::now();

	int i = 0;
	int p, q, alpha_num_non_pos;
	auto status = SolveStatus::MaxIter;

	// ============== Main loop ==============
	do {
		// ========= Entering variable =========

		t.ev_start = Clock::now();
		// y = c_b * B_inv
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, m, m, &alpha, d_c_b, 1, d_B_inv, m, &beta, d_y, 1);
		// y_aug[1..m] = y
		cudaMemcpy(d_y_aug + 1, d_y, m * sizeof(real), cudaMemcpyDeviceToDevice);
		// e = [1 y] * [-c; A] 
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, n, m+1, &alpha, d_y_aug, 1, d_D, m+1, &beta, d_e, 1);
		t.blas_end = Clock::now();

		p = entering_var(d_e, n, blocks_for_n, d_curr, d_next, d_ix);
		t.ev_end = Clock::now();
		if (p < 0) {
			status = SolveStatus::OptimumFound;
			break;
		}

		// ============ Leaving variable ============

		t.lv_start = Clock::now();
		// get column A_p from A
		cudaMemcpy(d_A_p, d_A + p * m, m * sizeof(real), cudaMemcpyDeviceToDevice);
		// alpha = B_inv * A_p
		cublasSgemv(handle, CUBLAS_OP_N,
			m, m, &alpha, d_B_inv, m, d_A_p, 1, &beta, d_alpha, 1);
		// reset the number of non-positive elements in alpha
		cudaMemset(d_alpha_num_non_pos, 0, sizeof(int));

		compute_theta<<<blocks_for_m, BS_1D>>>
			(d_x_b, d_alpha, d_theta, d_theta_flags, m, d_alpha_num_non_pos);
		cudaMemcpy(&alpha_num_non_pos, d_alpha_num_non_pos, sizeof(int), cudaMemcpyDeviceToHost);
		if (alpha_num_non_pos == m) {
			status = SolveStatus::Unbounded;
			break;
		}
		q = get_min_ix(d_theta, m, blocks_for_m, nullptr, d_curr, d_next, d_ix);
		t.lv_end = Clock::now();

		// ============ Update the basis ============

		t.B_start = Clock::now();
		if (compute_E(d_E, d_alpha, m, blocks_for_m, q, d_alpha_q)) {
			status = SolveStatus::ThetaOverflow;
			break;
		}
		// new_B_inv = E * B_inv
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			m, m, m, &alpha, d_E, m, d_B_inv, m, &beta, d_new_B_inv, m);
		std::swap(d_B_inv, d_new_B_inv);
		t.B_end = Clock::now();
		
		// ======= Update the cost, indices and solution =======

		cudaMemcpy(d_c_b + q, d_c + p, sizeof(real), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_b_ixs + q, &p, sizeof(int), cudaMemcpyHostToDevice);
		// x_b = B_inv * b
		cublasSgemv(handle, CUBLAS_OP_N,
			m, m, &alpha, d_B_inv, m, d_b, 1, &beta, d_x_b, 1);

	} while (++i < MAX_ITER);

	real z;
	if (status == SolveStatus::OptimumFound) {
		cublasSdot(handle, m, d_c_b, 1, d_x_b, 1, &z);
		cudaMemcpy(x_b, d_x_b, m * sizeof(real), cudaMemcpyDeviceToHost);
		cudaMemcpy(b_ixs, d_b_ixs, m * sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	t.dealloc_start = Clock::now();
	cublasDestroy(handle);
	for (auto &[ptr,_] : real_allocs)
		if (ptr) cudaFree(ptr);
	for (auto &[ptr,_] : int_allocs)
		if (ptr) cudaFree(ptr);
	t.dealloc_end = Clock::now();
	
	return std::make_pair(z, status);
}

/* ===================== MAIN ===================== */

int main(int argc, char *argv[]) {
	std::ios_base::sync_with_stdio(false);
	
	if (argc < 2) {
		std::cerr << "Please, specify an input file.\n";
		return 1;
	}
	
	std::ifstream file(argv[1]);
	if (!file.is_open()) {
		std::cerr << "Could not open " << argv[1] << ".\n";
		return 1;
	}

	TimePoints t;
	t.start = Clock::now();

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
	t.host_alloc_end = Clock::now();

	t.file_read_start = Clock::now();
	load_array(file, A, m, n);
	load_array(file, b, m, 1);
	load_array(file, c, 1, n);
	file.close();
	t.file_read_end = Clock::now();

	auto [z, status] = solve(A, b, c, x_b, b_ixs, m, n, t);

	switch (status) {
		case SolveStatus::OptimumFound:
			std::cout << "Optimum found: " << z << '\n';
			for (int i = 0; i < m; ++i)
				std::cout << "x_" << b_ixs[i] << " = " << x_b[i] << "\n";
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
	}

	cudaFreeHost(A);
	cudaFreeHost(b);
	cudaFreeHost(c);
	cudaFreeHost(x_b);
	cudaFreeHost(b_ixs);

	t.end = Clock::now();

	print_elapsed_time("Total", t.start, t.end);
	print_elapsed_time("Read", t.file_read_start, t.file_read_end);
	print_elapsed_time("Host alloc", t.host_alloc_start, t.host_alloc_end);
	print_elapsed_time("BLAS entering var", t.ev_start, t.blas_end);
	print_elapsed_time("Entering var", t.ev_start, t.end);
	print_elapsed_time("Alloc", t.alloc_start, t.alloc_end);
	print_elapsed_time("Dealloc", t.dealloc_start, t.dealloc_end);
	print_elapsed_time("Init", t.init_start, t.init_end);
	print_elapsed_time("Leaving var", t.lv_start, t.lv_end);
	print_elapsed_time("Binv update", t.B_start, t.B_end);

	return 0;
}
