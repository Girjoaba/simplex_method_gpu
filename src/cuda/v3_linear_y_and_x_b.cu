#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
constexpr int MAX_ITER = 5;

// #define PRINT

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
		std::cerr << "cudaMallocHost failed for " << name << ": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t err = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		std::cerr << "cudaMalloc failed for " << name << ": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for " << name << ": " << cudaGetErrorString(err) << "\n";
		std::exit(EXIT_FAILURE);
	}
}

template<typename T>
void load_matrix_impl(std::ifstream& file, T* a, int m, int n, const char* name) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!(file >> a[R2C(i, j, m)])) {
				std::cerr <<"Failed to read (" << i << "," << j << ") for " << name << "\n";
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
#ifndef PRINT
	std::cout << "# Iteration " << ++i << '\n';
#else
	std::cout << "# Iteration " << ++i << "\n\n";
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
	if (j < n) D[R2C(0,j,m)] = -c[j];
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

__global__ void compute_scalar(real* c_p, real* c_b_q, real* scalar) {
	scalar[0] += c_p[0] - c_b_q[0];
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
		int flag = alpha[j] > 0;
		flags[j] = flag;
		theta[j] = flag ? (x_b[j] / alpha[j]) : INFINITY;	// explosion?
		atomicAdd(num_non_pos, !flag);
	}
}

__global__ void compute_E_q(real* E_q, real* alpha, real* alpha_q, int q, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m)
		E_q[i] = (i != q) ? (-alpha[i] / alpha_q[0]) : (1.0 / alpha_q[0] - 1.0);
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

/* ===================== SOLVER ===================== */

std::pair<real, SolveStatus> solve(real* A, real* b, real* c, real* x_b, int* b_ixs, int m, int n, TimeStruct& t) {
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasCreate failed.\n";
		std::exit(EXIT_FAILURE);
	}
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	real *d_A, *d_b, *d_c;
	real *d_B_inv, *d_c_b, *d_c_b_q, *d_x_b;
	real *d_scalar, *d_y_aug, *d_D, *d_e;
	real *d_alpha, *d_theta;
	real *d_alpha_q, *d_B_inv_q, *d_E_q;
	real *d_next, *d_curr;
	int *d_b_ixs;
	int *d_theta_flags, *d_alpha_num_non_pos;
	int *d_ix;

	int blocks_for_n = num_blocks_1D(n);
	int blocks_for_m = num_blocks_1D(m);

	const real one = 1.0f, zero = 0.0f;

	PtrAlloc<real> real_allocs[] = {
		{d_A, m * n}, {d_b, m}, {d_c, n}, {d_c_b_q, 1}, {d_B_inv, m * m},
		{d_c_b, m}, {d_x_b, n}, {d_scalar, 1}, {d_y_aug, m + 1},
		{d_D, (m + 1) * n}, {d_e, n}, {d_alpha, m}, {d_theta, m}, {d_alpha_q, 1},
		{d_B_inv_q, m}, {d_E_q, m}, {d_next, blocks_for_n}, {d_curr, n}
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
	
	// ============ Initialization ============

	t.init_start = Clock::now();
	cuda_memcpy(d_A, A, m * n, cudaMemcpyHostToDevice);
	cuda_memcpy(d_b, b, m, cudaMemcpyHostToDevice);
	cuda_memcpy(d_c, c, n, cudaMemcpyHostToDevice);
	init_I<<<dim3(blocks_for_m, blocks_for_m), dim3(BS_2D, BS_2D)>>>(d_B_inv, m);
	cuda_memcpy(d_c_b, d_c + n - m, m, cudaMemcpyDeviceToDevice);
	cuda_memcpy(d_x_b, d_b, m, cudaMemcpyDeviceToDevice);
	init_b_ixs<<<blocks_for_m, BS_1D>>>(d_b_ixs, m, n);
	cuda_memcpy(d_y_aug, &one, 1, cudaMemcpyHostToDevice);
	cuda_memcpy(d_y_aug + 1, d_c_b, n - m, cudaMemcpyDeviceToDevice);
	init_D_from_c<<<blocks_for_n, BS_1D>>>(d_c, d_D, m+1, n);
	init_D_from_A<<<dim3(blocks_for_n, blocks_for_m + 1), dim3(BS_2D, BS_2D)>>>(d_A, d_D, m, n);
	t.init_end = Clock::now();

	int i = 0;
	int p, q, alpha_num_non_pos;
	auto status = SolveStatus::MaxIter;

	// ============== Main loop ==============
	do {
		print_iteration(i);
		// ========= Entering variable =========

		// e = [1 y] * [-c; A] 
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, n, m+1, &one, d_y_aug, 1, d_D, m+1, &zero, d_e, 1);
		print_matrix(d_e, 1, n);

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
		cublasScopy(handle, m, d_B_inv + q, m, d_B_inv_q, 1);
		compute_E_q<<<blocks_for_m, BS_1D>>>(d_E_q, d_alpha, d_alpha + q, q, m);
		cublasSger(handle, m, m, &one, d_E_q, 1, d_B_inv_q, 1, d_B_inv, m);
		t.B_inv_duration += duration(t.B_inv_start, Clock::now());
		print_matrix(d_B_inv, m, m);
		
		// ======= Update the cost, indices, solution and y =======

		cudaMemcpy(d_c_b_q, d_c_b + q, sizeof(real), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_c_b + q, d_c + p, sizeof(real), cudaMemcpyDeviceToDevice);
		print_matrix(d_c_b, 1, m);
		cudaMemcpy(d_b_ixs + q, &p, sizeof(int), cudaMemcpyHostToDevice);
		print_matrix(d_b_ixs, 1, m);

		t.x_b_start = Clock::now();
		// x_b = B_inv * b
		cublasSdot(handle, m, d_B_inv_q, 1, d_b, 1, d_scalar);
		cublasSaxpy(handle, m, d_scalar, d_E_q, 1, d_x_b, 1);
		t.x_b_duration += duration(t.x_b_start, Clock::now());
		print_matrix(d_x_b, m, 1);

		t.y_start = Clock::now();
		// y += ((c_p - c_b_q) + c_b * E_q) * B_inv_q
		cublasSdot(handle, m, d_c_b, 1, d_E_q, 1, d_scalar);
		compute_scalar<<<1,1>>>(d_c + p, d_c_b_q, d_scalar);
		cublasSaxpy(handle, m, d_scalar, d_B_inv_q, 1, d_y_aug + 1, 1);
		t.y_duration += duration(t.y_start, Clock::now());

	} while (++i < MAX_ITER);
	print_endline();

	real z;
	if (status == SolveStatus::OptimumFound) {
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
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