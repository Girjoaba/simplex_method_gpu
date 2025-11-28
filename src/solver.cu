#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <format>
#include <iomanip>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <utility>

using real = double;

constexpr int BS_1D = 256;
constexpr int BS_2D = 16;
constexpr int MAX_ITER = 5000;
constexpr real EPS = 1E-8;
constexpr real one = 1.0, minus_one = -1.0, zero = 0.0;

#define cuda_malloc_host(ptr, n) cuda_malloc_host_impl(&ptr, n, #ptr)
#define cuda_malloc(d_ptr, n) cuda_malloc_impl(&d_ptr, n, #d_ptr)
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
#define load_matrix(ptr, m, n) load_matrix_impl(ptr, m, n, #ptr)

template<typename T>
struct PtrAlloc {
	T*& ptr;
	int size;
};

enum class SolveStatus {
	MaxIter,
	OptimumFound,
	Unbounded
};

__host__ __device__ __forceinline__
constexpr int AT(int i, int j, int s) { return i * s + j; }

__host__ __device__ __forceinline__
constexpr int R2C(int i, int j, int m) { return j * m + i; }

__host__ __device__ __forceinline__
constexpr int num_blocks(int n, int BS) { return (n + BS - 1) / BS; }

/* ===================== UTILITIES ===================== */

template<typename T>
void cuda_malloc_host_impl(T** ptr, int n, const char* name) {
	cudaError_t err = cudaMallocHost((void**)ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		std::cerr << std::format("cudaMallocHost failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t err = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (err != cudaSuccess) {
		std::cerr << std::format("cudaMalloc failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (err != cudaSuccess) {
		std::cerr << std::format("cudaMemcpy failed for {}: {}\n", name, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

template<typename T>
void load_matrix_impl(T* a, int m, int n, const char* name) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!(std::cin >> a[R2C(i, j, m)])) {
				std::cerr << std::format("Failed to read ({},{}) for {}\n", i, j, name);
				std::exit(EXIT_FAILURE);
			}
		}
	}
}

template <typename T>
void swap_values(T* a, T* b, T* tmp, int p, int q) {
	cudaMemcpy(tmp, a + q, sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(a + q, b + p, sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(b + p, tmp, sizeof(T), cudaMemcpyHostToDevice);
}

/* ===================== KERNELS ===================== */

__global__ void init_identity(real* I, int m) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < m)
		I[R2C(i,j,m)] = (i == j) ? 1.0 : 0.0;
}

__global__ void init_indices(int* N_ixs, int artificial_end) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < artificial_end) N_ixs[j] = j;
}

__global__ void init_cost_phase_one(real* c_phase_one, int artificial_start, int artificial_end) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < artificial_end)
		c_phase_one[j] = (j < artificial_start) ? 0.0 : -1.0;
}

__global__ void init_cost_phase_two(int* N_ixs, int* B_ixs, real* c, real* c_N, real* c_B, int m, int new_end) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < m)       c_B[j] = c[B_ixs[j]];
	if (j < new_end) c_N[j] = c[N_ixs[j]];
}

__global__ void compute_theta(real* x_B, real* alpha, real* theta, int m) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < m)
		theta[j] = alpha[j] > 0.0 ? x_B[j] / alpha[j] : INFINITY;
}

__global__ void compute_E_q(real* E, real* alpha, int m, int q, real alpha_q) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		E[R2C(i, q, m)] = (i != q) ? (-alpha[i] / alpha_q) : (1.0 / alpha_q);
}

/* ===================== TRANSITION ===================== */

void transition(real* d_N, int* d_N_ixs, int* d_B_ixs, real* c, real* d_c, real* d_c_N,
	real* d_c_B, int m, int identity_start, int artificial_start, int artificial_end
) {
	std::vector<int> N_ixs; N_ixs.resize(identity_start);
	cudaMemcpy(N_ixs.data(), d_N_ixs, identity_start * sizeof(int), cudaMemcpyDeviceToHost);

	// make the non-artificial columns of N contiguous in memory and update the indices
	int i = 0, new_end = identity_start - (artificial_end - artificial_start);
	for (int j = new_end; j < identity_start; ++j)	{
		if (N_ixs[j] >= artificial_start) continue;
		while (N_ixs[i] < artificial_start) ++i;
		cudaMemcpy(d_N + i * m, d_N + j * m, m * sizeof(real), cudaMemcpyDeviceToDevice);
		N_ixs[i++] = N_ixs[j];
	}
	cudaMemcpy(d_N_ixs, N_ixs.data(), new_end * sizeof(int), cudaMemcpyHostToDevice);

	cuda_memcpy(d_c, c, artificial_start, cudaMemcpyHostToDevice);
	init_cost_phase_two<<<num_blocks(std::max(m, new_end), BS_1D), BS_1D>>>
		(d_N_ixs, d_B_ixs, d_c, d_c_N, d_c_B, m, new_end);
	cudaDeviceSynchronize();
}

/* ====================== CORE ====================== */

std::pair<real, SolveStatus> core(
	cublasHandle_t handle,
	real* d_B, real* d_N, real* d_b, real*& d_B_inv, real* d_c_B,
	real* d_c_N, real* d_x_B, real* d_y, real* d_e, real* d_alpha,
	real* d_theta, real* d_swap_column, real*& d_new_B_inv, real* d_E,
	int* d_B_ixs, int* d_N_ixs, real* d_swap_real, int* d_swap_int,
	thrust::device_ptr<real> thrust_e,
	thrust::device_ptr<real> thrust_alpha,
	thrust::device_ptr<real> thrust_theta,
	int m, int n, int blocks_1d_m, int blocks_2d_m
) {	// n := # of non-basic columns

	int i = 0, p, q;
	auto status = SolveStatus::MaxIter;
	thrust::device_ptr<real> iterator;
	real alpha_q;

	do {
		// y = c_B * B_inv
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, m, m, &one, d_c_B, 1, d_B_inv, m, &zero, d_y, 1);
		// e = y * N 
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			1, n, m, &one, d_y, 1, d_N, m, &zero, d_e, 1);
		// e = e - c_N
		cublasDaxpy(handle, n, &minus_one, d_c_N, 1, d_e, 1);

		iterator = thrust::min_element(thrust_e, thrust_e + n);
		p = iterator - thrust_e;
		if (*iterator > -EPS) {
			status = SolveStatus::OptimumFound;
			break;
		}

		// ============ Leaving variable ============

		// alpha = B_inv * N_p
		cublasDgemv(handle, CUBLAS_OP_N,
			m, m, &one, d_B_inv, m, d_N + p * m, 1, &zero, d_alpha, 1);

		if (*thrust::max_element(thrust_alpha, thrust_alpha + m) <= EPS) {
			status = SolveStatus::Unbounded;
			break;
		}

		compute_theta<<<blocks_1d_m, BS_1D>>>(d_x_B, d_alpha, d_theta, m);
		cudaDeviceSynchronize();
		iterator = thrust::min_element(thrust_theta, thrust_theta + m);
		q = iterator - thrust_theta;

		// ============ Update the basis and inverse ============

		// swap N[p] and B[q]
		cudaMemcpy(d_swap_column, d_B + q * m, m * sizeof(real), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_B + q * m, d_N + p * m, m * sizeof(real), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_N + p * m, d_swap_column, m * sizeof(real), cudaMemcpyDeviceToDevice);

		cudaMemcpy(&alpha_q, d_alpha + q, sizeof(real), cudaMemcpyDeviceToHost);
		init_identity<<<dim3(blocks_2d_m, blocks_2d_m), dim3(BS_2D, BS_2D)>>>(d_E, m);
		cudaDeviceSynchronize();
		compute_E_q<<<blocks_1d_m, BS_1D>>>(d_E, d_alpha, m, q, alpha_q);
		cudaDeviceSynchronize();
		// new_B_inv = E * B_inv
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			m, m, m, &one, d_E, m, d_B_inv, m, &zero, d_new_B_inv, m);
		std::swap(d_B_inv, d_new_B_inv);
		
		// ======= Update the cost, indices and solution =======

		swap_values<real>(d_c_B, d_c_N, d_swap_real, p, q);
		swap_values<int>(d_B_ixs, d_N_ixs, d_swap_int, p, q);
		// x_B = B_inv * b
		cublasDgemv(handle, CUBLAS_OP_N,
			m, m, &one, d_B_inv, m, d_b, 1, &zero, d_x_B, 1);

	} while (++i < MAX_ITER);
	std::cout << "# Iterations " << i << '\n';

	real z;
	if (status == SolveStatus::OptimumFound)
		cublasDdot(handle, m, d_c_B, 1, d_x_B, 1, &z);
	
	return std::make_pair(z, status);
}

/* ===================== SOLVER ===================== */

std::pair<real, SolveStatus> solve(
	real* N, real* b, real* c, int* B_ixs,
	int m, int n, int num_surplus, int num_slack
) {
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasCreate failed.\n";
		std::exit(EXIT_FAILURE);
	}

	real *d_B, *d_N, *d_b, *d_c, *d_c_phase_one, *d_B_inv;
	real *d_c_B, *d_c_N, *d_x_B, *d_y, *d_e, *d_alpha, *d_theta;
	real *d_swap_column, *d_E, *d_new_B_inv, *d_swap_real;
	int *d_B_ixs, *d_N_ixs, *d_swap_int;

	int identity_start = n + num_surplus;
	int artificial_start = identity_start + num_slack;
	int artificial_end = identity_start + m;
	int blocks_1d_n = num_blocks(n, BS_1D);
	int blocks_1d_m = num_blocks(m, BS_1D);
	int blocks_2d_m = num_blocks(m, BS_2D);
	int blocks_1d_artificial_end = num_blocks(artificial_end, BS_1D);

	PtrAlloc<real> real_allocs[] = {
		{d_B, m * m}, {d_N, m * identity_start}, {d_b, m}, {d_c, artificial_start},
		{d_c_phase_one, artificial_end}, {d_B_inv, m * m}, {d_c_N, artificial_end},
		{d_x_B, m}, {d_y, m}, {d_e, identity_start}, {d_alpha, m}, {d_theta, m},
		{d_swap_column, m}, {d_E, m * m}, {d_new_B_inv, m * m}, {d_swap_real, 1}
	};
	PtrAlloc<int> int_allocs[] = { {d_N_ixs, artificial_end}, {d_swap_int, 1} };
	
	// ============== Allocation ==============

	for (auto &[ptr, size] : real_allocs)
		cuda_malloc(ptr, size);
	for (auto &[ptr, size] : int_allocs)
		cuda_malloc(ptr, size);

	d_B_ixs = d_N_ixs + identity_start;
	d_c_B = d_c_N + identity_start;

	thrust::device_ptr<real> thrust_e(d_e);
	thrust::device_ptr<real> thrust_alpha(d_alpha);
	thrust::device_ptr<real> thrust_theta(d_theta);
	
	// ============ Initialization ============

	init_identity<<<dim3(blocks_2d_m, blocks_2d_m), dim3(BS_2D, BS_2D)>>>(d_B_inv, m);
	init_identity<<<dim3(blocks_2d_m, blocks_2d_m), dim3(BS_2D, BS_2D)>>>(d_B, m);
	init_indices<<<blocks_1d_artificial_end, BS_1D>>>(d_N_ixs, artificial_end);
	init_cost_phase_one<<<blocks_1d_artificial_end, BS_1D>>>(d_c_phase_one, artificial_start, artificial_end);
	cuda_memcpy(d_N, N, m * identity_start, cudaMemcpyHostToDevice);
	cuda_memcpy(d_b, b, m, cudaMemcpyHostToDevice);
	cuda_memcpy(d_x_B, d_b, m, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cuda_memcpy(d_c_N, d_c_phase_one, artificial_end, cudaMemcpyDeviceToDevice);

	// =============== Phase I ===============

	auto [sum_artificial, status_phase_one] = core(
		handle, d_B, d_N, d_b, d_B_inv, d_c_B, d_c_N, d_x_B, d_y, d_e, d_alpha, d_theta,
		d_swap_column, d_new_B_inv, d_E, d_B_ixs, d_N_ixs, d_swap_real, d_swap_int,
		thrust_e, thrust_alpha, thrust_theta, m, identity_start, blocks_1d_m, blocks_2d_m
	);

	if (status_phase_one != SolveStatus::OptimumFound || fabs(sum_artificial) > EPS) {
		std::cout << "Phase I failed, the optimum is " << sum_artificial << '\n';
		std::exit(EXIT_FAILURE);
	}

	cudaMemcpy(B_ixs, d_B_ixs, m * sizeof(int), cudaMemcpyDeviceToHost);
	bool has_artificials = false;
	for (int ix : std::span{B_ixs, B_ixs + m}) {
		if (ix >= artificial_start) {
			std::cerr << "!! Index " << ix << " >= " << artificial_start << '\n';
			has_artificials = true;
		}	
	}
	if (has_artificials) std::exit(EXIT_FAILURE);

	// =============== Phase II ===============

	transition(d_N, d_N_ixs, d_B_ixs, c, d_c, d_c_N, d_c_B,
		m, identity_start, artificial_start, artificial_end);

	auto [z, status] = core(
		handle, d_B, d_N, d_b, d_B_inv, d_c_B, d_c_N, d_x_B, d_y, d_e, d_alpha, d_theta,
		d_swap_column, d_new_B_inv, d_E, d_B_ixs, d_N_ixs, d_swap_real, d_swap_int,
		thrust_e, thrust_alpha, thrust_theta, m, identity_start - m + num_slack, blocks_1d_m, blocks_2d_m
	);

	// ============== Deallocation ==============

	cublasDestroy(handle);
	for (auto &[ptr,_] : real_allocs)
		cudaFree(ptr);
	for (auto &[ptr,_] : int_allocs)
		cudaFree(ptr);
	
	return std::make_pair(z, status);
}

/* ===================== MAIN ===================== */

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);

	real optimum;
	int m, n, num_surplus, num_slack;
	if (!(std::cin >> m >> n >> num_surplus >> num_slack >> optimum)) {
		std::cerr << "Failed to read the file\n";
		std::exit(EXIT_FAILURE);
	}

	int identity_start = n + num_surplus;
	int artificial_start = identity_start + num_slack;
	real *N, *b, *c;
	int* B_ixs;

	cuda_malloc_host(N, m * identity_start);
	cuda_malloc_host(b, m);
	cuda_malloc_host(c, artificial_start);
	cuda_malloc_host(B_ixs, m);

	load_matrix(N, m, identity_start);
	load_matrix(b, m, 1);
	load_matrix(c, 1, artificial_start);

	auto [z, status] = solve(N, b, c, B_ixs, m, n, num_surplus, num_slack);

	switch (status) {
		case SolveStatus::OptimumFound:
			std::cout << std::scientific << std::uppercase << std::setprecision(3)
								<< "Optimum found: " << -z << '\n'
								<< "Optimum known: " << optimum << '\n';
			break;

		case SolveStatus::Unbounded:
			std::cout << "Problem unbounded.\n";
			break;

		case SolveStatus::MaxIter:
			std::cout << "MAX_ITER exceeded.\n";
			break;
	}

	cudaFreeHost(N);
	cudaFreeHost(c);
	cudaFreeHost(b);

	return 0;
}