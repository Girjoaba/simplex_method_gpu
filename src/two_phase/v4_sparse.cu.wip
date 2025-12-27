#include <cfloat>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <iostream>
#include <iomanip>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

#include <utility>

#define MAX_ITERS 100000
#define SOLUTION_PERIOD 100

constexpr double OPTIMALITY_TOL = 1e-6;
constexpr double PIVOT_TOL      = 1e-5;
constexpr double ROW_TOL        = 1e-12;

constexpr double ONE = 1.0;
constexpr double MINUS_ONE = -1.0;
constexpr double ZERO = 0.0;

constexpr int BLOCK_DIM_1D = 256;
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

constexpr int div_up(int n, int block_dim) { return (n + block_dim - 1) / block_dim; }

enum class SolveStatus { MaxIter, OptimumFound, Unbounded };

/* ===================== CUDA CHECKS ===================== */

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(code) << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cusolverCheckError(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
	if (code != CUSOLVER_STATUS_SUCCESS) {
		std::cerr << "cuSOLVER Error: " << code << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cublasCheckError(ans) { cublasAssert((ans), __FILE__, __LINE__); }
void cublasAssert(cublasStatus_t code, const char *file, int line) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS Error: " << code << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cusparseCheckError(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
	if (code != CUSPARSE_STATUS_SUCCESS) {
		std::cerr << "cuSPARSE Error: " << cusparseGetErrorString(code) << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cuda_malloc(d_ptr, n) { cuda_malloc_impl(&d_ptr, n, #d_ptr); }
template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t code = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (code != cudaSuccess) {
		std::cerr << "cudaMalloc failed for " << name << ": " << cudaGetErrorString(code) << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t code = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (code != cudaSuccess) {
		std::cerr << "cudaMempcy failed for " << name << ": " << cudaGetErrorString(code) << '\n';
		std::exit(EXIT_FAILURE);
	}
}

/* ===================== COMPRESSED SPARSE FORMAT ===================== */

struct CS {
	int m, n, nnz;
	bool is_csr;

	double *values;
	int *indices;
	int *starts;

	CS(int m, int n, int nnz, bool is_csr = false) :
		m(m), n(n), nnz(nnz), is_csr(is_csr) {
		cuda_malloc(values, nnz);
		cuda_malloc(indices, nnz);
		cuda_malloc(starts, (is_csr ? m : n) + 1);
	}

	void upload(
		const std::vector<double>& h_values,
		const std::vector<int>& h_indices,
		const std::vector<int>& h_starts
	) {
		cuda_memcpy(values, h_values.data(), nnz, cudaMemcpyHostToDevice);
		cuda_memcpy(indices, h_indices.data(), nnz, cudaMemcpyHostToDevice);
		cuda_memcpy(starts, h_starts.data(), (is_csr ? m : n) + 1, cudaMemcpyHostToDevice);
	}

	~CS() { cudaFree(values); cudaFree(indices); cudaFree(starts); }
};

struct CS_basis : CS {
	int capacity, *column_nnz, *thread_to_column_map;

	CS_basis(int m, int n, int nnz) : CS(m, n, nnz) {
		capacity = nnz;
		cuda_malloc(column_nnz, m);
		cuda_malloc(thread_to_column_map, nnz);
	}

	void adjust_capacity(int new_nnz) {
		if (new_nnz <= capacity) return;

		do { capacity *= 2; }
		while (new_nnz > capacity);

		double *new_values;
		int *new_indices, *new_map;

		cudaFree(values);
		cudaFree(indices);
		cudaFree(thread_to_column_map);

		cuda_malloc(new_values, capacity);
		cuda_malloc(new_indices, capacity);
		cuda_malloc(new_map, capacity);

		values = new_values;
		indices = new_indices;
		thread_to_column_map = new_map;
	}

	~CS_basis() { cudaFree(column_nnz); cudaFree(thread_to_column_map); }
};

/* ===================== KERNELS ===================== */

__global__ void assemble_basis(const double* __restrict__ A, double* __restrict__ B, const int* __restrict__ B_ids, int m) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < m && j < m)
		B[i + j * m] = A[i + B_ids[j] * m];
}	// how much do restrict and const help?

__global__ void mask_basis(double* vec, const int* B_ids, double val, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		vec[B_ids[i]] = val;
}

__global__ void update_xB(double* __restrict__ xB, const double* d, int leave, double theta_min, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		xB[i] = (i != leave) ? (xB[i] - theta_min * d[i]) : theta_min;
}

__global__ void gather_cost(double *cB, const double* c, const int* B_ids, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		cB[i] = c[B_ids[i]];
}

__global__ void compute_B_column_nnz(int m, int *B_ids, int *B_column_nnz, int *A_starts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		B_column_nnz[i] = A_starts[B_ids[i] + 1] - A_starts[B_ids[i]];
}

__global__ void extract_B(
    double* A_values, int* A_indices, int* A_starts,
    double* B_values, int* B_indices, int* B_starts,
    int* B_ids, int* thread_to_column_map, int nnz
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nnz) {
		int col_in_B = thread_to_column_map[i] - 1; 
		int col_in_A = B_ids[col_in_B];
		int local_offset = i - B_starts[col_in_B];
		int src_idx = A_starts[col_in_A] + local_offset;

		B_values[i] = A_values[src_idx];
		B_indices[i] = A_indices[src_idx];
	}
}

/* ===================== DEVICE RESOURCES ===================== */

template<typename T>
struct PtrAlloc { T*& ptr; int size; };

struct DeviceResources {
	cusolverDnHandle_t handle;
	cublasHandle_t handle_cublas;
	cusparseHandle_t handle_cusparse;

	CS A_csc, A_csr;
	CS_basis B_csc;
	dim3 block_dim, grid_dim, grid_dim_1D;

	double *b, *c, *xB;
	double *d;   // direction
	double *rc;  // reduced cost
	int m, *B_ids;

	double *B, *work, *y;
	int lwork, *ipiv, *info;

	std::vector<PtrAlloc<double>> double_allocs;
	std::vector<PtrAlloc<int>> int_allocs;

	DeviceResources(int m, int n, int nnz) :
		m(m), grid_dim(dim3(div_up(m, BLOCK_DIM_X), div_up(m, BLOCK_DIM_Y))),
		block_dim(dim3(BLOCK_DIM_X, BLOCK_DIM_Y)), grid_dim_1D(dim3(div_up(m, BLOCK_DIM_1D))),
		A_csc(m, n, nnz), A_csr(m, n, nnz, true), B_csc(m, m, m),
		double_allocs {{b, m}, {c, n}, {xB, m}, {d, m}, {rc, n}, {B, m * m}, {y, m}},
		int_allocs {{ipiv, m}, {B_ids, m}, {info, 1}} {

		cusolverCheckError(cusolverDnCreate(&handle));
		cublasCheckError(cublasCreate(&handle_cublas));
		cusparseCheckError(cusparseCreate(&handle_cusparse));

		for (auto &[ptr, size] : double_allocs) cuda_malloc(ptr, size);
		for (auto &[ptr, size] : int_allocs)    cuda_malloc(ptr, size);

		cusolverCheckError(cusolverDnDgetrf_bufferSize(handle, m, m, B, m, &lwork));
		cuda_malloc(work, lwork);
	}

	void csc2csr(CS& csc, CS& csr) {
		size_t bufferSize;
		void *buffer;
		
		cusparseCheckError(cusparseCsr2cscEx2_bufferSize(
			handle_cusparse, csc.n, csc.m, csc.nnz,
			csc.values, csc.starts, csc.indices,
			csr.values, csr.starts, csr.indices,
			CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
			&bufferSize
		));

		cudaMalloc(&buffer, bufferSize);

		cusparseCheckError(cusparseCsr2cscEx2(
			handle_cusparse, csc.n, csc.m, csc.nnz,
			csc.values, csc.starts, csc.indices,
			csr.values, csr.starts, csr.indices,
			CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
			buffer
		));

		cudaFree(buffer);
	}

	void assemble_basis() {
		compute_B_column_nnz<<<BLOCK_DIM_1D, grid_dim_1D>>>(m, B_ids, B_csc.column_nnz, A_csc.starts);
		thrust::exclusive_scan(thrust::device, B_csc.column_nnz, B_csc.column_nnz + m + 1, B_csc.starts);

		int nnz;
		cuda_memcpy(&nnz, B_csc.starts + m, 1, cudaMemcpyDeviceToHost);

		B_csc.adjust_capacity(nnz);

		thrust::upper_bound(
			thrust::device,
			B_csc.starts, B_csc.starts + m + 1,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(0) + nnz,
			B_csc.thread_to_column_map
		);

		extract_B<<<BLOCK_DIM_1D, div_up(nnz, BLOCK_DIM_1D)>>>(
			A_csc.values, A_csc.indices, A_csc.starts,
			B_csc.values, B_csc.indices, B_csc.starts,
			B_ids, B_csc.thread_to_column_map, nnz
		);
	}

	~DeviceResources() {
		for (auto &[ptr,_] : double_allocs) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)    cudaFree(ptr);
		cusolverDnDestroy(handle);
		cublasDestroy(handle_cublas);
		cusparseDestroy(handle_cusparse);
	}
};

/* ===================== THRUST ===================== */

struct RatioTestUnaryOp {
	const double *xB, *d;

	__device__ thrust::pair<double, int> operator()(int i) const {
		double d_i = d[i];
		return d_i > PIVOT_TOL ? thrust::make_pair(xB[i] / d_i, i) : thrust::make_pair(DBL_MAX, -1);
	}
};

struct MinPairOp {
	__device__ thrust::pair<double, int> operator()
	(const thrust::pair<double, int>& a, const thrust::pair<double, int>& b) const {
		if (a.first < b.first) return a;
		if (a.first > b.first) return b;
		return (a.second < b.second) ? a : b;
	}
};

struct IsNonZero {
	__device__ bool operator()(double v) const { return fabs(v) > ROW_TOL; }
};

/* ===================== CORE LOGIC ===================== */

std::pair<double, SolveStatus> core(
	DeviceResources& gpu, int m, int n
) {

	auto status = SolveStatus::MaxIter;
	int iteration;

	for (iteration = 1; iteration <= MAX_ITERS; ++iteration) {
		// solve for y: B^T * y^T = cB^T
		gather_cost<<<BLOCK_DIM_1D, gpu.grid_dim_1D>>>(gpu.y, gpu.c, gpu.B_ids, m);
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_T, m, 1, gpu.B, m, gpu.ipiv, gpu.y, m, gpu.info));

		// compute rc^T = -A^T * y^T + c^T
		cuda_memcpy(gpu.rc, gpu.c, n, cudaMemcpyDeviceToDevice);
		cublasCheckError(cublasDgemv(gpu.handle_cublas, CUBLAS_OP_T, m, n, &MINUS_ONE, gpu.A, m, gpu.y, 1, &ONE, gpu.rc, 1));

		// select entering variable
		mask_basis<<<BLOCK_DIM_1D, gpu.grid_dim_1D>>>(gpu.rc, gpu.B_ids, -1.0, m);
		thrust::device_ptr<double> thrust_rc(gpu.rc);
		auto iterator = thrust::max_element(thrust_rc, thrust_rc + n);
		int enter = iterator - thrust_rc;
		
		if (*iterator <= OPTIMALITY_TOL) {
			status = SolveStatus::OptimumFound;
			break;
		}
		
		// solve for d: B * d = A_enter
		cuda_memcpy(gpu.d, gpu.A + (enter * m), m, cudaMemcpyDeviceToDevice);
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.d, m, gpu.info));
		
		// ratio test
		auto [theta_min, leave] = thrust::transform_reduce(
			thrust::device,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(m),
			RatioTestUnaryOp{gpu.xB, gpu.d},
			thrust::make_pair(DBL_MAX, -1),       // init value
			MinPairOp()
		);
		
		if (leave == -1 || theta_min >= DBL_MAX) {
			status = SolveStatus::Unbounded;
			break;
		}

		// update B_ids, factorise B
		cuda_memcpy(gpu.B_ids + leave, &enter, 1, cudaMemcpyHostToDevice);
		assemble_basis<<<grid_dim, block_dim>>>(gpu.A, gpu.B, gpu.B_ids, m);
		cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.B, m, gpu.work, gpu.ipiv, gpu.info));

		// update xB
		if (iteration % SOLUTION_PERIOD) {
			update_xB<<<BLOCK_DIM_1D, gpu.grid_dim_1D>>>(gpu.xB, gpu.d, leave, theta_min, m);
		} else {
			cuda_memcpy(gpu.xB, gpu.b, m, cudaMemcpyDeviceToDevice);
			cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.xB, m, gpu.info));
		}
	}
	std::cout << "Iterations: " << std::min(iteration, MAX_ITERS) << '\n';

	double z;
	if (status != SolveStatus::Unbounded) {
		gather_cost<<<BLOCK_DIM_1D, gpu.grid_dim_1D>>>(gpu.y, gpu.c, gpu.B_ids, m);
		cublasDdot(gpu.handle_cublas, m, gpu.y, 1, gpu.xB, 1, &z);
	}

	return std::make_pair(z, status);
}

/* ===================== SOLVER ===================== */

std::pair<DeviceResources, std::vector<int>> initialise_device_resources(
	int m, int n, int nnz,
	const std::vector<double>& h_values,
	const std::vector<int>& h_indices,
	const std::vector<int>& h_starts,
	const std::vector<double>& b,
	const std::vector<double>& c,
	int identity_start, int artificial_start, int artificial_end
) {
	std::vector<int> B_ids(m);
	for (int i = 0; i < m; ++i)
	  B_ids[i] = identity_start + i;

	std::vector<double> cost_phase_one(artificial_end);
	for (int i = 0; i < artificial_end; ++i)
		cost_phase_one[i] = i < artificial_start ? 0.0 : -1.0;

	DeviceResources gpu(m, artificial_end, nnz);
	gpu.A_csr.upload(h_values, h_indices, h_starts);
	gpu.csc2csr(gpu.A_csc, gpu.A_csr);
	cuda_memcpy(gpu.b, b.data(), m, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.c, cost_phase_one.data(), artificial_end, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.B_ids, B_ids.data(), m, cudaMemcpyHostToDevice);

	return std::make_pair(gpu, B_ids);
}

void transition(
	const DeviceResources& gpu,
	const std::vector<double>& b,
	const std::vector<double>& c,
	std::vector<int>& B_ids,
	int m, int identity_start, int artificial_start
) {
	cuda_memcpy(B_ids.data(), gpu.B_ids, m, cudaMemcpyDeviceToHost);

	for (int i = 0; i < m; ++i) {
		int ix = B_ids[i];
		if (ix < artificial_start) continue;

		// build a unit vector to extract a row of B_inv
		int row = ix - identity_start;
		cudaMemset(gpu.y, 0, m * sizeof(double));
		cuda_memcpy(gpu.y + row, &ONE, 1, cudaMemcpyHostToDevice);
		// solve for y: B^T * y^T = e_row
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_T, m, 1, gpu.B, m, gpu.ipiv, gpu.y, m, gpu.info));
		// compute rc = A^T * y^T
		cublasCheckError(cublasDgemv(gpu.handle_cublas, CUBLAS_OP_T, m, artificial_start, &ONE, gpu.A, m, gpu.y, 1, &ZERO, gpu.rc, 1));
		// mask basic variables
		mask_basis<<<BLOCK_DIM_1D, gpu.grid_dim_1D>>>(gpu.rc, gpu.B_ids, 0.0, m);
		// pick the first element with a non-zero coefficient
		thrust::device_ptr<double> first(gpu.rc);
		auto last = first + artificial_start;
		auto iterator = thrust::find_if(first, last, IsNonZero());
		int enter = iterator == last ? -1 : (iterator - first);

		if (enter == -1) {
			std::cerr << "Var " << ix << ": " << "constraint " << row << " is redundant\n";
			std::exit(EXIT_FAILURE);
		}

		// update B_ids and factorise B
		cuda_memcpy(gpu.B_ids + i, &enter, 1, cudaMemcpyHostToDevice);
		assemble_basis<<<grid_dim, block_dim>>>(gpu.A, gpu.B, gpu.B_ids, m);
		cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.B, m, gpu.work, gpu.ipiv, gpu.info));
	}

	// update xB and the cost
	cuda_memcpy(gpu.xB, b.data(), m, cudaMemcpyHostToDevice);
	cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.xB, m, gpu.info));
	cuda_memcpy(gpu.c, c.data(), artificial_start, cudaMemcpyHostToDevice);
}

std::pair<double, SolveStatus> solve(
	int m, int n, int nnz,
	const std::vector<double>& h_values,
	const std::vector<int>& h_indices,
	const std::vector<int>& h_starts,
	const std::vector<double>& b,
	const std::vector<double>& c,
	int identity_start, int artificial_start, int artificial_end
) {
	auto [gpu, B_ids] = initialise_device_resources(
		m, n, nnz, h_values, h_indices, h_starts, b, c, 
		identity_start, artificial_start, artificial_end
	);

	gpu.assemble_basis();
	cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.B, m, gpu.work, gpu.ipiv, gpu.info));
	cuda_memcpy(gpu.xB, b.data(), m, cudaMemcpyHostToDevice);
	cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.xB, m, gpu.info));

	// =============== Phase I and Phase II ===============
	
	auto [sum_artificials, status_phase_one] = core(gpu, m, artificial_end);

	if (status_phase_one != SolveStatus::OptimumFound || fabs(sum_artificials) > OPTIMALITY_TOL) {
		std::cerr << "Phase I failed, the optimum is " << sum_artificials << '\n';
		std::exit(EXIT_FAILURE);
	}

	transition(gpu, b, c, B_ids, m, identity_start, artificial_start);

	return core(gpu, m, artificial_start);
}

int main() {
	std::ios_base::sync_with_stdio(false);

	int m, n, nnz, n_surplus, n_slack;
	double offset;

	std::cin >> m >> n >> nnz >> n_surplus >> n_slack >> offset;

	int identity_start = n + n_surplus;
	int artificial_start = identity_start + n_slack;
	int artificial_end = identity_start + m;
	
	std::vector<double> h_values(nnz);
	std::vector<int> h_indices(nnz);
	std::vector<int> h_starts(artificial_end + 1);

	for (int i = 0; i < nnz; ++i)                std::cin >> h_values[i];
	for (int i = 0; i < nnz; ++i)                std::cin >> h_indices[i];
	for (int i = 0; i < artificial_end + 1; ++i) std::cin >> h_starts[i];

	std::vector<double> b(m);
	std::vector<double> c(artificial_start);

	for (int i = 0; i < m; ++i)                std::cin >> b[i];
	for (int i = 0; i < artificial_start; ++i) std::cin >> c[i];

	auto [z, status] = solve(
		m, n, nnz, h_values, h_indices, h_starts, b, c, 
		identity_start, artificial_start, artificial_end
	);

	switch (status) {
		case SolveStatus::OptimumFound:
			std::cout << std::scientific << std::uppercase << std::setprecision(10)
			          << "Optimum found: " << (z + offset) << '\n';
			return EXIT_SUCCESS;

		case SolveStatus::Unbounded:
			std::cout << "Problem unbounded.\n";
			return EXIT_SUCCESS;

		case SolveStatus::MaxIter:
			std::cerr << "Reached " << MAX_ITERS << " iterations.\n";
			return EXIT_FAILURE;
	}
}
