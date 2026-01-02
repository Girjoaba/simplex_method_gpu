#include <algorithm>
#include <cfloat>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

#include <utility>

constexpr int xB_update_interval = 100;
constexpr int max_iterations     = 100000;

constexpr double optimality_tol = 1e-6;
constexpr double pivot_tol      = 1e-5;
constexpr double row_tol        = 1e-12;

constexpr int block_dim_1d = 256;
constexpr int block_dim_x  = 16;
constexpr int block_dim_y  = 16;

constexpr double one       =  1.0;
constexpr double neg_one   = -1.0;
constexpr double zero      =  0.0;

enum class SolveStatus {
	MaxIter,
	OptimumFound,
	Unbounded
};

// Integer division rounded up
constexpr int div_up(int n, int k) {
	return (n + k - 1) / k;
}

/* ===================== CUDA CHECKS ===================== */

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(code)
		          << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cusolverCheckError(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
	if (code != CUSOLVER_STATUS_SUCCESS) {
		std::cerr << "cuSOLVER Error: " << code
		          << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cublasCheckError(ans) { cublasAssert((ans), __FILE__, __LINE__); }
void cublasAssert(cublasStatus_t code, const char *file, int line) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS Error: " << code
		          << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cusparseCheckError(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
	if (code != CUSPARSE_STATUS_SUCCESS) {
		std::cerr << "cuSPARSE Error: " << cusparseGetErrorString(code)
		          << ' ' << file << ' ' << line << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cuda_malloc(d_ptr, n) { cuda_malloc_impl(&d_ptr, n, #d_ptr); }
template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t code = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (code != cudaSuccess) {
		std::cerr << "cudaMalloc failed for " << name << ": "
		          << cudaGetErrorString(code) << '\n';
		std::exit(EXIT_FAILURE);
	}
}
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t code = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (code != cudaSuccess) {
		std::cerr << "cudaMempcy failed for " << name << ": "
		          << cudaGetErrorString(code) << '\n';
		std::exit(EXIT_FAILURE);
	}
}

/* ===================== COMPRESSED SPARSE FORMAT ===================== */

// Supports both CSC and CSR, distinguished via the is_csr flag
struct CS {
	int m, n, nnz;
	bool is_csr;

	double *values;
	int *indices;
	int *starts;

	CS(int m, int n, int nnz, bool is_csr = false) :
		m(m), n(n), nnz(nnz), is_csr(is_csr) {
		cuda_malloc(values,  nnz);
		cuda_malloc(indices, nnz);
		cuda_malloc(starts,  (is_csr ? m : n) + 1);
	}

	void upload(
		const std::vector<double>& h_values,
		const std::vector<int>&    h_indices,
		const std::vector<int>&    h_starts
	) {
		cuda_memcpy(values,  h_values.data(),  nnz,                  cudaMemcpyHostToDevice);
		cuda_memcpy(indices, h_indices.data(), nnz,                  cudaMemcpyHostToDevice);
		cuda_memcpy(starts,  h_starts.data(),  (is_csr ? m : n) + 1, cudaMemcpyHostToDevice);
	}

	~CS() {
		cudaFree(values);
		cudaFree(indices);
		cudaFree(starts);
	}
};

// Additional members needed because the basis changes each iteration
struct CS_basis : CS {
	// Memory allocated for B
	int  capacity;
	// Non-zero count per column
	int *column_nnz;
	// Maps each non-zero value to its column
	int *thread_to_column_map;

	CS_basis(int m, int n, int nnz) :
		CS(m, n, nnz) {
		capacity = nnz;
		cuda_malloc(column_nnz, m);
		cuda_malloc(thread_to_column_map, nnz);
	}

	// Doubles capacity if current memory cannot hold new B
	void ensure_capacity(int new_nnz) {
		if (new_nnz <= capacity)
			return;

		do { capacity *= 2; }
		while (new_nnz > capacity);

		double *new_values;
		int *new_indices;
		int *new_map;

		cudaFree(values);
		cudaFree(indices);
		cudaFree(thread_to_column_map);

		cuda_malloc(new_values,  capacity);
		cuda_malloc(new_indices, capacity);
		cuda_malloc(new_map,     capacity);

		values = new_values;
		indices = new_indices;
		thread_to_column_map = new_map;
	}

	~CS_basis() {
		cudaFree(column_nnz);
		cudaFree(thread_to_column_map);
	}
};

/* ===================== KERNELS ===================== */

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

// Computes non-zero counts per column
__global__ void compute_B_column_nnz(int m, int *B_ids, int *B_column_nnz, int *A_starts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		B_column_nnz[i] = A_starts[B_ids[i] + 1] - A_starts[B_ids[i]];
}

// Gathers B from A_csc using B_ids and thread_to_column_map
__global__ void extract_B(
    double* A_values, int* A_indices, int* A_starts,
    double* B_values, int* B_indices, int* B_starts,
    int* B_ids, int* thread_to_column_map, int nnz
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nnz) {
		// -1 since thrust::upper_bound returns the index of the next column
		int col_in_B = thread_to_column_map[i] - 1; 
		int col_in_A = B_ids[col_in_B];
		int col_offset = i - B_starts[col_in_B];
		int src_idx = A_starts[col_in_A] + col_offset;

		B_values[i] = A_values[src_idx];
		B_indices[i] = A_indices[src_idx];
	}
}

/* ===================== DEVICE RESOURCES ===================== */

// Holds a pointer and the number of elements in the allocated memory
template<typename T>
struct PtrAlloc {
	T*& ptr;
	int size;
};

// Holds all device pointers and CUDA library handles
struct DeviceResources {
	// Number of rows
	int m;

	// Library handles
	cusolverDnHandle_t handle;
	cublasHandle_t     handle_cublas;
	cusparseHandle_t   handle_cusparse;

	// Sparse matrix structures
	CS A_csc, A_csr;
	CS_basis B_csc;

	// Block and grid dimensions for CUDA kernels
	dim3 block_dim_2D, grid_dim_1D, grid_dim_2D;

	// Dense vectors
	double *b, *c, *xB;
	double *d;   // direction
	double *rc;  // reduced cost
	int *B_ids;

	// cuSOLVER's variables for factorisation
	double *B, *work, *y;
	int lwork, *ipiv, *info;

	// Pointers grouped by type for easy iteration
	std::vector<PtrAlloc<double>> double_allocs;
	std::vector<PtrAlloc<int>>       int_allocs;

	DeviceResources(int m, int n, int nnz) :
		m(m),
		block_dim_2D(dim3(block_dim_x, block_dim_y)),
		grid_dim_1D(dim3(div_up(m, block_dim_1d))),
		grid_dim_2D(dim3(div_up(m, block_dim_x), div_up(m, block_dim_y))),
		A_csc(m, n, nnz), A_csr(m, n, nnz, true),
		B_csc(m, m, m),
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

	// CSR or CSC format depending on source's is_csr flag
	void convert_sparse(CS& src, CS& dst) {
		int algo_m = src.is_csr ? src.m : src.n;
    int algo_n = src.is_csr ? src.n : src.m;

		size_t bufferSize;
		void *buffer;

		cusparseCheckError(cusparseCsr2cscEx2_bufferSize(
			handle_cusparse,
			algo_m, algo_n, src.nnz,
			src.values, src.starts, src.indices,
			dst.values, dst.starts, dst.indices,
			CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
			&bufferSize
		));

		cudaMalloc(&buffer, bufferSize);

		cusparseCheckError(cusparseCsr2cscEx2(
			handle_cusparse,
			algo_m, algo_n, src.nnz,
			src.values, src.starts, src.indices,
			dst.values, dst.starts, dst.indices,
			CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
			buffer
		));

		cudaFree(buffer);
	}

	// Assemble sparse B from A_csc
	void assemble_basis() {

		// Compute number of non-zeros per column of B
		compute_B_column_nnz<<<block_dim_1d, grid_dim_1D>>>(m, B_ids, B_csc.column_nnz, A_csc.starts);

		// Compute column starts (prefix sum)
		thrust::exclusive_scan(
			thrust::device,
			B_csc.column_nnz,
			B_csc.column_nnz + m + 1,
			B_csc.starts
		);

		// Total number of non-zeros in B
		int nnz;
		cuda_memcpy(&nnz, B_csc.starts + m, 1, cudaMemcpyDeviceToHost);

		B_csc.ensure_capacity(nnz);

		// Map each non-zero to its column (upper_bound returns the next column)
		thrust::upper_bound(
			thrust::device,
			B_csc.starts, B_csc.starts + m + 1,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(0) + nnz,
			B_csc.thread_to_column_map
		);

		// Gather values and row indices from A into B
		extract_B<<<block_dim_1d, div_up(nnz, block_dim_1d)>>>(
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
		return d_i > pivot_tol ? thrust::make_pair(xB[i] / d_i, i) : thrust::make_pair(DBL_MAX, -1);
	}
};

// Reduction operator to find the minimum (value, index) pair
struct MinPairOp {
	__device__ thrust::pair<double, int> operator() (
		const thrust::pair<double, int>& a,
		const thrust::pair<double, int>& b
	) const {
		if (a.first < b.first) return a;
		if (a.first > b.first) return b;
		return (a.second < b.second) ? a : b;
	}
};

struct IsNonZero {
	__device__ bool operator()(double v) const {
		return fabs(v) > row_tol;
	}
};

/* ===================== CORE LOGIC ===================== */

std::pair<double, SolveStatus> core(
	DeviceResources& dev, int m, int n
) {

	auto status = SolveStatus::MaxIter;
	int iteration;

	for (iteration = 1; iteration <= max_iterations; ++iteration) {
		// solve for y: B^T * y^T = cB^T
		gather_cost<<<block_dim_1d, dev.grid_dim_1D>>>(dev.y, dev.c, dev.B_ids, m);
		// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_T, m, 1, dev.B, m, dev.ipiv, dev.y, m, dev.info));

		// compute rc^T = -A^T * y^T + c^T
		cuda_memcpy(dev.rc, dev.c, n, cudaMemcpyDeviceToDevice);
		// cublasCheckError(cublasDgemv(dev.handle_cublas, CUBLAS_OP_T, m, n, &neg_one, dev.A, m, dev.y, 1, &one, dev.rc, 1));

		// select entering variable
		mask_basis<<<block_dim_1d, dev.grid_dim_1D>>>(dev.rc, dev.B_ids, -1.0, m);
		thrust::device_ptr<double> thrust_rc(dev.rc);
		auto iterator = thrust::max_element(thrust_rc, thrust_rc + n);
		int enter = iterator - thrust_rc;
		
		if (*iterator <= optimality_tol) {
			status = SolveStatus::OptimumFound;
			break;
		}
		
		// solve for d: B * d = A_enter
		// cuda_memcpy(dev.d, dev.A + (enter * m), m, cudaMemcpyDeviceToDevice);
		// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.d, m, dev.info));
		
		// ratio test
		auto [theta_min, leave] = thrust::transform_reduce(
			thrust::device,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(m),
			RatioTestUnaryOp{dev.xB, dev.d},
			thrust::make_pair(DBL_MAX, -1),
			MinPairOp()
		);
		
		if (leave == -1 || theta_min >= DBL_MAX) {
			status = SolveStatus::Unbounded;
			break;
		}

		// update B_ids, factorise B
		cuda_memcpy(dev.B_ids + leave, &enter, 1, cudaMemcpyHostToDevice);
		dev.assemble_basis();
		// cusolverCheckError(cusolverDnDgetrf(dev.handle, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));

		// update xB
		if (iteration % xB_update_interval) {
			update_xB<<<block_dim_1d, dev.grid_dim_1D>>>(dev.xB, dev.d, leave, theta_min, m);
		} else {
			cuda_memcpy(dev.xB, dev.b, m, cudaMemcpyDeviceToDevice);
			// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.xB, m, dev.info));
		}
	}
	std::cout << "Iterations: " << std::min(iteration, max_iterations) << '\n';

	double z;
	if (status != SolveStatus::Unbounded) {
		gather_cost<<<block_dim_1d, dev.grid_dim_1D>>>(dev.y, dev.c, dev.B_ids, m);
		cublasDdot(dev.handle_cublas, m, dev.y, 1, dev.xB, 1, &z);
	}

	return std::make_pair(z, status);
}

/* ===================== LP PROBLEM ===================== */

struct LP_problem {
	int m, n, nnz;
	
	int identity_start;
	int artificial_start;
	int artificial_end;
	
	std::vector<double>	values;
	std::vector<int>    indices;
	std::vector<int>    starts;

	std::vector<double> b;
	std::vector<double> c;

	LP_problem(
		int m, int n, int nnz,
		int identity_start,
		int artificial_start,
		int artificial_end
	) :
		m(m), n(n), nnz(nnz),
		identity_start(identity_start),
		artificial_start(artificial_start),
		artificial_end(artificial_end)
	{
		values.resize(nnz);
		indices.resize(nnz);
		starts.resize(artificial_end + 1);

		b.resize(m);
		c.resize(artificial_start);
	}
};

DeviceResources get_device_resources(LP_problem lp_problem) {

	int m = lp_problem.m;
	int id_start  = lp_problem.identity_start;
	int art_start = lp_problem.artificial_start;
	int art_end   = lp_problem.artificial_end;

	std::vector<int> B_ids(m);
	std::iota(B_ids.begin(), B_ids.end(), id_start);

	std::vector<double> cost_phase_one(art_end, 0.0);
	std::fill(
		cost_phase_one.begin() + art_start,
		cost_phase_one.end(),
		-1.0
	);

	DeviceResources dev(m, art_end, lp_problem.nnz);

	dev.A_csc.upload(
		lp_problem.values,
		lp_problem.indices,
		lp_problem.starts
	);

	dev.convert_sparse(dev.A_csc, dev.A_csr);

	cuda_memcpy(dev.b,     lp_problem.b.data(),   m,       cudaMemcpyHostToDevice);
	cuda_memcpy(dev.c,     cost_phase_one.data(), art_end, cudaMemcpyHostToDevice);
	cuda_memcpy(dev.B_ids, B_ids.data(),          m,       cudaMemcpyHostToDevice);

	return dev;
}

void transition(DeviceResources& dev, LP_problem lp_problem) {

	int m = lp_problem.m;
	int id_start  = lp_problem.identity_start;
	int art_start = lp_problem.artificial_start;
	int art_end   = lp_problem.artificial_end;

	std::vector<int> B_ids(m);
	cuda_memcpy(B_ids.data(), dev.B_ids, m, cudaMemcpyDeviceToHost);

	for (int i = 0; i < m; ++i) {
		int ix = B_ids[i];
		if (ix < art_start)
			continue;

		int row = ix - id_start;

		// build a unit vector
		cudaMemset(dev.y, 0, m * sizeof(double));
		cuda_memcpy(dev.y + row, &one, 1, cudaMemcpyHostToDevice);

		// solve for y: B^T * y^T = e_row (i.e., extract a row of B_inv)
		// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_T, m, 1, dev.B, m, dev.ipiv, dev.y, m, dev.info));

		// compute rc = A^T * y^T
		// cublasCheckError(cublasDgemv(dev.handle_cublas, CUBLAS_OP_T, m, art_start, &one, dev.A, m, dev.y, 1, &zero, dev.rc, 1));

		// mask basic variables
		mask_basis<<<block_dim_1d, dev.grid_dim_1D>>>(dev.rc, dev.B_ids, 0.0, m);

		// pick the first element with a non-zero coefficient
		thrust::device_ptr<double> first(dev.rc);
		auto last = first + art_start;
		auto iterator = thrust::find_if(first, last, IsNonZero());
		int enter = iterator == last ? -1 : (iterator - first);

		if (enter == -1) {
			std::cerr << "Var " << ix << ": " << "constraint " << row << " is redundant\n";
			std::exit(EXIT_FAILURE);
		}

		// update B_ids and factorise B
		cuda_memcpy(dev.B_ids + i, &enter, 1, cudaMemcpyHostToDevice);
		dev.assemble_basis();
		// cusolverCheckError(cusolverDnDgetrf(dev.handle, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));
	}

	// update the cost and xB
	cuda_memcpy(dev.c, lp_problem.c.data(), art_start, cudaMemcpyHostToDevice);
	cuda_memcpy(dev.xB, lp_problem.b.data(), m, cudaMemcpyHostToDevice);
	// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.xB, m, dev.info));
}

std::pair<double, SolveStatus> solve(LP_problem lp_problem) {

	int m = lp_problem.m;
	int id_start  = lp_problem.identity_start;
	int art_start = lp_problem.artificial_start;
	int art_end   = lp_problem.artificial_end;

	DeviceResources dev = get_device_resources(lp_problem);

	dev.assemble_basis();

	cuda_memcpy(dev.xB, lp_problem.b.data(), m, cudaMemcpyHostToDevice);
	// cusolverCheckError(cusolverDnDgetrf(dev.handle, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));
	// cusolverCheckError(cusolverDnDgetrs(dev.handle, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.xB, m, dev.info));

	// Phase I	
	auto [art_sum, status] = core(dev, m, art_end);

	// Check feasability (and correctness)
	if (status != SolveStatus::OptimumFound || fabs(art_sum) > optimality_tol) {
		std::cerr << "Phase I failed: "
		          << (status == SolveStatus::Unbounded ? "Unbounded" : "MaxIter")
		          << ", " << art_sum << '\n';
	}

	// Transition between the phases
	transition(dev, lp_problem);

	// Phase II
	return core(dev, m, art_start);
}

int main() {
	std::ios_base::sync_with_stdio(false);

	int m, n, nnz;
	int n_surplus, n_slack;
	double offset;

	std::cin >> m >> n >> nnz
	         >> n_surplus >> n_slack
					 >> offset;

	int identity_start = n + n_surplus;
	int artificial_start = identity_start + n_slack;
	int artificial_end = identity_start + m;

	LP_problem lp_problem(
		m, n, nnz,
		identity_start,
		artificial_start,
		artificial_end
	);

	for (auto& value : lp_problem.values)   std::cin >> value;
	for (auto& index : lp_problem.indices)  std::cin >> index;
	for (auto& start : lp_problem.starts)   std::cin >> start;

	for (auto& value : lp_problem.b)        std::cin >> value;
	for (auto& value : lp_problem.c)        std::cin >> value;

	auto [obj_val, status] = solve(lp_problem);

	switch (status) {
		case SolveStatus::OptimumFound:
			std::cout << std::scientific << std::uppercase << std::setprecision(10)
			          << "Optimum found: " << (obj_val + offset) << '\n';
			return EXIT_SUCCESS;

		case SolveStatus::Unbounded:
			std::cout << "Problem unbounded.\n";
			return EXIT_SUCCESS;

		case SolveStatus::MaxIter:
			std::cerr << "Reached " << max_iterations << " iterations.\n";
			return EXIT_FAILURE;
	}
}
