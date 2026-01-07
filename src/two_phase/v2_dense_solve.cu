// TODO:

// Replace only one element in cB instead.
// Try to remove mask_basis similar to transform_reduce.

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

constexpr int max_iterations     = 1000;

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
		std::cerr << "cuSOLVER Error: " << code << ' '
		          << file << ' ' << line << '\n';
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
		artificial_end(artificial_end),
		values(nnz),
		indices(nnz),
		starts(artificial_end + 1),
		b(m),
		c(artificial_start)
	{}
};

/* ===================== COMPRESSED SPARSE FORMAT ===================== */

// Supports both CSC and CSR, distinguished via the is_csr flag
struct CS {
	int m, n, nnz = 0;
	// The capacity needed to hold the basis since nnz changes each iteration
	int capacity;
	bool is_csr;

	double *values;
	int    *indices;
	int    *starts;

	CS(int m, int n, int capacity, bool is_csr = false) :
		m(m), n(n),
		capacity(capacity),
		is_csr(is_csr)
	{
		cuda_malloc(values,  capacity);
		cuda_malloc(indices, capacity);
		cuda_malloc(starts,  (is_csr ? m : n) + 1);
	}

	void upload(
		const std::vector<double>& h_values,
		const std::vector<int>&    h_indices,
		const std::vector<int>&    h_starts
	) {
		// Ideally, should also check for equal sizes and nnz <= capacity
		nnz = h_values.size();
		cuda_memcpy(values,  h_values.data(),  nnz,                  cudaMemcpyHostToDevice);
		cuda_memcpy(indices, h_indices.data(), nnz,                  cudaMemcpyHostToDevice);
		cuda_memcpy(starts,  h_starts.data(),  (is_csr ? m : n) + 1, cudaMemcpyHostToDevice);
	}

	// Increases capacity and deletes all data by default !!
	void increase_capacity(int new_capacity, bool keep_data = false) {
		if (new_capacity <= capacity)
			return;

		double *new_values;
		int    *new_indices;

		cuda_malloc(new_values,  new_capacity);
		cuda_malloc(new_indices, new_capacity);

		if (keep_data) {
			cuda_memcpy(new_values,  values,  nnz, cudaMemcpyDeviceToDevice);
			cuda_memcpy(new_indices, indices, nnz, cudaMemcpyDeviceToDevice);
		}

		cudaFree(values);
		cudaFree(indices);

		values = new_values;
		indices = new_indices;

		capacity = new_capacity;
	}

	~CS() {
		cudaFree(values);
		cudaFree(indices);
		cudaFree(starts);
	}
};

/* ===================== BASIS ===================== */

// Holds the basis both in CSC and CSR in addition to
// extra members required to assemble the basis efficiently.
struct Basis {
	CS csc, csr;
	// Non-zero count per column (for B_csc)
	int *column_nnz;
	// Maps each non-zero value to its column (for B_csc)
	int *thread_to_column_map;

	Basis(int m) :
		csc(m, m, m),
		csr(m, m, m, true)
	{
		cuda_malloc(column_nnz, m);
		cuda_malloc(thread_to_column_map, m);
	}

	// Ensure that the new basis will fit into memory
	void ensure_capacity(int new_nnz) {
		int capacity = csc.capacity;

		if (new_nnz <= capacity)
			return;

		do { capacity *= 2; }
		while (new_nnz > capacity);

		csc.increase_capacity(capacity);
		csr.increase_capacity(capacity);

		cudaFree(thread_to_column_map);
		cuda_malloc(thread_to_column_map, capacity);
	}

	~Basis() {
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
__global__ void compute_B_column_nnz(int m, int *B_ids, int *B_col_nnz, int *A_starts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		B_col_nnz[i] = A_starts[B_ids[i] + 1] - A_starts[B_ids[i]];
}

// Gather dense B from A_csc
__global__ void extract_B(
		double* B,
    double* A_values, int* A_indices, int* A_starts,
    int* B_starts,
    int* B_ids, int* thread_to_column_map, int nnz, int m
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nnz) {
		// -1 since thrust::upper_bound returns the index of the next column
		int col_in_B = thread_to_column_map[i] - 1; 
		int col_in_A = B_ids[col_in_B];
		int col_offset = i - B_starts[col_in_B];
		int src_idx = A_starts[col_in_A] + col_offset;

		B[m * col_in_B + A_indices[src_idx]] = A_values[src_idx];
	}
}

// Scatters sparse A_p into dense (d)irection
__global__ void scatter_column(double* d, double* A_values, int* A_indices, int A_starts_enter, int col_nnz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < col_nnz) {
		int row = A_indices[A_starts_enter + i];
		d[row] = A_values[A_starts_enter + i];
	}
}

/* ===================== DEVICE RESOURCES ===================== */

// Holds a pointer and the number of elements in the allocated memory
template<typename T>
struct PtrAlloc {
	T*& ptr;
	int size;
};

// Holds all device-related resources
struct DeviceResources {
	// Number of rows
	int m;

	// Block and grid dimensions for CUDA kernels
	dim3 block_dim_2D, grid_dim_1D, grid_dim_2D;

	// Library handles
	cublasHandle_t     handle_cublas;
	cusparseHandle_t   handle_cusparse;
	cusolverDnHandle_t handle_cusolver;

	// Sparse matrix structures
	CS A_csc;
	Basis basis;

	// cuSPARSE data
	cusparseSpMatDescr_t A_desc;
	cusparseDnVecDescr_t y_desc;
	cusparseDnVecDescr_t rc_desc;
	size_t bufferSize = 0;
	void* externalBuffer;

	// Dense vectors
	double *c, *b;
	double *cB, *xB;
	double *rhs;  // for solving Ay = rhs with cuDSS
	double *d;    // direction
	double *rc;   // reduced cost
	int *B_ids;

	double *B, *work, *y;
	int lwork, *ipiv, *info;

	// Pointers grouped by type for easy iteration
	std::vector<PtrAlloc<double>> double_allocs;
	std::vector<PtrAlloc<int>>       int_allocs;

	DeviceResources(int m, int art_end, int nnz) :
		m(m),
		block_dim_2D(dim3(block_dim_x, block_dim_y)),
		grid_dim_1D(dim3(div_up(m, block_dim_1d))),
		grid_dim_2D(dim3(div_up(m, block_dim_x), div_up(m, block_dim_y))),
		A_csc(m, art_end, nnz),
		basis(m),
		double_allocs {{c, art_end}, {b, m}, {xB, m}, {cB, m}, {rhs, m}, {d, m}, {rc, art_end}, {B, m * m}, {y, m}},
		int_allocs {{ipiv, m}, {B_ids, m}, {info, 1}} {

		for (auto &[ptr, size] : double_allocs) cuda_malloc(ptr, size);
		for (auto &[ptr, size] : int_allocs)    cuda_malloc(ptr, size);

		cusolverCheckError(cusolverDnCreate(&handle_cusolver));
		cublasCheckError(cublasCreate(&handle_cublas));
		cusparseCheckError(cusparseCreate(&handle_cusparse));

		cusolverCheckError(cusolverDnDgetrf_bufferSize(handle_cusolver, m, m, B, m, &lwork));
		cuda_malloc(work, lwork);
	}

	~DeviceResources() {
		for (auto &[ptr,_] : double_allocs) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)    cudaFree(ptr);

		// Destroy handles
		cusolverDnDestroy(handle_cusolver);
		cublasDestroy(handle_cublas);
		cusparseDestroy(handle_cusparse);

		// Destroy sparse descriptors and workspace
		cusparseDestroySpMat(A_desc);
		cusparseDestroyDnVec(y_desc);
		cusparseDestroyDnVec(rc_desc);
		cudaFree(externalBuffer);
	}

	void prepare_sparse_workspace(LP_problem& lp_problem) {

		cusparseCheckError(cusparseCreateCsc(
			&A_desc, m, lp_problem.artificial_end, lp_problem.nnz,
			A_csc.starts, A_csc.indices, A_csc.values,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
		));

		// Must be called after cuda_malloc(y, rc) so that valid pointers are passed.
		cusparseCheckError(cusparseCreateDnVec(&y_desc, m, y, CUDA_R_64F));
		cusparseCheckError(cusparseCreateDnVec(&rc_desc, lp_problem.artificial_end, rc, CUDA_R_64F));

		// SpMV workspace
		cusparseCheckError(cusparseSpMV_bufferSize(
			handle_cusparse,
			CUSPARSE_OPERATION_TRANSPOSE,
			&neg_one, A_desc, y_desc,
			&one, rc_desc,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			&bufferSize
		));
		cudaMalloc(&externalBuffer, bufferSize);

		// Analyse A to speed up all future multiplications
		cusparseCheckError(cusparseSpMV_preprocess(
			handle_cusparse,
			CUSPARSE_OPERATION_TRANSPOSE,
			&neg_one, A_desc, y_desc,
			&one, rc_desc,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			externalBuffer
		));
	}

	void setup(LP_problem& lp_problem) {

		int m = lp_problem.m;
		int id_start  = lp_problem.identity_start;
		int art_start = lp_problem.artificial_start;
		int art_end   = lp_problem.artificial_end;

		// Must be called before prepare_sparse_workspace!
		A_csc.upload(
			lp_problem.values,
			lp_problem.indices,
			lp_problem.starts
		);

		prepare_sparse_workspace(lp_problem);

		std::vector<int> h_B_ids(m);
		std::iota(h_B_ids.begin(), h_B_ids.end(), id_start);

		std::vector<double> cost_phase_one(art_end, 0.0);
		std::fill(
			cost_phase_one.begin() + art_start,
			cost_phase_one.end(),
			-1.0
		);

		cuda_memcpy(b,     lp_problem.b.data(),   m,       cudaMemcpyHostToDevice);
		cuda_memcpy(c,     cost_phase_one.data(), art_end, cudaMemcpyHostToDevice);
		cuda_memcpy(B_ids, h_B_ids.data(),        m,       cudaMemcpyHostToDevice);
	}

	// Assemble basis.csc from A_csc
	void assemble_basis() {

		// Compute number of non-zeros per column of B
		compute_B_column_nnz<<<grid_dim_1D, block_dim_1d>>>(m, B_ids, basis.column_nnz, A_csc.starts);

		// Compute column starts (prefix sum)
		thrust::exclusive_scan(
			thrust::device,
			basis.column_nnz,
			basis.column_nnz + m + 1,
			basis.csc.starts
		);

		// Total number of non-zeros in B
		int nnz;
		cuda_memcpy(&nnz, basis.csc.starts + m, 1, cudaMemcpyDeviceToHost);

		basis.ensure_capacity(nnz);

		// Map each non-zero to its column (upper_bound returns the next column)
		thrust::upper_bound(
			thrust::device,
			basis.csc.starts, basis.csc.starts + m + 1,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(0) + nnz,
			basis.thread_to_column_map
		);

		cudaMemset(B, 0, m * m * sizeof(double));

		// Gather the basis
		extract_B<<<div_up(nnz, block_dim_1d), block_dim_1d>>>(
			B,
			A_csc.values, A_csc.indices, A_csc.starts,
			basis.csc.starts,
			B_ids, basis.thread_to_column_map, nnz, m
		);

		basis.csc.nnz = nnz;
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

struct MinPairOp {
	__device__ thrust::pair<double, int> operator()
	(const thrust::pair<double, int>& a, const thrust::pair<double, int>& b) const {
		if (a.first < b.first) return a;
		if (a.first > b.first) return b;
		return (a.second < b.second) ? a : b;
	}
};

struct IsNonZero {
	__device__ bool operator()(double v) const { return fabs(v) > row_tol; }
};

/* ===================== CORE SIMPLEX LOGIC ===================== */

std::pair<double, SolveStatus> core(
	int m, int n_cols,
	DeviceResources& dev,
	LP_problem& lp_problem
) {

	auto status = SolveStatus::MaxIter;
	int iteration = 1;

	for (; iteration <= max_iterations; ++iteration) {

		gather_cost<<<dev.grid_dim_1D, block_dim_1d>>>(dev.y, dev.c, dev.B_ids, m);

		// Solve for y: B^T * y^T = cB^T
		cusolverCheckError(cusolverDnDgetrs(dev.handle_cusolver, CUBLAS_OP_T, m, 1, dev.B, m, dev.ipiv, dev.y, m, dev.info));

		// Compute rc^T = -A^T * y^T + c^T
		cuda_memcpy(dev.rc, dev.c, n_cols, cudaMemcpyDeviceToDevice);
		cusparseSpMV(
			dev.handle_cusparse,
			CUSPARSE_OPERATION_TRANSPOSE,
			&neg_one, dev.A_desc, dev.y_desc,
			&one, dev.rc_desc,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			dev.externalBuffer
		);

		// Select entering variable
		mask_basis<<<dev.grid_dim_1D, block_dim_1d>>>(dev.rc, dev.B_ids, -1.0, m);
		thrust::device_ptr<double> thrust_rc(dev.rc);
		auto iterator = thrust::max_element(thrust_rc, thrust_rc + n_cols);
		int enter = iterator - thrust_rc;
		
		if (*iterator <= optimality_tol) {
			status = SolveStatus::OptimumFound;
			break;
		}
		
		// Assemble A_enter from A_csc
		cudaMemset(dev.d, 0, m * sizeof(double));
		int starts_enter = lp_problem.starts[enter];
		int col_nnz = lp_problem.starts[enter + 1] - starts_enter;
		scatter_column<<<div_up(col_nnz, block_dim_1d), block_dim_1d>>>(
			dev.d, dev.A_csc.values, dev.A_csc.indices, starts_enter, col_nnz
		);

		// Solve for d: B * d = A_enter
		cusolverCheckError(cusolverDnDgetrs(dev.handle_cusolver, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.d, m, dev.info));
		
		// Ratio test
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

		// Update B_ids
		cuda_memcpy(dev.B_ids + leave, &enter, 1, cudaMemcpyHostToDevice);

		// Assemble dense B and factorise
		dev.assemble_basis();
		cusolverCheckError(cusolverDnDgetrf(dev.handle_cusolver, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));

		update_xB<<<dev.grid_dim_1D, block_dim_1d>>>(dev.xB, dev.d, leave, theta_min, m);
	}
	std::cout << "Iterations: " << std::min(iteration, max_iterations) << '\n';

	double z;
	if (status != SolveStatus::Unbounded) {
		gather_cost<<<dev.grid_dim_1D, block_dim_1d>>>(dev.y, dev.c, dev.B_ids, m);
		cublasDdot(dev.handle_cublas, m, dev.y, 1, dev.xB, 1, &z);
	}

	return std::make_pair(z, status);
}

/* ===================== PHASE TRANSITION ===================== */

void phase_transition(DeviceResources& dev, LP_problem& lp_problem) {

	int m = lp_problem.m;
	int id_start  = lp_problem.identity_start;
	int art_start = lp_problem.artificial_start;
	int art_end   = lp_problem.artificial_end;

	// Since we discard the columns corresponding to artificials,
	// we must shrink A_csc and rewrap the sparse descriptors.

	// Destroy old descriptors
	cusparseDestroySpMat(dev.A_desc);
	cusparseDestroyDnVec(dev.rc_desc);

	// Shrink A_csc
	dev.A_csc.nnz -= art_end - art_start;
	dev.A_csc.n = art_start;

	// Rewrap A_csc and rc
	cusparseCheckError(cusparseCreateCsc(
		&dev.A_desc, m, dev.A_csc.n, dev.A_csc.nnz,
		dev.A_csc.starts, dev.A_csc.indices, dev.A_csc.values,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
	));
	cusparseCheckError(cusparseCreateDnVec(
		&dev.rc_desc, art_start, dev.rc, CUDA_R_64F
	));

	// Reallocate SpMV workspace
	cudaFree(dev.externalBuffer);
	cusparseCheckError(cusparseSpMV_bufferSize(
		dev.handle_cusparse,
		CUSPARSE_OPERATION_TRANSPOSE,
		&neg_one, dev.A_desc, dev.y_desc,
		&one, dev.rc_desc,
		CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
		&dev.bufferSize
	));
	cudaMalloc(&dev.externalBuffer, dev.bufferSize);

	// Analyse A again
	cusparseCheckError(cusparseSpMV_preprocess(
		dev.handle_cusparse, 
		CUSPARSE_OPERATION_TRANSPOSE,
		&neg_one, dev.A_desc, dev.y_desc, &one, dev.rc_desc, 
		CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, 
		dev.externalBuffer
	));

	// Now we remove artificials from the constraint matrix

	std::vector<int> B_ids(m);
	cuda_memcpy(B_ids.data(), dev.B_ids, m, cudaMemcpyDeviceToHost);

	for (int i = 0; i < m; ++i) {
		
		int ix = B_ids[i];
		if (ix < art_start)
			continue;

		// Build a unit vector
		int row = ix - id_start;
		cudaMemset(dev.y, 0, m * sizeof(double));
		cuda_memcpy(dev.y + row, &one, 1, cudaMemcpyHostToDevice);

		// Solve for y: B^T * y^T = e_row
		// i.e., extract a row of the inverse
		cusolverCheckError(cusolverDnDgetrs(dev.handle_cusolver, CUBLAS_OP_T, m, 1, dev.B, m, dev.ipiv, dev.y, m, dev.info));

		// Compute rc = A^T * y^T
		cusparseSpMV(
			dev.handle_cusparse,
			CUSPARSE_OPERATION_TRANSPOSE,
			&one, dev.A_desc, dev.y_desc,
			&zero, dev.rc_desc,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			dev.externalBuffer
		);

		// Mask basic variables
		mask_basis<<<dev.grid_dim_1D, block_dim_1d>>>(dev.rc, dev.B_ids, 0.0, m);
		// Pick the first element with a non-zero coefficient
		thrust::device_ptr<double> first(dev.rc);
		auto last = first + art_start;
		auto iterator = thrust::find_if(first, last, IsNonZero());
		int enter = iterator == last ? -1 : (iterator - first);

		if (enter == -1) {
			std::cerr << "Var " << ix << ": " << "constraint " << row << " is redundant\n";
			std::exit(EXIT_FAILURE);
		}

		// Update B_ids
		cuda_memcpy(dev.B_ids + i, &enter, 1, cudaMemcpyHostToDevice);

		// Assemble B and factorise
		dev.assemble_basis();
		cusolverCheckError(cusolverDnDgetrf(dev.handle_cusolver, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));
	}

	// Update xB
	cuda_memcpy(dev.xB, dev.b, m, cudaMemcpyDeviceToDevice);
	cusolverCheckError(cusolverDnDgetrs(dev.handle_cusolver, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.xB, m, dev.info));

	// Update the cost
	cuda_memcpy(dev.c, lp_problem.c.data(), art_start, cudaMemcpyHostToDevice);
}

/* ===================== SOLVE ===================== */

std::pair<double, SolveStatus> solve(LP_problem& lp_problem) {

	int m = lp_problem.m;
	int art_end = lp_problem.artificial_end;

	DeviceResources dev(m, art_end, lp_problem.nnz);
	dev.setup(lp_problem);

	dev.assemble_basis();
	cusolverCheckError(cusolverDnDgetrf(dev.handle_cusolver, m, m, dev.B, m, dev.work, dev.ipiv, dev.info));

	cuda_memcpy(dev.xB, dev.b, m, cudaMemcpyDeviceToDevice);
	cusolverCheckError(cusolverDnDgetrs(dev.handle_cusolver, CUBLAS_OP_N, m, 1, dev.B, m, dev.ipiv, dev.xB, m, dev.info));

	// Phase I	
	auto [art_sum, status] = core(m, art_end, dev, lp_problem);

	// Check feasability (and correctness)
	if (status != SolveStatus::OptimumFound || fabs(art_sum) > optimality_tol) {
		std::cerr << "Phase I failed: "
		          << (status == SolveStatus::Unbounded ? "Unbounded" : "MaxIter")
		          << ", " << art_sum << '\n';
		std::exit(EXIT_FAILURE);
	}

	// Transition between the two phases
	phase_transition(dev, lp_problem);

	return core(m, lp_problem.artificial_start, dev, lp_problem);
}

/* ===================== MAIN ===================== */

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
			std::cerr << "Phase II failed: reached " << max_iterations << " iterations.\n";
			return EXIT_FAILURE;
	}
}
