#include <cfloat>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <Eigen/Dense>

#include <iostream>
#include <iomanip>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

#include <utility>

#define MAX_ITERS 5000

constexpr double OPTIMALITY_TOL = 1e-10;
constexpr double PIVOT_TOL      = 1e-5;
constexpr double ROW_TOL        = 1e-12;

constexpr double ONE = 1.0;
constexpr double MINUS_ONE = -1.0;
constexpr double ZERO = 0.0;

constexpr int BLOCK_DIM_1D = 256;
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

enum class SolveStatus { MaxIter, OptimumFound, Unbounded };

/* ===================== UTILITIES ===================== */

void equilibrate(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& c) {
	int m = A.rows();
	int artificial_end = A.cols();

	for (int i = 0; i < m; ++i) {
		double max_val = std::abs(b(i));
		for (int j = 0; j < artificial_end - m; ++j)
			max_val = std::max(max_val, std::abs(A(i, j)));

		double scale = 1.0 / std::max(1.0, max_val);
		A.row(i) *= scale;
		b(i) *= scale;
	}

	for (int j = 0; j < artificial_end - m; j++) {
		double max_val = 0.0;
		for (int i = 0; i < m; i++)
			max_val = std::max(max_val, std::abs(A(i, j)));

		double scale = 1.0 / max_val;
		A.col(j) *= scale;
		c(j) *= scale;
	}
}

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

constexpr int div_up(int n, int block_dim) { return (n + block_dim - 1) / block_dim; }

/* ===================== DEVICE RESOURCES ===================== */

template<typename T>
struct PtrAlloc { T*& ptr; int size; };

struct DeviceResources {
	cusolverDnHandle_t handle;
	cublasHandle_t handle_cublas;

	double *A, *b, *c, *xB;
	double *d;   // direction
	double *rc;  // reduced cost
	int m, *B_ids;

	double *B, *work, *y;
	int lwork, *ipiv, *info;

	std::vector<PtrAlloc<double>> double_allocs;
	std::vector<PtrAlloc<int>> int_allocs;

	DeviceResources(int m, int n) : m(m),
		double_allocs {{A, m * n}, {b, m}, {c, n}, {xB, m}, {d, m}, {rc, n}, {B, m * m}, {y, m}},
		int_allocs {{ipiv, m}, {B_ids, m}, {info, 1}} {

		cusolverCheckError(cusolverDnCreate(&handle));
		cublasCheckError(cublasCreate(&handle_cublas));

		for (auto &[ptr, size] : double_allocs) cuda_malloc(ptr, size);
		for (auto &[ptr, size] : int_allocs)    cuda_malloc(ptr, size);

		cusolverCheckError(cusolverDnDgetrf_bufferSize(handle, m, m, B, m, &lwork));
		cuda_malloc(work, lwork);
	}

	~DeviceResources() {
		for (auto &[ptr,_] : double_allocs) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)    cudaFree(ptr);
		cusolverDnDestroy(handle);
		cublasDestroy(handle_cublas);
	}
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
	DeviceResources& gpu, int m, int n,
	const dim3 block_dim, const dim3 grid_dim_1D, const dim3 grid_dim
) {

	auto status = SolveStatus::MaxIter;
	int iteration;

	for (iteration = 1; iteration <= MAX_ITERS; ++iteration) {
		// solve for y: B^T * y^T = cB^T
		gather_cost<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.y, gpu.c, gpu.B_ids, m);
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_T, m, 1, gpu.B, m, gpu.ipiv, gpu.y, m, gpu.info));

		// compute rc^T = -A^T * y^T + c^T
		cuda_memcpy(gpu.rc, gpu.c, n, cudaMemcpyDeviceToDevice);
		cublasCheckError(cublasDgemv(gpu.handle_cublas, CUBLAS_OP_T, m, n, &MINUS_ONE, gpu.A, m, gpu.y, 1, &ONE, gpu.rc, 1));

		// select entering variable
		mask_basis<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.rc, gpu.B_ids, -1.0e20, m);
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

		// update xB and B_ids, factorise B
		cuda_memcpy(gpu.B_ids + leave, &enter, 1, cudaMemcpyHostToDevice);
		assemble_basis<<<grid_dim, block_dim>>>(gpu.A, gpu.B, gpu.B_ids, m);
		cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.B, m, gpu.work, gpu.ipiv, gpu.info));
		cuda_memcpy(gpu.xB, gpu.b, m, cudaMemcpyDeviceToDevice);
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.xB, m, gpu.info));
	}
	std::cout << "Iterations: " << std::max(iteration, MAX_ITERS) << '\n';

	double z;
	if (status == SolveStatus::OptimumFound) {
		gather_cost<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.y, gpu.c, gpu.B_ids, m);
		cublasDdot(gpu.handle_cublas, m, gpu.y, 1, gpu.xB, 1, &z);
	}

	return std::make_pair(z, status);
}

/* ===================== SOLVER ===================== */

std::pair<double, SolveStatus> solve(
	const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
	int m, int n, int identity_start, int artificial_start, int artificial_end
) {

	std::vector<int> B_ids(m);
	for (int i = 0; i < m; ++i)
	  B_ids[i] = identity_start + i;

	std::vector<double> cost_phase_one(artificial_end);
	for (int i = 0; i < artificial_end; ++i)
		cost_phase_one[i] = i < artificial_start ? 0.0 : -1.0;

	DeviceResources gpu(m, artificial_end);
	cuda_memcpy(gpu.A, A.data(), m * artificial_end, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.b, b.data(), m, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.c, cost_phase_one.data(), artificial_end, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.B_ids, B_ids.data(), m, cudaMemcpyHostToDevice);

	const dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
	const dim3 grid_dim(div_up(m, block_dim.x), div_up(m, block_dim.y));
	const dim3 grid_dim_1D(div_up(m, BLOCK_DIM_1D));

	assemble_basis<<<grid_dim, block_dim>>>(gpu.A, gpu.B, gpu.B_ids, m);
	cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.B, m, gpu.work, gpu.ipiv, gpu.info));
	cuda_memcpy(gpu.xB, b.data(), m, cudaMemcpyHostToDevice);
	cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.B, m, gpu.ipiv, gpu.xB, m, gpu.info));

	// =============== Phase I and Phase II ===============
	
	auto [sum_artificials, status_phase_one] = core(gpu, m, artificial_end, block_dim, grid_dim_1D, grid_dim);
	if (status_phase_one != SolveStatus::OptimumFound || fabs(sum_artificials) > OPTIMALITY_TOL) {
		std::cerr << "!! Phase I failed, the optimum is " << sum_artificials << '\n';
		std::exit(EXIT_FAILURE);
	}

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
		mask_basis<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.rc, gpu.B_ids, 0.0, m);
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

	return core(gpu, m, artificial_start, block_dim, grid_dim_1D, grid_dim);
}

int main() {
	int m, n, n_surplus, n_slack;
	double offset;

	std::cin >> m >> n >> n_surplus >> n_slack >> offset;

	int identity_start = n + n_surplus;
	int artificial_start = identity_start + n_slack;
	int artificial_end = identity_start + m;
	
	Eigen::MatrixXd A(m, artificial_end);
	Eigen::VectorXd b(m), c(artificial_start);

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < artificial_end; ++j)
			std::cin >> A(i, j);

	for (int i = 0; i < m; ++i)
		std::cin >> b(i);

	for (int i = 0; i < artificial_start; ++i)
		std::cin >> c(i);

	equilibrate(A, b, c);
	auto [z, status] = solve(A, b, c, m, n, identity_start, artificial_start, artificial_end);

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