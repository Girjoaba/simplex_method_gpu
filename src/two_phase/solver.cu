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
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

#include <utility>

#define MAX_ITERS 5000

constexpr double OPTIMALITY_TOL = 1e-6;
constexpr double PIVOT_TOL      = 1e-5;

constexpr double ONE = 1.0;
constexpr double MINUS_ONE = -1.0;

constexpr int BLOCK_DIM_1D = 256;
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

enum class SolveStatus { MaxIter, OptimumFound, Unbounded };

/* ===================== UTILITIES ===================== */

void equilibrate(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
	int m = A.rows();
	int n = A.cols();

	for (int i = 0; i < m; ++i) {
		double max_val = 1.0;
		for (int j = 0; j < n; ++j)
			max_val = std::max(max_val, std::abs(A(i, j)));

		if (max_val > 1.0) {
			double scale = 1.0 / max_val;
			A.row(i) *= scale;
			b(i) *= scale;
		}
	}
}

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}
#define cusolverCheckError(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
	if (code != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "cuSOLVER Error: %d %s %d\n", code, file, line);
		exit(code);
	}
}
#define cublasCheckError(ans) { cublasAssert((ans), __FILE__, __LINE__); }
void cublasAssert(cublasStatus_t code, const char *file, int line) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cuBLAS Error: %d %s %d\n", code, file, line);
		exit(code);
	}
}
#define cuda_malloc(d_ptr, n) { cuda_malloc_impl(&d_ptr, n, #d_ptr); }
template <typename T>
void cuda_malloc_impl(T** d_ptr, int n, const char* name) {
	cudaError_t code = cudaMalloc((void**)d_ptr, n * sizeof(T));
	if (code != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for %s: %s\n", name, cudaGetErrorString(code));
		exit(code);
	}
}
#define cuda_memcpy(dst, src, n, kind) cuda_memcpy_impl(dst, src, n, kind, #dst)
template <typename T>
void cuda_memcpy_impl(T* dst, const T* src, int size, cudaMemcpyKind kind, const char* name) {
	cudaError_t code = cudaMemcpy((void*)dst, (void*)src, size * sizeof(T), kind);
	if (code != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for %s: %s\n", name, cudaGetErrorString(code));
		exit(code);
	}
}

__device__ __forceinline__
constexpr int div_up(int n, int block_dim) { return (n + block_dim - 1) / block_dim; }

/* ===================== DEVICE RESOURCES ===================== */

template<typename T>
struct PtrAlloc { T*& ptr; int size; };

struct DeviceResources {
	cusolverDnHandle_t handle;
	cublasHandle_t handle_cublas;

	double *d_A, *d_xB, *d_c;
	double *d_d;   // direction
	double *d_rc;  // reduced cost
	int m, *d_B_ids;

	double *d_B, *d_work, *d_y;
	int lwork, *d_ipiv, *d_info;

	std::vector<PtrAlloc<double>> double_allocs;
	std::vector<PtrAlloc<int>> int_allocs;

	DeviceResources(int m, int n) : m(m),
		double_allocs {{d_A, m * n}, {d_B, m * m}, {d_y, m}, {d_xB, m}, {d_d, m}, {d_c, n}, {d_rc, n}},
		int_allocs {{d_ipiv, m}, {d_B_ids, m}, {d_info, 1}} {

		cusolverCheckError(cusolverDnCreate(&handle));
		cublasCheckError(cublasCreate(&handle_cublas));

		for (auto &[ptr, size] : double_allocs) cuda_malloc(ptr, size);
		for (auto &[ptr, size] : int_allocs)    cuda_malloc(ptr, size);

		cusolverCheckError(cusolverDnDgetrf_bufferSize(handle, m, m, d_B, m, &lwork));
		cuda_malloc(d_work, lwork);
	}

	~DeviceResources() {
		for (auto &[ptr,_] : double_allocs) cudaFree(ptr);
		for (auto &[ptr,_] : int_allocs)    cudaFree(ptr);
		cusolverDnDestroy(handle);
		cublasDestroy(handle_cublas);
	}
};

/* ===================== KERNELS ===================== */

// how much do restrict and const help?
__global__ void assemble_basis(const double* __restrict__ A, double* __restrict__ B, const int* __restrict__ B_ids, int m) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < m && j < m)
		B[i + j * m] = A[i + B_ids[j] * m];
}

__global__ void mask_basis(double *rc, const int* B_ids, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
		rc[B_ids[i]] = -1.0e20;
}

// learn the maths; is it better to use LU?
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

/* ===================== CORE LOGIC ===================== */

SolveStatus core(
	DeviceResources& gpu,
	int m, int n,
	const dim3 block_dim, const dim3 grid_dim_1D, const dim3 grid_dim
) {

	auto status = SolveStatus::MaxIter;
	int iteration;

	for (iteration = 1; iteration <= MAX_ITERS; ++iteration) {
		// factorise B
		assemble_basis<<<grid_dim, block_dim>>>(gpu.d_A, gpu.d_B, gpu.d_B_ids, m);
		cudaDeviceSynchronize();
		cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.d_B, m, gpu.d_work, gpu.d_ipiv, gpu.d_info));

		// solve for y: B^T * y^T = cB^T
		gather_cost<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.d_y, gpu.d_c, gpu.d_B_ids, m);
		cudaDeviceSynchronize();
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_T, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_y, m, gpu.d_info));

		// compute rc^T = -A^T * y^T + c^T
		cuda_memcpy(gpu.d_rc, gpu.d_c, n, cudaMemcpyDeviceToDevice);
		cublasCheckError(cublasDgemv(gpu.handle_cublas, CUBLAS_OP_T, m, n, &MINUS_ONE, gpu.d_A, m, gpu.d_y, 1, &ONE, gpu.d_rc, 1));

		// select entering variable
		mask_basis<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.d_rc, gpu.d_B_ids, m);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_rc(gpu.d_rc);
		auto iterator = thrust::max_element(thrust_rc, thrust_rc + n);
		int enter = iterator - thrust_rc;
		
		if (*iterator <= OPTIMALITY_TOL) {
			status = SolveStatus::OptimumFound;
			break;
		}  
		
		// solve for d: B * d = A_enter
		cuda_memcpy(gpu.d_d, gpu.d_A + (enter * m), m, cudaMemcpyDeviceToDevice);
		cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_d, m, gpu.d_info));
		
		// ratio test
		auto [theta_min, leave] = thrust::transform_reduce(
			thrust::device,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(m),
			RatioTestUnaryOp{gpu.d_xB, gpu.d_d},
			thrust::make_pair(DBL_MAX, -1),                 // init value
			MinPairOp()
		);
		
		if (leave == -1 || theta_min >= DBL_MAX) {
			status = SolveStatus::Unbounded;
			break;
		}
		
		update_xB<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.d_xB, gpu.d_d, leave, theta_min, m);
		cudaDeviceSynchronize();
		cuda_memcpy(gpu.d_B_ids + leave, &enter, 1, cudaMemcpyHostToDevice);
	}
	std::cout << "# Iterations " << iteration << '\n';

	return status;
}

/* ===================== SOLVER ===================== */

std::pair<double, SolveStatus> solve(
	const Eigen::MatrixXd& A,
	const Eigen::VectorXd& b,
	const Eigen::VectorXd& c,
	int m, int n,
	int identity_start, int artificial_start, int artificial_end
) {

	std::vector<int> B_ids(m);
	for (int i = 0; i < m; ++i)
	  B_ids[i] = identity_start + i;

	std::vector<double> cost_phase_one(artificial_end);
	for (int i = 0; i < artificial_end; ++i)
		cost_phase_one[i] = i < artificial_start ? 0.0 : -1.0;

	DeviceResources gpu(m, artificial_end);
	cuda_memcpy(gpu.d_A, A.data(), m * artificial_end, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.d_c, cost_phase_one.data(), artificial_end, cudaMemcpyHostToDevice);
	cuda_memcpy(gpu.d_B_ids, B_ids.data(), m, cudaMemcpyHostToDevice);

	const dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
	const dim3 grid_dim(div_up(m, block_dim.x), div_up(m, block_dim.y));
	const dim3 grid_dim_1D(div_up(m, BLOCK_DIM_1D));

	assemble_basis<<<grid_dim, block_dim>>>(gpu.d_A, gpu.d_B, gpu.d_B_ids, m);
	cudaDeviceSynchronize();
	cusolverCheckError(cusolverDnDgetrf(gpu.handle, m, m, gpu.d_B, m, gpu.d_work, gpu.d_ipiv, gpu.d_info));
	cuda_memcpy(gpu.d_xB, b.data(), m, cudaMemcpyHostToDevice);
	cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_xB, m, gpu.d_info));

	// =============== Phase I ===============
	
	SolveStatus _ = core(gpu, m, artificial_end, block_dim, grid_dim_1D, grid_dim);

	// if the basis contains artificials, pivot them out; for now, just exit
	cuda_memcpy(B_ids.data(), gpu.d_B_ids, m, cudaMemcpyDeviceToHost);
	bool has_artificials = false;
	for (int ix : B_ids) {
		if (ix >= artificial_start) {
			std::cerr << "!! Index " << ix << " >= " << artificial_start << '\n';
			has_artificials = true;
		}	
	}
	if (has_artificials) std::exit(EXIT_FAILURE);

	// =============== Phase II ===============

	cuda_memcpy(gpu.d_c, c.data(), artificial_start, cudaMemcpyHostToDevice);

	SolveStatus status = core(gpu, m, artificial_start, block_dim, grid_dim_1D, grid_dim);

	double z;
	if (status == SolveStatus::OptimumFound) {
		gather_cost<<<BLOCK_DIM_1D, grid_dim_1D>>>(gpu.d_y, gpu.d_c, gpu.d_B_ids, m);
		cudaDeviceSynchronize();
		cublasDdot(gpu.handle_cublas, m, gpu.d_y, 1, gpu.d_xB, 1, &z);
	}

	return std::make_pair(z, status);
}

int main() {
	int m, n, n_surplus, n_slack;
	// double optimum, offset;

	std::cin >> m >> n >> n_surplus >> n_slack; //  >> optimum >> offset;

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

	equilibrate(A, b);
	auto [z, status] = solve(A, b, c, m, n, identity_start, artificial_start, artificial_end);

	std::cout << std::scientific << std::uppercase << std::setprecision(5)
	          << "Optimum found: " << z << '\n';
	          // << "Optimum known: " << optimum << '\n';
}
