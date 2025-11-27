#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cfloat>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

// --- THRUST INCLUDES ---
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> 
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>       // For max_element, min_element
#include <thrust/copy.h>          // For thrust::copy
#include <thrust/transform.h>     // For thrust::transform
#include <thrust/functional.h>

#include <Eigen/Dense>
#include <Eigen/LU>

// 

// ---------------------------
// Util. Functions
// ---------------------------

void equilibrate(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& c) {
    int m = A.rows();
    int n = A.cols();

    // Geometric Mean Row Scaling
    for (int i = 0; i < m; i++) {
        double max_val = 1.0;
        for (int j = 0; j < n; j++) {
            max_val = std::max(max_val, std::abs(A(i, j)));
        }

        if (max_val > 1e-12) {
            double scale = 1.0 / max_val;
            A.row(i) *= scale;
            b(i) *= scale;
        }
    }

    // Geometric Mean Column Scaling
    for (int j = 0; j < n; j++) {
        double max_val = 1.0;
        for (int i = 0; i < m; i++) {
            max_val = std::max(max_val, std::abs(A(i, j)));
        }

        if (max_val > 1e-12) {
            double scale = 1.0 / max_val;
            A.col(j) *= scale;
            c(j) *= scale; // Must scale the cost to match the new "units"
        }
    }
}


// ---------------------------
// Cuda Util. Functions
// ---------------------------

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define cusolverCheckError(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSolver Error: %d %s %d\n", code, file, line);
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

void print_gpu_info() {
    int deviceCount;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA capable devices found." << std::endl;
        return;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaCheckError(cudaGetDeviceProperties(&prop, dev));

        std::cout << "\n==================================================" << std::endl;
        std::cout << " Device " << dev << ": " << prop.name << std::endl;
        std::cout << "==================================================" << std::endl;

        // --- 1. General Info ---
        std::cout << "--- General Information ---" << std::endl;
        std::cout << "Compute Capability:            " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Multiprocessors (SMs):         " << prop.multiProcessorCount << std::endl;
        std::cout << "Concurrent Kernels:            " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "Can Map Host Memory:           " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "Integrated GPU:                " << (prop.integrated ? "Yes" : "No") << std::endl;

        // --- 2. Memory Info ---
        std::cout << "\n--- Memory Information ---" << std::endl;
        std::cout << "Total Global Memory:           " << (double)prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total Constant Memory:         " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "Shared Memory per Block:       " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Registers per Block:           " << prop.regsPerBlock << std::endl;
        std::cout << "L2 Cache Size:                 " << prop.l2CacheSize / 1024 << " KB" << std::endl;

        // --- 3. Thread & Block Constraints (Crucial for Kernels) ---
        std::cout << "\n--- Thread & Block Constraints ---" << std::endl;
        std::cout << "Max Threads per Block:         " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Dim (Block):       [" << prop.maxThreadsDim[0] << ", " 
                                                          << prop.maxThreadsDim[1] << ", " 
                                                          << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "Max Grid Size:                 [" << prop.maxGridSize[0] << ", " 
                                                          << prop.maxGridSize[1] << ", " 
                                                          << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "Warp Size:                     " << prop.warpSize << std::endl;

        // --- 4. Clocks & Bus ---
        std::cout << "\n--- Clock & Bus ---" << std::endl;
        std::cout << "Memory Bus Width:              " << prop.memoryBusWidth << " bits" << std::endl;

        std::cout << "==================================================\n" << std::endl;
    }
}

struct CusolverResources {
    cusolverDnHandle_t handle;
    cublasHandle_t handle_cublas;
    double *d_B;        // The basis matrix
    double *d_x;        // Solution vector
    int *d_ipiv;        // Pivot array
    double *d_c;        // Cost vector
    double *d_s;        // Reduced cost vector on GPU (output)
    double *d_solution;

    double *d_work;     // Workspace for cuSolver
    int *d_info;


    int lwork;          // Workspace size
    int m;              // dimension
    double *d_A;
    int *d_basis_ids;

    CusolverResources(int m_dim, int n_dim) : m(m_dim) {
        cusolverCheckError(cusolverDnCreate(&handle));
        cublasCheckError(cublasCreate(&handle_cublas));

        cudaCheckError(cudaMalloc((void**)&d_B, sizeof(double) * m * m));
        cudaCheckError(cudaMalloc((void**)&d_A, sizeof(double) * m * n_dim))
        cudaCheckError(cudaMalloc((void**)&d_x, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_ipiv, sizeof(int) * m));
        cudaCheckError(cudaMalloc((void**)&d_basis_ids, sizeof(int) * m));
        cudaCheckError(cudaMalloc((void**)&d_c, sizeof(double) * n_dim));
        cudaCheckError(cudaMalloc((void**)&d_s, sizeof(double) * n_dim));
        cudaCheckError(cudaMalloc((void**)&d_solution, sizeof(double) * m));

        cudaCheckError(cudaMalloc((void**)&d_info, sizeof(int)));

        // get workspace size for LU Factorization
        cusolverCheckError(cusolverDnDgetrf_bufferSize(handle, m, m, d_B, m, &lwork));
    
        cudaCheckError(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
    }

    ~CusolverResources() {
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_ipiv);
        cudaFree(d_basis_ids);
        cudaFree(d_c);
        cudaFree(d_s);

        cudaFree(d_info);
        cudaFree(d_work);
        cusolverDnDestroy(handle);
        cublasDestroy(handle_cublas);
    }
};

// Basically an all-gather on creating the matrix A
__global__ void assemble_basis_kernel(const double* __restrict__ A,
                                      double* __restrict__ B,
                                      const int* __restrict__ basis_indices,
                                      int m, int m_stride_A) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < m) {
        int src_col_idx = basis_indices[col];
        B[row + col * m] = A[row + src_col_idx * m_stride_A];
    }
}

struct RatioTestFunctor {
    double tol;
    RatioTestFunctor(double _tol) : tol(_tol) {}

    __host__ __device__
    double operator()(const double& x_val, const double& d_val) const {
        if (d_val > tol) {
            return x_val / d_val;
        }
        return DBL_MAX; // Ignore this index
    }
};

// ---------------------------
// Main Algorithm
// ---------------------------


#define MAX_ITERS 5000

const double OPTIMALITY_TOL = 1e-6;
const double PIVOT_TOL      = 1e-5;

Eigen::VectorXd simplex_method(const Eigen::MatrixXd& A,
                               const Eigen::VectorXd& b,
                               const Eigen::VectorXd& c,
                               int n, int m) {

    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i) basis[i] = n - m + i;

    Eigen::VectorXd cB(m);
    
    CusolverResources gpu(m, n);
    cudaCheckError(cudaMemcpy(gpu.d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(gpu.d_c, c.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

    dim3 blockSize(32, 8);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    
    // Create Device Basis Vector
    thrust::device_vector<int> d_basis(basis.begin(), basis.end());
    
    // Helper vectors for Thrust inputs
    thrust::device_vector<double> d_ratios(m);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        
        // 1. Update CPU costs (needed if you process partial results on CPU)
        for (int i = 0; i < m; ++i) cB(i) = c(basis[i]);

        // BUG FIX: Ensure GPU has the current basis indices
        // Ideally, do the swap on GPU, but for now, just copy it to be safe.
        thrust::copy(basis.begin(), basis.end(), d_basis.begin());

        // 2. Assemble Basis Matrix on GPU
        int* d_basis_ptr = thrust::raw_pointer_cast(d_basis.data());
        assemble_basis_kernel<<<gridSize, blockSize>>>(gpu.d_A, gpu.d_B, d_basis_ptr, m, m);
        
        // 3. LU Factorization
        cusolverCheckError(cusolverDnDgetrf(
            gpu.handle, m, m, gpu.d_B, m,
            gpu.d_work, gpu.d_ipiv, gpu.d_info
        ));

        // 4. Calculate Lambda (Using Transpose)
        cudaCheckError(cudaMemcpy(gpu.d_x, cB.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_T, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info
        ));

        // 5. Calculate Reduced Costs (s)
        double alpha = -1.0;
        double beta  = 1.0;
        cudaCheckError(cudaMemcpy(gpu.d_s, gpu.d_c, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        
        cublasCheckError(cublasDgemv(
            gpu.handle_cublas, CUBLAS_OP_T, m, n,
            &alpha, gpu.d_A, m, gpu.d_x, 1, &beta, gpu.d_s, 1
        ));
        
        // 6. Find Entering Variable (Thrust)
        thrust::device_ptr<double> s_ptr(gpu.d_s);
        auto max_iter = thrust::max_element(s_ptr, s_ptr + n);
        double s_max = *max_iter; 
        int enter = max_iter - s_ptr;

        if (s_max <= OPTIMALITY_TOL) {
            // Reconstruct final solution
            // We need to solve for xB one last time because d_x currently holds lambda
            cudaCheckError(cudaMemcpy(gpu.d_x, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
            cusolverCheckError(cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info));
            
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd xB(m);
            cudaCheckError(cudaMemcpy(xB.data(), gpu.d_x, sizeof(double) * m, cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < m; ++i) x(basis[i]) = std::max(0.0, xB(i));
            return x;
        }

        // 7. Calculate Basic Solution (xB)
        cudaCheckError(cudaMemcpy(gpu.d_x, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_N, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info 
        ));

        // BUG FIX: Save xB into a separate buffer so we don't lose it when solving for d
        cudaCheckError(cudaMemcpy(gpu.d_solution, gpu.d_x, sizeof(double) * m, cudaMemcpyDeviceToDevice));

        // 8. Calculate Direction (d)
        // d_A is column-major (m x n). The column starts at d_A + enter * m.
        cublasCheckError(cublasDcopy(
            gpu.handle_cublas, m,
            gpu.d_A + (enter * m), 1, // Source: Column inside A
            gpu.d_x, 1                // Dest: d_x
        ));
        
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_N, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info
        ));

        // 9. Ratio Test (Thrust)
        // d_solution holds xB (Numerator)
        // d_x holds d (Denominator)
        thrust::device_ptr<double> xB_ptr(gpu.d_solution); 
        thrust::device_ptr<double> d_ptr(gpu.d_x);
        
        thrust::transform(xB_ptr, xB_ptr + m, d_ptr, d_ratios.begin(), RatioTestFunctor(PIVOT_TOL));

        auto min_iter = thrust::min_element(d_ratios.begin(), d_ratios.end());
        double theta_min = *min_iter;
        int leave = min_iter - d_ratios.begin();

        if (theta_min >= DBL_MAX) { // Check against max double, as defined in Functor
            std::cout << "Problem unbounded\n";
            return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        }
        
        // 10. Update Basis
        basis[leave] = enter;
    }
    
    return Eigen::VectorXd::Zero(n); // Limit reached
}


int main() {
    print_gpu_info();

    int n, m;
    // starts with n, m
    std::cin >> m >> n;
    
    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m), c(n);

    // followed by A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> A(i, j);
        }
    }
    // then, b
    for (int i = 0; i < m; i++) {
        std::cin >> b(i);
    }
    // then c
    for (int i = 0; i < n; i++) {
        std::cin >> c(i);
    }
    equilibrate(A, b, c);
    
    // std::cout << "DEBUG: First element of A: " << A(0,0) << "\n";
    // std::cout << "DEBUG: First element of b: " << b(0) << "\n";
    // std::cout << "DEBUG: Last element of c: " << c(n-1) << "\n"; // Should be -M or 0
    
    Eigen::VectorXd z = simplex_method(A, b, c, n, m);
    double optimum = c.dot(z);  // Compute c^T * z
    // std::cout << "Output:\n" << z.transpose() << "\n";
    std::cout << std::setprecision(15) << "Optimum found: " << optimum << "\n";

}