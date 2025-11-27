#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>       // For max_element, min_element

#include <Eigen/Dense>
#include <Eigen/LU>

//  105.168

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

    // Expect the identity on the right part!
    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i) {
      basis[i] = n - m + i;  // points to column in A
    } 

    Eigen::MatrixXd B(m, m);
    Eigen::VectorXd cB(m);
    
    CusolverResources gpu(m, n);
    cudaCheckError(cudaMemcpy(gpu.d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(gpu.d_c, c.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

    dim3 blockSize(32, 8);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);
    
    // =============================== |
    // --------- Main Loop ----------- |
    // =============================== |
    // See Algorithm 4. https://web.stanford.edu/class/msande310/Simplex-ref1.pdf
    
    int h_info = 0;
    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        
        for (int i = 0; i < m; ++i) {
            cB(i) = c(basis[i]);    // subset of the cost coeff.
        }

        // ========= CUDA BEGIN ==========  
        cudaCheckError(cudaMemcpy(gpu.d_basis_ids, basis.data(), sizeof(int) * m, cudaMemcpyHostToDevice));
        assemble_basis_kernel<<<gridSize, blockSize>>>(gpu.d_A, gpu.d_B, gpu.d_basis_ids, m, m);
        cudaCheckError(cudaGetLastError());
        // d_B becomes LU
        cusolverCheckError(cusolverDnDgetrf(
            gpu.handle, m, m, gpu.d_B, m,
            gpu.d_work, gpu.d_ipiv, gpu.d_info
        ));
        cudaCheckError(cudaMemcpy(&h_info, gpu.d_info, sizeof(int), cudaMemcpyDeviceToHost));

        cudaCheckError(cudaMemcpy(gpu.d_x, cB.data(), sizeof(double) * m, cudaMemcpyHostToDevice));

        // CUBLAS_OP_T: solve using transpose
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_T, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info
        ));

        double alpha = -1.0;
        double beta  = 1.0;
        cudaCheckError(cudaMemcpy(gpu.d_s, gpu.d_c, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        // Calculate s = -A^T * lambda + s
        cublasCheckError(cublasDgemv(
            gpu.handle_cublas, CUBLAS_OP_T, m, n,
            &alpha, gpu.d_A, m, gpu.d_x, 1, &beta, gpu.d_s, 1
        ));
        // =========== CUDA END ==========  

        thrust::device_ptr<double> s_ptr(gpu.d_s);
        auto max_iter = thrust::max_element(s_ptr, s_ptr + n);
        double s_max = *max_iter; 
        int enter = max_iter - s_ptr;

        
        Eigen::VectorXd xB(m);

        // ========= CUDA BEGIN ==========
        cudaCheckError(cudaMemcpy(gpu.d_x, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        // CUBLAS_OP_N: solve using no-transpose
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_N, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info 
        ));
        cudaCheckError(cudaMemcpy(xB.data(), gpu.d_x, sizeof(double) * m, cudaMemcpyDeviceToHost));
        // ========== CUDA END ===========  
        
        if (s_max <= OPTIMALITY_TOL) {
            std::cout << "Iteration: " << iter << "\n";
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < m; ++i) {
                x(basis[i]) = std::max(0.0, xB(i));
            }
            return x;
        }
        
        Eigen::VectorXd d(m);
        Eigen::VectorXd colEnter = A.col(enter);
        // ========= CUDA BEGIN ==========
        cudaCheckError(cudaMemcpy(gpu.d_x, colEnter.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        cusolverCheckError(cusolverDnDgetrs(
            gpu.handle, CUBLAS_OP_N, m, 1,
            gpu.d_B, m, gpu.d_ipiv, gpu.d_x, m, gpu.d_info
        ));
        cudaCheckError(cudaMemcpy(d.data(), gpu.d_x, sizeof(double) * m, cudaMemcpyDeviceToHost));
        // ========== CUDA END =========== 
        
        Eigen::Index leave = -1;
        double theta_min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            if (d(i) > PIVOT_TOL) {
                double theta = xB(i) / d(i);
                if (theta < theta_min) {
                    theta_min = theta;
                    leave = i;
                }
            }
        }
        
        if (leave == -1) {
            std::cout << "Problem unbounded\n";
            return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        }
        
        basis[leave] = enter;
        cB(leave)    = c(enter);
    }
    
    std::cerr << "Warning: Hit iteration limit\n";
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);
    Eigen::VectorXd x  = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd xB = lu.solve(b);
    for (int i = 0; i < m; ++i) x(basis[i]) = std::max(0.0, xB(i));
    return x;
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
    
    
    Eigen::VectorXd z = simplex_method(A, b, c, n, m);
    double optimum = c.dot(z);  // Compute c^T * z
    // std::cout << "Output:\n" << z.transpose() << "\n";
    std::cout << std::setprecision(15) << "Optimum found: " << optimum << "\n";

}