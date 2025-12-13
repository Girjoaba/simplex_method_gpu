#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cfloat>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

#include <Eigen/Dense>


inline int get_grid_size(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

// 20.4

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
    double *d_A;        // The entire matrix
    double *d_B;        // The basis matrix
    double *d_x;        // Solution vector
    double *d_xB;       // Make xB sol. persistent
    double *d_d;        // Make direction persistent
    int *d_ipiv;        // Pivot array
    double *d_c;        // Cost vector
    double *d_s;        // Reduced cost vector on GPU (output)

    double *d_work;     // Workspace for cuSolver
    int *d_info;


    int lwork;          // Workspace size
    int m;              // dimension
    int *d_basis_ids;

    // Sherman Morrison
    double *d_invB;    // Explicit inverse matrix
    double *d_lambda;  // Dual vector buffer
    double *d_row_p;   // Buffer to hold a row of B^-1 for the update
    double *d_scalar_val; // for pivoting the morrison

    dim3 block1D;
    dim3 grid1D;       // For vector operations (size m)
    dim3 gridFull;     // For matrix flattening (size m*m)
    dim3 blockSize;    // 2D Block
    dim3 gridSize;     // 2D Grid

    CusolverResources(int m_dim, int n_dim) : m(m_dim) {
        cusolverCheckError(cusolverDnCreate(&handle));
        cublasCheckError(cublasCreate(&handle_cublas));

        cudaCheckError(cudaMalloc((void**)&d_A, sizeof(double) * m * n_dim));
        cudaCheckError(cudaMalloc((void**)&d_B, sizeof(double) * m * m));
        cudaCheckError(cudaMalloc((void**)&d_x, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_xB, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_d, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_ipiv, sizeof(int) * m));
        cudaCheckError(cudaMalloc((void**)&d_basis_ids, sizeof(int) * m));
        cudaCheckError(cudaMalloc((void**)&d_c, sizeof(double) * n_dim));
        cudaCheckError(cudaMalloc((void**)&d_s, sizeof(double) * n_dim));
        cudaCheckError(cudaMalloc((void**)&d_info, sizeof(int)));

        // Sherman Morrison
        cudaCheckError(cudaMalloc((void**)&d_invB, sizeof(double) * m * m));
        cudaCheckError(cudaMalloc((void**)&d_lambda, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_row_p, sizeof(double) * m));
        cudaCheckError(cudaMalloc((void**)&d_scalar_val, sizeof(double)));

        cusolverCheckError(cusolverDnDgetrf_bufferSize(handle, m, m, d_B, m, &lwork));
        cudaCheckError(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

        // Blocking...
        block1D = dim3(256);
        blockSize = dim3(32, 8);

        // Grid size depends on m
        grid1D = dim3((m + block1D.x - 1) / block1D.x);
        gridFull = dim3((m * m + block1D.x - 1) / block1D.x);
        gridSize = dim3((m + blockSize.x - 1) / blockSize.x,
                        (m + blockSize.y - 1) / blockSize.y);
    }

    ~CusolverResources() {
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_x); 
        cudaFree(d_xB);
        cudaFree(d_d); 
        cudaFree(d_ipiv); 
        cudaFree(d_basis_ids);
        cudaFree(d_c); 
        cudaFree(d_s); 
        cudaFree(d_info); 
        cudaFree(d_work);
        
        // Sherman Morrison
        cudaFree(d_invB); 
        cudaFree(d_lambda); 
        cudaFree(d_row_p);
        cudaFree(d_scalar_val);
        
        cusolverDnDestroy(handle);
        cublasDestroy(handle_cublas);
    }
};

// ---------------------------
// Kernels
// ---------------------------

__global__ void set_identity_kernel(double* Matrix, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m * m) {
        int row = idx % m;
        int col = idx / m;
        Matrix[idx] = (row == col) ? 1.0 : 0.0;
    }
}

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

__global__ void mask_basis_kernel(double *d_s, const int* d_basis_ids, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int col_idx = d_basis_ids[idx];
        d_s[col_idx] = -1.0e20; // Mask out basis so they aren't selected
    }
}

__global__ void update_solution_kernel(double* __restrict__ xB, const double* d, int leave_idx, double theta, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        if (idx == leave_idx) {
            xB[idx] = theta; 
        } else {
            xB[idx] = xB[idx] - theta * d[idx];
        }
    }
}

__global__ void gather_vector_kernel(double *d_out, const double* d_in, const int* d_indices, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        d_out[idx] = d_in[d_indices[idx]];
    }
}

__global__ void prepare_sherman_morrison_data(double* d_d, int p, double* d_alpha) {
    if (threadIdx.x == 0) {
        double pivot = d_d[p];

        *d_alpha = -1.0 / pivot;
        d_d[p] = pivot - 1.0;
    }
}

// ---------------------------
// Cuda Functors
// ---------------------------

struct RatioTestUnaryOp {
    const double* xB;
    const double* d;
    double tol;

    __device__ thrust::pair<double, int> operator()(int i) const {
        double div = d[i];
        if (div > tol) {
            return thrust::make_pair(xB[i] / div, i);
        }
        return thrust::make_pair(DBL_MAX, -1);
    }
};

struct MinPairOp {
    __device__ thrust::pair<double, int> operator()(
                    const thrust::pair<double, int>& a,
                    const thrust::pair<double, int>& b) const {

        if (a.first < b.first) return a;
        if (a.first > b.first) return b;
        return (a.second < b.second) ? a : b;
    }
};

// ---------------------------
// Logic
// ---------------------------

// Recomputes B from scratch to establish numerical stability dominance 
void compute_basis_inverse(CusolverResources& gpu, const Eigen::VectorXd& b_host, int m) {

    // assemble B from A
    assemble_basis_kernel<<<gpu.gridSize, gpu.blockSize>>>(gpu.d_A, gpu.d_B, gpu.d_basis_ids, m, m);

    cusolverCheckError(cusolverDnDgetrf(
        gpu.handle, 
        m, m, 
        gpu.d_B, m,
        gpu.d_work, 
        gpu.d_ipiv, 
        gpu.d_info
    ));

    // refresh primal to fix drift
    cudaCheckError(cudaMemcpy(gpu.d_xB, b_host.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
    cusolverCheckError(cusolverDnDgetrs(
        gpu.handle, 
        CUBLAS_OP_N, 
        m, 1,
        gpu.d_B, m, 
        gpu.d_ipiv, 
        gpu.d_xB, m, 
        gpu.d_info
    ));

    // reset to identity
    set_identity_kernel<<<gpu.gridFull, gpu.block1D>>>(gpu.d_invB, m);
    
    // solve B * X = I. (X overwrites d_invB)
    cusolverCheckError(cusolverDnDgetrs(
        gpu.handle, 
        CUBLAS_OP_N, 
        m, m, 
        gpu.d_B, m, 
        gpu.d_ipiv, 
        gpu.d_invB, m, 
        gpu.d_info
    ));
}

#define MAX_ITERS 10000
#define REINVERSION_FREQ 30 // prevent drift with this frequency

const double OPTIMALITY_TOL = 1e-6;
const double PIVOT_TOL      = 1e-6;

Eigen::VectorXd simplex_method(const Eigen::MatrixXd& A,
                               const Eigen::VectorXd& b,
                               const Eigen::VectorXd& c,
                               int n, int m) {

    // initial basis
    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i) {
      basis[i] = n - m + i;  
    } 

    CusolverResources gpu(m, n);
    cudaCheckError(cudaMemcpy(gpu.d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(gpu.d_c, c.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    
    cudaCheckError(cudaMemcpy(gpu.d_basis_ids, basis.data(), sizeof(int) * m, cudaMemcpyHostToDevice));
    
    compute_basis_inverse(gpu, b, m);

    const double alpha_one = 1.0;
    const double alpha_zero = 0.0;
    const double alpha_minus_one = -1.0;

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
 
        // numerical stability
        if (iter > 0 && iter % REINVERSION_FREQ == 0) {
            compute_basis_inverse(gpu, b, m);
        }

        // lambda = (B^-1)^T * cB
        gather_vector_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_x, gpu.d_c, gpu.d_basis_ids, m);
        
        cublasCheckError(cublasDgemv(
            gpu.handle_cublas, CUBLAS_OP_T, 
            m, m,
            &alpha_one, 
            gpu.d_invB, m, 
            gpu.d_x, 1,         // cB
            &alpha_zero, 
            gpu.d_lambda, 1     // Result into d_lambda
        ));

        // pricing: s = c - A^T * lambda
        cudaCheckError(cudaMemcpy(gpu.d_s, gpu.d_c, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        cublasCheckError(cublasDgemv(
            gpu.handle_cublas, 
            CUBLAS_OP_T, 
            m, n,
            &alpha_minus_one, 
            gpu.d_A, m, 
            gpu.d_lambda, 1,    // Read from d_lambda
            &alpha_one, 
            gpu.d_s, 1
        ));

        // ----- Select entering variable
        mask_basis_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_s, gpu.d_basis_ids, m);

        thrust::device_ptr<double> s_ptr(gpu.d_s);
        auto max_iter = thrust::max_element(s_ptr, s_ptr + n);
        
        double s_max;
        int enter;
        int offset = max_iter - s_ptr;
        cudaCheckError(cudaMemcpy(&s_max, thrust::raw_pointer_cast(&*max_iter), sizeof(double), cudaMemcpyDeviceToHost));
        enter = offset;
        
        // ========== EXIT OPTIMALLY ===========  
        if (s_max <= OPTIMALITY_TOL) {
            std::cout << "Iteration: " << iter << "\n";
            Eigen::VectorXd xB(m);
            cudaCheckError(cudaMemcpy(xB.data(), gpu.d_xB, sizeof(double) * m, cudaMemcpyDeviceToHost));
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < m; ++i) {
                x(basis[i]) = std::max(0.0, xB(i));
            }
            return x;
        }
        
        // compute direction: d = B^-1 * A_enter
        cublasCheckError(cublasDgemv(
            gpu.handle_cublas,  
            CUBLAS_OP_N, 
            m, m,
            &alpha_one,
            gpu.d_invB, m,            
            gpu.d_A + (enter * m), 1, 
            &alpha_zero,
            gpu.d_d, 1
        ));
        
        // ratio test
        thrust::pair<double, int> result = thrust::transform_reduce(
            thrust::device,
            thrust::make_counting_iterator(0),             
            thrust::make_counting_iterator(m),             
            RatioTestUnaryOp{gpu.d_xB, gpu.d_d, PIVOT_TOL}, 
            thrust::make_pair(DBL_MAX, -1),                 
            MinPairOp()                                     
        );

        double theta_min = result.first;
        int leave = result.second;
        
        // ========== EXIT UNBOUNDED ===========
        if (leave == -1 || theta_min >= DBL_MAX) {
            std::cout << "Problem unbounded\n";
            return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        }
        
        // update solution xB
        update_solution_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_xB, gpu.d_d, leave, theta_min, m);
        cudaCheckError(cudaMemcpy(gpu.d_basis_ids + leave, &enter, sizeof(int), cudaMemcpyHostToDevice));

        // ===========================================
        // update inverse basis w/ Sherman-Morrison
        // Formula: B_new^-1 = B^-1 - ( (d - e_p) * row_p(B^-1) ) / d[p]
        
        // Extract p-th row of B^-1 into d_row_p
        cublasCheckError(cublasDcopy(
            gpu.handle_cublas, 
            m, 
            gpu.d_invB + leave, m,
            gpu.d_row_p, 1
        ));

        // Prepare u = (d - e_p). 
        // Update: d_d[p] -= 1.0
        prepare_sherman_morrison_data<<<1, 1>>>(gpu.d_d, leave, gpu.d_scalar_val);
        cublasCheckError(cublasSetPointerMode(gpu.handle_cublas, CUBLAS_POINTER_MODE_DEVICE));

        cublasCheckError(cublasDger(
            gpu.handle_cublas,
            m, m,
            gpu.d_scalar_val,
            gpu.d_d, 1,
            gpu.d_row_p, 1,
            gpu.d_invB, m
        ));
        cublasCheckError(cublasSetPointerMode(gpu.handle_cublas, CUBLAS_POINTER_MODE_HOST));

        basis[leave] = enter;
    }
    
    std::cerr << "Warning: Hit iteration limit\n";
    Eigen::VectorXd xB(m);
    cudaCheckError(cudaMemcpy(xB.data(), gpu.d_xB, sizeof(double)*m, cudaMemcpyDeviceToHost));
    Eigen::VectorXd x  = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < m; ++i) x(basis[i]) = std::max(0.0, xB(i));
    return x;
}

int main() {
    print_gpu_info();

    int n, m;
    if(!(std::cin >> m >> n)) return 0;
    
    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m), c(n);

    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) std::cin >> A(i, j);
    for (int i = 0; i < m; i++) std::cin >> b(i);
    for (int i = 0; i < n; i++) std::cin >> c(i);
    
    equilibrate(A, b, c);
    
    Eigen::VectorXd z = simplex_method(A, b, c, n, m);
    double optimum = c.dot(z);
    std::cout << std::setprecision(15) << "Optimum found: " << optimum << "\n";
}