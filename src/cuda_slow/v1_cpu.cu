#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/LU>

// ---------------------------
// Util. Functions
// ---------------------------

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
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

// void equilibrate(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& c) {

// }

// ---------------------------
// Main Algorithm
// ---------------------------


#define MAX_ITERS 200000

const double EPSILON = 1e-6;

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
    for (int j = 0; j < m; ++j) {
        B.col(j) = A.col(basis[j]); // columns in my basis
    }

    Eigen::VectorXd cB(m);
    for (int i = 0; i < m; ++i) {
        cB(i) = c(basis[i]);    // subset of the cost coeff.
    }

    // =============================== |
    // --------- Main Loop ----------- |
    // =============================== |
    // See Algorithm 4. https://web.stanford.edu/class/msande310/Simplex-ref1.pdf

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        /* Determine entering variable */
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);     // We use LU Decomposition
        Eigen::VectorXd lambda = lu.transpose().solve(cB);         // y[m] <- cB * B^-1
        Eigen::VectorXd s      = c - A.transpose() * lambda;    // e[n] <- [1, y] * [-c; A]
        
        std::vector<char> inBasis(n, 0);
        for (int i = 0; i < m; ++i) inBasis[basis[i]] = 1;
        
        // Most positive reduced cost
        Eigen::Index enter = -1;
        double s_max = EPSILON;  
        for (int j = 0; j < n; ++j) {
            if (!inBasis[j] && s(j) > s_max) {
                s_max = s(j);
                enter = j;
            }
        }
        
        Eigen::VectorXd xB = lu.solve(b);
        
        // If no more reductions, exit
        if (enter == -1) {
            std::cout << "Iteration: " << iter << "\n";
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < m; ++i) {
                x(basis[i]) = std::max(0.0, xB(i));
            }
            return x;
        }
        
        // how much each basis variable change if I increase the entering variable
        Eigen::VectorXd d = lu.solve(A.col(enter));
        
        // Find leaving variable
        Eigen::Index leave = -1;
        double theta_min = std::numeric_limits<double>::infinity();
        // theta_min = min { xB(i) / d(i) : d(i) > 0 }
        // the furthest you can go before a basis variable hits 0 (violates a constraint) 
        for (int i = 0; i < m; ++i) {
            if (d(i) > EPSILON) {
                double theta = xB(i) / d(i);
                if (theta < theta_min) {
                    theta_min = theta;
                    leave = i;
                }
            }
        }
        
        if (leave == -1) {
            std::cerr << "Problem unbounded\n";
            return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        }
        
        // Pivot
        basis[leave] = enter;
        B.col(leave) = A.col(enter);
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

    // Then, b
    for (int i = 0; i < m; i++) {
        std::cin >> b(i);
    }

    // Finally c
    for (int i = 0; i < n; i++) {
        std::cin >> c(i);
    }
    
    
    // std::cout << "DEBUG: First element of A: " << A(0,0) << "\n";
    // std::cout << "DEBUG: First element of b: " << b(0) << "\n";
    // std::cout << "DEBUG: Last element of c: " << c(n-1) << "\n"; // Should be -M or 0
    
    Eigen::VectorXd z = simplex_method(A, b, c, n, m);
    double optimum = c.dot(z);  // Compute c^T * z
    // std::cout << "Output:\n" << z.transpose() << "\n";
    std::cout << std::setprecision(15) << "Optimum found: " << optimum << "\n";

}