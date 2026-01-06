#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cfloat>
#include <algorithm> // for std::max
#include <utility>   // for std::pair

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include <cub/cub.cuh>

#include <Eigen/Dense>

// ---------------------------
// Structures & Utils
// ---------------------------

struct CscMatrix
{
    std::vector<double> values;
    std::vector<int> row_indices;
    std::vector<int> col_ptr;
    int nnz;
};

CscMatrix dense_to_csc(const Eigen::MatrixXd &A)
{
    CscMatrix mat;
    int m = A.rows();
    int n = A.cols();

    mat.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            double val = A(i, j);
            if (std::abs(val) > 1e-12)
            {
                mat.values.push_back(val);
                mat.row_indices.push_back(i);
            }
        }
        mat.col_ptr.push_back(mat.values.size());
    }
    mat.nnz = mat.values.size();
    return mat;
}

// ---------------------------
// CUDA Utils
// ---------------------------

#define cudaCheckError(ans)                    \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ---------------------------
// Kernels
// ---------------------------

__global__ void set_identity_kernel(double *Matrix, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * m)
    {
        Matrix[idx] = ((idx % m) == (idx / m)) ? 1.0 : 0.0;
    }
}

__global__ void assemble_basis_kernel(const double *__restrict__ A, double *__restrict__ B,
                                      const int *__restrict__ basis_indices, int m, int m_stride_A)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < m)
    {
        int src_col_idx = basis_indices[col];
        B[row + col * m] = A[row + src_col_idx * m_stride_A];
    }
}

__global__ void mask_basis_kernel(double *d_s, const int *d_basis_ids, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        int col_idx = d_basis_ids[idx];
        d_s[col_idx] = -1.0e20; // Mask out basis for ArgMax
    }
}

__global__ void update_solution_kernel(double *__restrict__ xB, const double *d, int leave_idx, double theta, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        if (idx == leave_idx)
            xB[idx] = theta;
        else
            xB[idx] = xB[idx] - theta * d[idx];
    }
}

__global__ void gather_vector_kernel(double *d_out, const double *d_in, const int *d_indices, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
        d_out[idx] = d_in[d_indices[idx]];
}

__global__ void prepare_sherman_morrison_data(double *d_d, int p, double *d_alpha)
{
    if (threadIdx.x == 0)
    {
        double pivot = d_d[p];
        *d_alpha = -1.0 / pivot;
        d_d[p] = pivot - 1.0;
    }
}

// --- FUSED KERNEL 1: GATHER COLUMN ---
__global__ void fused_gather_column_kernel(
    double *d_dense_out,
    const double *d_val,
    const int *d_row_ind,
    const int *d_col_ptr,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp)
{
    int enter_col = d_arg_max_kvp->key;
    int start = d_col_ptr[enter_col];
    int end = d_col_ptr[enter_col + 1];
    int count = end - start;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
    {
        d_dense_out[d_row_ind[start + idx]] = d_val[start + idx];
    }
}

// --- FUSED KERNEL 2: UPDATE BASIS ---
__global__ void update_basis_device_kernel(
    int *d_basis_ids,
    const int *d_leave_idx,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp)
{
    if (threadIdx.x == 0)
    {
        d_basis_ids[*d_leave_idx] = d_arg_max_kvp->key;
    }
}

// --- NEW KERNEL: COMPUTE RATIOS ---
// Calculates theta = xB[i] / d[i] for valid pivots.
// Writes DBL_MAX for invalid pivots to be ignored by ArgMin.
__global__ void compute_ratios_kernel(double *ratios, const double *xB, const double *d, int m, double tol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        double div = d[idx];
        // If div > tol, compute ratio, else set to Infinity
        ratios[idx] = (div > tol) ? (xB[idx] / div) : DBL_MAX;
    }
}

// ---------------------------
// GPU Resources
// ---------------------------

struct CusolverResources
{
    cusolverDnHandle_t handle;
    cublasHandle_t handle_cublas;
    cusparseHandle_t handle_cusparse;

    // Sparse Matrix
    double *d_A_val;
    int *d_A_rowInd;
    int *d_A_colPtr;
    cusparseSpMatDescr_t matA_desc;
    cusparseDnVecDescr_t vecLambda_desc;
    cusparseDnVecDescr_t vecS_desc;
    void *d_buffer_mv;
    size_t bufferSize_mv;

    // Dense Vectors / Matrices
    double *d_col_dense;
    double *d_A, *d_B, *d_x, *d_xB, *d_d, *d_c, *d_s;
    int *d_ipiv, *d_basis_ids, *d_info;
    double *d_work;
    int lwork;
    int *d_leave_idx;

    // Sherman Morrison
    double *d_invB, *d_lambda, *d_row_p, *d_scalar_val;

    // Ratio Test Buffer
    double *d_ratios;

    // --- CUB Storage ---
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Pricing (ArgMax) Output
    cub::KeyValuePair<int, double> *d_arg_max_out;
    cub::KeyValuePair<int, double> *h_arg_max_out;

    // Ratio Test (ArgMin) Output
    cub::KeyValuePair<int, double> *d_arg_min_out;
    cub::KeyValuePair<int, double> *h_arg_min_out;

    int m, n, nnz;
    dim3 block1D, grid1D, gridFull, blockSize, gridSize;

    CusolverResources(int m_dim, int n_dim, const CscMatrix &A_csc) : m(m_dim), n(n_dim), nnz(A_csc.nnz)
    {
        cusolverDnCreate(&handle);
        cublasCreate(&handle_cublas);
        cusparseCreate(&handle_cusparse);

        // Allocations
        cudaMalloc((void **)&d_A_val, sizeof(double) * nnz);
        cudaMalloc((void **)&d_A_rowInd, sizeof(double) * nnz);
        cudaMalloc((void **)&d_A_colPtr, sizeof(double) * (n + 1));
        cudaMemcpy(d_A_val, A_csc.values.data(), sizeof(double) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_rowInd, A_csc.row_indices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_colPtr, A_csc.col_ptr.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice);

        cusparseCreateCsc(&matA_desc, m, n, nnz, d_A_colPtr, d_A_rowInd, d_A_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cudaMalloc((void **)&d_A, sizeof(double) * m * n_dim);
        cudaMalloc((void **)&d_B, sizeof(double) * m * m);
        cudaMalloc((void **)&d_x, sizeof(double) * m);
        cudaMalloc((void **)&d_xB, sizeof(double) * m);
        cudaMalloc((void **)&d_d, sizeof(double) * m);
        cudaMalloc((void **)&d_ipiv, sizeof(int) * m);
        cudaMalloc((void **)&d_basis_ids, sizeof(int) * m);
        cudaMalloc((void **)&d_c, sizeof(double) * n_dim);
        cudaMalloc((void **)&d_s, sizeof(double) * n_dim);
        cudaMalloc((void **)&d_info, sizeof(int));
        cudaMalloc((void **)&d_col_dense, sizeof(double) * m);
        cudaMalloc((void **)&d_invB, sizeof(double) * m * m);
        cudaMalloc((void **)&d_lambda, sizeof(double) * m);
        cudaMalloc((void **)&d_row_p, sizeof(double) * m);
        cudaMalloc((void **)&d_scalar_val, sizeof(double));
        cudaMalloc((void **)&d_leave_idx, sizeof(int));

        // Allocate buffer for ratio test values
        cudaMalloc((void **)&d_ratios, sizeof(double) * m);

        cusparseCreateDnVec(&vecLambda_desc, m, d_lambda, CUDA_R_64F);
        cusparseCreateDnVec(&vecS_desc, n, d_s, CUDA_R_64F);
        double alpha = 1.0;
        cusparseSpMV_bufferSize(handle_cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, matA_desc, vecLambda_desc, &alpha, vecS_desc, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_mv);
        cudaMalloc(&d_buffer_mv, bufferSize_mv);

        cusolverDnDgetrf_bufferSize(handle, m, m, d_B, m, &lwork);
        cudaMalloc((void **)&d_work, sizeof(double) * lwork);

        // --- CUB Allocations ---
        cudaMalloc((void **)&d_arg_max_out, sizeof(cub::KeyValuePair<int, double>));
        cudaMallocHost((void **)&h_arg_max_out, sizeof(cub::KeyValuePair<int, double>));

        cudaMalloc((void **)&d_arg_min_out, sizeof(cub::KeyValuePair<int, double>));
        cudaMallocHost((void **)&h_arg_min_out, sizeof(cub::KeyValuePair<int, double>));

        // Determine max Temp Storage size for both ArgMax and ArgMin
        size_t size_argmax = 0;
        size_t size_argmin = 0;

        cub::DeviceReduce::ArgMax(nullptr, size_argmax, d_s, d_arg_max_out, n);
        cub::DeviceReduce::ArgMin(nullptr, size_argmin, d_ratios, d_arg_min_out, m);

        temp_storage_bytes = std::max(size_argmax, size_argmin);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Blocking
        block1D = dim3(256);
        grid1D = dim3((m + block1D.x - 1) / block1D.x);
        gridFull = dim3((m * m + block1D.x - 1) / block1D.x);
        blockSize = dim3(32, 8);
        gridSize = dim3((m + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    }

    ~CusolverResources()
    {
        cusparseDestroySpMat(matA_desc);
        cusparseDestroyDnVec(vecLambda_desc);
        cusparseDestroyDnVec(vecS_desc);
        cusparseDestroy(handle_cusparse);
        cusolverDnDestroy(handle);
        cublasDestroy(handle_cublas);
        cudaFree(d_A_val);
        cudaFree(d_A_rowInd);
        cudaFree(d_A_colPtr);
        cudaFree(d_buffer_mv);
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
        cudaFree(d_col_dense);
        cudaFree(d_invB);
        cudaFree(d_lambda);
        cudaFree(d_row_p);
        cudaFree(d_scalar_val);
        cudaFree(d_leave_idx);
        cudaFree(d_ratios);

        cudaFree(d_temp_storage);
        cudaFree(d_arg_max_out);
        cudaFreeHost(h_arg_max_out);
        cudaFree(d_arg_min_out);
        cudaFreeHost(h_arg_min_out);
    }
};

void compute_basis_inverse(CusolverResources &gpu, const Eigen::VectorXd &b_host, int m)
{
    assemble_basis_kernel<<<gpu.gridSize, gpu.blockSize>>>(gpu.d_A, gpu.d_B, gpu.d_basis_ids, m, m);
    cusolverDnDgetrf(gpu.handle, m, m, gpu.d_B, m, gpu.d_work, gpu.d_ipiv, gpu.d_info);
    cudaMemcpy(gpu.d_xB, b_host.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_xB, m, gpu.d_info);
    set_identity_kernel<<<gpu.gridFull, gpu.block1D>>>(gpu.d_invB, m);
    cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, m, gpu.d_B, m, gpu.d_ipiv, gpu.d_invB, m, gpu.d_info);
}

#define MAX_ITERS 100000
#define REINVERSION_FREQ 30
const double OPTIMALITY_TOL = 1e-6;
const double PIVOT_TOL = 1e-6;

std::pair<Eigen::VectorXd, int> simplex_method(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c, int n, int m)
{
    nvtxRangePush("First Iteration");
    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i)
        basis[i] = n - m + i;

    CscMatrix sparse_A = dense_to_csc(A);
    CusolverResources gpu(m, n, sparse_A);

    cudaMemcpy(gpu.d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.d_c, c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.d_basis_ids, basis.data(), sizeof(int) * m, cudaMemcpyHostToDevice);

    compute_basis_inverse(gpu, b, m);
    double alpha_one = 1.0, alpha_zero = 0.0, alpha_mone = -1.0;

    nvtxRangePop();
    for (int iter = 0; iter < MAX_ITERS; ++iter)
    {
        if (iter > 0 && iter % REINVERSION_FREQ == 0)
        {
            nvtxRangePush("Basis Reinversion");
            compute_basis_inverse(gpu, b, m);
            nvtxRangePop();
        }

        // 1. Duals Calculation
        nvtxRangePush("Dual Calculation");
        gather_vector_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_x, gpu.d_c, gpu.d_basis_ids, m);
        cublasDgemv(gpu.handle_cublas, CUBLAS_OP_T, m, m, &alpha_one, gpu.d_invB, m, gpu.d_x, 1, &alpha_zero, gpu.d_lambda, 1);
        nvtxRangePop();

        // 2. Pricing
        nvtxRangePush("Pricing");
        cudaMemcpy(gpu.d_s, gpu.d_c, sizeof(double) * n, cudaMemcpyDeviceToDevice);
        cusparseSpMV(gpu.handle_cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_mone, gpu.matA_desc, gpu.vecLambda_desc, &alpha_one, gpu.vecS_desc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, gpu.d_buffer_mv);
        nvtxRangePop();

        // 3. Selection
        nvtxRangePush("Selection");
        mask_basis_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_s, gpu.d_basis_ids, m);
        cub::DeviceReduce::ArgMax(gpu.d_temp_storage, gpu.temp_storage_bytes,
                                  gpu.d_s, gpu.d_arg_max_out, n);
        nvtxRangePop();

        // 4. Optimized Async Gather
        nvtxRangePush("Async Gather");
        cudaMemcpyAsync(gpu.h_arg_max_out, gpu.d_arg_max_out,
                        sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost);
        cudaMemsetAsync(gpu.d_col_dense, 0, sizeof(double) * m);
        fused_gather_column_kernel<<<gpu.grid1D, gpu.block1D>>>(
            gpu.d_col_dense, gpu.d_A_val, gpu.d_A_rowInd, gpu.d_A_colPtr, gpu.d_arg_max_out);
        nvtxRangePop();

        cudaDeviceSynchronize();
        double s_max = gpu.h_arg_max_out->value;
        int enter = gpu.h_arg_max_out->key;

        if (s_max <= OPTIMALITY_TOL)
        {
            Eigen::VectorXd xB(m);
            cudaMemcpy(xB.data(), gpu.d_xB, sizeof(double) * m, cudaMemcpyDeviceToHost);
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < m; ++i)
                x(basis[i]) = std::max(0.0, xB(i));
            return {x, iter};
        }

        // 5. Direction
        nvtxRangePush("Direction");
        cublasDgemv(gpu.handle_cublas, CUBLAS_OP_N, m, m, &alpha_one, gpu.d_invB, m, gpu.d_col_dense, 1, &alpha_zero, gpu.d_d, 1);
        nvtxRangePop();

        // 6. Ratio Test (CUB Implementation)
        // Compute ratios in a kernel: ratios[i] = xB[i] / d[i] (or DBL_MAX if invalid)
        nvtxRangePush("Ratio Test");
        compute_ratios_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_ratios, gpu.d_xB, gpu.d_d, m, PIVOT_TOL);
        nvtxRangePop();

        // Find min ratio and its index using CUB
        cub::DeviceReduce::ArgMin(gpu.d_temp_storage, gpu.temp_storage_bytes,
                                  gpu.d_ratios, gpu.d_arg_min_out, m);

        // Fetch result (async)
        cudaMemcpyAsync(gpu.h_arg_min_out, gpu.d_arg_min_out,
                        sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        double theta_min = gpu.h_arg_min_out->value;
        int leave = gpu.h_arg_min_out->key;

        if (theta_min >= DBL_MAX)
        {
            std::cout << "Problem unbounded\n";
            return {Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity()), iter};
        }

        // 7. Update Solution & Basis
        nvtxRangePush("Update Solution & Basis");
        update_solution_kernel<<<gpu.grid1D, gpu.block1D>>>(gpu.d_xB, gpu.d_d, leave, theta_min, m);

        // Copy leave index to device to allow device-side basis update
        cudaMemcpy(gpu.d_leave_idx, &leave, sizeof(int), cudaMemcpyHostToDevice);
        update_basis_device_kernel<<<1, 1>>>(gpu.d_basis_ids, gpu.d_leave_idx, gpu.d_arg_max_out);

        // Update Inverse (Sherman-Morrison)
        cublasDcopy(gpu.handle_cublas, m, gpu.d_invB + leave, m, gpu.d_row_p, 1);
        prepare_sherman_morrison_data<<<1, 1>>>(gpu.d_d, leave, gpu.d_scalar_val);
        cublasSetPointerMode(gpu.handle_cublas, CUBLAS_POINTER_MODE_DEVICE);
        cublasDger(gpu.handle_cublas, m, m, gpu.d_scalar_val, gpu.d_d, 1, gpu.d_row_p, 1, gpu.d_invB, m);
        cublasSetPointerMode(gpu.handle_cublas, CUBLAS_POINTER_MODE_HOST);

        basis[leave] = enter;
        nvtxRangePop();
    }

    std::cerr << "Warning: Hit iteration limit\n";
    Eigen::VectorXd xB(m);
    cudaMemcpy(xB.data(), gpu.d_xB, sizeof(double) * m, cudaMemcpyDeviceToHost);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < m; ++i)
        x(basis[i]) = std::max(0.0, xB(i));
    return {x, MAX_ITERS};
}

int main()
{
    int m, n, n_surplus, n_slack;
    double offset, M;
    if (!(std::cin >> m >> n >> n_surplus >> n_slack >> offset >> M))
        return 0;

    int last_col = n + n_surplus + m;
    Eigen::MatrixXd A(m, last_col);
    Eigen::VectorXd b(m), c(last_col);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < last_col; j++)
            std::cin >> A(i, j);
    for (int i = 0; i < m; i++)
        std::cin >> b(i);
    for (int i = 0; i < last_col; i++)
        std::cin >> c(i);

    c.tail(m - n_slack) *= M;

    std::pair<Eigen::VectorXd, int> result = simplex_method(A, b, c, last_col, m);
    Eigen::VectorXd z = result.first;
    int iterations = result.second;
    double optimum = c.dot(z);
    std::cout << std::scientific << std::uppercase << std::setprecision(10)
              << "Optimum found: " << (optimum + offset) << "\n"
              << "Iterations: " << iterations << "\n";
    return 0;
}