#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <utility>   // for std::pair

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include <cub/cub.cuh>
#include <Eigen/Dense>

// ---------------------------
// Constants
// ---------------------------
constexpr double OPTIMALITY_TOL = 1e-6;
constexpr int BATCH_SIZE = 30;
constexpr int MAX_ITERS = 100000;

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
// Kernels (Modified for Device Guards)
// ---------------------------

__global__ void set_identity_kernel(double *Matrix, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * m)
        Matrix[idx] = ((idx % m) == (idx / m)) ? 1.0 : 0.0;
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
        d_s[d_basis_ids[idx]] = -1.0e20;
}

__global__ void gather_vector_kernel(double *d_out, const double *d_in, const int *d_indices, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
        d_out[idx] = d_in[d_indices[idx]];
}

__global__ void fused_gather_column_kernel(
    double *d_dense_out,
    const double *d_val,
    const int *d_row_ind,
    const int *d_col_ptr,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp)
{
    // GUARD: If optimal, do not gather (keeps d_col_dense as 0)
    if (d_arg_max_kvp->value <= OPTIMALITY_TOL)
        return;

    int enter_col = d_arg_max_kvp->key;
    int start = d_col_ptr[enter_col];
    int end = d_col_ptr[enter_col + 1];

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (end - start); idx += gridDim.x * blockDim.x)
    {
        d_dense_out[d_row_ind[start + idx]] = d_val[start + idx];
    }
}

__global__ void compute_ratios_kernel(double *ratios, const double *xB, const double *d, int m, double tol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        double div = d[idx];
        ratios[idx] = (div > tol) ? (xB[idx] / div) : DBL_MAX;
    }
}

__global__ void update_solution_graph_kernel(
    double *__restrict__ xB,
    const double *d,
    const cub::KeyValuePair<int, double> *d_arg_min_kvp,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp, // Added for guard
    int m)
{
    // GUARD: If optimal, do not update solution
    if (d_arg_max_kvp->value <= OPTIMALITY_TOL)
        return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int leave_idx = d_arg_min_kvp->key;
    double theta = d_arg_min_kvp->value;

    if (idx < m)
    {
        if (idx == leave_idx)
            xB[idx] = theta;
        else
            xB[idx] = xB[idx] - theta * d[idx];
    }
}

__global__ void extract_inverse_row_kernel(
    double *d_row_out,
    const double *d_invB,
    const cub::KeyValuePair<int, double> *d_arg_min_kvp,
    int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = d_arg_min_kvp->key;
    if (idx < m)
    {
        d_row_out[idx] = d_invB[row_idx + idx * m];
    }
}

__global__ void prepare_sherman_morrison_data_graph(
    double *d_d,
    double *d_scalar_val,
    const cub::KeyValuePair<int, double> *d_arg_min_kvp,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp // Added for guard
)
{
    if (threadIdx.x == 0)
    {
        // GUARD: If optimal, set alpha to 0.0 so cublasDger does nothing
        if (d_arg_max_kvp->value <= OPTIMALITY_TOL)
        {
            *d_scalar_val = 0.0;
        }
        else
        {
            int p = d_arg_min_kvp->key;
            double pivot = d_d[p];
            *d_scalar_val = -1.0 / pivot;
            d_d[p] = pivot - 1.0;
        }
    }
}

__global__ void update_basis_device_graph_kernel(
    int *d_basis_ids,
    const cub::KeyValuePair<int, double> *d_arg_min_kvp,
    const cub::KeyValuePair<int, double> *d_arg_max_kvp)
{
    if (threadIdx.x == 0)
    {
        // GUARD: If optimal, do not change basis
        if (d_arg_max_kvp->value > OPTIMALITY_TOL)
        {
            d_basis_ids[d_arg_min_kvp->key] = d_arg_max_kvp->key;
        }
    }
}

// ---------------------------
// GPU Resources & Graphs
// ---------------------------

struct CusolverResources
{
    cusolverDnHandle_t handle;
    cublasHandle_t handle_cublas;
    cusparseHandle_t handle_cusparse;
    cudaStream_t stream;

    cudaGraph_t graph_pricing, graph_update;
    cudaGraphExec_t exec_pricing, exec_update;
    bool graphs_created = false;

    // ... (Sparse Matrix pointers same as before)
    double *d_A_val;
    int *d_A_rowInd;
    int *d_A_colPtr;
    cusparseSpMatDescr_t matA_desc;
    cusparseDnVecDescr_t vecLambda_desc;
    cusparseDnVecDescr_t vecS_desc;
    void *d_buffer_mv;
    size_t bufferSize_mv;

    // ... (Dense Data pointers same as before)
    double *d_col_dense, *d_A, *d_B, *d_x, *d_xB, *d_d, *d_c, *d_s;
    int *d_ipiv, *d_basis_ids, *d_info;
    double *d_work;
    int lwork;

    // Sherman Morrison
    double *d_invB, *d_lambda, *d_row_p, *d_scalar_val;
    double *d_ratios;

    // CUB
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::KeyValuePair<int, double> *d_arg_max_out, *h_arg_max_out;
    cub::KeyValuePair<int, double> *d_arg_min_out, *h_arg_min_out;

    int m, n, nnz;
    dim3 block1D, grid1D, gridFull, blockSize, gridSize;
    double alpha_one = 1.0, alpha_zero = 0.0, alpha_mone = -1.0;

    CusolverResources(int m_dim, int n_dim, const CscMatrix &A_csc) : m(m_dim), n(n_dim), nnz(A_csc.nnz)
    {
        cusolverDnCreate(&handle);
        cublasCreate(&handle_cublas);
        cusparseCreate(&handle_cusparse);
        cudaStreamCreate(&stream);

        cusolverDnSetStream(handle, stream);
        cublasSetStream(handle_cublas, stream);
        cusparseSetStream(handle_cusparse, stream);

        // Memory Allocations (Same as provided code)
        cudaMalloc((void **)&d_A_val, sizeof(double) * nnz);
        cudaMalloc((void **)&d_A_rowInd, sizeof(double) * nnz);
        cudaMalloc((void **)&d_A_colPtr, sizeof(double) * (n + 1));
        cudaMemcpyAsync(d_A_val, A_csc.values.data(), sizeof(double) * nnz, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_A_rowInd, A_csc.row_indices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_A_colPtr, A_csc.col_ptr.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice, stream);

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
        cudaMalloc((void **)&d_ratios, sizeof(double) * m);

        cusparseCreateDnVec(&vecLambda_desc, m, d_lambda, CUDA_R_64F);
        cusparseCreateDnVec(&vecS_desc, n, d_s, CUDA_R_64F);

        cusparseSpMV_bufferSize(handle_cusparse, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha_one, matA_desc, vecLambda_desc, &alpha_one, vecS_desc, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_mv);
        cudaMalloc(&d_buffer_mv, bufferSize_mv);

        cusolverDnDgetrf_bufferSize(handle, m, m, d_B, m, &lwork);
        cudaMalloc((void **)&d_work, sizeof(double) * lwork);

        // CUB Output & Temp
        cudaMalloc((void **)&d_arg_max_out, sizeof(cub::KeyValuePair<int, double>));
        cudaMallocHost((void **)&h_arg_max_out, sizeof(cub::KeyValuePair<int, double>));
        cudaMalloc((void **)&d_arg_min_out, sizeof(cub::KeyValuePair<int, double>));
        cudaMallocHost((void **)&h_arg_min_out, sizeof(cub::KeyValuePair<int, double>));

        size_t s1 = 0, s2 = 0;
        cub::DeviceReduce::ArgMax(nullptr, s1, d_s, d_arg_max_out, n);
        cub::DeviceReduce::ArgMin(nullptr, s2, d_ratios, d_arg_min_out, m);
        temp_storage_bytes = std::max(s1, s2);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        block1D = dim3(256);
        grid1D = dim3((m + block1D.x - 1) / block1D.x);
        gridFull = dim3((m * m + block1D.x - 1) / block1D.x);
        blockSize = dim3(32, 8);
        gridSize = dim3((m + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    }

    void captureGraphs()
    {
        if (graphs_created)
            return;

        // --- 1. CAPTURE PRICING GRAPH ---
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        gather_vector_kernel<<<grid1D, block1D, 0, stream>>>(d_x, d_c, d_basis_ids, m);
        cublasDgemv(handle_cublas, CUBLAS_OP_T, m, m, &alpha_one, d_invB, m, d_x, 1, &alpha_zero, d_lambda, 1);
        cudaMemcpyAsync(d_s, d_c, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream);
        cusparseSpMV(handle_cusparse, CUSPARSE_OPERATION_TRANSPOSE, &alpha_mone, matA_desc, vecLambda_desc, &alpha_one, vecS_desc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_mv);
        mask_basis_kernel<<<grid1D, block1D, 0, stream>>>(d_s, d_basis_ids, m);
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_s, d_arg_max_out, n, stream);

        cudaStreamEndCapture(stream, &graph_pricing);
        cudaGraphInstantiate(&exec_pricing, graph_pricing, NULL, NULL, 0);

        // --- 2. CAPTURE UPDATE GRAPH ---
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        cudaMemsetAsync(d_col_dense, 0, sizeof(double) * m, stream);

        // Fused Gather
        fused_gather_column_kernel<<<grid1D, block1D, 0, stream>>>(d_col_dense, d_A_val, d_A_rowInd, d_A_colPtr, d_arg_max_out);

        cublasDgemv(handle_cublas, CUBLAS_OP_N, m, m, &alpha_one, d_invB, m, d_col_dense, 1, &alpha_zero, d_d, 1);

        compute_ratios_kernel<<<grid1D, block1D, 0, stream>>>(d_ratios, d_xB, d_d, m, 1e-6);
        cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_ratios, d_arg_min_out, m, stream);

        // Update Solution
        update_solution_graph_kernel<<<grid1D, block1D, 0, stream>>>(d_xB, d_d, d_arg_min_out, d_arg_max_out, m);

        // Update Basis
        update_basis_device_graph_kernel<<<1, 1, 0, stream>>>(d_basis_ids, d_arg_min_out, d_arg_max_out);

        extract_inverse_row_kernel<<<grid1D, block1D, 0, stream>>>(d_row_p, d_invB, d_arg_min_out, m);

        // Prepare Sherman-Morrison (Guarded: sets scalar to 0.0 if optimal)
        prepare_sherman_morrison_data_graph<<<1, 1, 0, stream>>>(d_d, d_scalar_val, d_arg_min_out, d_arg_max_out);

        cublasSetPointerMode(handle_cublas, CUBLAS_POINTER_MODE_DEVICE);
        cublasDger(handle_cublas, m, m, d_scalar_val, d_d, 1, d_row_p, 1, d_invB, m);
        cublasSetPointerMode(handle_cublas, CUBLAS_POINTER_MODE_HOST);

        cudaStreamEndCapture(stream, &graph_update);
        cudaGraphInstantiate(&exec_update, graph_update, NULL, NULL, 0);

        graphs_created = true;
    }

    ~CusolverResources()
    {
        if (graphs_created)
        {
            cudaGraphExecDestroy(exec_pricing);
            cudaGraphDestroy(graph_pricing);
            cudaGraphExecDestroy(exec_update);
            cudaGraphDestroy(graph_update);
        }
        cusparseDestroySpMat(matA_desc);
        cusparseDestroyDnVec(vecLambda_desc);
        cusparseDestroyDnVec(vecS_desc);
        cusparseDestroy(handle_cusparse);
        cusolverDnDestroy(handle);
        cublasDestroy(handle_cublas);
        cudaStreamDestroy(stream);
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
    // Standard re-inversion (Same as provided)
    assemble_basis_kernel<<<gpu.gridSize, gpu.blockSize, 0, gpu.stream>>>(gpu.d_A, gpu.d_B, gpu.d_basis_ids, m, m);
    cusolverDnDgetrf(gpu.handle, m, m, gpu.d_B, m, gpu.d_work, gpu.d_ipiv, gpu.d_info);
    cudaMemcpyAsync(gpu.d_xB, b_host.data(), sizeof(double) * m, cudaMemcpyHostToDevice, gpu.stream);
    cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, 1, gpu.d_B, m, gpu.d_ipiv, gpu.d_xB, m, gpu.d_info);
    set_identity_kernel<<<gpu.gridFull, gpu.block1D, 0, gpu.stream>>>(gpu.d_invB, m);
    cusolverDnDgetrs(gpu.handle, CUBLAS_OP_N, m, m, gpu.d_B, m, gpu.d_ipiv, gpu.d_invB, m, gpu.d_info);
}

// ---------------------------
// Main Loop (Batched)
// ---------------------------

std::pair<Eigen::VectorXd, int> simplex_method(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c, int n, int m)
{
    nvtxRangePush("First Iteration");
    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i)
        basis[i] = n - m + i;

    CscMatrix sparse_A = dense_to_csc(A);
    CusolverResources gpu(m, n, sparse_A);

    cudaMemcpyAsync(gpu.d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice, gpu.stream);
    cudaMemcpyAsync(gpu.d_c, c.data(), sizeof(double) * n, cudaMemcpyHostToDevice, gpu.stream);
    cudaMemcpyAsync(gpu.d_basis_ids, basis.data(), sizeof(int) * m, cudaMemcpyHostToDevice, gpu.stream);

    compute_basis_inverse(gpu, b, m);
    gpu.captureGraphs();
    
    nvtxRangePop();
    for (int iter = 0; iter < MAX_ITERS; iter += BATCH_SIZE)
    {
        if (iter > 0 && iter % BATCH_SIZE == 0)
        {
            nvtxRangePush("Basis Reinversion");
            compute_basis_inverse(gpu, b, m);
            nvtxRangePop();
        }

        // --- BATCHED EXECUTION ---
        for (int k = 0; k < BATCH_SIZE; ++k)
        {
            cudaGraphLaunch(gpu.exec_pricing, gpu.stream);
            cudaGraphLaunch(gpu.exec_update, gpu.stream);
        }

        // --- CHECK TERMINATION ONCE PER BATCH ---
        // We read the LAST iteration's status.
        // If the algorithm finished at step k < BATCH, subsequent steps were identity no-ops.
        cudaMemcpyAsync(gpu.h_arg_max_out, gpu.d_arg_max_out, sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost, gpu.stream);
        cudaMemcpyAsync(gpu.h_arg_min_out, gpu.d_arg_min_out, sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost, gpu.stream);

        cudaStreamSynchronize(gpu.stream); // Sync only once every BATCH_SIZE iterations

        // Check 1: Optimality
        if (gpu.h_arg_max_out->value <= OPTIMALITY_TOL)
        {
            nvtxRangePush("EXIT: Optimality");
            Eigen::VectorXd xB(m);
            cudaMemcpy(xB.data(), gpu.d_xB, sizeof(double) * m, cudaMemcpyDeviceToHost);
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            cudaMemcpy(basis.data(), gpu.d_basis_ids, sizeof(int) * m, cudaMemcpyDeviceToHost);
            for (int i = 0; i < m; ++i)
                x(basis[i]) = std::max(0.0, xB(i));

            nvtxRangePop();
            return {x, iter};
        }

        // Check 2: Unbounded
        if (gpu.h_arg_min_out->value >= DBL_MAX)
        {
            std::cout << "Problem unbounded\n";
            return {Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity()), iter};
        }
    }

    return {Eigen::VectorXd::Zero(n), MAX_ITERS};
}

// Main function remains exactly the same as in your snippet
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