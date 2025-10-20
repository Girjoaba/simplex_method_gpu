#include "matrix_types.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

bool is_basis_column(const Matrix& A, size_t col_index)
{
    size_t nonnegative_count = 0;
    for (size_t i = 0; i < A.rows(); ++i) {
        if (A[i][col_index] > 0.0) {
            nonnegative_count++;

            if (nonnegative_count > 1) {
                return false;
            }
        }
    }
    return true;
}

std::vector<size_t> get_basis_indices(const Matrix& A)
{
    std::vector<size_t> basis_indices;
    for (size_t j = 0; j < A.cols(); ++j) {
        if (is_basis_column(A, j)) {
            basis_indices.push_back(j);
        }
    }
    return basis_indices;
}

Matrix initialize_basis(const Matrix& A, const std::vector<size_t>& basis_indices)
{
    size_t m = A.rows();
    size_t n = A.cols();
    Matrix B(m, m);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            B[i][j] = A[i][basis_indices[j]];
        }
    }

    return B;
}

Vector objective_basis(const Vector& c, const std::vector<size_t>& basis_indices)
{
    size_t m = basis_indices.size();
    Vector cb(m);
    for (size_t i = 0; i < m; ++i) {
        cb[i] = c[basis_indices[i]];
    }
    return cb;
}

size_t choose_leaving_variable(const Vector& xb, const Vector& d)
{
    size_t leaving_index = std::numeric_limits<size_t>::max();
    double min_ratio = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < xb.size(); ++i) {
        if (d[i] > 0) {
            double ratio = xb[i] / d[i];
            if (ratio < min_ratio) {
                min_ratio = ratio;
                leaving_index = i;
            }
        }
    }

    return leaving_index;
}

void update_basis(Matrix& B, const Matrix& A, size_t entering_index, size_t leaving_index)
{
    for (size_t i = 0; i < B.rows(); ++i) {
        B[i][leaving_index] = A[i][entering_index];
    }
}

Vector revised_simplex(const Matrix& A, const Vector& b, const Vector& c)
{
    size_t m = A.rows();
    size_t n = A.cols();

    std::vector<size_t> basis_indices = get_basis_indices(A);
    Matrix B = initialize_basis(A, basis_indices);
    Vector cb = objective_basis(c, basis_indices);
    Vector xb(n); // Initialize solution vector with zeros

    while (true) {
        Matrix B_inv = B.inverse();

        // Determine the entering variable
        Vector c_tilde = cb * B_inv * A - c;
        size_t entering_index = std::min_element(c_tilde.begin(), c_tilde.end()) - c_tilde.begin();
        xb = B_inv * b;

        if (c_tilde[entering_index] >= 0) {
            // Optimal solution found
            Vector tmp(n);
            for (size_t i = 0; i < m; ++i) {
                tmp[basis_indices[i]] = xb[i];
            }
            return tmp;
        }

        // Determine the leaving variable
        Vector d = B_inv * A.column(entering_index);
        size_t leaving_index = choose_leaving_variable(xb, d);

        if (leaving_index == std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Unbounded solution.");
        }

        update_basis(B, A, entering_index, leaving_index);
        cb[leaving_index] = c[entering_index];
        basis_indices[leaving_index] = entering_index;
    }
}


// possible improvements:
// - implement a more efficient way to update B_inv instead of computing it from scratch each iteration
// - use column major storage for matrices



int main() {
    // Define the problem data
    // Maximize z = 3x1 + 2x2
    // Subject to:
    // x1 + x2 <= 4
    // 2x1 + x2 <= 5
    // x1, x2 >= 0

    Matrix A = {
        {1, 1, 1, 0},
        {2, 1, 0, 1}
    };

    Vector b = {4, 5};
    Vector c = {3, 2, 0, 0};

    try {
        Vector solution = revised_simplex(A, b, c);
        std::cout << "Optimal solution found:" << std::endl;
        solution.print();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}