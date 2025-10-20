#ifndef MATRIX_TYPES_HPP
#define MATRIX_TYPES_HPP

#include <algorithm>
#include <iostream>
#include <vector>

// forward declaration
class Matrix;

class Vector {

public:
    std::vector<double> data;
    Vector(size_t size)
        : data(size, 0.0)
    {
    }
    Vector(const std::vector<double>& d)
        : data(d)
    {
    }
    Vector(std::initializer_list<double> init)
        : data(init)
    {
    }

    size_t size() const { return data.size(); }

    double& operator[](size_t index) { return data[index]; }
    const double& operator[](size_t index) const { return data[index]; }
    void print() const
    {
        for (const auto& val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    Vector operator+(const Vector& other) const
    {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector sizes do not match for addition.");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    Vector operator-(const Vector& other) const
    {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector sizes do not match for subtraction.");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    Vector scalarMultiply(double scalar) const
    {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }

    Vector operator*(const Matrix& A) const;
};

class Matrix {

public:
    std::vector<std::vector<double>> data;
    Matrix(size_t rows, size_t cols)
        : data(rows, std::vector<double>(cols, 0.0))
    {
    }
    Matrix(const std::vector<std::vector<double>>& d)
        : data(d)
    {
    }
    Matrix(std::initializer_list<std::initializer_list<double>> init)
    {
        for (const auto& row : init) {
            data.emplace_back(row);
        }
    }

    size_t rows() const { return data.size(); }
    size_t cols() const { return data[0].size(); }

    std::vector<double>& operator[](size_t index) { return data[index]; }
    const std::vector<double>& operator[](size_t index) const { return data[index]; }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }

    void print() const
    {
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    Vector column(size_t col_index) const
    {
        Vector column(rows());
        for (size_t i = 0; i < rows(); ++i) {
            column[i] = data[i][col_index];
        }
        return column;
    }

    Matrix transpose() const
    {
        Matrix result(cols(), rows());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const
    {
        if (cols() != other.rows()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        Matrix result(rows(), other.cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < other.cols(); ++j) {
                for (size_t k = 0; k < cols(); ++k) {
                    result[i][j] += data[i][k] * other[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const
    {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }
        Matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i][j] = data[i][j] + other[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const
    {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }
        Matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i][j] = data[i][j] - other[i][j];
            }
        }
        return result;
    }

    Matrix scalarMultiply(double scalar) const
    {
        Matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const
    {
        return scalarMultiply(scalar);
    }

    Vector operator*(const std::vector<double>& vec) const
    {
        if (cols() != vec.size()) {
            std::cout << "rows: " << rows() << " cols: " << cols() << " vec size: " << vec.size() << std::endl;
            throw std::invalid_argument("Matrix columns must match vector size for multiplication.");
        }
        Vector result(rows());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }
        return result;
    }

    Vector operator*(const Vector& vec) const {
        return (*this) * vec.data;
    }

    Matrix inverse() const
    {
        size_t n = rows();
        if (n != cols()) {
            throw std::invalid_argument("Only square matrices can be inverted.");
        }

        Matrix augmented(n, 2 * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                augmented[i][j] = data[i][j];
            }
            augmented[i][n + i] = 1.0;
        }

        for (size_t i = 0; i < n; ++i) {
            double pivot = augmented[i][i];
            if (pivot == 0) {
                throw std::runtime_error("Matrix is singular and cannot be inverted.");
            }
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= pivot;
            }
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (size_t j = 0; j < 2 * n; ++j) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        Matrix inv(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inv[i][j] = augmented[i][n + j];
            }
        }
        return inv;
    }
};

// Now that Matrix is a complete type, provide the implementation of
// Vector::operator*(const Matrix&)
inline Vector Vector::operator*(const Matrix& A) const
{
    // right multiplication: Vector * Matrix
    if (size() != A.rows()) {
        throw std::invalid_argument("Vector size must match Matrix rows for multiplication.");
    }

    Vector result(A.cols());
    for (size_t j = 0; j < A.cols(); ++j) {
        for (size_t i = 0; i < size(); ++i) {
            result[j] += data[i] * A[i][j];
        }
    }
    return result;
}

#endif // MATRIX_TYPES_HPP