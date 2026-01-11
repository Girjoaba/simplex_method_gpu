/**
 * GLOP Canonical LP Solver
 *
 * Reads canonical LP format and solves using Google OR-Tools GLOP.
 * Compatible with test_solver_correctness.bash testing infrastructure.
 *
 * Input format (canonical file):
 *   Line 1: m n (number of constraints, number of variables)
 *   Next m lines: A matrix rows (space-separated)
 *   Next line: b vector (space-separated)
 *   Next line: c vector (space-separated)
 *
 * Problem formulation:
 *   Maximize: c^T x
 *   Subject to: Ax = b, x >= 0
 *
 * Output format (to stdout):
 *   Optimum found: <objective_value>
 *   (followed by solution if optimal)
 */

#include "glop_solver.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

constexpr double MAX_MATRIX_VALUE = 1e15;
constexpr double MIN_POSITIVE_VALUE = 1e-15;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <canonical_file>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    // Read dimensions
    int m, n;
    if (!(file >> m >> n)) {
        std::cerr << "Error: Failed to read dimensions m and n\n";
        return 1;
    }

    if (m <= 0 || n <= 0) {
        std::cerr << "Error: Invalid dimensions m=" << m << ", n=" << n
                  << " (must be positive)\n";
        return 1;
    }

    if (m > n) {
        std::cerr << "Warning: m > n (" << m << " > " << n
                  << "), problem may be overconstrained\n";
    }

    // Allocate matrices
    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m);
    Eigen::VectorXd c(n);

    // Read matrix A (row by row, matching canonical format)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> A(i, j))) {
                std::cerr << "Error: Failed to read A(" << i << "," << j << ")\n";
                return 1;
            }

            // Validate matrix values
            if (!std::isfinite(A(i, j))) {
                std::cerr << "Error: A(" << i << "," << j << ") is not finite\n";
                return 1;
            }
            if (std::abs(A(i, j)) > MAX_MATRIX_VALUE) {
                std::cerr << "Warning: A(" << i << "," << j << ") = " << A(i, j)
                          << " is very large\n";
            }
        }
    }

    // Read vector b
    for (int i = 0; i < m; ++i) {
        if (!(file >> b(i))) {
            std::cerr << "Error: Failed to read b(" << i << ")\n";
            return 1;
        }
        if (!std::isfinite(b(i))) {
            std::cerr << "Error: b(" << i << ") is not finite\n";
            return 1;
        }
    }

    // Read vector c
    for (int j = 0; j < n; ++j) {
        if (!(file >> c(j))) {
            std::cerr << "Error: Failed to read c(" << j << ")\n";
            return 1;
        }
        if (!std::isfinite(c(j))) {
            std::cerr << "Error: c(" << j << ") is not finite\n";
            return 1;
        }
    }

    file.close();

    // Solve using GLOP
    GLOPSolver solver(A, b, c, 1e-10);

    // Solve with high iteration limit (10000 matches CUDA solver)
    auto [x, obj_value, status, iterations] = solver.solve(10000);

    // Output results in format expected by test_solver_correctness.bash
    if (status == "optimal") {
        // Use high precision to match groundtruth comparison tolerance
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Optimum found: " << obj_value << '\n';

        // Optionally output solution (matches CUDA solver format)
        std::cout << std::scientific;
        for (int j = 0; j < n; ++j) {
            if (std::abs(x(j)) > MIN_POSITIVE_VALUE) {
                std::cout << "x[" << j << "] = " << x(j) << '\n';
            }
        }
        std::cout << "Iterations: " << iterations << '\n';

        return 0;
    } else if (status == "unbounded") {
        std::cout << "Problem unbounded.\n";
        return 2;
    } else if (status == "infeasible") {
        std::cout << "Problem infeasible.\n";
        return 3;
    } else {
        std::cout << "MAX_ITER exceeded or solver error. Status: " << status << '\n';
        return 4;
    }
}
