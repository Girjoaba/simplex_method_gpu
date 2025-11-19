/**
 * Example usage of GLOPSolver (Google OR-Tools GLOP wrapper)
 */

#include "glop_solver.hpp"
#include <iostream>
#include <iomanip>

void print_solution(const Eigen::VectorXd& x, double z,
                   const std::string& status, int iters) {
    std::cout << "Status: " << status << '\n';
    std::cout << "Iterations: " << iters << '\n';
    std::cout << "Objective value: " << std::fixed << std::setprecision(4) << z << '\n';
    std::cout << "Solution: ";
    for (int i = 0; i < x.size(); ++i) {
        if (x(i) > 1e-6) {  // Only print non-zero values
            std::cout << "x" << i << " = " << x(i) << "  ";
        }
    }
    std::cout << '\n';
}

int main() {
    std::cout << "=== GLOP (Google OR-Tools) Simplex Solver ===\n\n";

    // Example problem from PDF (Lab setting)
    // max 700*xg + 900*xt
    // s.t. 500*xg + 3400*xt <= 6000
    //      250*xg + 200*xt <= 500
    //      xg, xt >= 0

    Eigen::MatrixXd A(2, 4);
    A << 500, 3400, 1, 0,    // First constraint with slack
         250, 200, 0, 1;      // Second constraint with slack

    Eigen::VectorXd b(2);
    b << 6000, 500;

    Eigen::VectorXd c(4);
    c << 700, 900, 0, 0;  // Objective coefficients

    std::cout << "Problem: maximize 700*xg + 900*xt\n";
    std::cout << "         subject to 500*xg + 3400*xt <= 6000\n";
    std::cout << "                    250*xg + 200*xt <= 500\n";
    std::cout << "                    xg, xt >= 0\n\n";

    std::cout << "--- Solving with GLOP ---\n";
    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();
    print_solution(x, z, status, iters);

    std::cout << "\n=== Configuration ===\n";
    std::cout << "Numerical tolerance: 1e-10\n";
    std::cout << "\nGLOP is the reference implementation for benchmarking custom solvers.\n";

    return 0;
}
