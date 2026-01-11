#ifndef GLOP_SOLVER_HPP
#define GLOP_SOLVER_HPP

#include <Eigen/Dense>
#include <string>
#include <tuple>

/**
 * Wrapper around Google OR-Tools GLOP (Google's Linear Optimizer)
 * Provides same interface as SimplexSolver for benchmarking.
 * Solves LP in canonical augmented form: min c^T x s.t. Ax = b, x >= 0
 */
class GLOPSolver {
public:
    /**
     * @param A m x n constraint matrix (includes slack variables)
     * @param b m x 1 right-hand side vector
     * @param c n x 1 cost coefficients
     * @param eps Numerical tolerance (default: 1e-10)
     */
    GLOPSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, double eps = 1e-10);

    /**
     * Solve the LP problem using GLOP's revised simplex implementation.
     *
     * @param max_iter Maximum iterations (default: 1000)
     * @return tuple of (solution, objective_value, status, iteration_count)
     *         status: "optimal", "unbounded", "infeasible", or "max_iterations"
     */
    std::tuple<Eigen::VectorXd, double, std::string, int> solve(int max_iter = 1000);

private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    Eigen::VectorXd c_;
    double eps_;
    int m_, n_;
};

#endif // GLOP_SOLVER_HPP
