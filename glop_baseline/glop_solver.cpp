#include "glop_solver.hpp"
#include "ortools/lp_data/lp_data.h"
#include "ortools/glop/lp_solver.h"

using namespace operations_research::glop;

GLOPSolver::GLOPSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       const Eigen::VectorXd& c, double eps)
    : A_(A), b_(b), c_(c), eps_(eps), m_(A.rows()), n_(A.cols()) {

    if (m_ != b.size()) {
        throw std::invalid_argument("A.rows() must equal b.size()");
    }
    if (n_ != c.size()) {
        throw std::invalid_argument("A.cols() must equal c.size()");
    }
}

std::tuple<Eigen::VectorXd, double, std::string, int>
GLOPSolver::solve(int max_iter) {
    // Build LinearProgram from Eigen matrices
    LinearProgram lp;

    // Create variables: x_j >= 0
    for (int j = 0; j < n_; ++j) {
        ColIndex col = lp.CreateNewVariable();
        lp.SetVariableBounds(col, 0.0, kInfinity);
        lp.SetObjectiveCoefficient(col, c_(j));
    }

    // Create constraints: Ax = b (equality constraints)
    for (int i = 0; i < m_; ++i) {
        RowIndex row = lp.CreateNewConstraint();
        lp.SetConstraintBounds(row, b_(i), b_(i));  // lower = upper for equality

        // Set coefficients (only non-zero for efficiency)
        for (int j = 0; j < n_; ++j) {
            if (std::abs(A_(i, j)) > eps_) {
                lp.SetCoefficient(row, ColIndex(j), A_(i, j));
            }
        }
    }

    lp.SetMaximizationProblem(true);  // Maximization
    lp.CleanUp();  // Remove duplicates, sort

    // Configure solver parameters
    LPSolver solver;
    GlopParameters params;
    params.set_max_number_of_iterations(max_iter);
    params.set_primal_feasibility_tolerance(eps_);
    params.set_dual_feasibility_tolerance(eps_);
    params.set_provide_strong_optimal_guarantee(true);
    solver.SetParameters(params);

    // Solve
    ProblemStatus status = solver.Solve(lp);

    // Extract results
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n_);
    double obj = 0.0;
    std::string status_str;
    int iters = solver.GetNumberOfSimplexIterations();

    // Map GLOP status to string
    if (status == ProblemStatus::OPTIMAL) {
        status_str = "optimal";
        obj = solver.GetObjectiveValue();

        // Extract solution vector
        const DenseRow& vals = solver.variable_values();
        for (int j = 0; j < n_; ++j) {
            x(j) = vals[ColIndex(j)];
        }
    } else if (status == ProblemStatus::PRIMAL_UNBOUNDED) {
        status_str = "unbounded";
    } else if (status == ProblemStatus::PRIMAL_INFEASIBLE) {
        status_str = "infeasible";
    } else if (status == ProblemStatus::DUAL_UNBOUNDED) {
        status_str = "unbounded";  // Dual unbounded = primal infeasible, but we report unbounded
    } else {
        status_str = "max_iterations";
    }

    return {x, obj, status_str, iters};
}
