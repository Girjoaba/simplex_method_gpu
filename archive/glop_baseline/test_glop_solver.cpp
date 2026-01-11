/**
 * Test suite for GLOPSolver - validate GLOP correctness
 */

#include "glop_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>

bool is_close(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) < tolerance;
}

class GLOPSolverTest : public ::testing::Test {};

/**
 * Test Lab example from PDF (pages 23-25)
 * max 700*xg + 900*xt
 * s.t. 500*xg + 3400*xt <= 6000
 *      250*xg + 200*xt <= 500
 * Expected: xg ≈ 0.667, xt ≈ 1.667, z ≈ 1967
 */
TEST_F(GLOPSolverTest, LabExample) {
    Eigen::MatrixXd A(2, 4);
    A << 500, 3400, 1, 0,
         250, 200, 0, 1;

    Eigen::VectorXd b(2);
    b << 6000, 500;

    Eigen::VectorXd c(4);
    c << 700, 900, 0, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "optimal") << "Expected optimal solution";
    EXPECT_TRUE(is_close(x(0), 0.666666667)) << "Expected xg ≈ 0.667, got " << x(0);
    EXPECT_TRUE(is_close(x(1), 1.666666667)) << "Expected xt ≈ 1.667, got " << x(1);
    EXPECT_TRUE(is_close(z, 1966.666667, 1.0)) << "Expected z ≈ 1967, got " << z;
    EXPECT_GT(iters, 0) << "Should take at least one iteration";
}

/**
 * Test simple 2D LP
 * max 3*x1 + 2*x2
 * s.t. x1 + x2 <= 4, x1 <= 2, x2 <= 3
 * Expected: x1 = 2, x2 = 2, z = 10
 */
TEST_F(GLOPSolverTest, Simple2DProblem) {
    Eigen::MatrixXd A(3, 5);
    A << 1, 1, 1, 0, 0,
         1, 0, 0, 1, 0,
         0, 1, 0, 0, 1;

    Eigen::VectorXd b(3);
    b << 4, 2, 3;

    Eigen::VectorXd c(5);
    c << 3, 2, 0, 0, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "optimal");
    EXPECT_TRUE(is_close(x(0), 2.0)) << "Expected x1 = 2, got " << x(0);
    EXPECT_TRUE(is_close(x(1), 2.0)) << "Expected x2 = 2, got " << x(1);
    EXPECT_TRUE(is_close(z, 10.0)) << "Expected z = 10, got " << z;
}

/**
 * Test unbounded LP
 * max x1 + x2
 * s.t. -x1 + x2 <= 1
 */
TEST_F(GLOPSolverTest, UnboundedProblem) {
    Eigen::MatrixXd A(1, 3);
    A << -1, 1, 1;

    Eigen::VectorXd b(1);
    b << 1;

    Eigen::VectorXd c(3);
    c << 1, 1, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "unbounded") << "Expected unbounded, got " << status;
}

/**
 * Test degenerate basic solution
 * max x1 + x2
 * s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
 * Expected: x1 = 1, x2 = 1, z = 2
 */
TEST_F(GLOPSolverTest, DegenerateProblem) {
    Eigen::MatrixXd A(3, 5);
    A << 1, 1, 1, 0, 0,
         1, 0, 0, 1, 0,
         0, 1, 0, 0, 1;

    Eigen::VectorXd b(3);
    b << 2, 1, 1;

    Eigen::VectorXd c(5);
    c << 1, 1, 0, 0, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "optimal");
    EXPECT_TRUE(is_close(x(0), 1.0));
    EXPECT_TRUE(is_close(x(1), 1.0));
    EXPECT_TRUE(is_close(z, 2.0));
}

/**
 * Test single variable problem
 * max 5*x1
 * s.t. x1 <= 3
 * Expected: x1 = 3, z = 15
 */
TEST_F(GLOPSolverTest, SingleVariable) {
    Eigen::MatrixXd A(1, 2);
    A << 1, 1;

    Eigen::VectorXd b(1);
    b << 3;

    Eigen::VectorXd c(2);
    c << 5, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "optimal");
    EXPECT_TRUE(is_close(x(0), 3.0));
    EXPECT_TRUE(is_close(z, 15.0));
}

/**
 * Test initial basis already optimal
 * max 0*x1 + 0*x2
 * s.t. x1 + x2 <= 1
 * Expected: x1 = 0, x2 = 0, z = 0
 */
TEST_F(GLOPSolverTest, InitialBasisOptimal) {
    Eigen::MatrixXd A(1, 3);
    A << 1, 1, 1;

    Eigen::VectorXd b(1);
    b << 1;

    Eigen::VectorXd c(3);
    c << 0, 0, 0;

    GLOPSolver solver(A, b, c);
    auto [x, z, status, iters] = solver.solve();

    EXPECT_EQ(status, "optimal");
    EXPECT_TRUE(is_close(x(0), 0.0));
    EXPECT_TRUE(is_close(x(1), 0.0));
    EXPECT_TRUE(is_close(z, 0.0));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
