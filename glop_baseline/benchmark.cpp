#include "glop_solver.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
    int m, n_orig;
    int iterations;
    double time_ms;
    std::string status;
};

// Generate random LP problem: max c^T x s.t. Ax <= b, x >= 0
void generate_random_problem(int m, int n_orig, Eigen::MatrixXd& A,
                            Eigen::VectorXd& b, Eigen::VectorXd& c,
                            unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 10.0);

    int n = n_orig + m;  // Include slack variables

    // Generate constraint matrix (exclude slack columns)
    A = Eigen::MatrixXd(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n_orig; ++j) {
            A(i, j) = dis(gen);
        }
        // Add slack variables
        for (int j = n_orig; j < n; ++j) {
            A(i, j) = (j - n_orig == i) ? 1.0 : 0.0;
        }
    }

    // Generate RHS
    b = Eigen::VectorXd(m);
    for (int i = 0; i < m; ++i) {
        b(i) = dis(gen) * 10.0;
    }

    // Generate cost coefficients (positive for maximization)
    c = Eigen::VectorXd(n);
    for (int j = 0; j < n_orig; ++j) {
        c(j) = dis(gen);
    }
    // Slack variables have zero cost
    for (int j = n_orig; j < n; ++j) {
        c(j) = 0.0;
    }
}

BenchmarkResult run_benchmark(int m, int n_orig, const Eigen::MatrixXd& A,
                              const Eigen::VectorXd& b,
                              const Eigen::VectorXd& c) {
    GLOPSolver solver(A, b, c);

    auto start = Clock::now();
    auto [x, z, status, iters] = solver.solve(10000);
    auto end = Clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {m, n_orig, iters, time_ms, status};
}

void print_header() {
    std::cout << std::string(70, '=') << '\n';
    std::cout << "GLOP (Google OR-Tools) Simplex Solver Benchmark\n";
    std::cout << std::string(70, '=') << '\n';
}

void print_problem_header() {
    std::cout << "\n" << std::left << std::setw(15) << "Problem Size"
              << std::setw(15) << "Iterations"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Status" << '\n';
    std::cout << std::string(70, '-') << '\n';
}

void print_result(const BenchmarkResult& result) {
    std::string size_str = std::to_string(result.m) + "×" +
                          std::to_string(result.n_orig) +
                          " (+" + std::to_string(result.m) + " slack)";

    std::cout << std::left << std::setw(15) << size_str
              << std::setw(15) << result.iterations
              << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
              << std::setw(20) << result.status << '\n';
}

int main() {
    print_header();

    // Test on various problem sizes
    std::vector<std::pair<int, int>> problem_sizes = {
        {5, 10},     // Small
        {10, 20},    // Medium
        {20, 40},    // Large
        {50, 100},   // Very Large
        {100, 200},  // Extra Large
    };

    print_problem_header();

    for (const auto& [m, n_orig] : problem_sizes) {
        Eigen::MatrixXd A;
        Eigen::VectorXd b, c;
        generate_random_problem(m, n_orig, A, b, c);

        auto result = run_benchmark(m, n_orig, A, b, c);
        print_result(result);
    }

    std::cout << std::string(70, '=') << '\n';

    return 0;
}
