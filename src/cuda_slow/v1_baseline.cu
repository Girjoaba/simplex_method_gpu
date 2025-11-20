#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <iomanip>

#define MAX_ITERS 200000

const double EPSILON = 1e-10;
#include <limits>
#include <vector>
#include <Eigen/Dense>

Eigen::VectorXd simplex_method(const Eigen::MatrixXd& A,
                               const Eigen::VectorXd& b,
                               const Eigen::VectorXd& c,
                               int n, int m) {

    // Expect the identity on the right part!
    std::vector<int> basis(m);
    for (int i = 0; i < m; ++i) {
      basis[i] = n - m + i;  // points to column in A
    } 

    Eigen::MatrixXd B(m, m);
    for (int j = 0; j < m; ++j) {
        B.col(j) = A.col(basis[j]); // columns in my basis
    }

    Eigen::VectorXd cB(m);
    for (int i = 0; i < m; ++i) {
        cB(i) = c(basis[i]);    // subset of the cost coeff.
    }

    // =============================== |
    // --------- Main Loop ----------- |
    // =============================== |
    // See Algorithm 4. https://web.stanford.edu/class/msande310/Simplex-ref1.pdf

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        /* Determine entering variable */
        Eigen::MatrixXd Binv   = B.inverse();
        Eigen::VectorXd lambda = Binv.transpose() * cB;         // y[m] <- cB * B^-1
        Eigen::VectorXd s      = c - A.transpose() * lambda;    // e[n] <- [1, y] * [-c; A]
        
        std::vector<char> inBasis(n, 0);
        for (int i = 0; i < m; ++i) inBasis[basis[i]] = 1;
        
        // Most positive reduced cost
        Eigen::Index enter = -1;
        double s_max = EPSILON;  
        for (int j = 0; j < n; ++j) {
            if (!inBasis[j] && s(j) > s_max) {
                s_max = s(j);
                enter = j;
            }
        }
        
        Eigen::VectorXd xB = Binv * b;
        
        // If no more reductions, exit
        if (enter == -1) {
            std::cout << "Iteration: " << iter << "\n";
            Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < m; ++i) {
                x(basis[i]) = std::max(0.0, xB(i));
            }
            return x;
        }
        
        // how much each basis variable change if I increase the entering variable
        Eigen::VectorXd d = Binv * A.col(enter);
        
        // Find leaving variable
        Eigen::Index leave = -1;
        double theta_min = std::numeric_limits<double>::infinity();
        // theta_min = min { xB(i) / d(i) : d(i) > 0 }
        // the furthest you can go before a basis variable hits 0 (violates a constraint) 
        for (int i = 0; i < m; ++i) {
            if (d(i) > EPSILON) {
                double theta = xB(i) / d(i);
                if (theta < theta_min) {
                    theta_min = theta;
                    leave = i;
                }
            }
        }
        
        if (leave == -1) {
            std::cerr << "Problem unbounded\n";
            return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        }
        
        // Pivot
        basis[leave] = enter;
        B.col(leave) = A.col(enter);
        cB(leave)    = c(enter);
    }
    
    std::cerr << "Warning: Hit iteration limit\n";
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd xB = B.inverse() * b;
    for (int i = 0; i < m; ++i) x(basis[i]) = std::max(0.0, xB(i));
    return x;
}


int main() {
    int n, m;
    // starts with n, m
    std::cin >> m >> n;
    
    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m), c(n);

    // followed by A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> A(i, j);
        }
    }

    // Then, b
    for (int i = 0; i < m; i++) {
        std::cin >> b(i);
    }

    // Finally c
    for (int i = 0; i < n; i++) {
        std::cin >> c(i);
    }
    
    
    // std::cout << "DEBUG: First element of A: " << A(0,0) << "\n";
    // std::cout << "DEBUG: First element of b: " << b(0) << "\n";
    // std::cout << "DEBUG: Last element of c: " << c(n-1) << "\n"; // Should be -M or 0
    
    Eigen::VectorXd z = simplex_method(A, b, c, n, m);
    double optimum = c.dot(z);  // Compute c^T * z
    // std::cout << "Output:\n" << z.transpose() << "\n";
    std::cout << std::setprecision(15) << "Optimum found: " << optimum << "\n";

}