#include <glpk.h>
#include <vector>
#include <string>
#include <iostream>

struct LPProblem {
    std::vector<std::vector<double>> A;  // m x n matrix
    std::vector<double> b;               // RHS vector
    std::vector<double> c;               // Objective coefficients
    std::vector<double> lb;              // Lower bounds
    std::vector<double> ub;              // Upper bounds
    int m;                               // Number of constraints
    int n;                               // Number of variables
};

LPProblem lp_from_mps(const std::string& filename) {
    LPProblem lpProblem;
    // Make glp silent
    glp_term_out(GLP_OFF);

    glp_prob* lp = glp_create_prob();
    int ret = glp_read_mps(lp, GLP_MPS_FILE, NULL, filename.c_str());
    if (ret != 0) {
        std::cerr << "Error reading MPS file: " << filename << std::endl;
        exit(1);
    }

    // Get dimensions
    int m = glp_get_num_rows(lp);
    int n = glp_get_num_cols(lp);
    lpProblem.m = m;
    lpProblem.n = n;

    // Resize vectors/matrix
    lpProblem.A.resize(m, std::vector<double>(n, 0.0));
    lpProblem.b.resize(m);
    lpProblem.c.resize(n);
    lpProblem.lb.resize(n);
    lpProblem.ub.resize(n);

    for (int j = 1; j <= n; ++j) {
        lpProblem.lb[j-1] = glp_get_col_lb(lp, j);
        lpProblem.ub[j-1] = glp_get_col_ub(lp, j);
    }

    for (int i = 1; i <= m; ++i) {
        int type = glp_get_row_type(lp, i);
        if (type == GLP_UP)        lpProblem.b[i-1] = glp_get_row_ub(lp, i);
        else if (type == GLP_LO)   lpProblem.b[i-1] = glp_get_row_lb(lp, i);
        else if (type == GLP_FX)   lpProblem.b[i-1] = glp_get_row_lb(lp, i);
        else if (type == GLP_FR)   lpProblem.b[i-1] = 0.0;
    }

    // Fill c (objective coefficients)
    for (int j = 1; j <= n; ++j) {
        lpProblem.c[j-1] = glp_get_obj_coef(lp, j);
    }

    // Fill A matrix (GLPK uses 1-based indexing for sparse representation)
    int* ind = new int[n + 1];       // column indices
    double* val = new double[n + 1]; // values

    for (int i = 1; i <= m; ++i) {
        int nnz = glp_get_mat_row(lp, i, ind, val);
        for (int k = 1; k <= nnz; ++k) {
            int colIndex = ind[k];       // column number
            double value = val[k];       // matrix value
            lpProblem.A[i-1][colIndex-1] = value;
        }
    }

    delete[] ind;
    delete[] val;

    // Free GLPK problem
    glp_delete_prob(lp);
    return lpProblem;
}

void output_lp(LPProblem *lp) {
    // Output matches form of solver.cu
    // Output of m n 
    std::cout << lp->m << lp->n << "\n";

    // Output of A
    for (const auto& row : lp->A) {
        for (double v : row) std::cout << v << " ";
    }
    std::cout << "\n";

    // Output of b
    for (double v : lp->b) std::cout << v << " ";
    std::cout << "\n";

    // Output of c
    for (double v : lp->c) std::cout << v << " ";
    std::cout << "\n";
}

int main() {
    // Example .mps file
    LPProblem lp = lp_from_mps("problems/timtab1.mps");
    output_lp(&lp);
    return 0;
}

