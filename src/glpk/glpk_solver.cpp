#include <glpk.h>
#include <iostream>

int main(int argc, char *argv[]) {
    glp_term_out(GLP_OFF); 
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file.mps\n";
        return 1;
    }
    const char *fname = argv[1];

    glp_prob *lp = glp_create_prob();
    glp_set_prob_name(lp, argv[1]);

    int err = glp_read_mps(lp, GLP_MPS_DECK, NULL, fname);
    if (err != 0) {
        std::cerr << "Error reading MPS file: " << err << "\n";
        glp_delete_prob(lp);
        return 2;
    }

    // Different solvers are available
    glp_simplex(lp, NULL);

    // Retrieve results
    int status = glp_get_status(lp);
    if (status == GLP_OPT) {
        int n = glp_get_num_cols(lp);  // number of variables
        double obj = glp_get_obj_val(lp);

        for (int i = 1; i <= n; ++i) {  // 1-based indexing
            double x_i = glp_get_col_prim(lp, i);
            std::cout << "x[" << i << "] = " << x_i << "\n";
        }

        std::cout << "Optimal objective: " << obj << "\n";
    } else {
        std::cout << "Problem status: " << status << "\n";
    }

    glp_delete_prob(lp);
    return 0;
}
