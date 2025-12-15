# Revised simplex algorithm in CUDA
Initial repository for the DPHPC course at ETH.

# How to compile and run the program

In order to compile all binaries, simpyl use:
```bash
make
```
Don't get confused by Eigen warnings, those just clutter the make output.

If you want to change compiler flags, please only work with the Makefile in order to allow for batch compilation before benchmarking.

IMPORTANT: All two-phase binaries should be prefixed with `tp_` and all big-m binaries with `bm_`. This is used during testing and benchmarking to derive the preprocessing steps, removing the prefix **will** crash.

## Testing
### Generating testing ground truth
Since the solvers require different inputs and we still compare with the gurobi output (We probably should instead compare with the netlib recorded optimal values!). You have to run the `scripts/generate_test_environment.bash` once. This is already done correctly and **can not** be done on the container, since it runs gurobi... If there's need to generate the grounId truth again, due to changes in the `src/gurobi/interface_*_gutobi.py` files report quickly in the chat and I can regnerate the values on my local machine.

### Running the correctness tests
If the groundtruth is available the tests can be run by running `scripts/correctness.bash {your_binary} {small|medium|large}`.
The size indicator (last argument) indicates on what size of problems should be benchmarked. Right now we determine size by simply looking at the diskspace of the `.mps` for a test. This will be redone to have more meaningful test suites. For the mean time I would just run it on `medium` to get a reasonable correctness check.

I will also add a time indication per test run in order to provide more information on how well the test does. 


# TODO:
- [ ] include unbounded problems as correct
- [ ] take larger problems (use gurobi license)
- [ ] gpu opt

# Optimizations
## Algorithmic
- [ ] Steepest edge
- [ ] Steepest edge with a recurrence
- [x] Quadratic update of B_inv without E
- [x] Linear update of y and x_b
## High-level:
- [x] CUB reduction
- [ ] Move the logic around between CPU and GPU
- [ ] Employ CUDA streams
## Low-level:
- [ ] Optimize kernels (warps, cache, sync, atomic, …)
- [ ] Tune BS differently for distinct tasks
- [ ] Combine kernels to avoid restarts (CUDA graphs)
## Other:
- [ ] What if x_b_t < 0 for some t (compute_theta)
- [ ] Division by a small number (two spots)
- [ ] Explore different data storage (sparse, CSR)
- [ ] Branch CUBLAS for "real"
