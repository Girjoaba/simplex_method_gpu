# Revised simplex algorithm in CUDA
Initial repository for the DPHPC course at ETH.

# How to compile and run the program
```bash
nvcc --std=c++20 src/solver.cu -o bin/solver.out -ccbin /usr/bin/g++-13 -lcublas
./bin/solver.out input/sample.txt
```

# Remarks
Works on Andrei's machine (Ubuntu x64, g++ 13.4, 1050 Ti, Driver 580, CUDA Toolkit 12.9)
Works on the student cluster (after adding module `cuda/13.0.2`)

# Optimizations
## Algorithmic
- [ ] Steepest edge
- [ ] Steepest edge with a recurrence
- [x] Quadratic update of B_inv without E
- [ ] Linear update of y and x_b
- [ ] Save c_b * B_inv directly to y_aug
## High-level:
- [ ] CUB reduction
- [ ] Explore different data storage
- [ ] Move the logic around between CPU and GPU
- [ ] Explore CUDA streams and std::async
## Low-level:
- [ ] Optimize kernels (warps, cache, sync, atomic, …)
- [ ] Tune BS differently for distinct tasks
- [ ] Combine kernels to avoid restarts
