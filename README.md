# simplex_method_gpu
Initial repository for the DPHPC course at ETH.


# To run the code
```bash
nvcc --std=c++20 solver.cu -o simplex -ccbin /usr/bin/g++-13 -lcublas && ./simplex simple_input.txt
```

Tested that it works on CUDA 12.9