nvcc --std=c++20 src/solver.cu \
     -o bin/solver.out \
     -ccbin /usr/bin/g++-13 \
     -I ~/local/include \
     -lcublas -lcusolver \
     --expt-relaxed-constexpr