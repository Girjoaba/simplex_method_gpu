output_file="$1"

if [ -z "$output_file" ]; then
  output_file="solver"
fi

nvcc --std=c++20 src/solver.cu -o bin/${output_file}.out -ccbin /usr/bin/g++-13 -lcublas -lcusolver