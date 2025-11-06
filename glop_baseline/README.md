# GLOP Baseline - Google OR-Tools Simplex Solver

**To Build**
```bash
make -j4
```
Or use all CPU cores:
```bash
make -j$(sysctl -n hw.ncpu)  # macOS
make -j$(nproc)              # Linux
```

### Build Options

# Build specific targets
make glop_baseline_lib   # Just the library
make glop_test           # Just tests
make glop_benchmark      # Just benchmark


**Conversion Flow**:
```
Eigen Matrices → GLOP LinearProgram → LPSolver → Extract Solution → Eigen Vectors
```

### GLOP Configuration

Default parameters set in `glop_solver.cpp`:
```cpp
GlopParameters params;
params.set_max_number_of_iterations(max_iter);
params.set_primal_feasibility_tolerance(eps);
params.set_dual_feasibility_tolerance(eps);
params.set_provide_strong_optimal_guarantee(true);
```

Run tests:
```bash
./glop_test
```

**Run benchmark**:
```bash
./glop_benchmark
```

## License

OR-Tools: Apache 2.0 License
- https://github.com/google/or-tools
- Copyright Google

## References

- [OR-Tools Documentation](https://developers.google.com/optimization)
- [GLOP Source Code](https://github.com/google/or-tools/tree/stable/ortools/glop)
- [LinearProgram API](https://or-tools.github.io/docs/cpp/classoperations__research_1_1glop_1_1LinearProgram.html)
- [LPSolver API](https://or-tools.github.io/docs/cpp/classoperations__research_1_1glop_1_1LPSolver.html)
- [RevisedSimplex API](https://or-tools.github.io/docs/cpp/classoperations__research_1_1glop_1_1RevisedSimplex.html)
