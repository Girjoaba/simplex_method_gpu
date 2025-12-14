# Revised Simplex Algorithm in CUDA

## How to handle input, compile the program and run it
```bash
./compile
./extract afiro
./run afiro
```

## How to perform the two-phase method
0. assume that n >= m
1. assume for simplicity that LB>=0 and UP=+inf for all vars; this is checked for in convert.py
2. multiply each constraint that has a negative RHS by -1; remember to reverse the direction
3. add surplus vars for >=, slack vars for <= and artificial vars for >= and =
4. in practice, you make <= rows the first rows of A, append surplus columns and then the identity matrix
5. solve z = max -\sum a_i where a_i's are artificial vars
6. if z < 0, the problem is infeasible
7. otherwise assume for simplicity that the solution does not contain artificial vars
8. update c_B and solve the original problem