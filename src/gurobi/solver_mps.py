import gurobipy as gp

model = gp.read("test/input/e226.mps")
model.optimize()

if model.status == gp.GRB.OPTIMAL:
    # Get the objective constant
    obj_const = model.ObjCon
    true_objective = model.objVal - obj_const
    
    print(f"Gurobi reported objective: {model.objVal}")
    print(f"Objective constant: {obj_const}")
    print(f"True objective value: {true_objective}")
else:
    print(f"Optimization status: {model.status}")