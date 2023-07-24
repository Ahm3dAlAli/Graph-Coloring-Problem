import requests
from io import StringIO
import numpy as np
import time
from gurobipy import *
import pandas as pd

# Declare instance sets
small_instances = {'myciel3'}  # , 'anna', 'myciel7', 'queen8_8'}
medium_instances = {'fpsol2.i.3', 'zeroin.i.1', 'le450_5a'}
large_instances = {'inithx.i.1', 'latin_square_10', 'queen16_16'}
diff_nodes_sim_edges = {'fpsol2.i.1', 'fpsol2.i.2', 'fpsol2.i.3'}
inc_edges_const_nodes = {'le450_15a', 'le450_25a', 'le450_5a'}

options = {
    'small': small_instances,
    'medium': medium_instances,
    'large': large_instances,
    'diff_nodes': diff_nodes_sim_edges,
    'inc_edges': inc_edges_const_nodes
}

def read_dimacs_graph(instance_name):
    # Fetch DIMACS data from the web
    url = f"https://mat.tepper.cmu.edu/COLOR/instances/{instance_name}.col"
    response = requests.get(url)
    data = StringIO(response.text)

    # Initialize an empty adjacency list
    adjacency_list = {}

    for line in data:
        # Strip leading/trailing whitespace
        line = line.strip()

        # DIMACS lines start with a character that denotes the type of line.
        if line.startswith('c'):
            continue
        elif line.startswith('p'):
            _, _, vertices, _ = line.split()
            for i in range(1, int(vertices) + 1):
                adjacency_list[i] = []
        elif line.startswith('e'):
            _, start, end = line.split()
            adjacency_list[int(start)].append(int(end))
            adjacency_list[int(end)].append(int(start))

    return adjacency_list


def Intial_MIP_Formulation(graph):
    # MIP graph coloring problem formulation
    n = len(graph)
    model = Model("mip_graph_coloring")

    # Define variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # if vertex v is assigned color i
    w = model.addVars(n, vtype=GRB.BINARY, name="w")  # if color i is used

    # Define objective
    model.setObjective(w.sum(), GRB.MINIMIZE)

    # Unique Color Constraint: each vertex is assigned only a single color
    model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), name="unique_color")

    # Proper Coloring Constraint: no two adjacent vertices share the same color
    for v, neighbors in graph.items():
        for u in neighbors:
            model.addConstrs((x[v - 1, i] + x[u - 1, i] <= w[i] for i in range(n)), name=f"proper_coloring_{v}_{u}")

    # Optimize model
    start_time = time.time()
    model.optimize()
    elapsed_time = time.time() - start_time

    if model.Status == GRB.INFEASIBLE:
        print("The problem is not solvable. Please check your constraints.")
        return None, None, elapsed_time
    elif model.Status == GRB.OPTIMAL:
        print("An optimal solution was found.")
        colors = [x[i, :].X for i in range(n)]
        num_colors = model.objVal
        return colors, num_colors, elapsed_time
    else:
        print("The problem status is: ", model.Status)
        return None, None, elapsed_time


def Quadratically_Constrained_MIP_Formulation(graph):
    n = len(graph)
    model = Model("forward_elimination")

    # Define variables
    x = model.addVars(n+1, n+1, vtype=GRB.BINARY, name="x")  # plus one for 1-indexing
    w = model.addVars(n+1, vtype=GRB.BINARY, name="w")  # plus one for 1-indexing

    # Define objective
    model.setObjective(w.sum(), GRB.MINIMIZE)

    # Constraints for each node to have one color
    model.addConstrs((x.sum(i, '*') == 1 for i in range(1, n+1)), name="unique_color")  # plus one for 1-indexing

    # Constraints for adjacent nodes not to share a color
    for v in range(1, n+1):  # plus one for 1-indexing
        for u in graph[v]:
            model.addConstrs((x[v, i] + x[u, i] <= w[i] for i in range(1, n+1)), name=f"proper_coloring_{v}_{u}")  # plus one for 1-indexing

    # Constraints and quadratic constraints
    constraints = []
    quadratic_constraints = []
    
    # Add constraints and quadratic constraints
    for v in range(1, n+1):  # plus one for 1-indexing
        for i in range(1, n+1):  # plus one for 1-indexing
            for j in range(1, n+1):  # plus one for 1-indexing
                if i != j:
                    qc = model.addQConstr(x[v, i] * x[v, j] == 0, name=f"quadratic_{v}_{i}_{j}")
                    quadratic_constraints.append(qc)
                if j in graph[v]:
                    qc = model.addQConstr(x[v, i] * x[j, i] == 0, name=f"quadratic_{v}_{i}_{j}")
                    quadratic_constraints.append(qc)

    # xui * wi = xui , ∀u ∈ V, ∀i ∈ {1, . . . , n}
    for u in range(1, n+1):
        model.addConstrs((x[u, i] * w[i] == x[u, i] for i in range(1, n+1)), name=f"xui_wi_{u}")

    # x2ui − xui = 0, ∀u ∈ V, ∀i ∈ {1, . . . , n}
    for u in range(1, n+1):
        model.addConstrs(((x[u, i]*x[u, i] - x[u, i] == 0) for i in range(1, n+1)), name=f"x2ui_{u}")

    # w2i − wi = 0, ∀i ∈ {1, . . . , n}
    for i in range(1, n+1):
        model.addQConstr((w[i]*w[i] - w[i] == 0), name=f"w2i_{i}")

        # Optimize model
    start_time = time.time()
    model.optimize()
    elapsed_time = time.time() - start_time

    if model.Status == GRB.INFEASIBLE:
        print("The problem is not solvable. Please check your constraints.")
        return None, None, elapsed_time
    elif model.Status == GRB.OPTIMAL:
        print("An optimal solution was found.")
        colors = [x[i, :].X for i in range(n)]
        num_colors = model.objVal
        return colors, num_colors, elapsed_time
    else:
        print("The problem status is: ", model.Status)
        return None, None, elapsed_time
    
 

def Constraint_Elimination(model, constraints, quadratic_constraints):
    performance = []
    lower_bounds = []
    constraint_names = []
    threshold = 1
    
    # Save the names of the constraints for the table
    for constr in constraints:
        constraint_names.append(constr.getAttr(GRB.Attr.ConstrName))

    for qc in quadratic_constraints:
        model.addQConstr(qc)
        start_time = time.time()
        model.setParam('TimeLimit', threshold)
        model.optimize()
        elapsed_time = time.time() - start_time
        performance.append(elapsed_time)
        lower_bounds.append(model.objVal if model.status == GRB.OPTIMAL else None)

    to_remove = [i for i, perf in enumerate(performance) if perf > threshold]
    for i in reversed(to_remove):
        model.remove(constraints[i])

    model.setParam('TimeLimit', float('inf'))  # remove the time limit
    model.optimize()

    # Append the final results to the table
    performance.append(model.Runtime)
    lower_bounds.append(model.objVal if model.status == GRB.OPTIMAL else None)
    constraint_names.append("Final Objective")
    
    # Create the DataFrame
    data = {
        "Constraint": constraint_names,
        "Elapsed Time": performance,
        "Lower Bound": lower_bounds
    }
    df = pd.DataFrame(data)
    
    return model.objVal, model.Status, df



# Get user's choice
choice = input("Choose option (small, medium, large, diff_nodes, inc_edges): ")

# Read graph data for instances in chosen set
if choice in options:
    for instance_name in options[choice]:
        adjacency_list = read_dimacs_graph(instance_name)
        colors, num_colors, elapsed_time = mip_graph_coloring(adjacency_list)
        print(f"For {instance_name}, used {num_colors} colors in {elapsed_time} seconds.")
else:
    print(f"Invalid option: {choice}")



def lp_relaxation(graph):
    n = len(graph)
    model = Model("lp_relaxation")

    # Define variables
    x = model.addVars(n+1, n+1, lb=0, ub=1, name="x")  # plus one for 1-indexing
    w = model.addVars(n+1, lb=0, ub=1, name="w")  # plus one for 1-indexing

    # Define objective
    model.setObjective(w.sum(), GRB.MINIMIZE)

    # Constraints for each node to have one color
    model.addConstrs((x.sum(i, '*') == 1 for i in range(1, n+1)), name="unique_color")  # plus one for 1-indexing

    # Constraints for adjacent nodes not to share a color
    for v in range(1, n+1):  # plus one for 1-indexing
        for u in graph[v]:
            model.addConstrs((x[v, i] + x[u, i] <= w[i] for i in range(1, n+1)), name=f"proper_coloring_{v}_{u}")  # plus one for 1-indexing

    # Additional quadratic constraints
    for v in range(1, n+1):  # plus one for 1-indexing
        for i in range(1, n+1):  # plus one for 1-indexing
            for j in range(1, n+1):  # plus one for 1-indexing
                if i != j:
                    model.addQConstr(x[v, i] * x[v, j] == 0, name=f"quadratic_{v}_{i}_{j}")
                if j in graph[v]:
                    model.addQConstr(x[v, i] * x[j, i] == 0, name=f"quadratic_{v}_{i}_{j}")

    # xui * wi = xui , ∀u ∈ V, ∀i ∈ {1, . . . , n}
    for u in range(1, n+1):
        model.addConstrs((x[u, i] * w[i] == x[u, i] for i in range(1, n+1)), name=f"xui_wi_{u}")

    # x2ui − xui = 0, ∀u ∈ V, ∀i ∈ {1, . . . , n}
    for u in range(1, n+1):
        model.addConstrs(((x[u, i]*x[u, i] - x[u, i] == 0) for i in range(1, n+1)), name=f"x2ui_{u}")

    # w2i − wi = 0, ∀i ∈ {1, . . . , n}
    for i in range(1, n+1):
        model.addQConstr((w[i]*w[i] - w[i] == 0), name=f"w2i_{i}")

    model.optimize()
    return model.objVal, model.Status


result_value, result_status = lp_relaxation(adjacency_list)
print("Objective Value (LP Relaxation): ", result_value)
print("Model Status: ", result_status)


choice = input("Choose option (small, medium, large, diff_nodes, inc_edges): ")

# Read graph data for instances in chosen set
if choice in options:
    for instance_name in options[choice]:
        adjacency_list = read_dimacs_graph(instance_name)
        colors, num_colors, elapsed_time = mip_graph_coloring(adjacency_list)
        print(f"For {instance_name}, used {num_colors} colors in {elapsed_time} seconds.")
        result_value, result_status = forward_elimination(adjacency_list)

        # Get the quadratic constraints from the forward_elimination model
        quadratic_constraints = []
        for v in range(1, len(adjacency_list) + 1):
            for i in range(1, len(adjacency_list) + 1):
                for j in range(1, len(adjacency_list) + 1):
                    if i != j:
                        qc = model.getConstrByName(f"quadratic_{v}_{i}_{j}")
                        quadratic_constraints.append(qc)

        # Apply LP Relaxation to the model with the tightest quadratic constraints
        if result_status == GRB.Status.OPTIMAL:
            lp_result_value, lp_result_status = lp_relaxation(adjacency_list, quadratic_constraints)
            print("Objective Value (Quadratic Constraints): ", result_value)
            print("Model Status (Quadratic Constraints): ", result_status)
            print("Objective Value (LP Relaxation): ", lp_result_value)
            print("Model Status (LP Relaxation): ", lp_result_status)

else:
    print(f"Invalid option: {choice}")


def lp_relaxation(model):
    # Change variable types to continuous for LP relaxation
    for var in model.getVars():
        if var.vtype == GRB.BINARY:
            var.vtype = GRB.CONTINUOUS

    model.update()



def sdp_relaxation(model):


    # Modify the existing quadratic constraints to SDP relaxation counterparts
    n = len(model.getVars())  # Get the number of decision variables

    for v in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    Xij = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"X_{v}_{i}_{j}")
                    model.addConstr(Xij == model.getVarByName(f"x{v}_{i}") * model.getVarByName(f"x{v}_{j}"))

    for u in range(1, n+1):
        for v in range(1, n+1):
            for i in range(1, n+1):
                if i != j and v in graph[u]:
                    Xij = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"X_{u}_{v}_{i}")
                    model.addConstr(Xij == model.getVarByName(f"x{u}_{i}") * model.getVarByName(f"x{v}_{i}"))

    model.update()



def compare_relaxations(initial_model, quadratic_model):
    sdp_model = initial_model.copy()
    lp_model = quadratic_model.copy()

    # Apply SDP Relaxation to the sdp_model
    sdp_relaxation(sdp_model)

    # Apply LP Relaxation to the lp_model
    lp_relaxation(lp_model)

    # Solve both models and compare the results
    sdp_model.optimize()
    lp_model.optimize()

    sdp_optimal_value = sdp_model.objVal
    sdp_status = sdp_model.Status
    sdp_execution_time = sdp_model.Runtime

    lp_optimal_value = lp_model.objVal
    lp_status = lp_model.Status
    lp_execution_time = lp_model.Runtime

    # Compare results and choose the best relaxation method
    if sdp_status == GRB.OPTIMAL and lp_status == GRB.OPTIMAL:
        if sdp_optimal_value <= lp_optimal_value:
            print("SDP Relaxation provides the best solution.")
            return sdp_model
        else:
            print("LP Relaxation provides the best solution.")
            return lp_model
    elif sdp_status == GRB.OPTIMAL:
        print("SDP Relaxation provides the best solution.")
        return sdp_model
    elif lp_status == GRB.OPTIMAL:
        print("LP Relaxation provides the best solution.")
        return lp_model
    else:
        print("Neither relaxation method provides a feasible solution.")
        return None

# Example usage
initial_model = create_initial_mip_model()  # Replace with your function to create the MIP model
quadratic_model = create_quadratic_model(initial_model)  # Replace with your function to create the quadratic model

best_relaxation_model = compare_relaxations(initial_model, quadratic_model)

if best_relaxation_model is not None:
    # Solve the best relaxation model
    best_relaxation_model.optimize()

    # Get the optimal solution and other relevant information
    optimal_solution = best_relaxation_model.objVal
    execution_time = best_relaxation_model.Runtime

    # Print the results
    print(f"Optimal Solution: {optimal_solution}")
    print(f"Execution Time: {execution_time}")