import requests
from io import StringIO
import numpy as np
import time
import pandas as pd
from mosek.fusion import *
import mosek
import time
from bs4 import BeautifulSoup
from itertools import chain, combinations
import re
import sys
import matplotlib.pyplot as plt
import itertools
pd.set_option('display.max_columns', None)

s = StringIO()
env = mosek.Env()
task = env.Task()
task.set_Stream(mosek.streamtype.log, print) # Print log to console 
task.putintparam(mosek.iparam.log, 3) # Increase verbosity


# Declare instance sets #myciel3.col','myciel4.col'} 'queen5_5.col','queen6_6.col'}
small_instances = {'myciel3.col','myciel4.col'}
medium_instances = {'myciel5.col'}
#large_instances = {'anna.col','david.col'}
#diff_nodes_sim_edges = {'fpsol2.i.1.col', 'fpsol2.i.2.col'}
#inc_edges_const_nodes = {'le450_15a.col', 'le450_25a.col'}
options = {
    'small': small_instances,
    'medium': medium_instances
    #'large': large_instances
    #'diff_nodes': diff_nodes_sim_edges,
    #'inc_edges': inc_edges_const_nodes
}

def Read_DIMACS_Graph(instance_name):
    # Fetch DIMACS data from the web
    url = f"https://mat.tepper.cmu.edu/COLOR/instances/{instance_name}"
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

def Get_Instance_Metadata(instance_name):
    # Fetch the data from the webpage
    url = "https://mat.tepper.cmu.edu/COLOR/instances.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # This is a pattern to find the filename, nodes, edges, optimal coloring, and source.
    pattern = r'<a href="instances/([^"]+)">\s+([^<]+)\s+</a>\s+\((\d+),(\d+)\),\s+([^<]+),\s+<a href="#[^"]+">([^<]+)</a>'
    matches = re.findall(pattern, str(soup))

    # Extract values from the regex matches and filter for the desired instance name
    data = [{'file': match[0],
             'name': match[1].strip(),
             'nodes': int(match[2]),
             'edges': int(match[3]),
             'optimal_coloring': int(match[4].strip()),
             'source': match[5]} for match in matches if match[1].strip() == instance_name]

    return data[0] if data else None

def compute_gap(Actual_Optimal_Value, Approximate_Optimal_Value):
    gap = ((Actual_Optimal_Value - Approximate_Optimal_Value) / Actual_Optimal_Value) * 100.0
    return gap

def Baseline_MIP_Formulation(graph):
    # MIP graph coloring problem formulation
    n = len(graph)
    # Define a model
    model = Model("mip_gpc")
    constraint_count = 0
    # Define variables
    x = model.variable("x", [n, n], Domain.binary())  # if vertex v is assigned color i
    w = model.variable("w", n, Domain.binary())  # if color i is used

    # Define objective
    model.objective(ObjectiveSense.Minimize, Expr.sum(w))

    # Unique Color Constraint: each vertex is assigned only a single color
    for i in range(n):
        model.constraint(f"unique_color_{i}", Expr.sum(x.slice([i,0],[i+1,n])), Domain.equalsTo(1))
        constraint_count += 1

    # Proper Coloring Constraint: no two adjacent vertices share the same color
    added_edges = set()
    for v, neighbors in graph.items():
        for u in neighbors:
            if (u,v) not in added_edges:
                for i in range(n):
                    model.constraint(f"proper_coloring_{v}_{u}_{i}", Expr.sub(Expr.add([x.index(v - 1, i), x.index(u - 1, i)]), w.index(i)), Domain.lessThan(0.0))
                    constraint_count += 1
                added_edges.add((u,v))
                added_edges.add((v,u))
    return model,constraint_count

def LP_Relaxation_Formulation(graph):
    # LP relaxation of graph coloring problem formulation
    n = len(graph)
    model = Model("lp_gpc")
    constraint_count = 0

    # Define variables
    x = model.variable("x", [n, n], Domain.inRange(0, 1.0))  # if vertex v is assigned color i
    w = model.variable("w", n, Domain.inRange(0, 1.0))  # if color i is used

    
    # Define objective
    model.objective(ObjectiveSense.Minimize, Expr.sum(w))

    # Unique Color Constraint: each vertex is assigned only a single color
    for i in range(n):
        model.constraint(f"unique_color_{i}", Expr.sum(x.slice([i,0],[i+1,n])), Domain.equalsTo(1))
        constraint_count += 1

    # Proper Coloring Constraint: no two adjacent vertices share the same color
    # Proper Coloring Constraint: no two adjacent vertices share the same color
    added_edges = set()
    for v, neighbors in graph.items():
        for u in neighbors:
            if (u,v) not in added_edges:
                for i in range(n):
                    model.constraint(f"proper_coloring_{v}_{u}_{i}", Expr.sub(Expr.add([x.index(v - 1, i), x.index(u - 1, i)]), w.index(i)), Domain.lessThan(0.0))
                    constraint_count += 1
                added_edges.add((u,v))
                added_edges.add((v,u))

    return model,constraint_count

def Baseline_SDP_Formulation(graph):
    n = len(graph)
    dim = n**2 + n + 1
    constraint_count =0

    # Define a model
    model = Model("Baseline_SDP")  

    # Define the matrix variable for Z
    Z = model.variable('Z',Domain.inPSDCone(dim))

    
    # Objective function
    model.objective(ObjectiveSense.Minimize, Expr.sum(Z.slice([(n**2)+n, n**2], [(n**2)+n, (n**2)+n-1])))

    # 1. n∑ i=1 Z[n^2 + n + 1, n(v − 1) + i ] = 1 ∀v ∈ V
    for v in graph.keys():
        model.constraint(f'Unique_Color_Usage_{v}',Expr.sum(Z.slice([n**2+n, n*(v-1)], [n**2+n+1, n*(v-1)+n])), Domain.equalsTo(1.0))
        constraint_count += 1

    # 2. Z[n^2 + n + 1, n(u − 1) + i ] + Z[n^2 + n + 1, n(v − 1) + i ] - Z[n^2 + n + 1, n^2 + i ] <= 0 ∀(u, v ) ∈ E, i ∈ {1, . . . , n}
    added_edges = set()
    for v, neighbors in graph.items():
        for u in neighbors:
            if (u,v) not in added_edges:
                for i in range(n):
                    sum_expr = Expr.sub(Expr.add(Z.index(n**2+n, n*(v-1)+i),   
                                                Z.index(n**2+n, n*(u-1)+i)),
                                    Z.index(n**2+n, n**2+i))
                    model.constraint(f'Color_{i+1}_Between_Edge_{v,u}',sum_expr, Domain.lessThan(0.0))  
                    constraint_count += 1
                added_edges.add((u,v))
                added_edges.add((v,u))
    
    
    # 3. Z[n^2 + n + 1, n(v − 1) + i ] -Z[n(v − 1) + i , n(v − 1) + i ]= 0 ∀v ∈ V, i ∈ {1, . . . , n}
    for v in graph.keys():
        for i in range(n):
            model.constraint(f'Binary_Enforce_Node_{v,i}',Expr.sub(Z.index(n**2+n, n*(v-1)+i), Z.index(n*(v-1)+i, n*(v-1)+i)), Domain.equalsTo(0.0))
            constraint_count += 1
    # 4. Z[n^2 + n + 1, n^2 + i ] -Z[n^2 + i , n^2 + i ] = 0∀, i ∈ {1, . . . , n}
    for i in range(n):
        model.constraint(f'Binary_Enforce_Color_{i}',Expr.sub(Z.index(n**2+n, n**2+i),Z.index(n**2+i, n**2+i)), Domain.equalsTo(0.0))
        constraint_count += 1

   
    # 5. Z[n^2 + n + 1, n^2 + n + 1] = 1
    model.constraint(f'SDP_Scalar',Z.index(n**2+n, n**2+n), Domain.equalsTo(1.0))
    constraint_count += 1

    return model,constraint_count

def Quadratically_Enhanced_SDP_Formulation(graph,constraints_to_include):
    n = len(graph)
    dim = n**2 + n + 1 
    constraint_count=0


    # Define a model
    model = Model("Enhanced_SDP")  

    # Define the matrix variable for Z
    Z = model.variable('Z',Domain.inPSDCone(dim))

    
    # Objective function
    model.objective(ObjectiveSense.Minimize, Expr.sum(Z.slice([(n**2)+n, n**2], [(n**2)+n, (n**2)+n-1])))

    # 1. n∑ i=1 Z[n^2 + n + 1, n(v − 1) + i ] = 1 ∀v ∈ V
    for v in graph.keys():
        model.constraint(f'Unique_Color_Usage_{v}',Expr.sum(Z.slice([n**2+n, n*(v-1)], [n**2+n+1, n*(v-1)+n])), Domain.equalsTo(1.0))
        constraint_count += 1
    

    # 2. Z[n^2 + n + 1, n(u − 1) + i ] + Z[n^2 + n + 1, n(v − 1) + i ] - Z[n^2 + n + 1, n^2 + i ] <= 0 ∀(u, v ) ∈ E, i ∈ {1, . . . , n}
    added_edges = set()
    for v, neighbors in graph.items():
        for u in neighbors:
            if (u,v) not in added_edges:
                for i in range(n):
                    sum_expr = Expr.sub(Expr.add(Z.index(n**2+n, n*(v-1)+i),   
                                                Z.index(n**2+n, n*(u-1)+i)),
                                    Z.index(n**2+n, n**2+i))
                    model.constraint(f'Color_{i+1}_Between_Edge_{v,u}',sum_expr, Domain.lessThan(0.0))  
                    constraint_count += 1
                added_edges.add((u,v))
                added_edges.add((v,u))
    
    # 3. Z[n^2 + n + 1, n(v − 1) + i ] -Z[n(v − 1) + i , n(v − 1) + i ]= 0 ∀v ∈ V, i ∈ {1, . . . , n}
    for v in graph.keys():
        for i in range(n):
            model.constraint(f'Binary_Enforce_Node_{v,i}',Expr.sub(Z.index(n**2+n, n*(v-1)+i), Z.index(n*(v-1)+i, n*(v-1)+i)), Domain.equalsTo(0.0))
            constraint_count += 1


    # QE1  xvi*xui=0, Z[n(v − 1) + i , n(v − 1) + j] = 0  ∀v ∈ V, ∀(i , j) ∈ {1, . . . , n}, i ̸ = j
    if "vertex_unique_color" in constraints_to_include:
        for v in graph.keys():
            for i in range(n):
                for j in range(i + 1, n):  # Only consider i != j
                    model.constraint(Z.index(n*(v-1)+i, n*(v-1)+j), Domain.equalsTo(0.0))
                    constraint_count += 1
    # QE2 xui*xvi=0, Z[n(v − 1) + i , n(u − 1) + i ] = 0 ∀(u, v ) ∈ E, ∀i ∈ {1, . . . , n}
    if "adjacent_vertex" in constraints_to_include:
        for (v, neighbors) in graph.items():
            for u in neighbors:  # For each neighbor of v
                for i in range(n):
                    model.constraint(Z.index( n*(v-1)+i, n*(u-1)+i), Domain.equalsTo(0.0))
                    constraint_count += 1
    # QE3 xui*wu=xui, Z[n2 + i , n(u − 1) + i ] = Z[n2 + n + 1, n(u − 1) + i ] ∀u ∈ V, ∀i ∈ {1, . . . , n}
    if "vertex_to_global_color" in constraints_to_include:
        for u in graph.keys():
            for i in range(n):
                model.constraint(Expr.sub(Z.index(n**2+i,n*(u-1)+i),Z.index(n**2+n,n*(u-1)+i)), Domain.equalsTo(0.0)) 
                constraint_count += 1

    # 4. Z[n^2 + n + 1, n^2 + i ] -Z[n^2 + i , n^2 + i ] = 0∀, i ∈ {1, . . . , n}
    for i in range(n):
        model.constraint(f'Binary_Enforce_Color_{i}',Expr.sub(Z.index(n**2+n, n**2+i),Z.index(n**2+i, n**2+i)), Domain.equalsTo(0.0))
        constraint_count += 1
   
    # 5. Z[n^2 + n + 1, n^2 + n + 1] = 1
    model.constraint(f'SDP_Scalar',Z.index(n**2+n, n**2+n), Domain.equalsTo(1.0))
    constraint_count += 1

    
    return model,constraint_count 


def Quadratic_Enhancemnts_Elimination(graph, Actual_Optimal_Value, sdp_sol,sdp_time):
    n = len(graph)
    best_result = None
    best_reformulation = []
    quadratic_enhancements = ["vertex_unique_color", "adjacent_vertex", "vertex_to_global_color"]
    result_reform = []
    baseline_gap = compute_gap(Actual_Optimal_Value, sdp_sol)
    baseline_time = sdp_time
    best_gap =[]
    current_gap=baseline_gap
    current_time=baseline_time

    avg_neighbors = sum(len(neighbors) for neighbors in graph.values()) / len(graph.keys())


    # Generate all combinations: 3 singles, 3 doubles, 1 triple = 7 combinations
    all_combinations = []
    for r in range(1, 4):
        for combination in itertools.combinations(quadratic_enhancements, r):
            all_combinations.append(list(combination))
    
    for enhancements in all_combinations:
        model,num_constr = Quadratically_Enhanced_SDP_Formulation(graph, constraints_to_include=enhancements)
        
        # Setting solver parameters and solving
        model.setSolverParam("intpntCoTolPfeas", 1e-4)
        model.setSolverParam("intpntCoTolDfeas", 1e-4)
        model.setSolverParam("intpntCoTolInfeas", 1e-4)
        start_time = time.time()
        model.solve()
        end_time = time.time()
        new_time = end_time - start_time
        Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
        index_of_one = np.where(np.round(Z_values[-1], 1) == 1.)[0][0]
        sol = np.sum(Z_values[-1][index_of_one - 3: index_of_one])
        new_gap = compute_gap(Actual_Optimal_Value, sol)
        beta = new_gap/baseline_gap
        gamma= new_time/baseline_time

        additional_constraints = 0
        if "vertex_unique_color" in enhancements:
            additional_constraints += len(graph.keys()) * (n * (n - 1) // 2)
        if "adjacent_vertex" in enhancements:
            additional_constraints += int(len(graph.keys()) * avg_neighbors * n)
        if "vertex_to_global_color" in enhancements:
            additional_constraints += len(graph.keys()) * n
    
        theta = additional_constraints / (num_constr)
    

        result_reform.append({'Reformulation': enhancements, 'LowerBound':sol, 'Beta': beta,'Gamma':gamma, 'Theta':theta})

        if new_gap < current_gap:
           current_gap=new_gap
           current_time=new_time
           best_result=sol
           best_gap=current_gap

    result_reform = pd.DataFrame(result_reform)

    

    return model,best_result, best_gap, current_time, result_reform,num_constr

def optimize_model(Formulation, adjacency_list,method):
    if method in ['sdp']:
        n=len(adjacency_list)
        model,num_constr = Formulation(adjacency_list)
        # Set primal feasibility tolerance 
        model.setSolverParam("intpntCoTolPfeas", 1e-6)
        # Set dual feasibility tolerance
        model.setSolverParam("intpntCoTolDfeas", 1e-6)
        # Set infeasibility
        model.setSolverParam("intpntCoTolInfeas", 1e-6)
        start_time = time.time()
        model.solve()
        end_time = time.time()
        Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
        # Find the index of "1"
        index_of_one = np.where(np.round(Z_values[-1],1)==1.)[0][0]
        # Extract the last 3 values before the index of "1"
        sol = Z_values[-1][index_of_one - 3: index_of_one]
        sol=np.sum(sol)
        status = model.getPrimalSolutionStatus()
        print("Status:", status)
        execution_time = end_time - start_time
        return model, sol, execution_time,num_constr
    else:
        n=len(adjacency_list)
        model,num_constr= Formulation(adjacency_list)
        start_time = time.time()
        model.solve()
        sol = model.primalObjValue()
        status = model.getPrimalSolutionStatus()
        print("Status:", status)
        end_time = time.time()
        execution_time = end_time - start_time
        return model, sol, execution_time,num_constr

def run_models(optimal_value,adjacency_list):
    results_dict = {}
    
    # Run MIP
    #_, mip_sol, mip_time,num_constr = optimize_model(Baseline_MIP_Formulation, adjacency_list, 'mip')
    #mip_gap = compute_gap(optimal_value, mip_sol)
    results_dict['mip'] = {'sol': 0, 'time': 0, 'gap': 0, 'complexity': 0/100}#{'sol': mip_sol, 'time': mip_time, 'gap': mip_gap, 'complexity': num_constr/100}

    # Run LP
    _, lp_sol, lp_time,num_constr = optimize_model(LP_Relaxation_Formulation, adjacency_list, 'lp')
    lp_gap = compute_gap(optimal_value, lp_sol)
    results_dict['lp'] = {'sol': lp_sol, 'time': lp_time, 'gap': lp_gap, 'complexity': num_constr/100}

    # Run SDP
    _, sdp_sol, sdp_time,num_constr = optimize_model(Baseline_SDP_Formulation, adjacency_list, 'sdp')
    sdp_gap = compute_gap(optimal_value, sdp_sol)
    results_dict['sdp'] = {'sol': sdp_sol, 'time': sdp_time, 'gap': sdp_gap, 'complexity': num_constr/100}

    # Run QESDP 
    _,qesdp_sol, qesdp_gap,qesdp_time, result_reform,num_constr  = Quadratic_Enhancemnts_Elimination(adjacency_list, optimal_value, sdp_sol,sdp_time)
    results_dict['qesdp'] = {'sol': qesdp_sol, 'time': qesdp_time, 'gap': qesdp_gap, 'complexity': num_constr/100}
 
    result_data = pd.DataFrame.from_dict(results_dict, orient='index')


    return result_data, result_reform


formulations = {
    "mip": Baseline_MIP_Formulation,   
    "lp": LP_Relaxation_Formulation,
    "sdp": Baseline_SDP_Formulation,
    "qesdp": Quadratic_Enhancemnts_Elimination
}

if __name__ == '__main__':

    # Initialize empty DataFrames
    results_df = pd.DataFrame()#columns=['Instance', 'Nodes', 'Edges', 'Method'])
    enhancements_df = pd.DataFrame(columns=['Instance', 'Reformulation','Beta','Gamma','Theta'])
    best_formulations_df = pd.DataFrame(columns=['Instance', 'Quadratic Enhancments', 'SDP Lower Bound', 'QESDP Lower Bound'])


    # Read graph data for instances in chosen set
    for choice in ['small','medium','large']:
        if choice in options:
            for instance_name in options[choice]:
                adjacency_list = Read_DIMACS_Graph(instance_name)
                Metadata=Get_Instance_Metadata(instance_name)
                Actual_Optimal_Value=Metadata['optimal_coloring']
                #adjacency_list = {1: [2, 3], 2: [1, 3], 3: [1, 2]}  # Example adjacency list
                #Metadata = {'nodes': 3, 'edges': 3, 'optimal_coloring': 3}
                result_data, result_reform = run_models(Metadata['optimal_coloring'],adjacency_list)

                temp_results = []
                for index, row in result_data.iterrows():
                    temp_results.append({
                        'Instance': instance_name,
                        'Nodes': Metadata['nodes'],
                        'Edges': Metadata['edges'],
                        'Method': index,
                        'OptimalValue':row['sol'],
                        'SolutionTime': row['time'],
                        'Gap': row['gap'],
                        'Complexity': row['complexity']
                    })
                results_df = pd.concat([results_df,pd.DataFrame(temp_results)], ignore_index=True)

                result_reform['Instance'] = instance_name
                enhancements_df = pd.concat([enhancements_df, result_reform.reset_index(drop=True)], ignore_index=True)
                
                min_beta_index = result_reform['Beta'].idxmin()
                best_reform_row = result_reform.loc[min_beta_index]
                
                best_formulations_df = pd.concat([best_formulations_df, pd.DataFrame([{
                    'Instance': instance_name,
                    'Quadratic Enhancments': best_reform_row['Reformulation'],
                    'SDP Lower Bound': result_data.loc['sdp', 'sol'] if 'sdp' in result_data.index else None,
                    'QESDP Lower Bound': result_data.loc['qesdp', 'sol'] if 'qesdp' in result_data.index else None,
                    'Beta': best_reform_row['Beta']
                }])], ignore_index=True)

        print(results_df)
        print(enhancements_df)
        print(best_formulations_df)


    # Optional: Save to CSV
    results_df.to_csv('results.csv', index=False)
    enhancements_df.to_csv('enhancements.csv', index=False)
    best_formulations_df.to_csv('best_formulations.csv', index=False)
     


  
 