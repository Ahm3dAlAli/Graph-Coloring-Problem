import numpy as np
import time
import pandas as pd
from Performance_Evaluation import *
from Formulations import *
import sys

pd.set_option('display.max_columns', None)


def Quadratic_Enhancemnts_Elimination(graph, Actual_Optimal_Value, sdp_sol,sdp_time):
    n = len(graph)
    best_result = None
    best_reformulation = []
    
    quadratic_enhancements = ["vertex_unique_color", "adjacent_vertex", "vertex_to_global_color"]
    baseline_gap = compute_gap(Actual_Optimal_Value, sdp_sol)
    baseline_time = sdp_time
    best_gap =[]
    current_gap=baseline_gap
    current_time=baseline_time

    avg_neighbors = sum(len(neighbors) for neighbors in graph.values()) / len(graph.keys())
    single_enhancements_results = []

    for enhancement in quadratic_enhancements:
        model,num_constr = Quadratically_Enhanced_SDP_Formulation(graph, constraints_to_include=[enhancement])
        
        # Setting solver parameters and solving
        model.setSolverParam("intpntCoTolPfeas", 1e-8)
        model.setSolverParam("intpntCoTolInfeas", 1e-8)
        model.setLogHandler(sys.stdout) # Add logging
        model.writeTask("QESDP.ptf") # Save problem in readable format
        start_time = time.time()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.solve()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        end_time = time.time()
        new_time = end_time - start_time
        solution_status = model.getPrimalSolutionStatus()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        if solution_status == SolutionStatus.Optimal:
            Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
            # Find the index of "1"
            index_of_one = np.where(np.round(Z_values[-1],1)==1.)[0][0]
            # Extract the last n values before the index of "1"
            sol = Z_values[-1][index_of_one - n: index_of_one]
            sol=np.sum(sol)
            status='Optimal Feasible'
        elif solution_status == SolutionStatus.Unknown:
            Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
            # Find the index of "1"
            index_of_one = np.where(np.round(Z_values[-1],1)==1.)[0][0]
            # Extract the last n values before the index of "1"
            sol = Z_values[-1][index_of_one - n: index_of_one]
            sol=np.sum(sol)
            status = 'Uncertain'
        else:
            sol = None
            status = 'Infeasible'

        new_gap = compute_gap(Actual_Optimal_Value, sol)
        beta = compute_beta(baseline_gap,new_gap)
        gamma= compute_gamma(baseline_time, new_time)

        additional_constraints = 0
        if "vertex_unique_color" in enhancement:
            additional_constraints += len(graph.keys()) * (n * (n - 1) // 2)
        if "adjacent_vertex" in enhancement:
            additional_constraints += int(len(graph.keys()) * avg_neighbors * n)
        if "vertex_to_global_color" in enhancement:
            additional_constraints += len(graph.keys()) * n
    
        theta = compute_theta(additional_constraints,num_constr)

        if new_gap < current_gap:
           current_gap=new_gap
           current_time=new_time
           best_result=sol
           best_gap=current_gap
    
    
        single_enhancements_results.append({'Reformulation': enhancement, 'LowerBound':sol, 'Beta': beta,'Gamma':gamma, 'Theta':theta})

    df_single = pd.DataFrame(single_enhancements_results)
    df_single = df_single.sort_values(by=['Beta', 'Gamma'], ascending=[True, True])
    best_single = df_single.iloc[0]['Reformulation']

    result_reform = single_enhancements_results
    
    new_combinations = [[best_single, x] for x in quadratic_enhancements if x != best_single]
    new_combinations.append([best_single] + [x for x in quadratic_enhancements if x != best_single])
    new_combinations.append([best_single])
    print(new_combinations)

    for enhancements in new_combinations:
        model,num_constr = Quadratically_Enhanced_SDP_Formulation(graph, constraints_to_include=enhancements)
        
        # Setting solver parameters and solving
        model.setSolverParam("intpntCoTolPfeas", 1e-8)
        model.setSolverParam("intpntCoTolInfeas", 1e-8)
        model.setLogHandler(sys.stdout) # Add logging
        model.writeTask("QESDP.ptf") # Save problem in readable format
        start_time = time.time()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.solve()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        end_time = time.time()
        new_time = end_time - start_time
        solution_status = model.getPrimalSolutionStatus()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        if solution_status == SolutionStatus.Optimal:
            Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
            # Find the index of "1"
            index_of_one = np.where(np.round(Z_values[-1],1)==1.)[0][0]
            # Extract the last n values before the index of "1"
            sol = Z_values[-1][index_of_one - n: index_of_one]
            sol=np.sum(sol)
            status='Optimal Feasible'
        elif solution_status == SolutionStatus.Unknown:
            Z_values = np.array(model.getVariable('Z').level()).reshape(n**2+n+1, n**2+n+1)
            # Find the index of "1"
            index_of_one = np.where(np.round(Z_values[-1],1)==1.)[0][0]
            # Extract the last n values before the index of "1"
            sol = Z_values[-1][index_of_one - n: index_of_one]
            sol=np.sum(sol)
            status = 'Uncertain'
        else:
            sol = None
            status = 'Infeasible'

        new_gap = compute_gap(Actual_Optimal_Value, sol)
        beta = compute_beta(baseline_gap,new_gap)
        gamma= compute_gamma(baseline_time, new_time)

        additional_constraints = 0
        if "vertex_unique_color" in enhancements:
            additional_constraints += len(graph.keys()) * (n * (n - 1) // 2)
        if "adjacent_vertex" in enhancements:
            additional_constraints += int(len(graph.keys()) * avg_neighbors * n)
        if "vertex_to_global_color" in enhancements:
            additional_constraints += len(graph.keys()) * n
    
        theta = compute_theta(additional_constraints,num_constr)
    
        result_reform.append({'Reformulation': enhancements, 'LowerBound':sol, 'Beta': beta,'Gamma':gamma, 'Theta':theta})
        print(result_reform)
        if new_gap < current_gap:
           current_gap=new_gap
           current_time=new_time
           best_result=sol
           best_gap=current_gap
    
    result_reform = pd.DataFrame(result_reform)

    

    return model,best_result, best_gap, current_time, result_reform,num_constr,status