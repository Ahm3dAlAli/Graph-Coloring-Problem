import pandas as pd
from Parse_Scrape_Graph import Read_DIMACS_Graph,Get_Instance_Metadata
from Formulations import *
import requests
from io import StringIO
import numpy as np
import time
import pandas as pd
from mosek.fusion import *
import sys
import time
from Constraint_Elimination import *
from Generate_Random_Graph import random_graph
pd.set_option('display.max_columns', None)
import random
random.seed(22)



def optimize_model(Formulation, adjacency_list,method):	
    n=len(adjacency_list)
    model,num_constr = Formulation(adjacency_list)
    # Set primal feasibility tolerance 
    model.setSolverParam("intpntCoTolPfeas", 1e-8)
    # Set infeasibility
    model.setSolverParam("intpntCoTolInfeas", 1e-8)
    # Set Gap toelrance 
    model.setSolverParam("intpntCoTolRelGap", 1.0e-8)
    # Access Any Solutions 
    model.acceptedSolutionStatus(AccSolutionStatus.Anything)

    if method in ['sdp']:
        model.setLogHandler(sys.stdout) # Add logging
        model.writeTask("sdp.ptf") # Save problem in readable format
        start_time = time.time()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.solve()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        end_time = time.time()
        solution_status = model.getPrimalSolutionStatus()

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
            sol = Z_values[-1][index_of_one -n: index_of_one]
            sol=np.sum(sol)
            status = 'Uncertain'
        else:
            sol = None
            status = 'Infeasible'

        execution_time = end_time - start_time
        return model, sol, execution_time,num_constr,status
    
    elif method in ['lp']:
        model.setLogHandler(sys.stdout) # Add logging
        model.writeTask("lp.ptf") # Save problem in readable format
        start_time = time.time()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.solve()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        end_time= time.time()


        solution_status = model.getPrimalSolutionStatus()

        #FIX
        if solution_status == SolutionStatus.Optimal:
            sol = sum(np.array(model.getVariable('w').level()).reshape(n,1))[0]#model.primalObjValue()
            status = 'Optimal Feasible'
        elif solution_status == SolutionStatus.Unknown:
            sol = sum(np.array(model.getVariable('w').level()).reshape(n,1))[0]
            status = 'Uncertain'
        else:
            sol = None
            status = 'Infeasible'

    elif method in ['mip']:
        # Set termination criteria (20 minutes = 1200 seconds)
        model.setSolverParam("mioMaxTime", 1200)
        model.setLogHandler(sys.stdout) # Add logging
        model.writeTask("mip.ptf") # Save problem in readable format
        start_time = time.time()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        model.solve()
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        end_time= time.time()


        solution_status = model.getPrimalSolutionStatus()
        
        if solution_status == SolutionStatus.Optimal:
            sol = model.primalObjValue()
            status = 'Optimal Feasible'
        elif solution_status == SolutionStatus.Feasible:
            sol = sum(np.array(model.getVariable('w').level()).reshape(n,1))[0]
            status = 'Sub-Optimal Feasible'
        elif solution_status == SolutionStatus.Undefined or solution_status == SolutionStatus.Unknown:
            sol = None
            status = 'Infeasible'

    execution_time = end_time - start_time
    return model, sol, execution_time,num_constr,status


def run_models(optimal_value,adjacency_list,mode):
    results_dict = {}
    if mode == 'Benchmark':
        # Run MIP
        _, mip_sol, mip_time,num_constr,status = optimize_model(Baseline_MIP_Formulation, adjacency_list, 'mip')
        mip_gap = compute_gap(optimal_value, mip_sol)
        results_dict['mip'] = {'sol': mip_sol, 'time': mip_time, 'gap': mip_gap, 'complexity': num_constr/100, 'status': status}

        # Run LP
        _, lp_sol, lp_time,num_constr,status = optimize_model(LP_Relaxation_Formulation, adjacency_list, 'lp')
        lp_gap = compute_gap(optimal_value, lp_sol)
        results_dict['lp'] = {'sol': lp_sol, 'time': lp_time, 'gap': lp_gap, 'complexity': num_constr/100, 'status': status}
        
        # Run SDP
        _, sdp_sol, sdp_time,num_constr,status = optimize_model(Baseline_SDP_Formulation, adjacency_list, 'sdp')
        sdp_gap = compute_gap(optimal_value, sdp_sol)
        results_dict['sdp'] = {'sol': sdp_sol, 'time': sdp_time, 'gap': sdp_gap, 'complexity': num_constr/100, 'status': status}
        # Run QESDP 
        _,qesdp_sol, qesdp_gap,qesdp_time, result_reform,num_constr,status  = Quadratic_Enhancemnts_Elimination(adjacency_list, optimal_value,sdp_sol,sdp_time)
        results_dict['qesdp'] = {'sol': qesdp_sol, 'time': qesdp_time, 'gap': qesdp_gap, 'complexity': num_constr/100, 'status': status}
    
        result_data = pd.DataFrame.from_dict(results_dict, orient='index')

        return result_data, result_reform
    elif mode== 'Simulation':
        # Run MIP
        _, mip_sol, mip_time,num_constr,status = optimize_model(Baseline_MIP_Formulation, adjacency_list, 'mip')
        optimal_value=mip_sol
        mip_gap = compute_gap(optimal_value, mip_sol)
        results_dict['mip'] = {'sol': mip_sol, 'time': mip_time, 'gap': mip_gap, 'complexity': num_constr/100, 'status': status}

        # Run LP
        _, lp_sol, lp_time,num_constr,status = optimize_model(LP_Relaxation_Formulation, adjacency_list, 'lp')
        lp_gap = compute_gap(optimal_value, lp_sol)
        results_dict['lp'] = {'sol': lp_sol, 'time': lp_time, 'gap': lp_gap, 'complexity': num_constr/100, 'status': status}

        # Run SDP
        _, sdp_sol, sdp_time,num_constr,status = optimize_model(Baseline_SDP_Formulation, adjacency_list, 'sdp')
        sdp_gap = compute_gap(optimal_value, sdp_sol)
        results_dict['sdp'] = {'sol': sdp_sol, 'time': sdp_time, 'gap': sdp_gap, 'complexity': num_constr/100, 'status': status}

        # Run QESDP 
        _,qesdp_sol, qesdp_gap,qesdp_time, result_reform,num_constr,status  = Quadratic_Enhancemnts_Elimination(adjacency_list, optimal_value, sdp_sol,sdp_time)
        results_dict['qesdp'] = {'sol': qesdp_sol, 'time': qesdp_time, 'gap': qesdp_gap, 'complexity': num_constr/100, 'status': status}
    
        result_data = pd.DataFrame.from_dict(results_dict, orient='index')

        return result_data, result_reform,optimal_value






# Declare Benchamrk instance sets 
mycie = {'myciel3.col','myciel4.col'}
queens = {'queen6_6.col','queen6_6.col'}
Insertions={'2-Insertions_3.col'} 
Full_Ins={'1-FullIns_3.col'}



options = {
    'mycie': mycie,
    'queens': queens,
    'insertion': Insertions,
    'full_ins':Full_Ins,
}

if __name__ == '__main__':

    mode = 'Simulation'

    if mode == 'Benchmark':
        results_df = pd.DataFrame()
        enhancements_df = pd.DataFrame(columns=['Instance', 'Reformulation','Beta','Gamma','Theta'])
        best_formulations_df = pd.DataFrame(columns=['Instance', 'Quadratic Enhancments', 'SDP Lower Bound', 'QESDP Lower Bound'])

        # Read graph data for instances in chosen set
        for choice in options:
                for instance_name in options[choice]:
                    adjacency_list = Read_DIMACS_Graph(instance_name)
                    print(adjacency_list)
                    #Trick or Cam
                    #Metadata=Get_Instance_Metadata(instance_name,'Trick')
                    Metadata=Get_Instance_Metadata('(*)'+instance_name,'Cam')
                    Actual_Optimal_Value=Metadata['optimal_coloring']
                    result_data, result_reform = run_models(Actual_Optimal_Value,adjacency_list,mode)
                    
                    temp_results = []
                    for index, row in result_data.iterrows():
                        temp_results.append({
                            'Instance': instance_name,
                            'Vertices': Metadata['nodes'],
                            'Edges': Metadata['edges'],
                            'Size': (Metadata['nodes']*(Metadata['nodes'] -1))/2,
                            'Density': (2* Metadata['edges'])/(Metadata['nodes']*(Metadata['nodes'] -1)),
                            'Method': index,
                            'status':row['status'],
                            'Actual Optimal Coloring':Metadata['optimal_coloring'],
                            'Optimal Coloring':row['sol'],
                            'Solution Time': row['time'],
                            'Gap': row['gap'],
                            'Complexity': row['complexity']

                        })
                    results_df = pd.concat([results_df,pd.DataFrame(temp_results)], ignore_index=True)

                    result_reform['Instance'] = instance_name
                    enhancements_df = pd.concat([enhancements_df, result_reform.reset_index(drop=True)], ignore_index=True)
                    
                    # Sort by Beta and use Gamma as tie-breaker
                    individual_results = result_reform.to_dict('records')
                    sorted_by_beta = sorted(individual_results, key=lambda x: x["Beta"])
                    best_individual = sorted_by_beta[0]
                    for result in sorted_by_beta[1:]:
                        if abs(result["Beta"] - best_individual["Beta"]) < 0.05:
                            if result["Gamma"] < best_individual["Gamma"]:
                                best_individual = result
                    
                    best_reform_row = pd.DataFrame([best_individual])

                    best_formulations_df = pd.concat([best_formulations_df, pd.DataFrame([{
                        'Instance': instance_name,
                        'Quadratic Enhancments': best_reform_row.iloc[0]['Reformulation'],
                        'SDP Lower Bound': result_data.loc['sdp', 'sol'] if 'sdp' in result_data.index else None,
                        'QESDP Lower Bound': result_data.loc['qesdp', 'sol'] if 'qesdp' in result_data.index else None,
                        'Beta': best_reform_row.iloc[0]['Beta'],
                        'Gamma': best_reform_row.iloc[0]['Gamma'],
                        'Theta': best_reform_row.iloc[0]['Theta']
                    }])], ignore_index=True)

                print(results_df)
                print(enhancements_df)
                print(best_formulations_df)


        # Optional: Save to CSV
        results_df.to_csv('results'+choice+'.csv', index=False)
        enhancements_df.to_csv('enhancements'+choice+'.csv', index=False)
        best_formulations_df.to_csv('best_formulations'+choice+'.csv', index=False)
        

    elif mode == 'Simulation':
        results_df = pd.DataFrame()
        enhancements_df = pd.DataFrame()
        best_formulations_df = pd.DataFrame()

        num_simulations=5
        vertex_list = [18]#,15,18]
        sparsity_list = [0.2,0.4,0.6,0.8]
    
        for sim in range(1, num_simulations + 1):
            for n in vertex_list:
                for s in sparsity_list:
                    instance_name = f"{n}_Vertices_{s}_Sparsity"
                    adjacency_list = random_graph(n, s)
                    print(instance_name)
                        
                    result_data, result_reform,optimal_val = run_models(0, adjacency_list,'Simulation')

                    Metadata = {'nodes': n, 'edges': sum(len(v) for v in adjacency_list.values()) // 2, 'optimal_coloring': optimal_val }
                    temp_results = []
                    for index, row in result_data.iterrows():
                        temp_results.append({
                            'Simulation': sim,
                            'Instance': instance_name,
                            'Vertices': Metadata['nodes'],
                            'Sparsity':s,
                            'Edges': Metadata['edges'],
                            'Size': (Metadata['nodes']*(Metadata['nodes'] -1))/2,
                            'Density': (2* Metadata['edges'])/(Metadata['nodes']*(Metadata['nodes'] -1)),
                            'Method': index,
                            'status':row['status'],
                            'Actual Optimal Coloring':Metadata['optimal_coloring'],
                            'Optimal Coloring':row['sol'],
                            'Solution Time': row['time'],
                            'Gap': row['gap'],
                            'Complexity': row['complexity']
                            })
                        results_df = pd.concat([results_df,pd.DataFrame(temp_results)], ignore_index=True)

                        result_reform['Instance'] = instance_name
                        result_reform['Simulation'] = sim
                        enhancements_df = pd.concat([enhancements_df, result_reform.reset_index(drop=True)], ignore_index=True)
                        
                        # Sort by Beta and use Gamma as tie-breaker
                        individual_results = result_reform.to_dict('records')
                        sorted_by_beta = sorted(individual_results, key=lambda x: x["Beta"])
                        best_individual = sorted_by_beta[0]
                        for result in sorted_by_beta[1:]:
                            if abs(result["Beta"] - best_individual["Beta"]) < 0.01:
                                if result["Gamma"] < best_individual["Gamma"]:
                                    best_individual = result
                        
                        best_reform_row = pd.DataFrame([best_individual])

                        best_formulations_df = pd.concat([best_formulations_df, pd.DataFrame([{
                            'Simulation':sim,
                            'Instance': instance_name,
                            'Quadratic Enhancments': best_reform_row.iloc[0]['Reformulation'],
                            'SDP Lower Bound': result_data.loc['sdp', 'sol'] if 'sdp' in result_data.index else None,
                            'QESDP Lower Bound': result_data.loc['qesdp', 'sol'] if 'qesdp' in result_data.index else None,
                            'Beta': best_reform_row.iloc[0]['Beta'],
                            'Gamma': best_reform_row.iloc[0]['Gamma'],
                            'Theta': best_reform_row.iloc[0]['Theta']
                        }])], ignore_index=True)
            print(results_df)
            print(enhancements_df)
            print(best_formulations_df)
        # Save to CSV
        results_df.to_csv('simulation'+str(n)+'vertices'+'_results_t2.csv', index=False)
        enhancements_df.to_csv('simulation'+str(n)+'vertices'+'_enhancements_t2.csv', index=False)
        best_formulations_df.to_csv('simulation'+str(n)+'vertices'+'_best_formulations.csv_t2', index=False)



        # Calculate average results per instance and simulation
        average_results_df = results_df.groupby(['Instance', 'Simulation']).mean().reset_index()
        average_enhancements_df = enhancements_df.groupby(['Instance', 'Simulation']).mean().reset_index()
        average_best_formulations_df = best_formulations_df.groupby(['Instance', 'Simulation']).mean().reset_index()

        # Save the averaged results to CSV
        average_results_df.to_csv('average_simulation_results.csv', index=False)
        average_enhancements_df.to_csv('average_simulation_enhancements.csv', index=False)
        average_best_formulations_df.to_csv('average_simulation_best_formulations.csv', index=False)

