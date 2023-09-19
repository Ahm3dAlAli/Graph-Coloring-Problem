
import numpy as np
import time
import pandas as pd
from mosek.fusion import *
import itertools
from Performance_Evaluation import *

pd.set_option('display.max_columns', None)


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
    model = Model("sdp_gpc")  

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
    model = Model("Enhanced_sdp_gpc")  

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

