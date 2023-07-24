# 🎨 Graph Coloring Problem (GCP) Solver 📊

This project contains a set of Python scripts that aim to solve the Graph Coloring Problem (GCP) using various optimization techniques and formulations such as Mixed Integer Programming (MIP), Quadratically Constrained Programming (QCP), Linear Programming (LP), and Semi-Definite Programming (SDP).

### 📋 Table of Contents
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>

<a name="introduction"></a>

### 🚀 Introduction
The aim of this project is to develop an algorithm that can effectively solve the Graph Coloring Problem. The solution approach involves formulating the problem in several ways and optimizing the resulting mathematical model using various formulations such as MIP, QCP, LP, and SDP.

<a name="features"></a>

### 🎯 Features
- Reads graph instances from the DIMACS benchmark suite.
- Constructs various problem formulations and relaxations (MIP, QCP, LP, SDP) for the GCP.
- Solves the GCP using the Gurobi Optimizer.
- Implements constraint elimination technique for improving the solution.
- Tests the performance of the GCP solver with different types of instances (small, medium, large, etc.).

<a name="getting-started"></a>

### ⚙️ Getting Started
#### Prerequisites
This project requires Python and the following Python libraries installed:
- numpy
- pandas
- requests
- gurobipy

To install these libraries, you can use pip:
```bash
pip install numpy pandas requests gurobipy
```
Clone this repo to your local machine:
```bash
git clone https://github.com/username/repo.git
```
Run the script:
```bash
python gcp_solver.py
<a name="project-structure"></a>
```
## 🗃️ Project Structure
Here's a short description of the key files in this project:

-  gcp_solver.py: The main script where the GCP solver is implemented.
-  read_dimacs_graph(): Function to read graph instances from the DIMACS benchmark suite.
-  Intial_MIP_Formulation(): Function that constructs and solves the GCP using the MIP formulation.
-  Quadratically_Constrained_MIP_Formulation(): Function that constructs and solves the GCP using the QCP formulation.
-  lp_relaxation(): Function that constructs and solves the GCP using the LP relaxation.
<a name="results"></a>

## 📊 Results
The results of the computational experiments are printed on the console. For each problem instance, the script prints the number of colors used and the time taken to find the solution.

<a name="contributing"></a>

## 👥 Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

<a name="license"></a>

## 📜 License
This project is licensed under the MIT License - see the LICENSE.md file for details.
