# 🎨 Graph Coloring Problem (GCP) Solver 📊

This project contains a set of Python scripts that aim to solve the Graph Coloring Problem (GCP) using various optimization techniques and formulations such as Mixed Integer Programming (MIP), Linear Programming (LP), Semi-Definite Programming (SDP) and Quadratically Constraint Semi-Definite Programming (QESDP) 

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
The aim of this project is to carry a computational analysis of the affect of binary refrmoulations of constraint on SDP Relaxation. The solution approach involves formulating the problem in several ways and optimizing the resulting mathematical model using various formulations.

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
- MOSEK

To install these libraries, you can use pip:
```bash
pip install numpy pandas requests mosek.fusion
```
Clone this repo to your local machine:
```bash
git clone https://github.com/username/repo.git
```
Run the script:
```bash
python run.py
<a name="project-structure"></a>
```
## 🗃️ Project Structure
Here's a short description of the key files in this project:

-  Run.py: Runs optimization of GCP
-  Parse_Scrape_Graph.py: Parsing and reading DIMACS formatted graphs and their metadata
-  Genreate_Rnadom_Graph: Genreate set of rnadom graph based on vertix number and sparsity.
-  Formulations.py: Contain set of predefined MOSEK formulations 
-  Constraint_Elimination.py: Modeling of QESDP
-  Pre_Proccess: Average Simulated Data.
-  Performance_Evalaution: Set of metrics for optimization performance evaluation.
-  Analysis and Visiualization: Graph presenting and statstical analysis mainly ANOVA and Multiple Linear Regression.
<a name="results"></a>

## 📊 Results
The results of the computational experiments are found in the paper.

<a name="contributing"></a>

## 👥 Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

<a name="license"></a>

## 📜 License
This project is licensed under the MIT License - see the LICENSE.md file for details.
