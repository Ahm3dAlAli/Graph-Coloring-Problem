import pandas as pd

def compute_gap(actual, approximate):
  gap = ((actual - approximate) / actual) * 100
  return gap

def compute_complexity(constraints):
  return constraints 


def compute_beta(baseline_gap, new_gap):
  return new_gap / baseline_gap


def compute_gamma(baseline_time, new_time):
  return new_time / baseline_time 


def compute_theta(additional_constraints, total_constraints):
  return additional_constraints / total_constraints
