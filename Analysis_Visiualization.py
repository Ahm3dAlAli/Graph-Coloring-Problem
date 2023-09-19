import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_rel
import numpy as np
from scipy.stats import f_oneway

pd.set_option('display.max_columns', 15)



# Data sets
Result_Formulation_df=pd.read_csv('Results_Benchmarks.csv')
col_names = ['Instance','Reformulation','Beta','Gamma','Theta','LowerBound']
Result_Qud_Enhan_df = pd.read_csv('Results_Enhancents_Benchmarks.csv', names=col_names, sep=';')


# Correlation Analysis
##########################
# Perform one-hot encoding on the 'Method' column
Cor_Anal = pd.get_dummies(Result_Formulation_df, columns=['Method'])

# Create a list of the numerical columns you are interested in for correlation
numerical_columns = ['Vertices', 'Edges', 'Size', 'Density', 'Actual Optimal Coloring', 
                     'Approximate Optimal Coloring', 'Solution Time', 'Gap', 'Complexity'] 

# Add the newly created one-hot encoded columns to the list
numerical_columns += [col for col in Cor_Anal.columns if 'Method_' in col]


# Calculate and round the correlation matrix for the numerical columns
correlation_matrix = Cor_Anal[numerical_columns].corr().round(1)
print(correlation_matrix)

# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm')
# Save the figure as a JPG file
plt.savefig("correlation_matrix.jpg",bbox_inches='tight')
plt.show()


# Extract rows related to methods only but keep all columns
methods_correlation = correlation_matrix.loc[
    [col for col in correlation_matrix.index if 'Method_' in col],
    : # All columns
]

# Display the extracted part of the correlation matrix
print("Correlation Matrix with Methods on Rows and All Other Data Columns:")
print(methods_correlation)

# Plot this extracted part using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(methods_correlation, annot=True, fmt=".1f", cmap='coolwarm')

# Save this figure as a separate JPG file
plt.savefig("methods_all_columns_correlation_matrix.jpg", bbox_inches='tight')
plt.show()


# Pefrmoance of Rleaxation Technique versus Optimal Method
###########################################################

Perf_Rel_Opt=Result_Formulation_df



# Grouping based on Graph Family
graph_families = {
    'Mycielski Graphs': ['myciel3.col', 'myciel4.col'],
    'Queen Graphs': ['queen5_5.col', 'queen6_6.col'],
    'Insertions': ['2-Insertions_3.col', '1-FullIns_3.col']
}

# Methods and Metrics
methods = ['mip', 'lp', 'sdp', 'qesdp']
metrics = ['Solution Time', 'Gap', 'Complexity']  

# Loop through metrics to create a figure for each
for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{metric} Across Graph Families")

    # Loop through each graph family
    for i, (family, instances) in enumerate(graph_families.items()):
        ax = axes[i]

        # Filter data based on the family
        subset = Perf_Rel_Opt[Perf_Rel_Opt['Instance'].isin(instances)]

        # Create a bar plot with metric values on the y-axis
        sns.barplot(x='Method', y=metric, data=subset, ax=ax)

        ax.set_title(family)

        # Hide y-label for all subplots except the left-most one
        if i != 0:
            ax.set_ylabel('')  
        else:
            ax.set_ylabel(metric)  # Show y-axis label only for the left-most subplot

        # Hide x-label for all subplots
        ax.set_xlabel('')

    # Show x-axis label only for the middle subplot
    axes[1].set_xlabel('Method')

    # Final plot adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Graph_Instances_'+metric+'.jpg', bbox_inches='tight')
    plt.show()

# Comp. Effifaicny of ADP and QESDP on Lower Bound Approxiamtions
################################################################
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

# Dataframe containing the results
Eff_Rel_Opt = Result_Formulation_df
Eff_Rel_Opt.rename(columns={"Approximate Optimal Coloring": "Lower Bound"}, inplace=True)
Eff_Rel_Opt['Solution Time'] = Eff_Rel_Opt['Solution Time'].round(2)  # Round solution time to 2 decimal places

# Filter only SDP and QESDP methods
Eff_Rel_Opt = Eff_Rel_Opt[Eff_Rel_Opt['Method'].isin(['sdp', 'qesdp'])]

#Based Grouping

instance_groups = {
    'Queens': ['queen5_5.col', 'queen6_6.col'],
    'Mycielski': ['myciel3.col', 'myciel4.col'],
    'Insertions': ['1-FullIns_3.col', '2-Insertions_3.col']
}

density_groups = {
    'Low/Moderate Density (0.2-0.4)': ['myciel3.col', 'myciel4.col', '2-Insertions_3.col'],
    'High Density (0.4-0.6)': ['1-FullIns_3.col', 'queen5_5.col', 'queen6_6.col']
}

vertex_groups = {
    'Low Count (10-25 vertices)': ['myciel3.col', 'myciel4.col', 'queen5_5.col'],
    'High Count (25-40 vertices)': ['queen6_6.col', '2-Insertions_3.col', '1-FullIns_3.col']
}


# Initialize empty list to store ANOVA results
anova_results = []

# Initialize metric
metric = 'Lower Bound'

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"Comparative Efficacy of SDP and QESDP on {metric}")

# Create a color bar
norm = Normalize(vmin=Eff_Rel_Opt['Solution Time'].min(), vmax=Eff_Rel_Opt['Solution Time'].max())
cbar_ax = fig.add_axes([0.15, 0.15, 0.02, 0.7])
cb = ColorbarBase(cbar_ax, cmap='coolwarm', orientation='vertical', norm=norm)
cbar_ax.set_title('Solution Time')

# Loop through density groups
for i, (group, instances) in enumerate(instance_groups.items()):
    ax = axes[i]
    
    # Filter data
    subset = Eff_Rel_Opt[Eff_Rel_Opt['Instance'].isin(instances)]

    # Box Plot for Lower Bound
    sns.boxplot(x='Method', y=metric, data=subset, ax=ax)
    
    # Overlay with Point Markers representing Solution Time
    sns.swarmplot(x='Method', y=metric, data=subset, ax=ax, size=8, hue='Solution Time', 
                  palette="coolwarm", dodge=True, hue_norm=norm)

    # Remove legend for swarmplot, as we already have a color bar
    ax.get_legend().remove()

    # Set subplot title and labels
    ax.set_title(f"{group}")
    ax.set_xlabel('Method')
    ax.set_ylabel(metric)

    # Perform ANOVA and store results
    anova_data = [subset[subset['Method'] == method][metric] for method in ['sdp', 'qesdp']]
    f_value, p_value = f_oneway(*anova_data)
    anova_results.append(f"For {group}, F-value: {f_value:.2f}, p-value: {p_value:.2f}")

# Final plot adjustments
plt.tight_layout(rect=[0.2, 0.03, 1, 0.95])  # Adjusted layout to accommodate color bar
plt.savefig(f"Comparative_Efficacy_of_{group}_on_{metric}.jpg", bbox_inches='tight')
plt.show()

# Print ANOVA results
for result in anova_results:
    print(result)



# Impact of Quadratic Reformulations on Lower-Bound Approximation
#####################################################################

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Assuming Result_Qud_Enhan_df is your original DataFrame
df = Result_Qud_Enhan_df.iloc[1:].copy()  # Make a copy to avoid SettingWithCopyWarning


new_columns = ['vertex_to_global_color', 'vertex_unique_color', 'adjacent_vertex']


# Convert the 'Reformulation' column
def convert_to_array(row):
        python_obj = ast.literal_eval(row)
        cleaned_str = python_obj.replace('[', '').replace(']', '').replace('"', '')
        split_list = cleaned_str.split(", ")
        return split_list


# Apply the function to the 'Reformulation' column
df['Reformulation'] = df['Reformulation'].apply(convert_to_array)

# Initialize new columns with zeros
for col in new_columns:
    df[col] = 0

# Loop over each row to populate new columns based on 'Reformulation'
for index, row in df.iterrows():
    reformulations = row['Reformulation']
    
    # Update new columns based on the list in 'Reformulation'
    for reform in reformulations:
        if reform in new_columns:
            df.loc[index, reform] = 1

# Show the updated DataFrame
print(df)



# Define Instance Types, Density Groups, and Vertex Groups
instance_groups = {
    'Queens': ['queen5_5.col', 'queen6_6.col'],
    'Mycielski': ['myciel3.col', 'myciel4.col'],
    'Insertions': ['1-FullIns_3.col', '2-Insertions_3.col']
}

density_groups = {
    'Low/Moderate Density (0.2-0.4)': ['myciel3.col', 'myciel4.col', '2-Insertions_3.col'],
    'High Density (0.4-0.6)': ['1-FullIns_3.col', 'queen5_5.col', 'queen6_6.col']
}


vertex_groups = {
    'Low Count (10-25 vertices)': ['myciel3.col', 'myciel4.col', 'queen5_5.col'],
    'High Count (25-40 vertices)': ['queen6_6.col', '2-Insertions_3.col', '1-FullIns_3.col']
}


# Initialize metrics
metrics = ['Beta', 'Gamma', 'Theta']


#
# Plot
#

def plot_figure(title, instance_type):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    fig.suptitle(f"{title} - {instance_type}")

    # Filter DataFrame for specific instance type
    subset = df[df['Instance'].isin(density_groups[instance_type])].copy()

    # Convert metrics to numeric values, if they are not already
    for metric in metrics:
        subset[metric] = pd.to_numeric(subset[metric], errors='coerce')

    for j, metric in enumerate(metrics):
        ax = axes[j]

        # Create a new column that combines enhancements to a single string
        subset['Enhancements'] = subset.apply(
            lambda row: ",".join([k for k, v in {'E1': row['vertex_unique_color'], 'E2': row['adjacent_vertex'], 'E3': row['vertex_to_global_color']}.items() if v == 1]),
            axis=1
        )

        sns.boxplot(x='Enhancements', y=metric, data=subset, ax=ax)
        ax.set_title(f"{metric}")
        ax.set_xlabel('Enhancements')
        ax.set_ylabel(metric)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig(f"{title}_{instance_type}.jpg")
    plt.show()

# Plot figures based on instance type
for instance_type in density_groups.keys():
    plot_figure("All Types of Enhancements", instance_type)


from statsmodels.stats.multicomp import pairwise_tukeyhsd

#
# ANOVA
#

# Function to perform ANOVA for each grouping
def perform_anova(df, grouping, metric):
    print(f"ANOVA for {metric} with different groupings:")
    for group in grouping.keys():
        subset = df[df['Instance'].isin(grouping[group])].copy()
        
        # Convert metrics to numeric values, if they are not already
        subset[metric] = pd.to_numeric(subset[metric], errors='coerce')

        # Create a column for Enhancements which is a combination of the binary features in your DataFrame
        subset['Enhancements'] = subset.apply(lambda row: tuple(row[['vertex_to_global_color', 'vertex_unique_color', 'adjacent_vertex']].values), axis=1)

        # Perform ANOVA
        model = ols(f'{metric} ~ C(Enhancements)', data=subset).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(f"For group {group}:\n{anova_table}\n")

# ANOVA for different groupings and metrics
groupings = [instance_groups]  # You can add more grouping dictionaries here if needed
for grouping in groupings:
    for metric in metrics:
        perform_anova(df, grouping, metric)

#
# Regression Analysis 
#

# Assign Instance and Density groups
for group, instances in instance_groups.items():
    df.loc[df['Instance'].isin(instances), 'Instance_Family'] = group

for group, instances in density_groups.items():
    df.loc[df['Instance'].isin(instances), 'Density'] = group



from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm


def create_interaction_terms(row):
    return row['vertex_to_global_color'] * row['vertex_unique_color'], row['vertex_unique_color'] * row['adjacent_vertex'], row['vertex_to_global_color'] * row['adjacent_vertex'], row['vertex_to_global_color'] * row['vertex_unique_color'] * row['adjacent_vertex']


def perform_regression(df, grouping_col, metric):
    print(f"Multiple Linear Regression for {metric} with grouping based on {grouping_col}:")
    
    # Convert metrics to numeric values, if they are not already
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
    # Generate interaction terms
    df['vertex_global_vertex_unique'], df['vertex_unique_adjacent'], df['vertex_global_adjacent'], df['all_interactions'] = zip(*df.apply(create_interaction_terms, axis=1))
        
    # Generate dummies for grouping column (family or density)
    group_dummies = pd.get_dummies(df[grouping_col], prefix=grouping_col)
    
    
    # Convert boolean columns to integer (0 or 1)
    bool_cols = group_dummies.select_dtypes(include=[bool]).columns
    group_dummies[bool_cols] = group_dummies[bool_cols].astype(int)
    
    # Define dependent and independent variables
    Y = df[metric]
    X = df[['vertex_to_global_color', 'vertex_unique_color', 'adjacent_vertex', 'vertex_global_vertex_unique', 'vertex_unique_adjacent', 'vertex_global_adjacent', 'all_interactions']]
        
    # Add the group dummies to X
    X = pd.concat([X, group_dummies], axis=1)
        
    # Add a constant to the independent variable set to represent the intercept
    X = sm.add_constant(X)

    # Fit the model, according to the OLS (ordinary least squares) method with a dependent variable Y and an independent X
    model = sm.OLS(Y, X).fit()

    # Calculate VIF scores
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("VIF Scores:")
    print(vif_data)

    # Print out the statistics
    print(model.summary())

grouping_cols = ['Density']#, 'Density']  # Replace with your actual columns
metrics = ['Beta','Gamma']  # Replace with your actual metrics

for grouping_col in grouping_cols:
    for metric in metrics:
        perform_regression(df, grouping_col, metric)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_significance(p_value):
    return 'Significant' if p_value <= 0.05 else 'Not Significant'

# Data for Beta coefficients
df_beta = pd.DataFrame({
    'Variable': ['E1', 'E2', 'E3', 'E1E2', 'E1E3', 'E2E3', 'E1E2E3', 'Mycielski', 'Queens', 'Insertions'],
    'Coefficient': [0.034, 0.038, 0.019, 0.115, -0.041, -0.588, 0.1149, 0.2388, 0.311, 0.170],
    'Lower_CI': [-0.177, -0.172, -0.145, -0.110, -0.315, -0.863, -0.110, 0.102, 0.175, 0.034],
    'Upper_CI': [0.244, 0.249, 0.183, 0.340, 0.233, -0.314, 0.340, 0.375, 0.448, 0.307],
    'p_value': [0.746, 0.712, 0.814, 0.304, 0.761, 0.000, 0.304, 0.001, 0.000, 0.016]
})
df_beta['Significance'] = df_beta['p_value'].apply(set_significance)

df_beta = pd.DataFrame({
'Variable': ['E1', 'E2', 'E3', 'E1E2', 'E1E3', 'E2E3', 'E1E2E3', 'Low Den.','High Den.'],
    'Coefficient': [ 0.0048, 0.0195, 0.0243, -0.0270, 0.1149, -0.5743, 0.1149, 0.3318,0.3182],
    'Lower_CI': [ -0.161, -0.194, -0.189, -0.302, -0.112, -0.850, -0.112, 0.222,0.209],
    'Upper_CI': [ 0.170, 0.233, 0.238, 0.248, 0.342, -0.299, 0.342, 0.441,0.428],
    'p_value': [ 0.953, 0.853, 0.818, 0.842, 0.309, 0.000, 0.309, 0.000, 0.000]
})
df_beta['Significance'] = df_beta['p_value'].apply(set_significance)

# Plot for Beta Enhancements
sns.pointplot(x='Variable', y='Coefficient', hue='Significance', data=df_beta, join=False)
plt.errorbar(x=df_beta.index, y=df_beta['Coefficient'], yerr=(df_beta['Upper_CI'] - df_beta['Lower_CI']) / 2, fmt='none', c='black')
plt.title('Beta Coefficients for Enhancements')
plt.xlabel('Enhancement')
# Ensure legend has both categories
handles, labels = plt.gca().get_legend_handles_labels()
if len(labels) == 1:
    handles.append(handles[0])
    labels.append('Significant' if labels[0] == 'Not Significant' else 'Not Significant')
plt.legend(handles, labels, title='Significance')
plt.show()


# Data for Gamma coefficients
df_gamma = pd.DataFrame({
    'Variable': ['E1', 'E2', 'E3', 'E1E2', 'E1E3', 'E2E3', 'E1E2E3', 'Mycielski', 'Queens', 'Insertions'],
    'Coefficient': [16.090, 8.380, -4.995, 12.116, 7.684, 6.839, 12.116, -16.540, 7.560, 13.933],
    'Lower_CI': [-1.977, -9.686, -19.039, -7.153, -15.819, -16.664, -7.153, -28.249, -4.149, 2.223],
    'Upper_CI': [34.155, 26.446, 9.049, 31.385, 31.187, 30.341, 31.385, -4.831, 19.269, 25.642],
    'p_value': [0.079, 0.350, 0.472, 0.208, 0.509, 0.556, 0.208, 0.007, 0.197, 0.021]
})
df_gamma['Significance'] = df_gamma['p_value'].apply(set_significance)


# Data for Density Family (Gamma)
df_gamma = pd.DataFrame({
    'Variable': ['E1', 'E2', 'E3', 'E1E2', 'E1E3', 'E2E3', 'E1E2E3', 'Low Den.','High Den.'],
    'Coefficient': [ -5.0925, 15.9923, 8.2833, 7.7811, 12.1162, 6.9356, 12.1162, -3.1712,7.6377],
    'Lower_CI': [ -21.105, -4.653, -12.362, -18.851, -9.819, -19.696, -9.819, -13.746,-2.937],
    'Upper_CI': [10.920, 36.637, 28.928, 34.413, 34.052, 33.567, 34.052, 7.404,18.212],
    'p_value': [ 0.521, 0.124, 0.419, 0.555, 0.268, 0.598, 0.268, 0.544, 0.150]
})
df_gamma['Significance'] = df_gamma['p_value'].apply(set_significance)
# Plot for Gamma Enhancements
sns.pointplot(x='Variable', y='Coefficient', hue='Significance', data=df_gamma, join=False)
plt.errorbar(x=df_gamma.index, y=df_gamma['Coefficient'], yerr=(df_gamma['Upper_CI'] - df_gamma['Lower_CI']) / 2, fmt='none', c='black')
plt.title('Gamma Coefficients for Enhancements')
plt.xlabel('Enhancement')

plt.show()

# Ensure legend has both categories
handles, labels = plt.gca().get_legend_handles_labels()
if len(labels) == 1:
    handles.append(handles[0])
    labels.append('Significant' if labels[0] == 'Not Significant' else 'Not Significant')
plt.legend(handles, labels, title='Significance')
plt.show()


