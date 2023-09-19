import pandas as pd

# Read your data into a DataFrame (assuming it's stored in a CSV file)
df = pd.read_csv('simulation10vertices_results.csv')

# Remove duplicates
df = df.drop_duplicates()

# Average by simulation and instance
grouped_df = df.groupby(['Simulation', 'Instance']).mean().reset_index()

# You can write this DataFrame to a new CSV file, if needed
grouped_df.to_csv('grouped_data.csv', index=False)

print(grouped_df)