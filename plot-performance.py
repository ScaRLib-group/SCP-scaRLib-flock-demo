import pandas as pd
import matplotlib.pyplot as plt

# Specify the file path
file_path = 'data/performance.csv'

# Load the data
data = pd.read_csv(file_path)
data['time'] = data['time'] / 16 / 1000.0
first = data['time'][0]

# Get a list of agents
agents = data['agents']
# pow of two

# Calculate the speedup
data['speedup'] = data['time'] / first
# Plot the data
# Change the size of the plot
plt.rcParams['figure.figsize'] = [10, 5]
fig = data.plot(x='agents', y='time', kind='line', title='Execution time vs agents', xlabel='agents', ylabel='time (s)')
print("Plotting scalability charts")
fig.get_figure().savefig('charts/performance.pdf')
