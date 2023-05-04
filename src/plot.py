import numpy as np
import matplotlib.pyplot as plt

# Load data from text file
data = np.loadtxt('compare.txt')

# Extract y values for the two bars
y = data[:2]

# Set up x values for the two bars
x = np.arange(2)

# Create bar chart
plt.bar(x, y)

# Set axis labels and title
plt.xlabel('Bar Index')
plt.ylabel('Y Values')
plt.title('Bar Chart Example')
plt.xticks(x, ['Bar 1', 'Bar 2'])

# Show the plot
# plt.show()
plt.savefig('compare.png')