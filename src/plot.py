import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="model", required=False,default='resnet')
parser.add_argument('--row', help="row number", type=int,required=False,default=4)
parser.add_argument('--col', help="column number", type=int, required=False,default=4)

# get values from arguments
args = parser.parse_args()

# Load data from text file
data = np.loadtxt(f"{args.model}_compare_{args.row}_{args.col}.txt")

# Extract y values for the two bars
y = data[:2]

# Set up x values for the two bars
x = np.arange(0,1,0.5)

# Create bar chart
plt.bar(x, y, width = 0.2)

# Set axis labels and title
# plt.xlabel('Bar Index')
plt.ylabel('Delay (# of cycles)')
plt.title(args.model+' evaluation')
plt.xticks(x, ['Baseline', 'Ours'])

# Show the plot
# plt.show()
plt.savefig(f"{args.model}_compare_{args.row}_{args.col}.png")