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
y = data[:3]

# Set up x values for the two bars
x = np.arange(0,3)
# x = [0,0.5,1]

# Create bar chart
barlist=plt.bar(x, y, width = 0.4)
barlist[-1].set_color('r')

# Set axis labels and title
# plt.xlabel('Bar Index')
plt.ylabel('Delay (# of cycles)')
plt.title(f'{args.model.title()} Evaluation with {args.row}x{args.col} PEs')
plt.xticks(x, ['Baseline', 'Ours', "Lower-bound"])

# Show the plot
plt.tight_layout()
plt.savefig(f"{args.model}_compare_{args.row}_{args.col}.png")
plt.show()