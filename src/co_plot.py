import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="model", required=False,default='resnet')
parser.add_argument('--hw', type=str, required=False, help='Hardware the model is run on',
                    default='imc')
parser.add_argument('--par', help="partition algorithm", type=str, required=False, 
                    default='bfs')

# get values from arguments
args = parser.parse_args()
output_dir = '../' + args.model
result_dir = output_dir + '/result/'

file_list = os.listdir(result_dir)
file_pattern = '^' + args.model + '_compare_' + args.par + '_\d+_\d+\.txt$'

row_col = []
data_list = []
for file_name in file_list:
    if re.match(file_pattern, file_name):
        data_list.append(np.loadtxt(result_dir + file_name))
        row_col.append('x'.join(re.search('\d+_\d+', file_name).group().split('_')))

row_col, data_list = zip(*sorted(zip(row_col,data_list)))
row_col = list(row_col)
data_list = list(data_list)

baseline_time = data_list[0][0]
lower_bound = data_list[0][-1]

y = []
y.append(baseline_time)
for data_idx in range(len(data_list)):
    y.append(data_list[data_idx][1])
y.append(lower_bound)

# Set up x values for the two bars
x = np.arange(0,len(y))

# Create bar chart
barlist=plt.bar(x, y, width = 0.8)
barlist[-1].set_color('r')
barlist[0].set_color('darkorange')

# Set axis labels and title
# plt.xlabel('Bar Index')
plt.ylabel('Delay (# of cycles)', fontsize=16)
# plt.title(f'{args.model.title()} Evaluation with {args.par}', fontsize=20)
plt.xticks(x, ['Baseline'] + row_col + ["Lower Bound"], fontsize=13)
plt.yticks(fontsize=13)

# Show the plot
plt.tight_layout()
plt.savefig(result_dir + f"{args.model}_compare_{args.par}.png", dpi=1000)
plt.show()