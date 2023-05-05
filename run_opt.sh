#! /bin/bash
cd src
python3 optimization.py --model ${1:-"resnet"} --row $2 --col $3 --par ${4:-"bfs"} --hw ${5:-"imc"}
python3 plot.py --model ${1:-"resnet"} --row $2 --col $3 --par ${4:-"bfs"} --hw ${5:-"imc"}