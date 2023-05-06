#! /bin/bash
cd src
python3 anal_orig.py --model ${1:-"resnet"}
python3 split_critical_sec.py --model ${1:-"resnet"}
python3 create_yaml.py --model ${1:-"resnet"} --hw ${2:-"imc"}
python3 timespace.py --model ${1:-"resnet"} --hw ${2:-"imc"}