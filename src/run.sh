#!/bin/sh

python train.py --fold 0 --model decision_tree_gini
python train.py --fold 1 --model decision_tree_gini
python train.py --fold 2 --model decision_tree_gini
python train.py --fold 3 --model decision_tree_gini
python train.py --fold 4 --model decision_tree_gini