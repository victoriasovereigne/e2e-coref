#!/bin/bash

source ~/tensorflow-gpu-1.3/bin/activate
source ~/.bashrc

export CUDA_VISIBLE_DEVICES="0,1"
export GPU="1"

# python singleton.py best_categories
# python singleton.py best_phi
# python singleton.py best_categories_glove
# python singleton.py best_pos_phi_categories_glove
# python singleton.py best_phi

# python evaluator.py best_phi
# python evaluator.py best_categories
# python evaluator.py best_categories_glove
# python evaluator.py best_pos_phi_categories_glove
# python evaluator.py best_pos_phi_categories

python singleton.py no_pt_phi