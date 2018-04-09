#!/bin/bash

source ~/tensorflow-gpu-1.3/bin/activate
source ~/.bashrc

# export CUDA_VISIBLE_DEVICES="0,1"
# export GPU="0"

# python singleton.py best
# python singleton.py no_meta
# python singleton.py best_pos
# python singleton.py best_pos_phi
# python singleton.py best_pos_phi_categories
# python singleton.py best_phi_categories

# python evaluator.py best
# python evaluator.py no_meta
# python evaluator.py best_pos
# python evaluator.py best_pos_phi
# python evaluator.py best_pos_phi_categories
# python evaluator.py best_phi_categories

# python singleton.py no_pt_categories
# python singleton.py no_pt_categories_glove
# python singleton.py no_pt_pos
# python singleton.py no_pt_pos_phi
# python singleton.py no_pt_pos_phi_categories
# python singleton.py no_pt_pos_phi_categories_glove


# python evaluator.py no_pt_categories
# python evaluator.py no_pt_categories_glove

# python evaluator.py no_pt_pos
python evaluator.py no_pt_pos_phi
python evaluator.py no_pt_pos_phi_categories
python evaluator.py no_pt_pos_phi_categories_glove
python evaluator.py no_pt
# python evaluator.py no_pt_phi
