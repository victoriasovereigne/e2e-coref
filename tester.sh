#!/bin/bash

python evaluator.py mod
python evaluator.py mod_categories
python evaluator.py mod_categories_glove
python evaluator.py mod_pos

python evaluator.py mod_phi
python evaluator.py mod_g
python evaluator.py mod_pos_phi
python evaluator.py mod_pos_phi_categories_glove
