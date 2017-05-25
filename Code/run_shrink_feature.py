# -*- coding: utf-8 -*-
"""
@brief: script for shrink the feature space with clf_xgb_tree

"""

import os
import sys

from utils import time_utils

suffix = 'dropsome'
threshold = 0.05

cmd = "python get_feature_conf_nonlinear.py -d 10 -o feature_conf_nonlinear_%s.py"%suffix
os.system(cmd)

cmd = "python feature_combiner.py -l 1 -c feature_conf_nonlinear_%s -n basic_nonlinear_%s -t %.6f"%(suffix, suffix, threshold)
os.system(cmd)

cmd = "python task.py -m single -f basic_nonlinear_%s -l clf_xgb_tree_single -e 1 -p True"%suffix
os.system(cmd)
