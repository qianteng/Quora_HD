import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import ipdb
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')

import config
from utils import pkl_utils



suffix = 'extra'
threshold = 0.05
cmd = "python get_feature_conf_select.py -d 0 -o feature_conf_select_%s.py"%suffix
os.system(cmd)

cmd = "python feature_combiner.py -l 1 -c feature_conf_select_%s -n selected_%s -t %.6f"%(suffix, suffix, threshold)
os.system(cmd)


feature_name = "selected_%s"%suffix
fname = os.path.join(config.FEAT_DIR+"/Combine", feature_name+config.FEAT_FILE_SUFFIX)
data_dict = pkl_utils._load(fname)
X_train = data_dict["X_train_basic"]
X_test = data_dict["X_test"]
y_train = data_dict["y_train"]
splitter = data_dict["splitter"]
n_iter = data_dict["n_iter"]
i = n_iter - 1             # use the last splitter to split the cv
X_train_cv = data_dict["X_train_basic"][splitter[i][0], :]
X_valid_cv = data_dict["X_train_basic"][splitter[i][1], :]
y_train_cv = data_dict["y_train"][splitter[i][0]]
y_valid_cv = data_dict["y_train"][splitter[i][1]]

learner = ExtraTreesClassifier(n_estimators=50, criterion='gini', max_depth=5,
                               min_weight_fraction_leaf=0.0, max_features='auto',
                               n_jobs=-1, random_state=config.RANDOM_SEED, verbose=10)
learner.fit(X_train_cv, y_train_cv)
p_test = learner.predict_proba(X_valid_cv)
print(log_loss(y_valid_cv, p_test))
