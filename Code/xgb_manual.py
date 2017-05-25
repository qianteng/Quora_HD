import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import ipdb
from sklearn.metrics import log_loss
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')

import config
from utils import pkl_utils

feature_name = "basic_nonlinear_20170525"
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
#params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'eta': .02, 'max_depth': 7, 'silent': 1}
#params = {'base_score': 0.369197, 'booster': 'gbtree', 'colsample_bylevel': 0.4, 'colsample_bytree': 1,
#          'gamma': 0.08759523291635393, 'eta': 0.072, 'max_depth': 8, 'min_child_weight': 1.6010959785458306e-10,
#          'num_class': 2, 'objective': 'multi:softprob', 'alpha': 9.817830899287001,
#          'lambda': 0.7386562144326775, 'seed': 2017, 'subsample': 0.95, 'silent': 1}
params = {'base_score': 0.369197, 'booster': 'gbtree', 'colsample_bylevel': 0.4, 'colsample_bytree': 1,
          'gamma': 0.08759523291635393, 'eta': 0.072, 'max_depth': 6, 'min_child_weight': 1.6010959785458306e-10,
          'eval_metric': 'logloss', 'objective': 'binary:logistic', 'alpha': 9.817830899287001,
          'lambda': 0.7386562144326775, 'seed': 2017, 'subsample': 0.85, 'silent': 1}
num_round = 50
d_train = xgb.DMatrix(X_train_cv, label=y_train_cv, feature_names = data_dict["feature_names"])
d_valid = xgb.DMatrix(X_valid_cv, label=y_valid_cv, feature_names = data_dict["feature_names"])
watchlist = [(d_train, 'train_cv'), (d_valid, 'valid_cv')]
bst = xgb.train(params, d_train, num_round, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(X_test, feature_names = data_dict["feature_names"])
p_test = bst.predict(d_test)
sub = pd.DataFrame()
test = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
sub['test_id'] = test['test_id']
try:
    sub['is_duplicate'] = p_test
except:
    sub['is_duplicate'] = p_test[:,1]
path = os.path.join(config.OUTPUT_DIR + '/Subm', 'xgb_manual.csv')
sub.to_csv(path, index=False)

ax = xgb.plot_importance(bst)
ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, "feature_importance"))
yticklabels = ax.get_yticklabels()[::-1]
topn = len(yticklabels)
fname = "XGBClassifier_topn_features.txt"
with open(fname, "w") as f:
    for i in range(topn):
        f.write("%s\n"%yticklabels[i].get_text())

