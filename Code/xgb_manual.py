import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import ipdb
from sklearn.metrics import log_loss
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
from sklearn.cross_validation import train_test_split
import config
from utils import pkl_utils

combine_flag = True
if combine_flag:
    suffix = 'v0'
    threshold = 0.05
    cmd = "python get_feature_conf_magic.py -d 44 -o feature_conf_magic_%s.py"%suffix
    os.system(cmd)
    cmd = "python feature_combiner.py -l 1 -c feature_conf_magic_%s -n basic_magic_%s -t %.6f"%(suffix, suffix, threshold)
    os.system(cmd)
    
feature_name = "basic_magic_{}".format(suffix)
fname = os.path.join(config.FEAT_DIR+"/Combine", feature_name+config.FEAT_FILE_SUFFIX)
data_dict = pkl_utils._load(fname)
X_train = pd.DataFrame(data_dict["X_train_basic"], columns = data_dict["feature_names"])
X_test = pd.DataFrame(data_dict["X_test"], columns = data_dict["feature_names"])
y_train = data_dict["y_train"]


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)
#UPDownSampling
pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]
X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
print(np.mean(y_train))
del pos_train, neg_train

pos_valid = X_valid[y_valid == 1]
neg_valid = X_valid[y_valid == 0]
X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
print(np.mean(y_valid))
del pos_valid, neg_valid

#splitter = data_dict["splitter"]
#n_iter = data_dict["n_iter"]
#i = n_iter - 1             # use the last splitter to split the cv
#X_train_cv = data_dict["X_train_basic"][splitter[i][0], :]
#X_valid_cv = data_dict["X_train_basic"][splitter[i][1], :]
#y_train_cv = data_dict["y_train"][splitter[i][0]]
#y_valid_cv = data_dict["y_train"][splitter[i][1]]
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'eta': .02,
          'max_depth': 7,'subsample': 0.6, 'base_score': 0.2, 'silent': True}
#params = {'base_score': 0.369197, 'booster': 'gbtree', 'colsample_bylevel': 0.4, 'colsample_bytree': 1,
#          'gamma': 0.08759523291635393, 'eta': 0.072, 'max_depth': 8, 'min_child_weight': 1.6010959785458306e-10,
#          'num_class': 2, 'objective': 'multi:softprob', 'alpha': 9.817830899287001,
#          'lambda': 0.7386562144326775, 'seed': 2017, 'subsample': 0.95, 'silent': 1}

num_round = 5000
d_train_cv = xgb.DMatrix(X_train, label=y_train, feature_names = data_dict["feature_names"])
d_valid_cv = xgb.DMatrix(X_valid, label=y_valid, feature_names = data_dict["feature_names"])
watchlist = [(d_train_cv, 'train_cv'), (d_valid_cv, 'valid_cv')]
bst = xgb.train(params, d_train_cv, num_round, watchlist, early_stopping_rounds=50, verbose_eval=50)

d_train = xgb.DMatrix(pd.concat((X_train, X_valid), axis=0), label=np.concatenate((y_train, y_valid)), feature_names = data_dict["feature_names"])
d_test = xgb.DMatrix(X_test, feature_names = data_dict["feature_names"])
bst_refit = xgb.train(params, d_train, int(bst.attr('best_iteration')), verbose_eval=50)
p_test = bst_refit.predict(d_test)
sub = pd.DataFrame()
test = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
sub['test_id'] = test['test_id']
try:
    sub['is_duplicate'] = p_test
except:
    sub['is_duplicate'] = p_test[:,1]
path = os.path.join(config.OUTPUT_DIR + '/Subm', 'xgb_magic_' + suffix + '.csv')
sub.to_csv(path, index=False)

ax = xgb.plot_importance(bst)
ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, "feature_importance"))
yticklabels = ax.get_yticklabels()[::-1]
topn = len(yticklabels)
fname = "XGBClassifier_topn_features_magic.txt"
with open(fname, "w") as f:
    for i in range(topn):
        f.write("%s\n"%yticklabels[i].get_text())
