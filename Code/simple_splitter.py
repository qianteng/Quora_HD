# -*- coding: utf-8 -*-
"""
@brief: splitter for Homedepot project

"""

import numpy as np
import pandas as pd
import ipdb
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
#import matplotlib.pyplot as plt
#from matplotlib_venn import venn2
#plt.rcParams["figure.figsize"] = [5, 5]

import config
from utils import pkl_utils


## Simple splitter by Qian Teng
class SimpleSplitter:
    """stratified splitter"""
    def __init__(self, dfTrain, dfTest, n_iter=5, random_state=config.RANDOM_SEED):
        self.dfTrain = dfTrain
        self.dfTest = dfTest           # not used in the current version
        self.n_iter = n_iter
        self.random_state = random_state
        
    def __str__(self):
        return "SimpleSplitter"

    def split(self):
        splitter = StratifiedKFold(n_splits = self.n_iter, random_state = self.random_state)
        rs = splitter.split(np.zeros(len(self.dfTrain)), self.dfTrain[config.LABEL])
        self.splits = [0] * self.n_iter
        run = 0
        for trainInd, validInd in rs:
            #print('1 in train {:.3f}'.format(np.sum(self.dfTrain.loc[trainInd]['is_duplicate'] == 1) / float(len(trainInd))))
            #print('1 in valid {:.3f}'.format(np.sum(self.dfTrain.loc[validInd]['is_duplicate'] == 1) / float(len(validInd))))
            #target_ratio = 0.175
            #validInd = self.under_sample(validInd, target_ratio)
            #print('1 in valid {:.3f}'.format(np.sum(self.dfTrain.loc[validInd]['is_duplicate'] == 1) / float(len(validInd))))
            self.splits[run] = trainInd, validInd
            run += 1
        return self

    def under_sample(self, index, target_ratio):
        # under sample the index list to the target ratio. (17.5%)
        current_ratio = np.sum(self.dfTrain.loc[index]['is_duplicate'] == 1) / float(len(index))
        drop_num = int((current_ratio - target_ratio) / (1 - target_ratio) * len(index))
        index_np = np.array(index)
        np.random.seed(config.RANDOM_SEED + index[0])
        drop = np.random.choice(np.where(self.dfTrain.loc[index]['is_duplicate'] == 1)[0], drop_num, False)
        index = np.delete(index_np, drop).tolist()
        return index

    def save(self, fname):
        pkl_utils._save(fname, self.splits)            
        
class RollingTimeSplitter:
    """rolling time splitter"""
    def __init__(self, dfTrain, dfTest, n_iter=5, random_state=config.RANDOM_SEED):
        self.dfTrain = dfTrain
        self.dfTest = dfTest           # not used in the current version
        self.n_iter = n_iter
        self.random_state = random_state
        
    def __str__(self):
        return "SimpleSplitter"

    def split(self):
        splitter = TimeSeriesSplit(n_splits = self.n_iter)
        rs = splitter.split(np.zeros(len(self.dfTrain)), self.dfTrain[config.LABEL])
        self.splits = [0] * self.n_iter
        run = 0
        for trainInd, validInd in rs:
            qmax_index = self.get_qmax_index()
            self.splits[run] = qmax_index[trainInd], qmax_index[validInd]
            run += 1
        return self
    
    def get_qmax_index(self):
        self.dfTrain['qmax'] = self.dfTrain.apply( lambda row: max(row["qid1"], row["qid2"]), axis=1 )
        qmax_index = self.dfTrain.sort_values(by=["qmax"], ascending=True).index
        return qmax_index

    def save(self, fname):
        pkl_utils._save(fname, self.splits)      

def main():
    
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")     # Not used
    # splits for level1
    #splitter = SimpleSplitter(dfTrain=dfTrain,
    #                          dfTest=dfTest, 
    #                          n_iter=config.N_RUNS, 
    #                          random_state=config.RANDOM_SEED,
    #)
    splitter = RollingTimeSplitter(dfTrain=dfTrain, 
                                   dfTest=dfTest, 
                                   n_iter=config.N_RUNS,
                                   random_state=config.RANDOM_SEED,
    )
    splitter.split()
    splitter.save("%s/splits_level1.pkl"%config.SPLIT_DIR)

if __name__ == "__main__":
    main()
