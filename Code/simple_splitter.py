# -*- coding: utf-8 -*-
"""
@brief: splitter for Homedepot project

"""

import numpy as np
import pandas as pd
import ipdb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
plt.rcParams["figure.figsize"] = [5, 5]

import config
from utils import pkl_utils


## Simple splitter by Qian Teng
class SimpleSplitter:
    # This is not ideal yet
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
        #ipdb.set_trace()
        self.splits = [0] * self.n_iter
        run = 0
        for trainInd, validInd in rs:
            self.splits[run] = trainInd, validInd
            run += 1
        return self

    def save(self, fname):
        pkl_utils._save(fname, self.splits)            
        


def main():
    
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")    

    # splits for level1
    splitter = SimpleSplitter(dfTrain=dfTrain, 
                                dfTest=dfTest, 
                                n_iter=config.N_RUNS, 
                                random_state=config.RANDOM_SEED, 
                                )
    splitter.split()
    splitter.save("%s/splits_level1.pkl"%config.SPLIT_DIR)

if __name__ == "__main__":
    main()
