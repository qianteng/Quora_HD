# -*- coding: utf-8 -*-
"""
@brief: generate raw dataframe data

"""

import gc
import ipdb
import numpy as np
import pandas as pd

import config
from utils import pkl_utils

def get_q_map(train, test):
    """Get unique qid for questions in test.

    Args:
         train:
         test:
    Returns:
         q_map: Series, all questions as index, corresponding qid as value

    """    
    q1 = pd.DataFrame({'qid': train['qid1'], 'question': train['question1']})
    q2 = pd.DataFrame({'qid': train['qid2'], 'question': train['question2']})
    questions = pd.concat([q1, q2])
    questions.drop_duplicates(inplace = True)
    questions = questions[~questions['question'].duplicated()] # 571 questions in train have the same question string but different qid
    q_map = pd.Series(questions['qid'].values, index = questions['question'].values)
    new_id = max(q_map.values) + 1                    # keep the available new id 
    test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist())
    test_qs.drop_duplicates(inplace = True)
    test_map = pd.Series(index = test_qs.values)
    q_map = pd.concat([q_map, test_map])
    q_map = q_map[~q_map.index.duplicated()]
    q_map[q_map.isnull()] = range(new_id, new_id + np.sum(q_map.isnull()))
    q_map = q_map.astype(int)
    return q_map

def feature_qid(train, test):
    q_map = get_q_map(train, test)                    # calculate qid
    test['qid1'] = test['question1'].map(q_map)      
    test['qid2'] = test['question2'].map(q_map)
    
def main():
    # load provided data
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
    dfTrain.fillna(config.MISSING_VALUE_STRING, inplace=True)
    dfTest.fillna(config.MISSING_VALUE_STRING, inplace=True)
    
    #
    print("n_sample of Train {}".format(len(dfTrain)))
    print("n_sample of Test {}".format(len(dfTest)))

    # add qid, is_duplicate to test
    feature_qid(dfTrain, dfTest)
    dfTest["is_duplicate"] = np.zeros((config.TEST_SIZE))
    dfTest = dfTest.rename(columns = {'test_id':'id'})

    # concat train and test
    dfAll = pd.concat((dfTrain, dfTest), ignore_index=True)
    del dfTrain
    del dfTest
    gc.collect()

    # save data
    if config.TASK == "sample":
        dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy()
    pkl_utils._save(config.ALL_DATA_RAW, dfAll)

    # info
    dfInfo = dfAll[["id","is_duplicate"]].copy()
    pkl_utils._save(config.INFO_DATA, dfInfo)

if __name__ == "__main__":
    main()
