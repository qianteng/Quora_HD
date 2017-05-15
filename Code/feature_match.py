# -*- coding: utf-8 -*-
"""
@brief: match based features

"""

import re
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


class MatchQueryCount(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchQueryCount"

    def _str_whole_word(self, str1, str2, i_):
        cnt = 0
        if len(str1) > 0 and len(str2) > 0:
            try:
                while i_ < len(str2):
                    i_ = str2.find(str1, i_)
                    if i_ == -1:
                        return cnt
                    else:
                        cnt += 1
                        i_ += len(str1)
            except:
                pass
        return cnt

    def transform_one(self, obs, target, id):
        return self._str_whole_word(obs, target, 0)


class MatchQueryRatio(MatchQueryCount):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "MatchQueryRatio"

    def transform_one(self, obs, target, id):
        return np_utils._try_divide(super().transform_one(obs, target, id), len(target.split(" ")))


#------------- Longest match features -------------------------------
class LongestMatchSize(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "LongestMatchSize"

    def transform_one(self, obs, target, id):
        return dist_utils._longest_match_size(obs, target)

class LongestMatchRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        
    def __name__(self):
        return "LongestMatchRatio"

    def transform_one(self, obs, target, id):
        return dist_utils._longest_match_ratio(obs, target)

# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_match_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)
    
    generators = [
        MatchQueryCount, 
        MatchQueryRatio, 
        LongestMatchSize, 
        LongestMatchRatio, 
    ]
    obs_fields_list = []
    target_fields_list = []
    ## question1 in question2
    obs_fields_list.append( ['question1'] )
    target_fields_list.append( ['question2'])
    ## question2 in question1
    obs_fields_list.append( ['question2'] )
    target_fields_list.append( ['question1'] )
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            param_list = []
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
            pf.go()

if __name__ == "__main__":
    main()
