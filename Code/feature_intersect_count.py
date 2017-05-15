# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: intersect count features

"""

import re
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from utils import logging_utils, time_utils, pkl_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ----------------------------------------------------------------------------
# How many ngrams of obs are in target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# IntersectCount: 4 (i.e., AB, AB, AB, AC)
# IntersectRatio: 4/6
class IntersectCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "IntersectCount_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return s

class IntersectRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "IntersectRatio_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return np_utils._try_divide(s, len(obs_ngrams))


# ----------------------------------------------------------------------------
# How many cooccurrence ngrams between obs and target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# CooccurrenceCount: 7 (i.e., AB x 2 + AB x 2 + AB x 2 + AC x 1)
# CooccurrenceRatio: 7/(6 x 5)
class CooccurrenceCount_Ngram(BaseEstimator):
    """obs_corpus and target_corpus are symmetric"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "CooccurrenceCount_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return s

class CooccurrenceRatio_Ngram(BaseEstimator):
    """obs_corpus and target_corpus are symmetric"""
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="", str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "CooccurrenceRatio_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return np_utils._try_divide(s, len(obs_ngrams)*len(target_ngrams))
    
class IntersectCount_Nterm(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, nterm, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.nterm = nterm
        self.nterm_str = ngram_utils._nterm_str_map[self.nterm]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "IntersectCount_%s"%self.nterm_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_nterms = ngram_utils._nterms(obs_tokens, self.nterm)
        target_nterms = ngram_utils._nterms(target_tokens, self.nterm)
        s = 0.
        for w1 in obs_nterms:
            for w2 in target_nterms:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return s
    
class IntersectRatio_Nterm(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, nterm, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.nterm = nterm
        self.nterm_str = ngram_utils._nterm_str_map[self.nterm]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "IntersecRatio_%s"%self.nterm_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_nterms = ngram_utils._nterms(obs_tokens, self.nterm)
        target_nterms = ngram_utils._nterms(target_tokens, self.nterm)
        s = 0.
        for w1 in obs_nterms:
            for w2 in target_nterms:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return np_utils._try_divide(s, len(obs_nterms))

class CooccurrenceCount_Nterm(BaseEstimator):
    """obs_corpus and target_corpus are symmetric"""
    def __init__(self, obs_corpus, target_corpus, nterm, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.nterm = nterm
        self.nterm_str = ngram_utils._nterm_str_map[self.nterm]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "CooccurrenceCount_%s"%self.nterm_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_nterms = ngram_utils._nterms(obs_tokens, self.nterm)
        target_nterms = ngram_utils._nterms(target_tokens, self.nterm)
        s = 0.
        for w1 in obs_nterms:
            for w2 in target_nterms:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return s
    
class CooccurrenceRatio_Nterm(BaseEstimator):
    """obs_corpus and target_corpus are symmetric"""
    def __init__(self, obs_corpus, target_corpus, nterm, aggregation_mode="", 
        str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.nterm = nterm
        self.nterm_str = ngram_utils._nterm_str_map[self.nterm]
        self.str_match_threshold = str_match_threshold
        
    def __name__(self):
        return "CooccurrenceRatio_%s"%self.nterm_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_nterms = ngram_utils._nterms(obs_tokens, self.nterm)
        target_nterms = ngram_utils._nterms(target_tokens, self.nterm)
        s = 0.
        for w1 in obs_nterms:
            for w2 in target_nterms:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return np_utils._try_divide(s, len(obs_nterms)*len(target_nterms))    
    
# ---------------------------- Main --------------------------------------
def main():
    logname = "generate_feature_intersect_count_%s.log"%time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

    # Ngram
    generators = [
        IntersectCount_Ngram, 
        IntersectRatio_Ngram, 
    ]
    obs_fields_list = [['question1'], ['question2']]
    target_fields_list = [['question2'], ['question1']]
    ngrams = [1, 2, 3, 4, 5, 12, 123]     # only 1,2,3,4,5,12,123 available, see ngram_utils.py
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                pf.go()

    # Ngram symmetric
    generators = [
        CooccurrenceCount_Ngram, 
        CooccurrenceRatio_Ngram,
        #CooccurrenceCount_Nterm,    # not used in Quora project, takes long to run
        #CooccurrenceRatio_Nterm, 
    ]
    obs_fields_list = [['question1']]
    target_fields_list = [['question2']]
    ngrams = [1, 2, 3, 4, 5, 12, 123]     # only 1,2,3,4,5,12,123 available, see ngram_utils.py
    nterms = [2, 3, 4]     # only 1,2,3,4 available,(uniterms is the same as unigrams) see ngram_utils.py
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            if generator.__name__[-5:] == 'Ngram':
                for ngram in ngrams:
                    param_list = [ngram]
                    pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                    pf.go()
            elif generator.__name__[-5:] == 'Nterm':
                for nterm in nterms:
                    param_list = [nterm]
                    pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger)
                    pf.go()
            else:
                 print("Wrong Generator")
                 pass

if __name__ == "__main__":
    main()
