# -*- coding: utf-8 -*-
"""
@brief: config for Quora project

"""

import os
import platform

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import os_utils


# ---------------------- Overall -----------------------
TASK = "all"
# # for testing data processing and feature generation
#TASK = "sample"
#SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = ".."

DATA_DIR = "%s/Data"%ROOT_DIR
CLEAN_DATA_DIR = "%s/Clean"%DATA_DIR

FEAT_DIR = "%s/Feat"%ROOT_DIR
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = "./conf"
FEAT_COMM_NONLINEAR_DIR = "./comm_nonlinear"         # comment out features
FEAT_COMM_LINEAR_DIR = "./comm_linear"
FEAT_SELECT_DIR = "./select" 

OUTPUT_DIR = "%s/Output"%ROOT_DIR
SUBM_DIR = "%s/Subm"%OUTPUT_DIR

LOG_DIR = "%s/Log"%ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR
TMP_DIR = "%s/Tmp"%ROOT_DIR
THIRDPARTY_DIR = "%s/Thirdparty"%ROOT_DIR

# word2vec/doc2vec/glove
WORD2VEC_MODEL_DIR = "%s/word2vec"%DATA_DIR
GLOVE_WORD2VEC_MODEL_DIR = "%s/glove/gensim"%DATA_DIR
DOC2VEC_MODEL_DIR = "%s/doc2vec"%DATA_DIR

# index split
SPLIT_DIR = "%s/split"%DATA_DIR

# dictionary
WORD_REPLACER_DATA = "%s/dict/word_replacer.csv"%DATA_DIR


# ------------------------ DATA ------------------------
# provided data

TRAIN_DATA = "%s/train.csv"%DATA_DIR
TEST_DATA = "%s/test.csv"%DATA_DIR
SAMPLE_DATA = "%s/sample_submission.csv"%DATA_DIR
LABEL = "is_duplicate"                    # name of the label column

ALL_DATA_RAW = "%s/all.raw.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED = "%s/all.lemmatized.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED_STEMMED = "%s/all.lemmatized.stemmed.csv.pkl"%CLEAN_DATA_DIR
INFO_DATA = "%s/info.csv.pkl"%CLEAN_DATA_DIR

# size
TRAIN_SIZE = 404290
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 2345796
TEST_RATIO = float(TEST_SIZE) / (TEST_SIZE + TRAIN_SIZE)
VALID_SIZE_MAX = 283000 # 0.7 * TRAIN_SIZE
TRAIN_MEAN = 0.369197      #  mean of train label
TRAIN_VAR = 0.232891
TEST_MEAN = 0.175
TEST_VAR = TRAIN_VAR
"""


# ------------------------SAMPLE DATA ------------------------
# provided data

TRAIN_DATA = "%s/train.csv.short"%DATA_DIR
TEST_DATA = "%s/test.csv.short"%DATA_DIR
SAMPLE_DATA = "%s/sample_submission.csv"%DATA_DIR
LABEL = "is_duplicate"                    # name of the label column
ALL_DATA_RAW = "%s/all.raw.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED = "%s/all.lemmatized.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED_STEMMED = "%s/all.lemmatized.stemmed.csv.pkl"%CLEAN_DATA_DIR
INFO_DATA = "%s/info.csv.pkl"%CLEAN_DATA_DIR

# size
TRAIN_SIZE = 1000
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 5000
TEST_RATIO = float(TEST_SIZE) / (TEST_SIZE + TRAIN_SIZE)
VALID_SIZE_MAX = 700
TRAIN_MEAN = 0.38      # mean of train label
TRAIN_VAR = 0.2358358
TEST_MEAN = 0.175
TEST_VAR = TRAIN_VAR
"""

# cv
N_RUNS = 5
N_FOLDS = 1

# intersect count/match
STR_MATCH_THRESHOLD = 0.85

# correct query with google spelling check dict
# turn this on/off to have two versions of features/models
# which is useful for ensembling
GOOGLE_CORRECTING_QUERY = True

# auto correcting query (quite time consuming; not used in final submission)
#AUTO_CORRECTING_QUERY = False


# bm25
BM25_K1 = 1.6
BM25_B = 0.75

# svd
SVD_DIM = 100
SVD_N_ITER = 5

# xgboost
# mean of label in training set
BASE_SCORE = TRAIN_MEAN

# word2vec/doc2vec
EMBEDDING_ALPHA = 0.025
EMBEDDING_LEARNING_RATE_DECAY = 0.5
EMBEDDING_N_EPOCH = 5
EMBEDDING_MIN_COUNT = 3
EMBEDDING_DIM = 100
EMBEDDING_WINDOW = 5
EMBEDDING_WORKERS = 6

# count transformer
COUNT_TRANSFORM = np.log1p

# missing value
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

# stop words
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# ------------------------ OTHER ------------------------
RANDOM_SEED = 2017
PLATFORM = platform.system()
NUM_CORES = 4 if PLATFORM == "Darwin" else 20

DATA_PROCESSOR_N_JOBS = 4 if PLATFORM == "Darwin" else 16
AUTO_SPELLING_CHECKER_N_JOBS = 4 if PLATFORM == "Darwin" else 16
# multi processing is not faster
AUTO_SPELLING_CHECKER_N_JOBS = 16

## rgf
#RGF_CALL_EXE = "%s/rgf1.2/test/call_exe.pl"%THIRDPARTY_DIR
#RGF_EXTENSION = ".exe" if PLATFORM == "Darwin" else ""
#RGF_EXE = "%s/rgf1.2/bin/rgf%s"%(THIRDPARTY_DIR, RGF_EXTENSION)


# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CLEAN_DATA_DIR]
DIRS += [SPLIT_DIR]
DIRS += [FEAT_DIR, FEAT_CONF_DIR, FEAT_COMM_NONLINEAR_DIR, FEAT_COMM_LINEAR_DIR]
DIRS += ["%s/All"%FEAT_DIR]
DIRS += ["%s/Run%d"%(FEAT_DIR,i+1) for i in range(N_RUNS)]
DIRS += ["%s/Combine"%FEAT_DIR]
DIRS += [OUTPUT_DIR, SUBM_DIR]
DIRS += ["%s/All"%OUTPUT_DIR]
DIRS += ["%s/Run%d"%(OUTPUT_DIR,i+1) for i in range(N_RUNS)]
DIRS += [LOG_DIR, FIG_DIR, TMP_DIR]
DIRS += [WORD2VEC_MODEL_DIR, DOC2VEC_MODEL_DIR, GLOVE_WORD2VEC_MODEL_DIR]

os_utils._create_dirs(DIRS)
