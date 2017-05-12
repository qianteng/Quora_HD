
# -*- coding: utf-8 -*-
"""
@brief: one feature conf

Generated by
python /Users/qt/Desktop/Notes/Quora_HD/Code/get_feature_conf_linear.py -d 1 -o feature_conf_linear_2017-05-08-00-18.py

Format:
FEATURE_NAME : (MANDATORY, TRANSFORM)

"""

import config
from feature_transformer import SimpleTransform, ColumnSelector

LSA_COLUMNS = range(1)

feature_dict = {

'CooccurrenceCount_Bigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceCount_Bigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceCount_Trigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceCount_Trigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceCount_Unigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceCount_Unigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'CooccurrenceRatio_Bigram_question1_x_question2_1D' : (False, SimpleTransform()),
'CooccurrenceRatio_Bigram_question2_x_question1_1D' : (False, SimpleTransform()),
'CooccurrenceRatio_Trigram_question1_x_question2_1D' : (False, SimpleTransform()),
'CooccurrenceRatio_Trigram_question2_x_question1_1D' : (False, SimpleTransform()),
'CooccurrenceRatio_Unigram_question1_x_question2_1D' : (False, SimpleTransform()),
'CooccurrenceRatio_Unigram_question2_x_question1_1D' : (False, SimpleTransform()),
'DigitCount_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'DigitCount_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'DigitRatio_question1_1D' : (False, SimpleTransform()),
'DigitRatio_question2_1D' : (False, SimpleTransform()),
'DocEntropy_question1_1D' : (False, SimpleTransform()),
'DocEntropy_question2_1D' : (False, SimpleTransform()),
'DocFreq_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'DocFreq_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'DocId_question1_1D' : (False, SimpleTransform()),
'DocId_question2_1D' : (False, SimpleTransform()),
'DocLen_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'DocLen_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Bigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Bigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Trigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Trigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Unigram_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectCount_Unigram_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectNormPosition_Bigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Bigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Trigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectNormPosition_Unigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectPosition_Bigram_Max_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Max_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Mean_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Mean_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Median_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Median_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Min_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Min_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Std_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Bigram_Std_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Max_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Max_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Mean_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Mean_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Median_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Median_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Min_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Min_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Std_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Trigram_Std_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Max_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Max_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Mean_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Mean_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Median_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Median_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Min_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Min_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Std_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectPosition_Unigram_Std_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'IntersectRatio_Bigram_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectRatio_Bigram_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectRatio_Trigram_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectRatio_Trigram_question2_x_question1_1D' : (False, SimpleTransform()),
'IntersectRatio_Unigram_question1_x_question2_1D' : (False, SimpleTransform()),
'IntersectRatio_Unigram_question2_x_question1_1D' : (False, SimpleTransform()),
'LongestMatchRatio_question1_x_question2_1D' : (False, SimpleTransform()),
'LongestMatchRatio_question2_x_question1_1D' : (False, SimpleTransform()),
'LongestMatchSize_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'LongestMatchSize_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'MatchQueryCount_question1_x_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'MatchQueryCount_question2_x_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'MatchQueryRatio_question1_x_question2_1D' : (False, SimpleTransform()),
'MatchQueryRatio_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Bigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Trigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocBM25_Unigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Bigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Trigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTFIDF_Unigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Bigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Trigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Max_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Max_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Mean_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Mean_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Median_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Median_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Min_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Min_question2_x_question1_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Std_question1_x_question2_1D' : (False, SimpleTransform()),
'StatCoocTF_Unigram_Std_question2_x_question1_1D' : (False, SimpleTransform()),
'UniqueCount_Bigram_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueCount_Bigram_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueCount_Trigram_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueCount_Trigram_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueCount_Unigram_question1_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueCount_Unigram_question2_1D' : (False, SimpleTransform(config.COUNT_TRANSFORM)),
'UniqueRatio_Bigram_question1_1D' : (False, SimpleTransform()),
'UniqueRatio_Bigram_question2_1D' : (False, SimpleTransform()),
'UniqueRatio_Trigram_question1_1D' : (False, SimpleTransform()),
'UniqueRatio_Trigram_question2_1D' : (False, SimpleTransform()),
'UniqueRatio_Unigram_question1_1D' : (False, SimpleTransform()),
'UniqueRatio_Unigram_question2_1D' : (False, SimpleTransform()),
}