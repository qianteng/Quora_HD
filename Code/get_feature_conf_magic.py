# -*- coding: utf-8 -*-
"""
@brief: generate feature conf for magic features and features calculated by me

@note: 
        - one can tune the MANDATORY_FEATS and COMMENT_OUT_FEATS to generate different feature subset

"""

import re
import os
import pandas as pd
from optparse import OptionParser

import config
from utils import time_utils


INCLUDE_FEATS = [
".+"
]


COUNT_FEATS = [
"Freq", 
"Len", 
"Count", 
"Size", 
"Position", 
]



NOT_COUNT_FEATS = ["Norm", "Ratio"]


MANDATORY_FEATS = [
    "LSA_magic_"         # pretend that the magic features are LSA
]

COMMENT_OUT_FEATS = [
".+"
]
for file in sorted(os.listdir(config.FEAT_COMM_NONLINEAR_DIR)):  # add the first 0.3 of features in the files
    feature_list = pd.read_table(config.FEAT_COMM_NONLINEAR_DIR + '/' + file, header = None).values.flatten().tolist()
    keep_ratio = 0.0
    MANDATORY_FEATS = MANDATORY_FEATS + feature_list[:int(len(feature_list) * keep_ratio)]

def _check_include(fname):
    for v in INCLUDE_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_count_feat(fname):
    for v in NOT_COUNT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return False
    for v in COUNT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_lsa_matrix(fname):
    pat = re.compile("^LSA")
    if len(re.findall(pat, fname)) > 0:
        return True
    return False


def _check_mandatory(fname):
    for v in MANDATORY_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


def _check_comment_out(fname):
    for v in COMMENT_OUT_FEATS:
        pat = re.compile(v)
        if len(re.findall(pat, fname)) > 0:
            return True
    return False


header_pattern = """
# -*- coding: utf-8 -*-
\"\"\"
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: one feature conf

Generated by
python %s -d %d -o %s

Format:
FEATURE_NAME : (MANDATORY, TRANSFORM)

\"\"\"

import config
from feature_transformer import SimpleTransform, ColumnSelector

LSA_COLUMNS = range(%d)

feature_dict = {

"""


def _create_feature_conf(lsa_columns, outfile):
    res = header_pattern%(__file__, int(lsa_columns), outfile, int(lsa_columns))

    folders = [config.FEAT_DIR, config.FEAT_DIR+"/All"]
    for folder in folders:
        try:
            for file in sorted(os.listdir(folder)):
                if config.FEAT_FILE_SUFFIX in file:
                    fname = file.split(".")[0]
                    if _check_include(fname):
                        line = ""
                        mandatory = _check_mandatory(fname)
                        if not mandatory and _check_comment_out(fname):
                            continue
                            line += "# "
                        line += "'%s' : "%fname
                        if mandatory:
                            line += "(True, "
                        else:
                            line += "(False, "
                        if _check_lsa_matrix(fname):
                            if int(lsa_columns) > 0:
                                line += "ColumnSelector(LSA_COLUMNS)),\n"
                            else:
                                continue
                        elif _check_count_feat(fname):
                            line += "SimpleTransform(config.COUNT_TRANSFORM)),\n"
                        else:
                            line += "SimpleTransform()),\n"
                        res += line
        except:
            pass
    res += "}\n"

    with open(os.path.join(config.FEAT_CONF_DIR, outfile), "w") as f:
        f.write(res)


def parse_args(parser):
    parser.add_option("-d", "--dim", default=1, type=int, dest="lsa_columns",
        help="lsa_columns")
    parser.add_option("-o", "--outfile", default="feature_conf_%s.py"%time_utils._timestamp(),
        type="string", dest="outfile", help="outfile")

    (options, args) = parser.parse_args()
    return options, args


def main(options):
    _create_feature_conf(lsa_columns=options.lsa_columns, outfile=options.outfile)


if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
