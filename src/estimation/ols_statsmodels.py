#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import statsmodels.api as sm
import os
import argparse

import data_functions


def ols_regression_sm(y_train, x_train):
    """
    INPUT:
    x_train, y_train: matrix and vector with logarithmized and preprocessed gravity data and (normal) PAX for training

    code performs Ordinary Least Squares regression, using statsmodels

    GravityFeatures:
    0 "Distance (km)"
    1 "Domestic (0=no)";
    2 "International (0 = no)";
    3 "Inter-EU-zone (0=no)";
    4 "same currency (0=no);
    5 population prod;
    6 population sum;
    7 catchment prod;
    8 catchment sum;
    9 GDP prod;
    10 GDP sum;
    11 PLI prod;
    12 PLI sum;
    13 nights prod;
    14 nights sum;
    15 coastal OR;
    16 island OR;
    17 poverty sum;
    18 poverty prod

    NO OUTPUT
    """
    mod = sm.OLS(y_train, x_train)
    res = mod.fit()
    print(res.summary())


def main(inFilePairStat, sos1, featureDecisionFile):
    decide = True if featureDecisionFile != '' else False
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile)
    ols_regression_sm(y_data_grav, x_data_grav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--sos1', action='store_true',
                        help='if true only sum or product is used in the gravity based models', default=False)
    parser.add_argument('--featureDecisionFile', help='directory to the feature decision file. Has to contain the '
                                                      'True-False-values for all variables.', default='')

    args = parser.parse_args()
    main(args.inFilePairStat, args.sos1, args.featureDecisionFile)
