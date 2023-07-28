#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import joblib
import argparse
import statsmodels.api as sm

import data_functions


def ppml_regression(y_train, y_vali, x_train, x_vali, kfold, fileName, modelName):
    """
    INPUT:
    x_train, y_train: matrix and vector with logarithmized gravity data and (normal) PAX for training
    x_vali, y_vali: matrix and vector with logarithmized gravity data and (normal) PAX for validation, if no validation
    should be performed enter 0
    x_trains, y_trains: not logarithmized data and PAX for evaluating
    fileName: name for the csv-file where the scores should be stored in
    modelName: name for the file where the model should be stored in
    kfold: If 0 there is no validation, so no validation data should be predicted

    code performs Ordinary Least Squares Regression

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

    NO OUTPUT: scores like R2, adjusted R2, max absolute loss, mean percentage loss and mean loss are saved in fileName
    and the belonging model in modelName.
    """
    numFeatures = np.shape(x_train)[1]
    pr = sm.Poisson(y_train, x_train)
    res = pr.fit(start_params=np.ones(numFeatures), maxiter=500, cov_type='HC0', disp=True)

    # evaluation and saving
    regr_coeffs = res.params
    pred_train_grav = res.predict(x_train)

    if kfold != 0.0:
        pred_vali_grav = res.predict(x_vali)
    else:
        pred_vali_grav = 0

    joblib.dump(res, modelName)  # to load write "loaded_model = joblib.load(modelName)"
    data_functions.getScores(fileName, y_train, y_vali, pred_train_grav, pred_vali_grav, modelName, len(regr_coeffs),
                             kfold)


def main(inFilePairStat, sos1, featureDecisionFile, fileName, modelName):
    decide = True if featureDecisionFile != '' else False
    # use demand that is not logarithmized
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile, False)
    ppml_regression(y_data_grav, [], x_data_grav, [], 0, fileName, modelName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--sos1', action='store_true',
                        help='if true only sum or product is used in the gravity based models', default=False)
    parser.add_argument('--featureDecisionFile', help='directory to the feature decision file. Has to contain the '
                                                      'True-False-values for all variables.', default='')
    parser.add_argument('--fileName', help='where to store the score-results, directory needs to exist', required=True)
    parser.add_argument('--modelName', help='where to store the trained models, directory needs to exist',
                        required=True)

    args = parser.parse_args()
    main(args.inFilePairStat, args.sos1, args.featureDecisionFile, args.fileName, args.modelName)    
