#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import argparse

import data_functions


def ols_regression(y_train, y_vali, y_trains, y_valis, x_train, x_vali, x_trains, x_valis, kfold, fileName, modelName):
    """
    performs Ordinary Least Squares regression and saves the scores and the trained model in a file. The training is
    performed on logarithmized data that include the features defined below.

    INPUT:
    x_train, y_train: matrix and vector with logarithmized gravity data and PAX for training
    x_vali, y_vali: matrix and vector with logarithmized gravity data and PAX for validation, if no validation
                    should be performed enter 0
    x_trains, y_trains: not logarithmized data and PAX for evaluating
    fileName: name for the csv-file where the scores should be stored in
    modelName: name for the file where the model should be stored in
    kfold: If 0 there is no validation, so no validation data should be predicted


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
    """

    lr = LinearRegression(fit_intercept=False)  # instantiating the model
    lr.fit(x_train, y_train)  # training

    # evaluation and saving
    pred_train_grav = lr.predict(x_train)
    regr_coeffs = lr.coef_
    pred_train = x_trains
    pred_t = np.power(pred_train, regr_coeffs)
    pred_train = np.prod(pred_t, axis=1)

    if kfold != 0.0:
        pred_vali = x_valis
        pred_v = np.power(pred_vali, regr_coeffs)
        pred_vali = np.prod(pred_v, axis=1)
        pred_vali_grav = lr.predict(x_vali)
        y_data = np.concatenate((y_vali, y_train))
        y_grav = np.concatenate((pred_vali_grav, pred_train_grav))
    else:
        pred_vali = 0
        pred_vali_grav = 0
        y_data = y_train
        y_grav = pred_train_grav

    joblib.dump(lr, modelName)  # to load write "loaded_model = joblib.load(modelName)"
    data_functions.getScores(fileName, y_trains, y_valis, pred_train, pred_vali, modelName, len(lr.coef_), kfold,
                             y_train, y_vali, pred_train_grav, pred_vali_grav)


def main(inFilePairStat, sos1, featureDecisionFile, fileName, modelName):
    decide = True if featureDecisionFile != '' else False
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile, True)
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat, sos1, decide,
                                                                               featureDecisionFile)
    ols_regression(y_data_grav, [], y_data_grav_score, [], x_data_grav, [], x_data_grav_score, [], 0, fileName,
                   modelName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--sos1', action='store_true',
                        help='if true only sum or product is used in the gravity based models', default=False)
    parser.add_argument('--featureDecisionFile', help='directory to the feature decision file. '
                                                      'Has to contain True-False-values for all variables.',  
                                                      default='')
    parser.add_argument('--fileName', help='where to store the score-results, directory to file has to exist',
                        required=True)
    parser.add_argument('--modelName', help='where to store the trained models, directory to file has to exist',
                        required=True)
    args = parser.parse_args()
    main(args.inFilePairStat, args.sos1, args.featureDecisionFile, args.fileName, args.modelName)    
