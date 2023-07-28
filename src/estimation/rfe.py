#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
import sklearn
import joblib
import argparse
import glob

import data_functions


def perform_rfe(y_data, y_datas, x_data, x_datas, kfold, fileName,
                modelName, cv=None):
    """
    code performs recurrent feature elimination with cross validation, decides on
    the best number of parameters and saves the scores, the model and the used 
    variables in different files. The training is performed on logarithmized
    data that include the features defined below.

    INPUT:
    x_data, y_data: matrix and vector with logarithmized data and PAX for training
    x_datas, y_datas: not logarithmized data and PAX for evaluating
    kfold: If 0 there is no validation, so no validation data should be predicted
    fileName: name for the csv-file where the scores should be stored in.
    modelName: name for the file where the model should be stored in.
    cv: int, cross-validation generator or an iterable. Determines the cross-validation
        splitting strategy. Possible inputs for cv are: None, to use the default 
        5-fold cross-validation, integer, to specify the number of folds, CV splitter, An
        iterable yielding (train, test) splits as arrays of indices. 
e 
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
    numFeatures = np.shape(x_datas)[1]

    # instantiating the model
    lr = LinearRegression(fit_intercept=False)
    rfe = RFECV(lr, cv=cv, scoring=sklearn.metrics.make_scorer(sklearn.metrics.r2_score))
    rfe = rfe.fit(x_data, y_data)
    lr.fit(rfe.transform(x_data), y_data)

    # evaluation and saving
    r = 0
    regr_coeffs = np.zeros(np.shape(x_datas)[1])
    for i in range(np.shape(x_datas)[1]):
        if rfe.support_[i]:
            regr_coeffs[i] = lr.coef_[r]
            r += 1

    k = rfe.n_features_
    act_modelName = '%s_k%i' % (modelName, k)
    var = rfe.support_

    joblib.dump(lr, '%s_lr' % act_modelName)  # to load write "lr = joblib.load({act_modelName}_lr)"
    joblib.dump(rfe, '%s_rfe' % act_modelName)  # to load write "rfe = joblib.load({act_modelName}_rfe)"
    # to use write f.e. lr.predict(rfe.transform(...)) or result = lr.score(rfe.transform(...), ...)

    if kfold != 0.0:
        for kfold_index in range(kfold):
            x_train_k = x_data[cv[kfold_index][0]]
            pred_train_grav = lr.predict(rfe.transform(x_train_k))
            pred_train = rfe.transform(x_datas[cv[kfold_index][0]])
            pred_t = np.power(pred_train, lr.coef_)
            pred_train = np.prod(pred_t, axis=1)
                
            pred_vali = rfe.transform(x_datas[cv[kfold_index][1]])
            pred_v = np.power(pred_vali, lr.coef_)
            pred_vali = np.prod(pred_v, axis=1)
            x_vali_k = x_data[cv[kfold_index][1]]
            pred_vali_grav = lr.predict(rfe.transform(x_vali_k))
                
            # save scores for k
            data_functions.getScores('%s_k%i' % (fileName, k), y_datas[cv[kfold_index][0]],
                                     y_datas[cv[kfold_index][1]], pred_train, pred_vali, [act_modelName, var], k, kfold,
                                     y_data[cv[kfold_index][0]], y_data[cv[kfold_index][1]], pred_train_grav,
                                     pred_vali_grav)
    else:
        pred_train_grav = lr.predict(rfe.transform(x_data))
        pred_train = rfe.transform(x_datas)
        pred_t = np.power(pred_train, lr.coef_)
        pred_train = np.prod(pred_t, axis=1)
        pred_vali = 0

        data_functions.getScores('%s_k%i' % (fileName, k), y_datas, 0, pred_train, 0, [act_modelName, var], k, kfold,
                                 y_data, 0, pred_train_grav, 0)
    # overview over the scores for different k
    # save which variables are used for k
    with open('%s_k%i_var_True_False.csv' % (fileName, k), mode='a') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(var) 
    for k in range(numFeatures + 1):
        filek = glob.glob('%s_k%i.csv' % (fileName, k), recursive=False)
        if len(filek) > 0:
            i, j, array = data_functions.readScoreInstance(filek[0])
            # overview over the scores for different k
            data_functions.writeScores('%s_k' % fileName, array[1], kfold, k)


def main(inFilePairStat, sos1, featureDecisionFile, fileName, modelName):
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, sos1, featureDecisionFile)
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat, sos1, sos1,
                                                                               featureDecisionFile)
    perform_rfe(y_data_grav,  y_data_grav_score, x_data_grav, x_data_grav_score, 0, fileName, modelName, 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--sos1', action='store_true',
                        help='if true in the gravity based models only sum or product is used', default=False)
    parser.add_argument('--featureDecisionFile', help='directory to the feature decision file for sos1. Has to contain '
                                                      'the True-False-values for all variables.', default='')
    parser.add_argument('--fileName', help='where to store the score-results, make sure the directory to the file '
                                           'exists', required=True)
    parser.add_argument('--modelName', help='where to store the trained models, make sure the directory to the file'
                                            ' exists', required=True)

    args = parser.parse_args()
    main(args.inFilePairStat, args.sos1, args.featureDecisionFile, args.fileName, args.modelName)
