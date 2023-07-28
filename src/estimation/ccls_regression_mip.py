#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import gurobipy as gp
from scipy import linalg
import os
import joblib
import argparse

import data_functions


def ccls_regression(y_train, y_vali, y_trains, y_valis, x_train, x_vali, x_trains, x_valis, sos1, kfold, fileName,
                    modelName):
    """
    code performs cardinality-constrained least-squares regression for automatic attraction, parameter selection
    computes R^2, maximal absolute loss, mean absolute percentage loss, mean loss and saves them.
    The CCLS problems are formulated as MIPs and solved exactly using gurobiPy;
    the problem is solved for all admissible numbers of nonzero regression coefficients (i.e., from 1 to 18):
        min ||Ax-b||_2^2 s.t. ||x||_0 <= k

    INPUT:
    x_train, y_train: matrix and vector with logarithmized gravity fata and PAX for training
    x_vali, y_vali: matrix and vector with logarithmized gravity fata and PAX for validation, if no validation
    should be performed enter 0
    x_trains, y_trains: not logarithmized data and PAX for evaluating
    x_valis, y_valis: not logarithmized gravity fata and PAX for validation, if no validation
    should be performed enter 0
    sos1: If True a constraint is added, which ensures that only one of the product or sum variables are used, not both.
    fileName: name for the csv-file where the scores should be stored in.
    modelName: name for the file where the model should be stored in.

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

    NO OUTPUT: scores like R2, adjusted R2, max absolute loss, mean percentage loss and mean loss are saved in the
    given fileName and the belonging model in modelName.
    """
    if os.path.exists('%s_k.csv' % fileName):
        os.remove('%s_k.csv' % fileName)

    numFeatures = np.shape(x_train)[1]  # numGravFeatures(False)

    regr_mat = x_train
    # add constraints to use sum sos1 multiply
    if sos1:
        I = np.zeros((6, 2 * numFeatures))
        I[0][5 + numFeatures] = 1  # population
        I[0][6 + numFeatures] = 1
        I[1][7 + numFeatures] = 1  # catchment
        I[1][8 + numFeatures] = 1
        I[2][9 + numFeatures] = 1  # GDP
        I[2][10 + numFeatures] = 1
        I[3][11 + numFeatures] = 1  # PLI
        I[3][12 + numFeatures] = 1
        I[4][13 + numFeatures] = 1  # nights
        I[4][14 + numFeatures] = 1
        I[5][17 + numFeatures] = 1  # poverty
        I[5][18 + numFeatures] = 1
        I = I[:, numFeatures:]
        
    # Model
    # cardinality-constrained LS regression:
    #   min ||Ax-b||_2^2 s.t. ||x||_0 <= k
    # use Gurobi as solver for the following Big-M MIP formulation of this problem:
    #      min x'(A'A)x - 2b'Ax s.t. -My <= x <= My, 1'y <= k, y binary
    # <=>  min x'(A'A)x - 2b'Ax s.t. [I,-M*I;-I,-M*I;0',1'][x;y] <= [0;0;k], y binary

    # build the model
    mip = gp.Model("MIP")
    mip.Params.MIPGap = 1e-9          # default 1e-4
    mip.Params.OutputFlag = 0
    bigM = 20

    # define A = [I,-M*I;-I,-M*I]
    A1 = np.eye(numFeatures)
    A2 = bigM * np.eye(numFeatures)
    A3 = np.concatenate([A1, -A2], axis=1)
    A4 = np.concatenate([-A1, -A2], axis=1)
    A_x = np.concatenate((A3, A4))

    # define Q = (A'A)
    Q = np.transpose(regr_mat) @ regr_mat

    regr_rhs = y_train
    rhs = np.zeros(2 * numFeatures)

    # define the lower and upper bound
    lb = np.concatenate([-bigM * np.ones(numFeatures), np.zeros(numFeatures)])
    ub = np.concatenate([bigM * np.ones(numFeatures), np.ones(numFeatures)])

    # define the variable type
    vtype = np.concatenate([67 * np.ones(numFeatures), 66 * np.ones(numFeatures)])
    vtype = vtype.astype(str)
    vtype = np.where(vtype == "66.0", "B", "C")

    # add variables
    x = mip.addMVar(numFeatures, ub=ub[0:numFeatures], lb=lb[0:numFeatures], vtype=vtype[0:numFeatures], name='x')
    y = mip.addMVar(numFeatures, ub=ub[numFeatures:], lb=lb[numFeatures:], vtype=vtype[numFeatures:], name='y')
    obj = -2 * np.dot(np.transpose(regr_rhs), regr_mat)  # = -2b'A

    # set objective
    mip.setObjective(x @ Q @ x + obj @ x)  # min x'(A'A)x - 2b'Ax
    # add constraint
    A_1x = A_x[:, :numFeatures]
    A_1y = A_x[:, numFeatures:]
    mip.addConstr(A_1x @ x + A_1y @ y <= rhs, "A")  # [I,-M*I;-I,-M*I][x,y] <= [0;0]

    # add constraint that either product or sum of the airport features are used
    if sos1:
        mip.addConstr(I @ y <= np.ones(6), "sos1")

    A_y = np.concatenate([np.zeros((1, numFeatures)), np.ones((1, numFeatures))], axis=1)[0]  # build 1'y <= k
    # produce results for all allowed numbers of nonzero regression coefficients
    K = 40
    K = max(1, min(numFeatures, K))
    for k in range(1, K, 1):
    #for k in range(K-1, 0, -1):
        mip.addConstr(A_y[numFeatures:] @ y <= k)  # adds the next 1'y <= k constraint
        mip.update()
        mip.optimize()
        regr_coeffs = x.X[0:numFeatures]
        mip.remove(mip.getConstrs()[-1])  # removes the last 1'y <= k constraint
        # evaluation and saving
        pred_train = x_trains
        np.power(pred_train[0], regr_coeffs)
        np.power(pred_train[1500:], regr_coeffs)
        pred_t = np.power(pred_train, regr_coeffs)
        pred_train = np.prod(pred_t, axis=1)

        pred_train_grav = np.dot(x_train, regr_coeffs)
        if kfold != 0:
            pred_vali = x_valis
            pred_v = np.power(pred_vali, regr_coeffs)
            pred_vali = np.prod(pred_v, axis=1)

            pred_vali_grav = np.dot(x_vali, regr_coeffs)
        else:
            pred_vali = 0
            pred_vali_grav = 0
        # scores etc.
        n = np.concatenate((x.X, y.X))
        variable = {'k': k, 'regr_coeffs': n}

        act_modelName = '%s_k%i' % (modelName, k)
        var = np.zeros(numFeatures, bool)
        for j in range(numFeatures):
            if n[j + numFeatures] == 1:
                var[j] = True
        joblib.dump(variable, act_modelName)  # to load write "loaded_model = joblib.load(modelName)"
        
        # save scores for k
        data_functions.getScores('%s_k%i' % (fileName, k), y_trains, y_valis, pred_train, pred_vali,
                                 [act_modelName, var], k, kfold, y_train, y_vali, pred_train_grav, pred_vali_grav)
        i, j, array = data_functions.readScoreInstance('%s_k%i.csv' % (fileName, k))
        # overview over the scores for different k
        data_functions.writeScores('%s_k' % fileName, array[1], kfold, k)
        if not sos1 or k in [10, 11, 12, 13]:
            # save which variables are used for k
            with open('%s_k%i_var_True_False.csv' % (fileName, k), mode='a') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                file_writer.writerow(var) 
        elif k == K - 1 and sos1:
            # save which variables are used for maximal k
            with open('%s_maxk_var_True_False.csv' % fileName, mode='a') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                file_writer.writerow(var)
    

def main(inFilePairStat, sos1, fileName, modelName):
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat)
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat)
    ccls_regression(y_data_grav, 0, y_data_grav_score, 0, x_data_grav, 0, x_data_grav_score, 0, sos1, 0, fileName,
                    modelName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--sos1', action='store_true',
                        help='if true only sum or product is used in the gravity based models', default=False)
    parser.add_argument('--fileName', help='where to store the score-results, the directory to the file must exist',
                        required=True)
    parser.add_argument('--modelName', help='where to store the trained models, the directory to the file must exist',
                        required=True)
    args = parser.parse_args()
    main(args.inFilePairStat, args.sos1, args.fileName, args.modelName)
