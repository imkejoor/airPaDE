#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import gurobipy as gp
from scipy import linalg
import os
import joblib
import argparse
from sklearn.linear_model import LinearRegression
import time

import data_functions
import ccls_regression_mip


stop_time = True
if stop_time:
    K = 20
    start = np.zeros(K)
    end = np.zeros(K)
    needed_time = np.zeros(K)


def ccls_group_regression(y_data, y_datas, x_data, x_datas, sos1, kfold, fileName, modelName, cv=None):
    """
    code performs cardinality-constrained least-squares regression for automatic attraction, parameter selection
    computes R^2, maximal absolute loss, mean absolute percentage loss, mean loss and saves them.
    The CCLS problems are formulated as MIPs and solved exactly using gurobiPy;
    the problem is solved for all admissible numbers of nonzero regression coefficients (i.e., from 1 to 18).
    For all given (train, test) yieldings (for example a 10-fold) the same nonzero regression coefficients are used, 
    but the real value can be different for each yielding. 
        min sum_i  ||Ax_i-b_i||_2^2 s.t. ||x_i||_0 <= k for all i
    The scores like R2, adjusted R2, max absolute loss, mean percentage loss and mean loss are saved in the
    given fileName and the belonging model in modelName.

    INPUT:
    x_data, y_data: matrix and vector with logarithmized data and PAX for training
    x_datas, y_datas: not logarithmized data and PAX for evaluating
    sos1: If True a constraint is added, which ensures that only one of the product or sum variables are used, not both.
    kfold: If 0 there is no validation, so no validation data should be predicted. 
    fileName: name for the csv-file where the scores should be stored in.
    modelName: name for the file where the model should be stored in.
    cv: If kfold != 0 An iterable yielding (train, test) splits as arrays of indices.
        If kfold == 0, cv is not used. 
    

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
    if os.path.exists('%s_k.csv' % fileName):
        os.remove('%s_k.csv' % fileName)

    if kfold == 0:
        ccls_regression_mip.ccls_regression(y_data, [], y_datas, [], x_data, [], x_datas, [], sos1, kfold, fileName,
                                            modelName)
    else:
        numFeatures = np.shape(x_data)[1]  # numGravFeatures(False)

        #regr_mat = x_train
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
        #   min sum_i  ||Ax_i-b_i||_2^2 s.t. ||x_i||_0 <= k for all i
        # use Gurobi as solver for the following Big-M MIP formulation of this problem:
        #      min sum_i  x_i'(A'A)x_i - 2b'Ax s.t. -My <= x_i <= My for all i, 1'y <= k, y binary 
        # <=>  min x_i'(A'A)x_i - 2b'Ax_i s.t. [I,-M*I;-I,-M*I;0',1'][x_i;y] <= [0;0;k] for all i, y binary

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

        #regr_rhs = y_train
        rhs = np.zeros(2 * numFeatures)

        # define the lower and upper bound
        lb = np.concatenate([-bigM * np.ones(numFeatures), np.zeros(numFeatures)])
        ub = np.concatenate([bigM * np.ones(numFeatures), np.ones(numFeatures)])

        # define the variable type
        vtype = np.concatenate([67 * np.ones(numFeatures), 66 * np.ones(numFeatures)])
        vtype = vtype.astype(str)
        vtype = np.where(vtype == "66.0", "B", "C")
        
        # add variables
        y = mip.addMVar(numFeatures, ub=ub[numFeatures:], lb=lb[numFeatures:], vtype=vtype[numFeatures:], name='y')
        obj = np.zeros((kfold, numFeatures))
        Q = np.zeros((kfold, numFeatures, numFeatures))
        x = []
        for k in range(kfold):
            x.append(mip.addMVar(numFeatures, ub=ub[0:numFeatures], lb=lb[0:numFeatures], vtype=vtype[0:numFeatures],
                                 name='x%s' % str([k])))
            x_train_k = x_data[cv[k][0]]
            y_train_k = y_data[cv[k][0]]
            obj[k] = -2 * np.dot(np.transpose(y_train_k), x_train_k)  # = -2b'A
            # define Q = (A'A)
            Q[k] = np.transpose(x_train_k) @ x_train_k
            # add constraint
            A_1x = A_x[:, :numFeatures]
            A_1y = A_x[:, numFeatures:]
            mip.addConstr(A_1x @ x[k] + A_1y @ y <= rhs, "A")  # [I,-M*I;-I,-M*I][x,y] <= [0;0]
    
        # set objective
        mip.setObjective(np.sum([x[k] @ Q[k] @ x[k] + obj[k] @ x[k] for k in range(kfold)]))  # min x'(A'A)x - 2b'Ax

        # add constraint that either product or sum of the airport features are used
        if sos1:
            mip.addConstr(I @ y <= np.ones(6), "sos1")

        A_y = np.concatenate([np.zeros((1, numFeatures)), np.ones((1, numFeatures))], axis=1)[0]  # build 1'y <= k
        # produce results for all allowed numbers of nonzero regression coefficients
        K = 40
        K = max(1, min(numFeatures, K))
        for k in range(1, K, 1):
        #for k in range(K-1, 0, -1):
            if stop_time:
                start[k] = time.time()
            mip.addConstr(A_y[numFeatures:] @ y <= k)  # adds the next 1'y <= k constraint
            mip.update()
            mip.optimize()
            mip.remove(mip.getConstrs()[-1])  # removes the last 1'y <= k constraint
            # evaluation and saving
            # scores etc.
            act_modelName = '%s_k%i' % (modelName, k)
            var = np.zeros(numFeatures, bool)

            x_data_transformed = x_data
            x_datas_transformed = x_datas
            for j in range(numFeatures - 1, -1, -1):
                if y.X[j] == 1:
                    var[j] = True
                else:    
                    x_data_transformed = np.delete(x_data_transformed, obj=j, axis=1)
                    x_datas_transformed = np.delete(x_datas_transformed, obj=j, axis=1)
            #print(np.shape(x_data_transformed))
            
            lr = LinearRegression(fit_intercept=False)
            lr.fit(x_data_transformed, y_data)
            regr_coeffs = lr.coef_
            n = np.concatenate((regr_coeffs, y.X))
            variable = {'k': k, 'regr_coeffs': n}
            joblib.dump(variable, act_modelName)  # to load write "loaded_model = joblib.load(modelName)"

            for kfold_index in range(kfold):
                pred_train = x_datas_transformed[cv[kfold_index][0]]
                pred_t = np.power(pred_train, regr_coeffs)
                pred_train = np.prod(pred_t, axis=1)
                
                x_train_k = x_data_transformed[cv[kfold_index][0]]
                pred_train_grav = np.dot(x_train_k, regr_coeffs)
                pred_vali = x_datas_transformed[cv[kfold_index][1]]
                pred_v = np.power(pred_vali, regr_coeffs)
                pred_vali = np.prod(pred_v, axis=1)
                x_vali_k = x_data_transformed[cv[kfold_index][1]]
                pred_vali_grav = np.dot(x_vali_k, regr_coeffs)
                
                # save scores for k
                data_functions.getScores('%s_k%i' % (fileName, k), y_datas[cv[kfold_index][0]],
                                         y_datas[cv[kfold_index][1]], pred_train, pred_vali, [act_modelName, var], k,
                                         kfold, y_data[cv[kfold_index][0]], y_data[cv[kfold_index][1]], pred_train_grav,
                                         pred_vali_grav)
            # overview over the scores for different k
            if not sos1 or k in range(6, 14):
                # save which variables are used for k
                with open('%s_k%i_var_True_False.csv' % (fileName, k), mode='a') as file:
                    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    file_writer.writerow(var) 
            elif k == K - 1 and sos1:
                # save which variables are used for maximal k
                with open('%s_maxk_var_True_False.csv' % fileName, mode='a') as file:
                    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    file_writer.writerow(var)
            i, j, array = data_functions.readScoreInstance('%s_k%i.csv' % (fileName, k))
            data_functions.writeScores('%s_k' % fileName, array[1], kfold, k)
            if stop_time:
                end[k] = time.time()
                needed_time[k] += end[k] - start[k]
        if stop_time:
            for k_time in range(K):
                print('%i: {:5.3f}s'.format(needed_time[k_time]) % k_time)
        
        
def main(inFilePairStat, sos1, fileName, modelName):
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat)
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat)
    ccls_group_regression(y_data_grav, y_data_grav_score, x_data_grav, x_data_grav_score, sos1, 0, fileName, modelName)


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
