#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import math
import os
from sklearn.metrics import *

# FeatureSettings
numAirportFeatures = 8  # -> Pop NUTS-3; Pop NUTS-2; GDP NUTS-3; PLI NUTS-1; Nights NUTS-2; Coastal NUTS-3;
# Island NUTS-3;Poverty percentage
numInterFeatures = 5  # -> "Distance (km)"; "Domestic (0=no)"; "International (0 = no)"; "Inter-EU-zone (0=no)";
# "same currency (0=no)"
numAllFeatures = 2 * numAirportFeatures + numInterFeatures
binaryGrav = [1, 2, 3, 4, 15, 16]
first = [5, 7, 9, 11, 13, 17]
featureGravNames = ["Distance (km)", "Domestic (0=no)", "International (0 = no)", "Inter-EU-zone (0=no)",
                    "same currency (0=no)", 'population prod', 'population sum', 'catchment prod', 'catchment sum',
                    'GDP prod', 'GDP sum', 'PLI prod', 'PLI sum', 'nights prod', 'nights sum', 'coastal OR',
                    'island OR', 'poverty sum', 'poverty prod']

# GravityFeatures:
# 0 "Distance (km)"
# 1 "Domestic (0=no)";
# 2 "International (0 = no)";
# 3 "Inter-EU-zone (0=no)";
# 4 "same currency (0=no); 
# 5 population prod;
# 6 population sum;
# 7 catchment prod;
# 8 catchment sum;
# 9 GDP prod;
# 10 GDP sum;
# 11 PLI prod;
# 12 PLI sum;
# 13 nights prod;
# 14 nights sum;
# 15 coastal OR;
# 16 island OR;
# 17 poverty sum;
# 18 poverty prod

binaryNoGrav = [1, 2, 3, 4, 10, 11, 18, 19]


# DirectFeatures:
# 0 "Distance (km)"
# 1 "Domestic (0=no)";
# 2 "International (0 = no)";
# 3 "Inter-EU-zone (0=no)";
# 4 "same currency (0=no);
# 5 "Pop NUTS-3" Airport A
# 6 "Pop NUTS-2" Airport A
# 7 "GDP NUTS-3" Airport A
# 8 "PLI NUTS-1" Airport A
# 9 "Nights NUTS-2" Airport A
# 10 "Coastal NUTS-3" Airport A
# 11 "Island NUTS-3" Airport A
# 12 "Poverty percentage" Airport A
# 13 "Pop NUTS-3" Airport B
# 14 "Pop NUTS-2" Airport B
# 15 "GDP NUTS-3" Airport B
# 16 "PLI NUTS-1" Airport B
# 17 "Nights NUTS-2" Airport B
# 18 "Coastal NUTS-3" Airport B
# 19 "Island NUTS-3" Airport B
# 20 "Poverty percentage" Airport B


def readInstance(name, prediction = False):
    """
    INPUT: 
    name: filepath to "pairwise-stat-data.csv"
    prediction: True, if the data is used for prediction, False, if it is used for training. 
                If set to True, "NA" is allowed in the PAX-column, else it is not and every
                row, that contains "NA" in this column is deleted. 

    OUTPUT: size of the read matrix as two integers and the matrix as an array. All rows, where
            some data is missing (shown by "NA"), are deleted. 
    """
    array = []
    i = -1
    j = 0
    PAX_col=-1
    with open(name, newline="\n") as f:
        csvreader = csv.reader(f, delimiter=";")
        new = True
        for row in csvreader:
            j = 0
            if new:
                i += 1
                array.append([])
            if i == 0:
                for value in row:
                    array[i].append([])
                    array[i][j] = value
                    if value == 'PAX':
                        PAX_col = j
                    j += 1
            else:
                for value in row:
                    if value != 'NA' or (prediction and j == PAX_col):
                        new = True
                        try:
                            new_value = float(value)
                        except ValueError:
                            if j == PAX_col:
                                new_value = 1
                            else:
                                new = False
                                break
                        if len(array[i]) > j:
                            array[i][j] = new_value
                        else:
                            array[i].append([])
                            array[i][j] = new_value
                        j += 1
                    else:
                        new = False
                        break
    return i, j, array


def readStatInstance(name):
    """
    INPUT: filepath
    OUTPUT: size of the read matrix as two integers and the matrix as an array
    """
    array = []
    i = -1
    j = 0
    r = 0
    with open(name, newline="\n") as f:
        csvreader = csv.reader(f, delimiter=";")
        new = True
        for row in csvreader:
            j = 1
            if new:
                i += 1
                array.append([])
                array[i].append([])
            if i == 0:
                array[0][0] = 0.0
                for value in row:
                    array[i].append([])
                    array[i][j] = value
                    j += 1
            else:
                for value in row:
                    if value != 'NA':
                        new = True
                        new_value = float(value)
                        if len(array[i]) > j:
                            array[i][j] = new_value
                        else:
                            array[i].append([])
                            array[i][j] = new_value
                        j += 1
                    else:
                        new = False
                        break
                    array[i][0] = r
            r += 1
    return i, j, array


def readScoreInstance(name):
    """
    INPUT: filepath
    OUTPUT: size of the read matrix as two integers and the matrix as an array
    """
    array = []
    i = -1
    j = 0
    with open(name, newline="\n") as f:
        csvreader = csv.reader(f, delimiter=",")
        new = True
        for row in csvreader:
            j = 0
            if new:
                i += 1
                array.append([])
            if i == 0:
                for value in row:
                    array[i].append([])
                    array[i][j] = value
                    j += 1
            else:
                for value in row:
                    if value != 'NA':
                        new = True
                        try:
                            new_value = float(value)
                        except ValueError:
                            new_value = value
                            if i == 1:
                                if j == 0:
                                    new_value = float('inf')
                        if len(array[i]) > j:
                            array[i][j] = new_value
                        else:
                            array[i].append([])
                            array[i][j] = new_value
                        j += 1
                    else:
                        new = False
                        break
        array[1].append([])
        array[1][-1] = 'SUMMARY'
    return i, j, array


def getScores(name, y_train, y_vali, pred_train, pred_vali, variable, num_variables, kfold=10, y_grav_train=(),
              y_grav_vali=(), pred_grav_train=(), pred_grav_vali=(), *, epochs=0):
    """
    code performs an evaluation of the model with scores [R2, adjusted R2, mean loss, max loss, mean loss percentage 
    (the training and the validation samples are evaluated separately)] where the corresponding variables are saved
    in a csv-file with a header and a summary of the evaluation.

    INPUT:
    name: place to save the scores
    y_train: demand array of the training samples
    y_vali: demand array of the validation samples
    pred_train: predicted demand of the training samples
    pred_vali: predicted demand of the validation samples
    variable: data to recreate the model 
    num_variables: how many trainable variables exist in this model?
    kfold: 0 if there was no validation, otherwise the k of kfold cross validation
    y_grav_train: demand array of the (logarithmized) training samples
    y_grav_vali: demand array of the (logarithmized) validation samples
    pred_grav_train: predicted demand of the (logarithmized) training samples
    pred_grav_vali: predicted demand of the (logarithmized) validation samples
    """
    if len(np.shape(pred_train)) > 2:
        pred_train = pred_train[:, 0, 0]
    if len(np.shape(y_train)) > 2:
        y_train = y_train[:, 0, 0]
    if len(np.shape(y_grav_train)) > 2:
            y_grav_train = y_grav_train[:, 0, 0]
    if len(np.shape(pred_grav_train)) > 2:
        y_grav_train = pred_grav_train[:, 0, 0]
    if kfold == 0:
        y_data = y_train
        y_pred = pred_train
        y_grav = y_grav_train
        pred_grav = pred_grav_train
    else:
        if len(np.shape(pred_vali)) > 2:
            pred_vali = pred_vali[:, 0, 0]
        if len(np.shape(y_vali)) > 2:
            y_vali = y_vali[:, 0, 0]
        if len(np.shape(y_grav_vali)) > 2:
            y_grav_vali = y_grav_vali[:, 0, 0]
        if len(np.shape(pred_grav_vali)) > 2:
            y_grav_vali = pred_grav_vali[:, 0, 0]
        y_data = np.concatenate((y_train, y_vali))
        y_pred = np.concatenate((pred_train, pred_vali))
        y_grav = np.concatenate((y_grav_train, y_grav_vali))
        pred_grav = np.concatenate((pred_grav_train, pred_grav_vali))
    r2 = r2_score(y_data, y_pred)
    adj_r2 = 1 - (1 - r2) * (y_data.shape[0] - 1) / (y_data.shape[0] - num_variables - 1)
    if len(y_grav_train) == 0 and len(pred_grav_train) == 0:
        r2_grav = 0
        adj_r2_grav = 0
    else:
        r2_grav = r2_score(y_grav, pred_grav)
        adj_r2_grav = 1 - (1 - r2_grav) * (y_grav.shape[0] - 1) / (y_grav.shape[0] - num_variables - 1)
    mean_loss = mean_absolute_error(y_data, y_pred)
    max_loss = max_error(y_data, y_pred)
    loss_perc = mean_absolute_percentage_error(y_data, y_pred)

    if kfold != 0.0:
        # calculate the values for validation and training data
        mean_loss_v = mean_absolute_error(y_vali, pred_vali)
        max_loss_v = max_error(y_vali, pred_vali)
        loss_perc_v = mean_absolute_percentage_error(y_vali, pred_vali)
        mean_loss_t = mean_absolute_error(y_train, pred_train)
        max_loss_t = max_error(y_train, pred_train)
        loss_perc_t = mean_absolute_percentage_error(y_train, pred_train)
        r2_v = r2_score(y_vali, pred_vali)
        adj_r2_v = 1 - (1 - r2_v) * (y_vali.shape[0] - 1) / (y_vali.shape[0] - num_variables - 1)
        if len(y_grav) == 0 and len(pred_grav) == 0:
            r2_grav_v = 0
            adj_r2_grav_v = 0
        else:
            r2_grav_v = r2_score(y_grav_vali, pred_grav_vali)
            adj_r2_grav_v = 1 - (1 - r2_grav_v) * (y_grav_vali.shape[0] - 1) / (y_grav_vali.shape[0] - num_variables - 1)
        r2_t = r2_score(y_train, pred_train)
        adj_r2_t = 1 - (1 - r2_t) * (y_train.shape[0] - 1) / (y_train.shape[0] - num_variables - 1)
        if len(y_grav) == 0 and len(pred_grav) == 0:
            r2_grav_t = 0
            adj_r2_grav_t = 0
        else:
            r2_grav_t = r2_score(y_grav_train, pred_grav_train)
            adj_r2_grav_t = 1 - (1 - r2_grav_t) * (y_grav_train.shape[0] - 1) / (y_grav_train.shape[0] - num_variables
                                                                               - 1)

    # write the file
    if os.path.exists('%s.csv' % name) and os.path.getsize('%s.csv' % name) > 0:
        with open('%s.csv' % name, mode='r') as file:
            file_reader = csv.reader(file, delimiter=',')
            lines = list(file_reader)
            line = lines[-1]
        length = int(line[0]) + 1  # number of the current model
    else:
        # if the file doesn't exist, a header and a provisional summary line is written
        with open('%s.csv' % name, mode='a') as file:
            file_writer_header = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if kfold != 0.0:
                headrow = ["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss", 
                                "mean loss percentage", "Validation-R2", "Validation-Adj R2", 
                                "Validation-Grav R2", "Validation-Grav Adj R2", "Validation-mean loss",
                                "Validation-max loss", "Validation-mean loss percentage", "Training-R2", 
                                "Training-Adj R2", "Training-Grav R2", "Training-Grav Adj R2", 
                                "Training-mean loss", "Training-max loss", "Training-mean loss percentage",
                                "epochs", "filename(s)"]
                file_writer_header.writerow(headrow)
                sumrow = ["Summary", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                file_writer_header.writerow(sumrow)
            else:
                headrow = ["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                            "mean loss percentage", "filename(s)"]
                file_writer_header.writerow(headrow)
                file_writer_header.writerow(["Summary", 0, 0, 0, 0, 0, 0, 0])
        length = 1  # number of the current model
        with open('%s.csv' % name, mode='r') as file:
            file_reader = csv.reader(file, delimiter=',')
            lines = list(file_reader)
    with open('%s.csv' % name, mode='w') as file:
        lines_new = lines
        if kfold != 0.0:
            # new line with all variables
            new_line = [length, r2, adj_r2, r2_grav, adj_r2_grav, mean_loss, max_loss, loss_perc, r2_v, adj_r2_v,
                        r2_grav_v, adj_r2_grav_v, mean_loss_v, max_loss_v, loss_perc_v, r2_t, adj_r2_t, r2_grav_t,
                        adj_r2_grav_t, mean_loss_t, max_loss_t, loss_perc_t, epochs, variable]
        else:
            new_line = [length, r2, adj_r2, r2_grav, adj_r2_grav, mean_loss, max_loss, loss_perc, variable]
        summary = lines[1]  # old summary line
        new_summary = summary  # new summary line
        if kfold != 0.0:
            for i in range(1, len(summary), 1):
                #print(new_line[i])
                new_summary[i] = (float(summary[i]) * (length - 1) + float(new_line[i])) / length  # calculate new summary
        else:
            for i in range(1, len(summary), 1):
                new_summary[i] = (float(summary[i]) * (length - 1) + float(new_line[i])) / length  # calculate new summary
        lines_new[1] = new_summary  # change the summary line
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerows(lines_new)  # write all rows
        file_writer.writerow(new_line)  # add new line


def writeScores(name, scores, kfold=0.1, first=None):
    """
    INPUT:
    name: place to save the scores
    scores: the scores that should be written
    kfold: 0 if there was no validation, otherwise the k of kfold cross validation
    first: A text the new line should begin with. (f.e. "SFS 2")
    
    NO OUTPUT
    the given scores are saved in the given file
    """
    if os.path.exists('%s.csv' % name) and os.path.getsize('%s.csv' % name) > 0:
        with open('%s.csv' % name, mode='r') as file:
            file_reader = csv.reader(file, delimiter=',')
            lines = list(file_reader)
            line = lines[-1]
        length = int(line[0]) + 1  # number of the current model
    else:
        # if the file doesn't exist, a header and a provisional summary line is written
        with open('%s.csv' % name, mode='a') as file:
            file_writer_header = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if kfold != 0.0:
                headrow = ["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss", 
                            "mean loss percentage", "Validation-R2", "Validation-Adj R2", 
                            "Validation-Grav R2", "Validation-Grav Adj R2", "Validation-mean loss",
                            "Validation-max loss", "Validation-mean loss percentage", "Training-R2", 
                            "Training-Adj R2", "Training-Grav R2", "Training-Grav Adj R2", 
                            "Training-mean loss", "Training-max loss", "Training-mean loss percentage",
                            "epochs", "filename(s)"]
                file_writer_header.writerow(headrow)
                if first is None:
                    sumrow = ["Summary", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                0]
                    file_writer_header.writerow(sumrow)
            else:
                file_writer_header.writerow(["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                                             "mean loss percentage", "filename(s)"])
                if first is None:
                    file_writer_header.writerow(["Summary", 0, 0, 0, 0, 0, 0, 0])
        length = 1  # number of the current model
        with open('%s.csv' % name, mode='r') as file:
            file_reader = csv.reader(file, delimiter=',')
            lines = list(file_reader)
    with open('%s.csv' % name, mode='w') as file:
        lines_new = lines
        new_line = scores
        if first is None:
            scores[0] = length  # new line with all variables
            summary = lines[1]  # old summary line
            new_summary = summary  # new summary line
            if kfold != 0.0:
                # calculate new summary
                for i in range(1, len(summary), 1):
                    new_summary[i] = (float(summary[i]) * (length - 1) + float(new_line[i])) / length
            else:
                for i in range(1, len(summary), 1):
                    new_summary[i] = (float(summary[i]) * (length - 1) + float(new_line[i])) / length
            lines_new[1] = new_summary  # change the summary line  
        else:
            scores[0] = first
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerows(lines_new)  # write all rows
        file_writer.writerow(new_line)  # add new line


def getGravityDataScores(inFilePairStat, sos1=False, decide=False, featureDecisionFile='', binary=False, prediction = False):
    """
    INPUT: 
    inFilePairStat: file, which contains the Pairwise Statistical Data for each connection
    sos1: True, if either the sum or the product is used.
    decide: True of the features should be selected in any way
    featureDecisionFile: file that contains True/False for each feature whether it's used or not.
                        Based on that file the features are selected.
    binary: True, if the binary features should be in {0,1}, False if they should be projected to {1,e}
    prediction: True, if the data is used for prediction, False, if it is used for training. 
    
    OUTPUT:
    not logarithmized matrix with the gravity features that should be used as columns
    demand: array with the number of persons who want to fly between the airports
    """

    # get data
    numFeatures = numGravFeatures(False)
    size, _, pairwise_data = readInstance(inFilePairStat, prediction)
    use = whichVariables(sos1, decide, featureDecisionFile)
    matrix = np.zeros([size - 1, numFeatures], dtype=np.float128)  # -1 because of the headline
    demand = np.zeros([size - 1])

    for i in range(size - 1):
        r = 0
        # the gravity features
        for j in range(numGravFeatures(False)):
            if j == numInterFeatures - 1:
                # the demand
                demand[i] = int(pairwise_data[i + 1][j + 2])
                continue
            value = pairwise_data[i + 1][j + 2]
            if r not in binaryGrav:
                matrix[i][r] = value
            else:  # transform 0,1 to 1,e
                if binary:
                    matrix[i][r] = value
                else:
                    matrix[i][r] = np.exp(value)
                if r == 1:  # add international feature
                    if binary:
                        matrix[i][r + 1] = 1 - value
                    else:
                        matrix[i][r + 1] = np.exp(1 - value)
                    r += 1
            r += 1
    for j in range(numGravFeatures(False) - 1, -1, -1):
        if not use[j]:
            matrix = np.delete(matrix, obj=j, axis=1)
    return matrix, demand


def getGravityPreprocessedDataScores(inFilePairStat, sos1=False, decide=False, featureDecisionFile='', binary=False, prediction = False):
    """
    INPUT: 
    inFilePairStat: file, which contains the Pairwise Statistical Data for each connection
    sos1: True if only one of the sum or the product is used
    decide: True if the features should be selected in any way
    featureDecisionFile: file that contains True/False for each feature whether it's used or not
                        Based on that file the features are selected
    binary: True, if the binary features should be in {0,1}, False if they should be projected to {1,e}
    prediction: True, if the data is used for prediction, False, if it is used for training. 
    
    OUTPUT:
    not logarithmized but preprocessed matrix with the gravity features that should be used as columns
    (normalized to mean 0 and variance 1)
    demand: array with the number of persons who want to fly between the airports
    """

    if sos1 and featureDecisionFile == '':
        raise IOError('featureDecisionFile not Found. If sos1 is true you should define it.')

    # get data
    size, _, pairwise_data = readInstance(inFilePairStat, prediction)
    use = whichVariables(sos1, decide, featureDecisionFile)

    matrix = np.zeros([size - 1, numGravFeatures(False)])  # -1 because of the headline
    demand = np.zeros([size - 1])

    normList = np.zeros([numGravFeatures(False), 2])
    vector = np.zeros([size - 1])

    for i in range(size - 1):
        r = 0
        # the gravity features
        for j in range(numGravFeatures(False)):
            if j == numInterFeatures - 1:
                # the demand
                demand[i] = int(pairwise_data[i + 1][j + 2])
                continue

            if r not in binaryGrav:
                if i == 0:  # get mean and variance for normalization
                    for n in range(size - 1):
                        vector[n] = pairwise_data[n + 1][j + 2]
                    normList[r][0] = np.mean(vector)
                    normList[r][1] = np.var(vector)
                    normList[r][1] = math.sqrt(normList[r][1])
                    # normalization to almost unit variance and zero mean
                if normList[r][1] == 0:
                    matrix[i][r] = pairwise_data[i + 1][j + 2]
                else:
                    matrix[i][r] = (pairwise_data[i + 1][j + 2] - normList[r][0]) / normList[r][1]
            else:  # transform 0,1 to 1,e
                value = pairwise_data[i + 1][j + 2]
                if binary:
                    matrix[i][r] = value
                else:
                    matrix[i][r + 1] = np.exp(value)
                if r == 1:  # add international feature
                    if binary:
                        matrix[i][r + 1] = 1 - value
                    else:
                        matrix[i][r + 1] = np.exp(1 - value)
                    r += 1
            r += 1
    for j in range(numGravFeatures(False) - 1, -1, -1):
        if not use[j]:
            matrix = np.delete(matrix, obj=j, axis=1)
    return matrix, demand


def getGravityData(inFilePairStat, sos1=False, decide=False, featureDecisionFile='', logDemand=True, prediction = False):
    """
    INPUT: 
    inFilePairStat: file, which contains the Pairwise Statistical Data for each connection
    sos1: True, if either the sum or the product is used.
    decide: True if the features should be selected in any way
    featureDecisionFile: file that contains True/False for each feature whether it is used or not.
                        Based on that file the features are selected.
    logDemand: True, if the demand should be logarithmized, False, if not
    prediction: True, if the data is used for prediction, False, if it is used for training.  
    
    OUTPUT:
    logarithmized matrix with the gravity features that should be used as columns
    demand: array with the (maybe logarithmized) number of persons who want to fly between the airports
    """

    # get data
    size, _, pairwise_data = readInstance(inFilePairStat, prediction)
    use = whichVariables(sos1, decide, featureDecisionFile)
    matrix = np.zeros([size - 1, numGravFeatures(False)])  # -1 because of the headline
    demand = np.zeros([size - 1])

    for i in range(size - 1):
        r = 0
        # the gravity features
        for j in range(numGravFeatures(False)):
            if j == numInterFeatures - 1:
                # the demand
                if logDemand:
                    demand[i] = np.log(pairwise_data[i + 1][j + 2])
                else:
                    demand[i] = pairwise_data[i + 1][j + 2]
                continue
            if r not in binaryGrav:
                matrix[i][r] = np.log(pairwise_data[i + 1][j + 2])
            else:  # binaryGrav = 0,1
                matrix[i][r] = pairwise_data[i + 1][j + 2]
                if r == 1:  # add international feature
                    matrix[i][r + 1] = 1 - pairwise_data[i + 1][j + 2]
                    r += 1
            r += 1
    for j in range(numGravFeatures(False) - 1, -1, -1):
        if not use[j]:
            matrix = np.delete(matrix, obj=j, axis=1)
    return matrix, demand


def getGravityPreprocessedData(inFilePairStat, sos1=False, decide=False, featureDecisionFile='', logDemand=True):
    """
    INPUT: 
    inFilePairStat: file, which contains the Pairwise Statistical Data for each connection
    sos1: True, if either the sum or the product is used.
    decide: True of the features should be selected in any way
    featureDecisionFile: file that contains True/False for each feature whether it's used or not.
    Based on that file the features are selected.
    logDemand: True, if the demand should be logarithmized, False, if not
    
    OUTPUT:
    logarithmized and preprocessed matrix with the gravity features that should be used as columns
    (normalized to mean 0 and variance 1)
    demand: array with the (maybe logarithmized) number of persons who want to fly between the airports
    """

    if sos1 and featureDecisionFile == '':
        raise IOError('featureDecisionFile not Found. If sos1 is true you should set it.')

    # get data
    size, _, pairwise_data = readInstance(inFilePairStat)
    use = whichVariables(sos1, decide, featureDecisionFile)

    matrix = np.zeros([size - 1, numGravFeatures(False)])  # -1 because of the headline
    demand = np.zeros([size - 1])

    normList = np.zeros([numGravFeatures(False), 2])
    vector = np.zeros([size - 1])

    for i in range(size - 1):
        r = 0
        # the gravity features
        for j in range(numGravFeatures(False)):
            if j == numInterFeatures:
                # the demand
                if logDemand:
                    demand[i] = np.log(pairwise_data[i + 1][j + 2])
                else:
                    demand[i] = pairwise_data[i + 1][j + 2]
                continue

            if r not in binaryGrav:
                if i == 0:  # get mean and variance for normalization
                    for n in range(size - 1):
                        vector[n] = np.log(pairwise_data[n + 1][j + 2])
                    normList[r][0] = np.mean(vector)
                    normList[r][1] = np.var(vector)
                    normList[r][1] = math.sqrt(normList[r][1])
                # normalization to almost unit variance and zero mean
                if normList[r][1] != 0:
                    matrix[i][r] = (np.log(pairwise_data[i + 1][j + 2]) - normList[r][0]) / normList[r][1]
                else:
                    matrix[i][r] = np.log(pairwise_data[i + 1][j + 2])
            else:  # binaryGrav = 0,1
                matrix[i][r] = pairwise_data[i + 1][j + 2]
                if r == 1:  # add international feature
                    matrix[i][r + 1] = 1 - pairwise_data[i + 1][j + 2]
                    r += 1
            r += 1
    for j in range(numGravFeatures(False) - 1, -1, -1):
        if not use[j]:
            matrix = np.delete(matrix, obj=j, axis=1)
    return matrix, demand


def whichVariables(sos1, decide, name):
    """
    sos1: If true the percentage of usage for each product and the belonging sum variable is compared and just one of
    them is used
    decide: If true only the features with a percentage of usage that is higher or equal to 0.5 is used
    name: file that contains True/False values, which imply whether a feature should be used or not
    """
    use = np.ones(2 * (numAirportFeatures - 1) + numInterFeatures, dtype=bool)
    if sos1:
        matrix = np.genfromtxt('%s' % name, delimiter=',', dtype=bool)
        matrix = matrix.astype(int)
        if np.shape(matrix)[0] != numGravFeatures(False):
            mean = np.mean(matrix, axis=0)
        else:
            mean = matrix
        for j in first:
            if mean[j] > mean[j + 1]:
                use[j + 1] = False
            else:
                use[j] = False
        return use
    elif decide:
        matrix = np.genfromtxt('%s' % name, delimiter=',', dtype=bool)
        matrix = matrix.astype(int)
        if np.shape(matrix)[0] != numGravFeatures(False):
            mean = np.mean(matrix, axis=0)
        else:
            mean = matrix
        for j in range(numGravFeatures(False)):
            if mean[j] < 0.5:
                use[j] = False
        return use
    else:
        return use


def numFeatures():
    """
    returns the number of features (not based on the gravity model)
    """
    return 2 * numAirportFeatures + numInterFeatures


def numGravFeatures(sos1):
    """
    returns the number of features based on the gravity model

    INPUT:
    sos1: True, if either the sum or the product is used, False if both are used. 
    """
    if sos1:
        return numAirportFeatures + numInterFeatures  # coastal and island both have no sum and prod, but just "OR"
    else:
        return 2 * (numAirportFeatures - 1) + numInterFeatures


def nameGravFeatures():
    """
    returns the name of the Gravity Features
    """
    return featureGravNames
