import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedKFold
import time
import os
import glob
import subprocess
import csv
import functools
import operator
import tensorflow as tf

import data_functions
import lav_regression
import ols_regression
import kr_regression
import sv_regression
import neural_network
import ols_noGrav
import ppml_regression


def show_correlation(inFilePairStat):
    """
    prints Correlation coefficients for product and sum matrices
    """
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, False)
    print("Correlation matrices for ln-Data")
    for i in [5, 7, 9, 11, 13, 17]:
        print(np.corrcoef(x_data_grav[:, i], x_data_grav[:, i+1]))
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat, False)
    print("Correlation matrices for normal Data")
    for i in [5, 7, 9, 11, 13, 17]:
        print(np.corrcoef(x_data_grav_score[:, i], x_data_grav_score[:, i+1]))
    
    print("Now random sum and product vectors are compared")
    a = np.random.randint(0, 100000, 10000)
    b = np.random.randint(0, 100000, 10000)
    prod = np.multiply(a, b)
    sums = a + b
    print(np.corrcoef(prod, sums))


def formatNumber(n, digits=4):
    """
    Returns the input number formatted and rounded to digits.

    Parameters
    ----------
    n: float, int
    digits: int

    Returns
    -------
    float
    """
    formatter = '{:.' + '{}'.format(digits) + 'f}'
    x = round(n, digits)
    return formatter.format(x)


def summary(fileNames, outFile, kfold, decide):
    """
    writes a summary of score-given files in another file 

    Parameters
    ----------
    fileNames: contains all filenames which should be summarized
    outFile: file where the summary should be stored in
    kfold: 0 if no  validation
    decide: True/ False If not all Features are used enter True
    """
    if os.path.isdir(outFile):
        outFile = '%s.csv' % outFile
    if os.path.exists(outFile):
        os.remove(outFile)
    with open(outFile, mode='a') as file:
        file_writer = csv.writer(file, delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if kfold != 0:
            file_writer.writerow(["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                                  "mean loss percentage", "Validation-R2", "Validation-Adj R2", "Validation-Grav R2",
                                  "Validation-Grav Adj R2", "Validation-mean loss", "Validation-max loss",
                                  "Validation-mean loss percentage", "Training-R2", "Training-Adj R2",
                                  "Training-Grav R2", "Training-Grav Adj R2", "Training-mean loss", "Training-max loss",
                                  "Training-mean loss percentage", "epochs", "filename(s)"])
        else:
            file_writer.writerow(["Model", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                                  "mean loss percentage", "filename(s)"])
    for model in fileNames:
        onlyFSModel = 'individual'
        modelCSV = '%s.csv' % model
        if os.path.exists(modelCSV):
            i, j, array = data_functions.readScoreInstance(modelCSV)
            modelCopy = model + '_copy'
            if os.path.exists('%s.csv' % modelCopy):
                os.remove('%s.csv' % modelCopy)
            for m in range(i - 1):
                if len(array[m+2]) == 23 and kfold > 0:
                    array[m+2].insert(-1,0)
                data_functions.writeScores(modelCopy, array[m+2], kfold)
            os.remove(modelCSV)
            os.rename('%s.csv' % modelCopy, modelCSV)
            summary = array[1]
            #print(array[1])
            if kfold != 0:
                if len(summary) == 26:
                    del summary[23:24]
                for l in range(1, 23):
                    if l in [5, 6, 12, 13, 19, 20, 22]:
                        summary[l] = round(float(summary[l]))
                    else:
                        summary[l] = formatNumber(float(summary[l]), 4)
            else:
                for l in range(1, 8):
                    if l in [5, 6]:
                        summary[l] = round(float(summary[l]))
                    else:
                        summary[l] = formatNumber(float(summary[l]), 4)
            head, model_tail = os.path.split(model)
            if model_tail.find('lav') != -1:
                summary[0] = 'LAV'
            elif model_tail.find('ols') != -1 and model.find('nG') == -1:
                summary[0] = 'OLS'
            elif model_tail.find('sv') != -1:
                summary[0] = 'SVR'
            elif model_tail.find('kr') != -1:
                summary[0] = 'KR'
            elif model_tail.find('nn') != -1:
                summary[0] = 'NN'
            elif model_tail.find('ppml') != -1:
                summary[0] = 'PPML'
            elif model_tail.find('ols') != -1 and model.find('nG') != -1:
                summary[0] = 'OLS nG'
            if decide:
                onlyFeatures = 'individual'
                if model_tail.find('ccls') != -1:
                    onlyFSModel = 'CCLS'
                elif model_tail.find('ccls_group') != -1:
                    onlyFSModel = 'CCLS GROUP'
                elif model_tail.find('rfe') != -1:
                    onlyFSModel = 'RFE'
                elif model_tail.find('sfs') != -1:
                    onlyFSModel = 'SFS'
                for k in range(data_functions.numGravFeatures(False)):
                    if modelCSV.find('k%i.csv' % k) != -1:
                        onlyFeatures = k
                        break
                summary[0] = '%s %s %s' % (summary[0], onlyFSModel, str(onlyFeatures))
            with open(outFile, mode='a') as file:
                file_writer = csv.writer(file, delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                file_writer.writerow(summary)


def createFeatureDecisionFiles(onlyFeatureNumber, onlyFeatureSelectionModel, outDirResults, number, inFilePairStat,
                               kfold=10):
    """
    creates Feature Decision Files for all given Parameters
    
    Parameters
    -----------
    onlyFeatureNumber: List of numbers, for which the decision file should be created
    onlyFeatureSelectionModel: List of models, that decide which features should be used
    outDirResults: Directory where the results of feature selection should be stored in
    number: the number of repeats of feature selection
    kfold: k for k-fold-cross validation
    inFilePairStat: input file for pairwise statistical data
    """
    fileNames = [[]]
    if kfold == 0:
        newnumber = number
    else:
        newnumber = np.ceil(number/kfold)
    i = 0
    for model in onlyFeatureSelectionModel:
        computed = False
        for featureNumber in onlyFeatureNumber:
            featureDecisionFile = os.path.join(args.outDirResults, 'feature_selection', '%i_fold_cross_validation'
                                               % kfold, '%s_k%i_var_True_False.csv' % (model, featureNumber))
            if not os.path.exists(featureDecisionFile):
                if not computed:
                    subprocess.run(['python3', 'run_feature_selection.py',  '--inFilePairStat', inFilePairStat,
                                    '--kfold', str(kfold), '--onlyModel', model, '--number',  str(int(newnumber)),
                                    '--outDirResults', outDirResults])
                    computed = True
            if os.path.exists(featureDecisionFile):
                if i >= len(fileNames):
                    fileNames.append([])
                fileNames[i].append(featureDecisionFile)
        i += 1
    return fileNames


def main(inFilePairStat, extendFiles, sos1, kfold, number, outDirResults, onlyModel, featureDecisionFile,
         decideAddFile, onlyFeatureNumber, onlyFeatureSelectionModel, fskfold, fsnumber, seed, *, onlySummary):
    """
    performes training with the given models on data with given features. For each model a score file and
    all trained models are saved. This main-function also computes an overview/ summary-file
    for all computations. 

    INPUT:
    inFilePairStat: path to input file for pairwise statistical data
    extendFiles: if true the existing files are extended, if false they will be removed
    sos1: if true either the sum or the product of the features/ parameters are used
    kfold: k for the k-fold cross validation k=0 implies no validation
    number: number of repetition for k=0 it is the number of calls for a model, else k*n is the number of calls
    outDirResults: path where the computational results should be stored
    onlyModel: the models that should be used for the training sublist of ['ols','lav','nn', 'ols_noGrav', 'kr','sfs']
    featureDecisionFile: a file that contains True/False values for all parameters
    decideAddFile: If true, depending on the following input variables featureDecisionFiles are created
                    and the models are trained on all of them.
    onlyFeatureNumber: limits the number of features/ parameters that can be used (list of int)
    onlyFeatureSelectionModel: specifies the model, which is used for feature-selection
    fskfold, fsnumber: same as kfold and number but for the feature selection
    seed: int if a random seed should be used, otherwise None
    """
    if seed is not None:  # use a random seed
        tf.keras.utils.set_random_seed(seed)

    decide = False
    if decideAddFile and not sos1:
        featureDecisionFiles = createFeatureDecisionFiles(onlyFeatureNumber, onlyFeatureSelectionModel, outDirResults,
                                                          fsnumber, inFilePairStat, fskfold)
        decide = True
    else:
        if featureDecisionFile != '' and not sos1:
            decide = True
        if featureDecisionFile != '':
            featureDecisionFiles = [featureDecisionFile]
        else:
            featureDecisionFiles = [['']]

    if decide:
        outDirResults = os.path.join(outDirResults, 'feature_selection_results')

    if kfold > 0:
        directory = os.path.join(outDirResults, '%s_fold_cross_validation' % kfold)
    else:
        directory = os.path.join(outDirResults, 'no_Validation')
    if sos1:
        directory = '%s_sos1' % directory
    modelDirectory = os.path.join(directory, 'models')
    Path(modelDirectory).mkdir(parents=True, exist_ok=True)

    for m in range(len(featureDecisionFiles)):
        if decide:
            fileNames = [[] for _ in range(len(onlyModel))]
        for featureDecisionFile in featureDecisionFiles[m]:
            j = 0
            if not onlySummary:
                x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile,
                                                                        True)
                # nld = not logarithmized demand
                x_data_grav_nlD, y_data_grav_nlD = data_functions.getGravityData(inFilePairStat, sos1, decide,
                                                                                featureDecisionFile, False)
                x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat, sos1, decide,
                                                                                        featureDecisionFile)
                x_data_grav_bin, y_data_grav_bin = data_functions.getGravityDataScores(inFilePairStat, sos1, decide,
                                                                                    featureDecisionFile, True)
                x_data_grav_pre_bin, y_data_grav_pre_bin = \
                    data_functions.getGravityPreprocessedDataScores(inFilePairStat, sos1, decide, featureDecisionFile, True)
            
            # declare filenames
            fileNameLAV = os.path.join(directory, 'lav_reg')
            fileNameOLS = os.path.join(directory, 'ols_reg')
            fileNameKR = os.path.join(directory, 'kr_reg')
            fileNameSVR = os.path.join(directory, 'sv_reg')
            fileNameNN = os.path.join(directory, 'nn')
            fileNameOLSnG = os.path.join(directory, 'ols_nG')
            fileNamePPML = os.path.join(directory, 'ppml')
            
            head, tail = os.path.split(featureDecisionFile)
            
            if featureDecisionFile != '' and not sos1:
                fileNameOLS = '%s_fs_%s' % (fileNameOLS, tail[:-19])
                fileNameLAV = '%s_fs_%s' % (fileNameLAV, tail[:-19])
                fileNameKR = '%s_fs_%s' % (fileNameKR, tail[:-19])
                fileNameSVR = '%s_fs_%s' % (fileNameSVR, tail[:-19])
                fileNameOLSnG = '%s_fs_%s' % (fileNameOLSnG, tail[:-19])
                fileNameNN = '%s_fs_%s' % (fileNameNN, tail[:-19])
                fileNamePPML = '%s_fs_%s' % (fileNamePPML, tail[:-19])
            
            # removeFiles
            if not extendFiles:
                if 'lav' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameLAV):
                        os.remove('%s.csv' % fileNameLAV)
                        fileList = glob.glob(os.path.join(modelDirectory, 'lav*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
                if 'ols' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameOLS):
                        os.remove('%s.csv' % fileNameOLS)
                        fileList = glob.glob(os.path.join(modelDirectory, 'ols*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
                if 'kr' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameKR):
                        os.remove('%s.csv' % fileNameKR)
                        fileList = glob.glob(os.path.join(modelDirectory, 'kr*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
                if 'svr' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameSVR):
                        os.remove('%s.csv' % fileNameSVR)
                        fileList = glob.glob(os.path.join(modelDirectory, 'svr*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
                if 'nn' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameNN):
                        os.remove('%s.csv' % fileNameNN)
                        fileList = glob.glob(os.path.join(modelDirectory, 'nn*'), recursive=False)
                        for filePath in fileList:
                            subprocess.run(['rm', '-r', filePath])
                if 'olsnG' in onlyModel:
                    if os.path.exists('%s.csv' % fileNameOLSnG):
                        os.remove('%s.csv' % fileNameOLSnG)
                        fileList = glob.glob(os.path.join(modelDirectory, 'ols_nG_*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
                if 'ppml' in onlyModel:
                    if os.path.exists('%s.csv' % fileNamePPML):
                        os.remove('%s.csv' % fileNamePPML)
                        fileList = glob.glob(os.path.join(modelDirectory, 'ppml*.z'), recursive=False)
                        for filePath in fileList:
                            os.remove(filePath)
            if decide:
                if 'ols' in onlyModel:
                    fileNames[j].append(fileNameOLS)
                    j += 1
                if 'lav' in onlyModel:
                    fileNames[j].append(fileNameLAV)
                    j += 1
                if 'kr' in onlyModel:
                    fileNames[j].append(fileNameKR)
                    j += 1
                if 'svr' in onlyModel:
                    fileNames[j].append(fileNameSVR)
                    j += 1
                if 'nn' in onlyModel:
                    fileNames[j].append(fileNameNN)
                    j += 1
                if 'olsnG' in onlyModel:
                    fileNames[j].append(fileNameOLSnG)
                    j += 1
                if 'ppml' in onlyModel:
                    fileNames[j].append(fileNamePPML)
                    j += 1

            if not onlySummary:
                rkf = RepeatedKFold(n_splits=kfold, n_repeats=number)
                
                # run algorithms
                if kfold > 0:
                    i = 0
                    for train_index, vali_index in rkf.split(x_data_grav):
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        x_train_grav = x_data_grav[train_index]
                        y_train_grav = y_data_grav[train_index]
                        x_train_grav_score = x_data_grav_score[train_index]
                        y_train_grav_score = y_data_grav_score[train_index]
                        x_train_grav_bin = x_data_grav_bin[train_index]
                        y_train_grav_bin = y_data_grav_bin[train_index]
                        x_train_grav_nlD = x_data_grav_nlD[train_index]
                        y_train_grav_nlD = y_data_grav_nlD[train_index]
                        x_vali_grav = x_data_grav[vali_index]
                        y_vali_grav = y_data_grav[vali_index]
                        x_vali_grav_score = x_data_grav_score[vali_index]
                        y_vali_grav_score = y_data_grav_score[vali_index]
                        x_vali_grav_bin = x_data_grav_bin[vali_index]
                        y_vali_grav_bin = y_data_grav_bin[vali_index]
                        x_vali_grav_nlD = x_data_grav_nlD[vali_index]
                        y_vali_grav_nlD = y_data_grav_nlD[vali_index]
                        x_train_grav_pre_bin = x_data_grav_pre_bin[train_index]
                        y_train_grav_pre_bin = y_data_grav_pre_bin[train_index]
                        x_vali_grav_pre_bin = x_data_grav_pre_bin[vali_index]
                        y_vali_grav_pre_bin = y_data_grav_pre_bin[vali_index]

                        if 'lav' in onlyModel:  # LAV-Regression
                            modelNameLAV = os.path.join(modelDirectory, 'lav_%s_%i.z' % (timestr, i))
                            lav_regression.lav_regression(y_train_grav, y_vali_grav, y_train_grav_score, y_vali_grav_score,
                                                        x_train_grav, x_vali_grav, x_train_grav_score, x_vali_grav_score,
                                                        kfold, fileNameLAV, modelNameLAV)
                            
                        if 'ols' in onlyModel:  # OLS-Regression
                            modelNameOLS = os.path.join(modelDirectory, 'ols_%s_%i.z' % (timestr, i))
                            ols_regression.ols_regression(y_train_grav, y_vali_grav, y_train_grav_score, y_vali_grav_score,
                                                        x_train_grav, x_vali_grav, x_train_grav_score, x_vali_grav_score,
                                                        kfold, fileNameOLS, modelNameOLS)
                            
                        if 'kr' in onlyModel:  # Kernel-Ridge-Regression
                            modelNameKR = os.path.join(modelDirectory, 'kr_%s_%i.z' % (timestr, i))
                            kr_regression.kr_regression(y_train_grav, y_vali_grav, y_train_grav_score, y_vali_grav_score,
                                                        x_train_grav, x_vali_grav, kfold, fileNameKR, modelNameKR)
                        if 'svr' in onlyModel:  # Support-Vector-Regression
                            modelNameSVR = os.path.join(modelDirectory, 'svr_%s_%i.z' % (timestr, i))
                            sv_regression.sv_regression(y_train_grav, y_vali_grav, y_train_grav_score, y_vali_grav_score,
                                                        x_train_grav, x_vali_grav, kfold, fileNameSVR, modelNameSVR)
                        if 'nn' in onlyModel:  # Neural Network
                            modelNameNN = os.path.join(modelDirectory, 'nn_%s_%i.z' % (timestr, i))
                            neural_network.neural_network(y_train_grav_pre_bin, y_vali_grav_pre_bin, x_train_grav_pre_bin,
                                                        x_vali_grav_pre_bin, kfold, fileNameNN, modelNameNN)
                        if 'olsnG' in onlyModel:
                            modelNameOLSnG = os.path.join(modelDirectory, 'ols_nG_%s_%i.z' % (timestr, i))
                            ols_noGrav.ols_noGrav(y_train_grav_bin, y_vali_grav_bin, x_train_grav_bin, x_vali_grav_bin,
                                                kfold, fileNameOLSnG, modelNameOLSnG)
                        if 'ppml' in onlyModel:
                            modelNamePPML = os.path.join(modelDirectory, 'ppml_%s_%i.z' % (timestr, i))
                            ppml_regression.ppml_regression(y_train_grav_nlD, y_vali_grav_nlD, x_train_grav_nlD,
                                                            x_vali_grav_nlD, kfold, fileNamePPML, modelNamePPML)
                        i = i + 1
                else: 
                    for i in range(number):
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        if 'lav' in onlyModel:
                            modelNameLAV = os.path.join(modelDirectory, 'lav_%s_%i.z' % (timestr, i))
                            lav_regression.lav_regression(y_data_grav, [], y_data_grav_score, [], x_data_grav, [],
                                                        x_data_grav_score, [], kfold, fileNameLAV, modelNameLAV)
                        if 'ols' in onlyModel:
                            modelNameOLS = os.path.join(modelDirectory, 'ols_%s_%i.z' % (timestr, i))
                            ols_regression.ols_regression(y_data_grav, [], y_data_grav_score, [], x_data_grav, [],
                                                        x_data_grav_score, [], kfold, fileNameOLS, modelNameOLS)
                        if 'kr' in onlyModel:  # Kernel-Ridge-Regression
                            modelNameKR = os.path.join(modelDirectory, 'kr_%s_%i.z' % (timestr, i))
                            kr_regression.kr_regression(y_data_grav, [], y_data_grav_score, [], x_data_grav, [],
                                                        kfold, fileNameKR, modelNameKR)
                        if 'svr' in onlyModel:  # Support-Vector-Regression
                            modelNameSVR = os.path.join(modelDirectory, 'svr_%s_%i.z' % (timestr, i))
                            sv_regression.sv_regression(y_data_grav, [], y_data_grav_score, [], x_data_grav, [],
                                                        kfold, fileNameSVR, modelNameSVR)
                        if 'nn' in onlyModel:  # Neural Network
                            modelNameNN = os.path.join(modelDirectory, 'nn_%s_%i' % (timestr, i))
                            neural_network.neural_network(y_data_grav_pre_bin, np.zeros([0]), x_data_grav_pre_bin,
                                                        np.zeros([0]), kfold, fileNameNN, modelNameNN)
                        if 'olsnG' in onlyModel:
                            modelNameOLSnG = os.path.join(modelDirectory, 'ols_nG_%s_%i.z' % (timestr, i))
                            ols_noGrav.ols_noGrav(y_data_grav_bin, [], x_data_grav_bin, [], kfold, fileNameOLSnG,
                                                modelNameOLSnG)
                        if 'ppml' in onlyModel:
                            modelNamePPML = os.path.join(modelDirectory, 'ppml_%s_%i.z' % (timestr, i))
                            ppml_regression.ppml_regression(y_data_grav_nlD, [], x_data_grav_nlD, [], kfold, fileNamePPML,
                                                            modelNamePPML)
        if decide:
            if onlyFeatureSelectionModel is None:
                outFile = '%s_%s.csv' % (directory, tail[:-19])
                summary(functools.reduce(operator.concat, fileNames), outFile, kfold, decide)
            else:
                for l in range(len(onlyModel)):
                    outFile = '%s_%s_%s.csv' % (directory, onlyModel[l], onlyFeatureSelectionModel[m])
                    summary(fileNames[l], outFile, kfold, decide)
    if not decide:
        summary([fileNameLAV, fileNameOLS, fileNameKR, fileNameSVR, fileNameNN, fileNameOLSnG, fileNamePPML],
                directory, kfold, decide)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--extendFiles', action='store_true',
                        help='extends all score-files, that usually would be removed by calling', default=False)
    parser.add_argument('--sos1', action='store_true',
                        help='if true only sum or product is used in the gravity based models', default=False)
    parser.add_argument('--fskfold', help='k for the k-fold cross validation k=0 implies no validation for the feature '
                                          'selection (of sos1-variables)', type=int, default=10)
    parser.add_argument('--fsnumber', help='number of repetition for k=0 it is the number of calls for a model, else '
                                           'k*n is the number of calls for the feature selection (of sos1-variables)',
                        type=int, default=10)
    parser.add_argument('--kfold', help='k for the k-fold cross validation k=0 implies no validation',
                        type=int, default=10)
    parser.add_argument('--number', help='number of repetition for k=0 it is the number of calls for a model, else k*n '
                                         'is the number of calls', type=int, default=100)
    parser.add_argument('--outDirResults', help='path to the Result-directory', default='results')
    parser.add_argument('--showCorrelation', action='store_true',
                        help='prints correlation matrices between the sum and product variables', default=False)
    parser.add_argument('--onlyModel', help='The prediction is just on the given models. If this command is not used, '
                                            'it is computed on every model', action='store',
                        choices=['ols', 'lav', 'kr', 'svr', 'nn', 'olsnG', 'ppml'], nargs='*',
                        default=['ols', 'lav', 'kr', 'svr', 'nn', 'olsnG', 'ppml'])
    parser.add_argument('--onlyFeatures',
                        help='Score all models with all possible limitations to features and the decision on the '
                             'features is based on all different feature-selection-models.',
                        action='store_true', default=False)
    parser.add_argument('--onlyFeatureNumber',
                        help='Only the given  number(s) of features is/are used for prediction. If no number is chosen,'
                             ' but onlyFeatures or a specific model is used/ chosen, its computed for every possible '
                             'number. If no number is chosen and onlyFeatures is not used, just the maximum number of '
                             'features is used.', action='store',
                        choices=range(1, data_functions.numGravFeatures(False) + 1), type=int,  nargs='*', default=None)
    parser.add_argument('--onlyFeatureSelectionModel',
                        help='Which features are used is based on the given model. If no model is chosen, but '
                             'onlyFeatures is used or a number of features is given, its computed for every possible '
                             'model.', action='store', choices=['ccls', 'sfs', 'rfe', 'ccls_group'], nargs='*',
                        default=None)
    parser.add_argument('--featureDecisionFile',
                        help='input file(s) for features that should be used. Contains either True-False-Values for '
                             'each feature or Numbers between 0 and 1. (>=0.5 are used)', action='store', nargs='*',
                        default='')
    parser.add_argument('--allPossibilities',
                        help='Every possible computation and evaluation for the paper is made.', action='store_true',
                        default=False)
    parser.add_argument('--randomSeed', help='random seed', nargs='?', type=int, const=42)
    parser.add_argument('--onlySummary', help='if true only the summaries are recomputed based on earliere computed files',
                        action='store_true', default=False)

    args = parser.parse_args()
    #onlySummary
    if args.onlySummary == True:
        args.extendFiles = True

    # featureDecisionFile
    if args.featureDecisionFile == []:
        args.featureDecisionFile = ''

    # any feature selection without a given file?
    if (args.onlyFeatureNumber is not None and args.onlyFeatureNumber != [data_functions.numGravFeatures(False)]) or \
            args.onlyFeatureSelectionModel is not None or args.onlyFeatures:
        decideAddFile = True
    else:
        decideAddFile = False
    
    # any feature selection with a given file?
    if args.featureDecisionFile != '':
        decideGivenFile = True
    else:
        decideGivenFile = False
    
    # errors
    if args.allPossibilities and (args.sos1 or decideAddFile or decideGivenFile or len(args.onlyModel) != 7):
        parser.error('You cannot use feature-selection-limitations and allPossibilities in common.')
    if (args.sos1 and decideAddFile) or (args.sos1 and decideGivenFile):
        parser.error('You cannot use the sos1 command and other commands that limit the features (only... commands) '
                     'in common.')
    if decideAddFile and decideGivenFile:
        parser.error('Run the given inputs not at the same time. First enter the featureDecisionFile and then limit the'
                     ' features without a given file.')
    
    # default handling for feature selection
    if args.onlyFeatures:
        if args.onlyFeatureNumber is None or args.onlyFeatureNumber == [data_functions.numGravFeatures(False)]:
            args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False))
        if args.onlyFeatureSelectionModel is None:
            args.onlyFeatureSelectionModel = ['ccls', 'ccls_group', 'sfs', 'rfe']
    else:
        if args.onlyFeatureNumber is None:
            if args.onlyFeatureSelectionModel is not None:
                args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False))
            else:
                args.onlyFeatureNumber = [data_functions.numGravFeatures(False)]
        elif min(args.onlyFeatureNumber) < data_functions.numGravFeatures(False):
            if args.onlyFeatureSelectionModel is None:
                args.onlyFeatureSelectionModel = ['ccls', 'ccls_group', 'sfs', 'rfe']
        else: 
            if args.onlyFeatureSelectionModel is not None:
                print('If you use all features you do not need a feature selection model.')
    
    # handling for sos1
    if args.sos1:
        if args.onlyFeatureSelectionModel is None:
            args.onlyFeatureSelectionModel = ['ccls_group']
        if args.onlyFeatureSelectionModel == ['ccls', 'ccls_group']:
            args.featureDecisionFile = [os.path.join(args.outDirResults, 'feature_selection',
                                                     '%i_fold_cross_validation_sos1' % args.fskfold,
                                                     'ccls_maxk_var_True_False.csv'),
                                        os.path.join(args.outDirResults, 'feature_selection',
                                                     '%i_fold_cross_validation_sos1' % args.fskfold,
                                                     'ccls_group_maxk_var_True_False.csv')]
            if not os.path.exists(args.featureDecisionFile[0]) and not args.onlySummary:
                print("The file that is needed to decide whether the sum or the product of the airport variables are "
                      "used does not exist, so the feature-selection is run first.")
                subprocess.run(['python3', 'run_feature_selection.py',  '--inFilePairStat', args.inFilePairStat,
                                '--sos1', '--kfold', str(args.fskfold), '--number',  str(args.fsnumber),
                                '--outDirResults', args.outDirResults, '--onlyModel', 'ccls'])
            if not os.path.exists(args.featureDecisionFile[1]) and not args.onlySummary:
                print("The file that is needed to decide whether the sum or the product of the airport variables are "
                      "used does not exist, so the feature-selection is run first.")
                subprocess.run(['python3', 'run_feature_selection.py',  '--inFilePairStat', args.inFilePairStat,
                                '--sos1', '--kfold', str(args.fskfold), '--number',  str(args.fsnumber),
                                '--outDirResults', args.outDirResults, '--onlyModel', 'ccls_group'])
        elif 'sfs' in args.onlyFeatureSelectionModel or 'rfe' in args.onlyFeatureSelectionModel:
            parser.error('sfs and rfe do not support sos1 constraints.')
        elif args.onlyFeatureSelectionModel == ['ccls']:
            args.featureDecisionFile = [os.path.join(args.outDirResults, 'feature_selection',
                                                     '%i_fold_cross_validation_sos1' % args.fskfold,
                                                     'ccls_maxk_var_True_False.csv')]
            if not os.path.exists(args.featureDecisionFile[0]) and not args.onlySummary:
                print("The file that is needed to decide whether the sum or the product of the airport variables are "
                      "used does not exist, so the feature-selection is run first.")
                subprocess.run(['python3', 'run_feature_selection.py',  '--inFilePairStat', args.inFilePairStat,
                                '--sos1', '--kfold', str(args.fskfold), '--number',  str(args.fsnumber),
                                '--outDirResults', args.outDirResults, '--onlyModel', 'ccls'])
        elif args.onlyFeatureSelectionModel == ['ccls_group']:
            args.featureDecisionFile = [os.path.join(args.outDirResults, 'feature_selection',
                                                     '%i_fold_cross_validation_sos1' % args.fskfold,
                                                     'ccls_group_maxk_var_True_False.csv')]
            if not os.path.exists(args.featureDecisionFile[0]) and not args.onlySummary:
                print("The file that is needed to decide whether the sum or the product of the airport variables are "
                      "used does not exist, so the feature-selection is run first.")
                subprocess.run(['python3', 'run_feature_selection.py',  '--inFilePairStat', args.inFilePairStat,
                                '--sos1', '--kfold', str(args.fskfold), '--number',  str(args.fsnumber),
                                '--outDirResults', args.outDirResults, '--onlyModel', 'ccls_group'])
    if args.allPossibilities:
        args.extendFiles = False
        decideAddFile = False
        args.number = 100
        args.fsnumber = args.number
        args.kfold = 10
        args.fskfold = args.kfold
        
        # Evaluate sos1
        print('sos1')
        args.sos1 = True
        args.featureDecisionFile = [os.path.join(args.outDirResults, 'feature_selection',
                                                 '%i_fold_cross_validation_sos1' % args.kfold,
                                                 'ccls_group_maxk_var_True_False.csv')]
        if not os.path.exists(args.featureDecisionFile[0]) and not args.onlySummary:
            subprocess.run(['python3', 'run_feature_selection.py', '--inFilePairStat', args.inFilePairStat, '--sos1',
                            '--kfold',  str(args.fskfold), '--number',  str(args.fsnumber), '--outDirResults',
                            args.outDirResults, '--onlyModel', 'ccls_group'])
        args.kfold = 0
        main(args.inFilePairStat, args.extendFiles, args.sos1, args.kfold, args.number, args.outDirResults,
             args.onlyModel, args.featureDecisionFile, decideAddFile, args.onlyFeatureNumber,
             args.onlyFeatureSelectionModel, args.fskfold, args.fsnumber, args.randomSeed, onlySummary = args.onlySummary)
        
        print('fs')
        # Evaluate feature selection
        args.sos1 = False
        decideAddFile = True
        args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False) + 1)
        args.onlyFeatureSelectionModel = ['ccls_group', 'sfs', 'rfe']
        args.kfold = 0
        main(args.inFilePairStat, args.extendFiles, args.sos1, args.kfold, args.number, args.outDirResults,
             args.onlyModel, args.featureDecisionFile, decideAddFile, args.onlyFeatureNumber,
             args.onlyFeatureSelectionModel, args.fskfold, args.fsnumber, args.randomSeed, onlySummary = args.onlySummary)
        
        # Evaluate normal
        print('normal')
        args.sos1 = False
        args.kfold = 10
        decideAddFile = False
        args.featureDecisionFile = ''
        main(args.inFilePairStat, args.extendFiles, args.sos1, args.kfold, args.number,
             args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile, args.onlyFeatureNumber,
             args.onlyFeatureSelectionModel, 0, 0, args.randomSeed, onlySummary = args.onlySummary)
        args.kfold = 0
        main(args.inFilePairStat, args.extendFiles, args.sos1, args.kfold, args.number,
             args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile, args.onlyFeatureNumber,
             args.onlyFeatureSelectionModel, 0, 0, args.randomSeed, onlySummary = args.onlySummary)
    else:
        main(args.inFilePairStat, args.extendFiles, args.sos1, args.kfold, args.number,
             args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile, args.onlyFeatureNumber,
             args.onlyFeatureSelectionModel, args.fskfold, args.fsnumber, args.randomSeed, onlySummary = args.onlySummary)
    if args.showCorrelation:
        show_correlation(args.inFilePairStat)
