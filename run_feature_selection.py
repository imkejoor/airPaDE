import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedKFold
import time
import os
import glob
import csv
import random

import rfe
import run
import data_functions
import ccls_regression_mip
import ccls_group_regression_mip
import sfs

# the following settings define the models that are used [the definition here is just for ease of use, so you do not
# have to scroll all the way down to the calls]
useCCLS = True
useCCLSGROUP = True
useRFE = True
useSFS = True


def summary(fileNames, outDir, kfold):
    """
    INPUT:
    fileNames: contains all fileNames which should be summarized
    outDir: directory where the summaries should be stored in
    kfold: 0 if no  validation
    """
    nameGravFeatures = data_functions.nameGravFeatures()
    for model in fileNames:
        if os.path.exists('%s.csv' % model):
            head, tail = os.path.split('%s.csv' % model)
            outFile_var = '%s_%s_var_percentage.csv' % (outDir, tail[:-4])
            outFile = '%s_%s.csv' % (outDir, tail[:-4])
            if os.path.exists(outFile_var):
                os.remove(outFile_var)
            if os.path.exists(outFile):
                os.remove(outFile)
            with open(outFile, mode='a') as file:
                file_writer = csv.writer(file, delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if kfold != 0:
                    file_writer.writerow(["Nr.", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                                          "mean loss percentage", "Validation-R2", "Validation-Adj R2",
                                          "Validation-Grav R2", "Validation-Grav Adj R2", "Validation-mean loss",
                                          "Validation-max loss", "Validation-mean loss percentage", "Training-R2",
                                          "Training-Adj R2", "Training-Grav R2", "Training-Grav Adj R2",
                                          "Training-mean loss", "Training-max loss", "Training-mean loss percentage",
                                          "filename(s)"])
                else:
                    file_writer.writerow(["Model", "R2", "Adj R2", "Grav R2", "Grav Adj R2", "mean loss", "max loss",
                                          "mean loss percentage", "filename(s)"])
            if tail.find('ccls_group') != -1:
                first = 'CCLS GROUP'
            elif tail.find('ccls') != -1:
                first = 'CCLS'
            elif tail.find('rfe') != -1:
                first = 'RFE'
            elif tail.find('sfs') != -1:
                first = 'SFS'
            with open(outFile_var, mode='a') as file:
                file_writer = csv.writer(file, delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                names = nameGravFeatures
                names_first = [first]
                names_first.extend(names)
                file_writer.writerow(names_first)
                for k in range(1, data_functions.numGravFeatures(False) + 1):
                    filek = glob.glob('%s%i_*var*.csv' % (model, k), recursive=False)
                    if len(filek) > 0:
                        matrix = np.genfromtxt('%s' % filek[0], delimiter=',', dtype=bool)
                        matrix = matrix.astype(int)
                        if len(np.shape(matrix)) == 2:
                            mean = list(np.mean(matrix, axis=0))
                        else:
                            mean = list(matrix)
                        mean.insert(0, round(sum(mean)))
                        for i in range(1, len(mean)):
                            mean[i] = run.formatNumber(mean[i])
                        file_writer.writerow(mean)
            for k in range(1, data_functions.numGravFeatures(False)+1):
                filek = glob.glob('%s%i.csv' % (model, k), recursive=False)
                if len(filek) > 0:
                    i, j, array = data_functions.readScoreInstance(filek[0])
                    summary = array[1]
                    if kfold != 0:
                        for l in range(1, 22):
                            if l in [5, 6, 12, 13, 19, 20]:
                                summary[l] = round(float(summary[l]))
                            else:
                                summary[l] = run.formatNumber(float(summary[l]), 4)
                    else:
                        for l in range(1, 8):
                            if l in [5, 6]:
                                summary[l] = round(float(summary[l]))
                            else:
                                summary[l] = run.formatNumber(float(summary[l]), 4)
                    if model.find('ccls_group') != -1:
                        onlyFSModel = 'CCLS GROUP'
                    elif model.find('ccls') != -1:
                        onlyFSModel = 'CCLS'
                    elif model.find('rfe') != -1:
                        onlyFSModel = 'RFE'
                    elif model.find('sfs') != -1:
                        onlyFSModel = 'SFS'
                    summary[0] = '%s %i' % (onlyFSModel, k)
                    with open(outFile, mode='a') as file:
                        file_writer = csv.writer(file, delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        file_writer.writerow(summary)


def main(inFilePairStat, extendFiles, sos1, onlyModel, kfold, number, outDirResults, seed):
    """
    performs feature-selection with the given models. For each model a score file and
    all trained models are saved. This main-function also computes an overview/ summary-file
    for all computations. 

    INPUT:
    inFilePairStat: path to input file for pairwise statistical data
    extendFiles: if true the existing files are extended, if false they will be removed
    sos1: if true either the sum or the product of the features/ parameters are used
    onlyModel: the models that should be used for feature selection
    kfold: k for the k-fold cross validation k=0 implies no validation
    number: number of repetition for k=0 it is the number of calls for a model, else k*n is the number of calls
    outDirResults: path where the computational results should be stored.
    """
    start = time.time()
    if seed is not None:  # use a random seed
        random.seed(seed)
        np.random.seed(seed)

    if kfold > 0:
        featureDirectory = os.path.join(outDirResults, 'feature_selection', '%s_fold_cross_validation' % kfold)
    else:
        featureDirectory = os.path.join(outDirResults, 'feature_selection', 'no_Validation')
    if sos1:
        featureDirectory = '%s_sos1' % featureDirectory
    modelFeatureDirectory = os.path.join(featureDirectory, 'models')
    Path(modelFeatureDirectory).mkdir(parents=True, exist_ok=True)
    
    x_data_grav_all, y_data_grav_all = data_functions.getGravityData(inFilePairStat)
    x_data_grav_score_all, y_data_grav_score_all = data_functions.getGravityDataScores(inFilePairStat)
    
    # declare filenames
    fileNameCCLS = os.path.join(featureDirectory, 'ccls')
    fileNameCCLSG = os.path.join(featureDirectory, 'ccls_group')
    fileNameRFE = os.path.join(featureDirectory, 'rfe')
    fileNameSFS = os.path.join(featureDirectory, 'sfs')
    
    # removeFiles
    if not extendFiles:
        if useCCLS and 'ccls' in onlyModel:
            if os.path.exists('%s_maxk_var_True_False.csv' % fileNameCCLS):
                os.remove('%s_maxk_var_True_False.csv' % fileNameCCLS)
                fileList = glob.glob('%s_k*.csv' % fileNameCCLS, recursive=False)
                for filePath in fileList:
                    os.remove(filePath)
                fileList = glob.glob(os.path.join(modelFeatureDirectory, 'ccls_*'), recursive=False)
                for filePath in fileList:
                    if 'group' not in filePath:
                        os.remove(filePath)
        if useCCLSGROUP and 'ccls_group' in onlyModel:
            if os.path.exists('%s_maxk_var_True_False.csv' % fileNameCCLS):
                os.remove('%s_maxk_var_True_False.csv' % fileNameCCLS)
            fileList = glob.glob('%s_k*.csv' % fileNameCCLSG, recursive=False)
            for filePath in fileList:
                os.remove(filePath)
            fileList = glob.glob(os.path.join(modelFeatureDirectory, 'ccls_group*'), recursive=False)
            for filePath in fileList:
                os.remove(filePath)
        if useRFE and not sos1 and 'rfe' in onlyModel:
            fileList = glob.glob('%s_k*.csv' % fileNameRFE, recursive=False)
            if len(fileList) > 0:
                for filePath in fileList:
                    os.remove(filePath)
                fileList = glob.glob(os.path.join(modelFeatureDirectory, 'rfe_*'), recursive=False)
                for filePath in fileList:
                    os.remove(filePath)
        if useSFS and not sos1 and 'sfs' in onlyModel:
            fileList = glob.glob('%s_k*.csv' % fileNameSFS, recursive=False)
            if len(fileList) > 0:
                fileList = glob.glob('%s_k*.csv' % fileNameSFS, recursive=False)
                for filePath in fileList:
                    os.remove(filePath)
                fileList = glob.glob(os.path.join(modelFeatureDirectory, 'sfs_*'), recursive=False)
                for filePath in fileList:
                    os.remove(filePath)

    i = 0
    if kfold > 0:
        rkf = RepeatedKFold(n_splits=kfold, n_repeats=number)  # random_state = 1
        rkf_copy = rkf
        rkfList = list(rkf_copy.split(x_data_grav_all))
        for train_index, vali_index in rkf.split(x_data_grav_all):
            if i == kfold * number:
                break
            timestr = time.strftime("%Y%m%d-%H%M%S")
            x_train_grav_all = x_data_grav_all[train_index]
            y_train_grav_all = y_data_grav_all[train_index]
            x_train_grav_score_all = x_data_grav_score_all[train_index]
            y_train_grav_score_all = y_data_grav_score_all[train_index]
            x_vali_grav_all = x_data_grav_all[vali_index]
            y_vali_grav_all = y_data_grav_all[vali_index]
            x_vali_grav_score_all = x_data_grav_score_all[vali_index]
            y_vali_grav_score_all = y_data_grav_score_all[vali_index]
            if i % kfold == 0:
                if useRFE and not sos1 and 'rfe' in onlyModel:
                    modelNameRFE = os.path.join(modelFeatureDirectory, 'rfe_%s_%i' % (timestr, i))
                    rfe.perform_rfe(y_data_grav_all, y_data_grav_score_all, x_data_grav_all, x_data_grav_score_all,
                                    kfold, fileNameRFE, modelNameRFE, rkfList[i:i + kfold])
                if useSFS and not sos1 and 'sfs' in onlyModel:
                    modelNameSFS = os.path.join(modelFeatureDirectory, 'sfs_%s_%i' % (timestr, i))
                    sfs.perform_sfs(y_data_grav_all, y_data_grav_score_all, x_data_grav_all, x_data_grav_score_all,
                                    kfold, fileNameSFS, modelNameSFS, rkfList[i:i + kfold])
                if useCCLSGROUP and 'ccls_group' in onlyModel:
                    modelNameCCLSG = os.path.join(modelFeatureDirectory, 'ccls_group_%s_%i' % (timestr, i))
                    ccls_group_regression_mip.ccls_group_regression(y_data_grav_all, y_data_grav_score_all,
                                                                    x_data_grav_all, x_data_grav_score_all, sos1, kfold,
                                                                    fileNameCCLSG, modelNameCCLSG, rkfList[i:i + kfold])
            if useCCLS and 'ccls' in onlyModel:
                modelNameCCLS = os.path.join(modelFeatureDirectory, 'ccls_%s_%i' % (timestr, i))
                ccls_regression_mip.ccls_regression(y_train_grav_all, y_vali_grav_all, y_train_grav_score_all,
                                                    y_vali_grav_score_all, x_train_grav_all, x_vali_grav_all,
                                                    x_train_grav_score_all, x_vali_grav_score_all, sos1, kfold,
                                                    fileNameCCLS, modelNameCCLS)
                     
            i = i + 1
    else: 
        for i in range(number):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if (useCCLS or useCCLSGROUP) and ('ccls' in onlyModel or 'ccls_group' in onlyModel):
                modelNameCCLS = os.path.join(modelFeatureDirectory, 'ccls_%s_%i' % (timestr, i))
                ccls_regression_mip.ccls_regression(y_data_grav_all, 0, y_data_grav_score_all, 0, x_data_grav_all, 0,
                                                    x_data_grav_score_all, 0, sos1, kfold, fileNameCCLS, modelNameCCLS)
            if useRFE and not sos1 and 'rfe' in onlyModel:
                modelNameRFE = os.path.join(modelFeatureDirectory, 'rfe_%s_%i' % (timestr, i))
                rfe.perform_rfe(y_data_grav_all, y_data_grav_score_all, x_data_grav_all, x_data_grav_score_all, kfold,
                                fileNameRFE, modelNameRFE)
            if useSFS and not sos1 and 'sfs' in onlyModel:
                modelNameSFS = os.path.join(modelFeatureDirectory, 'sfs_%s_%i' % (timestr, i))
                sfs.perform_sfs(y_data_grav_all, y_data_grav_score_all, x_data_grav_all, x_data_grav_score_all, kfold,
                                fileNameSFS, modelNameSFS)
    if not sos1:
        summary(['%s_k' % fileNameCCLSG, '%s_k' % fileNameCCLS, '%s_k' % fileNameRFE, '%s_k' % fileNameSFS],
                featureDirectory, kfold)
    else:
        summary(['%s_k' % fileNameCCLS, '%s_k' % fileNameCCLSG], featureDirectory, kfold)
    end = time.time()
    print('number: %i' % number)
    print('needed time: {:5.3f}s'.format(end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inFilePairStat', help='input file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--extendFiles', action='store_true',
                        help='extends all score-files, that usually would be removed by calling', default=False)
    parser.add_argument('--sos1', action='store_true',
                        help='if true in the gravity based models only sum or product is used', default=False)
    parser.add_argument('--kfold', help='k for the k-fold cross validation k=0 implies no validation', type=int,
                        default=10)
    parser.add_argument('--number', help='number of repetition for k=0 it is the number of calls for a model, else k*n '
                                         'is the number of calls', type=int, default=100)
    parser.add_argument('--onlyModel', help='The feature selection is performed with the given models.'
                                            'If no model is chosen, it is computed on every model', action='store',
                        choices=['ccls', 'ccls_group', 'rfe', 'sfs'], nargs='*', default=None)
    parser.add_argument('--outDirResults', help='path to the results directory', default='results')
    parser.add_argument('--randomSeed', help='random seed', nargs='?', type=int, const=42)

    args = parser.parse_args()

    if args.onlyModel is None:
        if args.sos1:
            args.onlyModel = ['ccls_group']
        else:
            args.onlyModel = ['ccls', 'ccls_group', 'rfe', 'sfs']
    else:
        if args.sos1:
            if 'sfs' in args.onlyModel or 'rfe' in args.onlyModel:
                parser.error('sfs and rfe do not support sos1 constraints.')

    main(args.inFilePairStat, args.extendFiles, args.sos1, args.onlyModel,  args.kfold, args.number, args.outDirResults,
         args.randomSeed)
