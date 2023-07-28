import argparse
import os
from pathlib import Path
import functools
import operator

import data_functions
import run
import run_feature_selection


def createFeatureDecisionFiles(onlyFeatureNumber, onlyFeatureSelectionModel, outDirResults, kfold=10):
    """
    creates feature decision files for all given parameters
    
    INPUT:
    onlyFeatureNumber: list of numbers for which the decision file should be created
    onlyFeatureSelectionModel: list of models that decide which features should be used
    outDirResults: directory where the results should be stored
    kfold: k for k-fold-crossvalidation
    """
    fileNames = [[]]
    i = 0
    for model in onlyFeatureSelectionModel:
        computed = False
        for featureNumber in onlyFeatureNumber:
            featureDecisionFile = os.path.join(args.outDirResults, 'feature_selection', '%i_fold_cross_validation'
                                               % kfold, '%s_k%i_var_True_False.csv' % (model, featureNumber))
            if not os.path.exists(featureDecisionFile):
                if not computed:
                    main_fs(False, kfold, outDirResults)
                    computed = True
            if os.path.exists(featureDecisionFile):
                if i >= len(fileNames):
                    fileNames.append([])
                fileNames[i].append(featureDecisionFile)
        i += 1
    return fileNames


def main_fs(sos1, kfold, outDirResults):
    if kfold > 0:
        featureDirectory = os.path.join(outDirResults, 'feature_selection', '%s_fold_cross_validation' % kfold)
    else:
        featureDirectory = os.path.join(outDirResults, 'feature_selection', 'no_Validation')
    if sos1:
        featureDirectory = '%s_sos1' % featureDirectory
    modelFeatureDirectory = os.path.join(featureDirectory, 'models')
    Path(modelFeatureDirectory).mkdir(parents=True, exist_ok=True)

    # declare filenames
    fileNameCCLS = os.path.join(featureDirectory, 'ccls')
    fileNameRFE = os.path.join(featureDirectory, 'rfe')
    fileNameSFS = os.path.join(featureDirectory, 'sfs')
    if os.path.exists('%s.csv' % featureDirectory):
        os.remove('%s.csv' % featureDirectory)
    run_feature_selection.summary(['%s_k' % fileNameCCLS, '%s_k' % fileNameRFE, '%s_k' % fileNameSFS],
                                  featureDirectory, kfold)


def main(sos1, kfold, outDirResults, onlyModel, featureDecisionFile, decideAddFile, onlyFeatureNumber,
         onlyFeatureSelectionModel):
    decide = False
    if decideAddFile and not sos1:
        featureDecisionFiles = createFeatureDecisionFiles(onlyFeatureNumber, onlyFeatureSelectionModel, outDirResults)
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
            fileNames = []
            for _ in onlyModel:
                fileNames.append([])

        for featureDecisionFile in featureDecisionFiles[m]:
            j = 0
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

        if decide:
            if onlyFeatureSelectionModel is None:
                outFile = '%s_%s.csv' % (directory, tail[:-19])
                run.summary(functools.reduce(operator.concat, fileNames), outFile, kfold, decide)
            else:
                for l in range(len(onlyModel)):
                    outFile = '%s_%s_%s.csv' % (directory, onlyModel[l], onlyFeatureSelectionModel[m])
                    run.summary(fileNames[l], outFile, kfold, decide)
    if not decide:
        if sos1:
            run.summary([fileNameLAV, fileNameOLS, fileNameKR, fileNameSVR, fileNameNN, fileNameOLSnG, fileNamePPML],
                        directory, kfold, decide)
        else:
            run.summary([fileNameLAV, fileNameOLS, fileNameKR, fileNameSVR, fileNameNN, fileNameOLSnG, fileNamePPML],
                        directory, kfold, decide)
    # else:
    # allFileNames = functools.reduce(operator.concat, allFiles)
    # outFile = 'all_Feature_results.csv'
    # outFile = os.path.join(directory, outFile)
    # run.summary(allFileNames, outFile, kfold, decide)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--extendFiles', action='store_true',
                        help='extends all score-files, that usually would be removed by calling', default=False)
    parser.add_argument('--sos1', action='store_true',
                        help='if true in the gravity based models only sum or product is used', default=False)
    parser.add_argument('--kfold', help='k for the k-fold cross validation k=0 implies no validation', type=int,
                        default=10)
    parser.add_argument('--number', help='number of repetition for k=0 it is the number of calls for a model, else '
                                         'k*n is the number of calls', type=int, default=100)
    parser.add_argument('--outDirResults', help='path to the Result-directory', default='results')
    parser.add_argument('--showCorrelation', action='store_true',
                        help='prints correlation matrices between the sum and product variables', default=False)
    parser.add_argument('--onlyModel', help='The prediction is just on the given models. If this command is not used, '
                                            'its computed on every model',
                        action='store', choices=['ols', 'lav', 'kr', 'svr', 'nn', 'olsnG', 'ppml'], nargs='*',
                        default=['ols', 'lav', 'kr', 'svr', 'nn', 'olsnG', 'ppml'])
    parser.add_argument('--onlyFeatures',
                        help='Score all models with all possible limitations to features and the decision on the '
                             'features is based on all different feature-selection-models.',
                        action='store_true', default=False)
    parser.add_argument('--onlyFeatureNumber',
                        help='Only the given  number(s) of features is/are used for prediction. If no number is chosen,'
                             ' but onlyFeatures or a specific model is used/ chosen, its computed for every possible '
                             'number. If no number is chosen and onlyFeatures is not used, just the maximum number of '
                             'features is used.',
                        action='store', choices=range(1, data_functions.numGravFeatures(False) + 1), type=int,
                        nargs='*', default=None)
    parser.add_argument('--onlyFeatureSelectionModel',
                        help='Which features are used is based on the given model. If no model is chosen, but '
                             'onlyFeatures is used or a number of features is given, its computed for every possible '
                             'model.', action='store', choices=['ccls', 'sfs', 'rfe'], nargs='*', default=None)
    parser.add_argument('--featureDecisionFile',
                        help='input file(s) for features that should be used. Contains either True-False-Values for '
                             'each feature or Numbers between 0 and 1. (>=0.5 are used)',
                        action='store', nargs='*', default='')
    parser.add_argument('--allPossibilities', help='Every possible computation and evaluation is made.',
                        action='store_true', default=False)
    args = parser.parse_args()

    if (args.onlyFeatureNumber is not None and args.onlyFeatureNumber != [data_functions.numGravFeatures(False)]) or \
            args.onlyFeatureSelectionModel is not None or args.onlyFeatures:
        decideAddFile = True
    else:
        decideAddFile = False
    if args.featureDecisionFile != '':
        decideGivenFile = True
    else:
        decideGivenFile = False
    if args.allPossibilities and (args.sos1 or decideAddFile or decideGivenFile or len(args.onlyModel) != 7):
        parser.error('You cannot use the only or sos1-limitations and allPossibilities in common.')
    if (args.sos1 and decideAddFile) or (args.sos1 and decideGivenFile):
        parser.error(
            'You cannot use the sos1 command and other commands that limit the features (only... commands) in common.')
    if decideAddFile and decideGivenFile:
        parser.error('Run the given inputs not at the same time. First enter the featureDecisionFile and then limit '
                     'the features without a given file.')
    if args.onlyFeatures:
        if args.onlyFeatureNumber is None or args.onlyFeatureNumber == [data_functions.numGravFeatures(False)]:
            args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False))
        if args.onlyFeatureSelectionModel is None:
            args.onlyFeatureSelectionModel = ['ccls', 'sfs', 'rfe']
    else:
        if args.onlyFeatureNumber is None:
            args.onlyFeatureNumber = [data_functions.numGravFeatures(False)]
            if args.onlyFeatureSelectionModel is not None:
                args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False))
        elif min(args.onlyFeatureNumber) < data_functions.numGravFeatures(False):
            if args.onlyFeatureSelectionModel is None:
                args.onlyFeatureSelectionModel = ['ccls', 'sfs', 'rfe']
            if 'nn' in args.onlyModel or 'olsnG' in args.onlyModel:
                parser.error('You cannot use feature selection with models that are not based on the gravity model.')
        else:
            if args.onlyFeatureSelectionModel is not None:
                print('If you use all features you do not need a feature selection model.')
    if args.sos1:
        args.featureDecisionFile = os.path.join(args.outDirResults, 'feature_selection', 'no_Validation_sos1',
                                                'ccls_maxk_var_True_False.csv')
        if not os.path.exists(args.featureDecisionFile):
            print("The file that is needed to decide whether the sum or the product of the airport variables are used "
                  "does not exist, so the feature-selection is run first.")
            main_fs(True, args.kfold, args.outDirResults)
    if args.allPossibilities:
        args.extendFiles = False
        decideAddFile = False
        args.number = 100

        # Evaluate sos1
        print('sos1')
        args.sos1 = True
        args.featureDecisionFile = os.path.join(args.outDirResults, 'feature_selection',
                                                '%i_fold_cross_validation_sos1' % args.kfold,
                                                'ccls_maxk_var_True_False.csv')
        if not os.path.exists(args.featureDecisionFile):
            main_fs(True, 10, args.outDirResults)
        args.kfold = 0
        main(args.sos1, args.kfold, args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile,
             args.onlyFeatureNumber, args.onlyFeatureSelectionModel)

        print('fs')
        # Evaluate feature selection
        args.sos1 = False
        decideAddFile = True
        args.onlyFeatureNumber = range(1, data_functions.numGravFeatures(False))
        args.onlyFeatureSelectionModel = ['ccls', 'sfs', 'rfe']
        args.kfold = 0
        main(args.sos1, args.kfold, args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile,
             args.onlyFeatureNumber, args.onlyFeatureSelectionModel)

        # Evaluate normal
        print('normal')
        args.sos1 = False
        args.kfold = 10
        decideAddFile = False
        args.featureDecisionFile = ''
        main(args.sos1, args.kfold, args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile,
             args.onlyFeatureNumber, args.onlyFeatureSelectionModel)
        args.kfold = 0
        main(args.sos1, args.kfold, args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile,
             args.onlyFeatureNumber, args.onlyFeatureSelectionModel)

    else:
        main(args.sos1, args.kfold, args.outDirResults, args.onlyModel, args.featureDecisionFile, decideAddFile,
             args.onlyFeatureNumber, args.onlyFeatureSelectionModel)
