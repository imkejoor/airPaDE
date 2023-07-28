# airPaDE
This code package provides tools to download and extract Eurostat data for various parameters used in causal air passenger demand estimation models, as well as implementations of several different approaches to estimate air passenger volumes between two cities.

## Description

This code package provides tools to download and extract Eurostat data
for various parameters used in causal air passenger demand estimation
models, as well as implementations of several different approaches to
estimate air passenger volumes between two cities.



## Installation

See setup.cfg for package requirements.

Example installation from scratch, using a virtual environment:

    install python3  
    virtualenv -p python3 venv  
    source venv/bin/activate  
    pip install -U  scipy numpy tensorflow sklearn gurobipy tensorflow_addons openpyxl joblib statsmodels


## Usage

### Data Download and Extraction

Path: `src/data_creation/`

+ download_data.py:
    + allows to automatically download the appropriate data files from the Eurostat database
+ extract_data.py:
    + extracts and curates the required parameter data for the demand estimation models from the basic data files (part of this package) and the external Eurostat data files;
    optionally also checks if data files have already been downloaded and does so, if not.

For details on the data files and sources, see `src/data/data_sources_readme.txt`


### Model Calibration and Prediction

#### Overview

Path: `src/estimation`

+ run.py and run_feature_selection.py  
    + contain the main functions that call all other functions from other files. 

+ ols_regression.py, ols_statsmodels.py (Ordinary Least-Squares Regression),
+ lav_regression.py (Least Absolute Value Regression),
+ kr_regression.py (Kernel Ridge Regression with RBF-Kernel),
+ sv_regression.py (Kernelized Support Vector Regression),
+ ppml_regression.py (Poisson Pseudo Maximum Likelihood Estimator),
+ ols_noGrav.py (Ordinary Least-Squares Regression on non-logarithmized data), and
+ neural_network.py (Neural Network Model on non-logarithmized data)
    + create and calibrate/train the estimation models.

+ sfs.py (Sequential Forward Selection),
+ rfe.py (Recursive Feature Elimination),
+ ccls_regression_mip.py (Cardinality-Constrained Least-Squares Regression), and
+ ccls_group_regression_mip.py (Group-Based CCLS)
    + contain the feature selection models/methods. 

+ data_functions.py
    + contains all functions for data (pre-)processing and trained-model scoring/evaluation. 


#### First steps

By entering

     python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --kfold 0

you run

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --onlyModel ols lav kr svr nn olsnG --outDirResults results

Here, each model is trained 100 times without any validation
(kfold = 0 implies that training data is not split).  
The respective scores (R2, Adjusted R2, R2 Grav, Adjusted R2 Grav, max
loss, mean loss, mean absolute percentage) are saved in
`results/no_Validation/<model>.csv` and the corresponding trained
models are saved in `results/no_Validation/models/<modelname>`.

In the score tables, the last column contains the model name,
usually of the form `<model_time_int>`. A summary of the computational
results (averaged scores) is stored in `results/no_Validation.csv`.

If you want to use/train just one of the models---say, e.g., OLS---enter:

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --onlyModel ols --outDirResults results

If you set kfold to an integer k>0, k-fold-cross-validation (CV)
will be used to train the models and obtain corresponding
(averaged) CV-scores.

Each model can be accessed directly via (for the example of LAV regression)

    python3 lav_regression.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --fileName test/lav_test --modelName test/lav_model_test

Then, just one model is trained and saved in `test/lav_model_test`. The scores for this model are saved in `test/lav_test`.

If you want to extend your score file, adding results from new runs while keeping those of previous ones, you can use the option
    `--extendFiles`.  
Otherwise, the model and score files will be overwritten.

To see all possible arguments enter

    python3 run.py -h
(Every function provides help on its mandatory and optional arguments when called with the `-h` or `--help` option.)


#### Manual feature selection

If you want to limit your features, one way is to create a file with 19 "True" or "False" strings, comma-separated in one line.

For example, say `test_fs.csv` contains the following line:  
False,True,True,True,False,True,False,True,False,True,True,True,True,True,True,True,True,False,True

If you enter 

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --featureDecisionFile test_fs.csv

every model is trained using only the features with a "True". 

The feature order is fixed as follows:

    1 Distance
    2 Domestic
    3 International
    4 Inter-EU-zone
    5 Same currency
    6 Population prod
    7 Population sum
    8 Catchment prod
    9 Catchment sum
    10 GDP prod
    11 GDP sum
    12 PLI prod
    13 PLI sum
    14 Nights prod
    15 Nights sum
    16 Coastal location
    17 Island location
    18 Poverty sum
    19 Poverty prod


#### Automatic feature selection

If you have no special features in mind but want to limit/reduce the number of utilized features, you can enter

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --onlyFeatureNumber 10 12

which amounts to

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --onlyFeatureNumber 10 12 --fskfold 10 --fsnumber 100

With recursive feature elimination (rfe), sequential feature selection (sfs) and two approaches for ccls-regression (ccls and ccls_group),
the best, say, 10 or 12 features are searched, respectively, and saved in
`results/feature_selection/10_fold_cross_validation/sfs_k=10_var_True_False.csv`.   
The scores are saved in
`results/feature_selection/10_fold_cross_validation/sfs_k=10.csv`.  
The results for all feature-reduced models trained on all available data are saved in
`results/feature_selection_results/no_Validation/<model>_fs_<fs_model>_k=<Integer>.csv`.
The scores are summarized for model and feature-selection model in
`results/feature_selection_results/no_Validation_<model>_<fs_model>.csv`.

If the feature selection shall be performed only once and only with, say, sfs, you can enter:

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 100 --kfold 0 --onlyFeatureNumber 10 12 --fskfold 10 --fsnumber 1 --onlyFeatureSelectionModel sfs

If you want to perform feature selection without any evaluation of the models like OLS, LAV, etc., you can enter 

    python3 run_feature_selection.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --number 1 --kfold 10 --onlyFeatureNumber 10 12 --onlyModel sfs


#### Automatic feature design via sos1-constraints

If you want to use either the sum or the product of certain parameters (not both), you can enter

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --kfold 0 --sos1

which runs

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --kfold 0 --sos1 --onlyFeatureSelectionModel ccls_group --fsnumber 100 --fskfold 10

The scores are saved in the `results/no_Validation_sos1/` folder,
and the produced "True"-"False"-tables in the `results/feature_selection/10_fold_cross_validation_sos1/` folder.


#### Accessing trained models

It is straightforward to access the trained (and stored) models,
inspect their properties (e.g., regression coefficients), and utilize
them to perform demand estimations on new data points. A python script
for these tasks should contain the following lines:

If you want to access a trained neural network, stored in `path_to_model`:

    import tensorflow as tf
    def custom_loss_function(y_true, y_pred):
        squared = keras.losses.MeanSquaredError()
        absPerc = keras.losses.MeanAbsolutePercentageError()
        return squared(y_true, y_pred)/300000000 + absPerc(y_true, y_pred)
    loaded_model = tf.keras.models.load_model(modelfile, custom_objects = {"custom_loss_function": custom_loss_function})

For all other models (stored in `path_to_model`, resp.):

    import joblib
    loaded_model = joblib.load(path_to_model)

Given the loaded model, the coefficents of ols, lav, olsnG and ppml regression models can be accessed directly:

    regr_coeffs = loaded_model.coef_  # for ols and olsnG
    regr_coeffs = loaded_model.x      # for lav
    regr_coeffs = loaded_model.params # for ppml

To predict the passenger demand, the data points of interest must be
provided in a file with the same structure as `pairwise-stat-data.csv`
(created by extract_data.py). In particular, for technical reasons,
positive non-zero integer values must be provided in the "PAX" column
even though these values will not be used for prediction (the PAX
numbers are what is being estimated).
Data import and (pre)processing is then done via:

    import data_functions
    # default parameters (modify/set as desired):
    sos1 = False
    decide = False
    featureDecisionFile = ''
    binary = False
    predictionBool = True
    # data preparation:
    # logarithmized data:
    x_data_grav, y_data_grav = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile, True, predictionBool)
    # non-logarithmized data:
    x_data_grav_nlD, y_data_grav_nlD = data_functions.getGravityData(inFilePairStat, sos1, decide, featureDecisionFile, False, predictionBool)
    # 1,e instead of 0,1 for indicator variables:
    x_data_grav_score, y_data_grav_score = data_functions.getGravityDataScores(inFilePairStat, sos1, decide, featureDecisionFile, binary, predictionBool)
    # 0,1 for indicator variables:
    x_data_grav_bin, y_data_grav_bin = data_functions.getGravityDataScores(inFilePairStat, sos1, decide, featureDecisionFile, True, predictionBool)
    # preprocessing with 0,1 for indicator variables:
    x_data_grav_pre_bin, y_data_grav_pre_bin = data_functions.getGravityPreprocessedDataScores(inFilePairStat, sos1, decide, featureDecisionFile, True, predictionBool)
    
For estimation using the different models, use:

    import numpy as np
    prediction = np.prod(np.power(x_data_grav_score, regr_coeffs), axis = 1) # for ols, lav
    prediction = np.dot(x_data_grav_bin, regr_coeffs)    # for olsnG
    prediction = loaded_model.predict(x_data_grav_nlD)          # for ppml
    prediction = np.exp(loaded_model.predict(x_data_grav))                   # for kr and svr
    prediction = loaded_model.predict(x_data_grav_pre_bin)                   # for nn
    if len(np.shape(prediction)) > 2:        # (only) for nn, to reshape 3-dim. tensor to vector
            prediction = prediction[:, 0, 0] 

The results of estimations are stored in the vector "prediction".

To save the results in a file write:

    import data_functions
    import csv
    # print zero-based airport index pairs and corresponding demand predictions into outFile:
    inFilePairStat  = <path_to_'pairwise-stat-data.csv'>
    outFile = <filename>
    size, _, pairwise_data = data_functions.readInstance(inFilePairStat, predictionBool)
    results = np.zeros([size - 1, 3], dtype=int)
    for i in range(size - 1):
        results[i][0] = int(pairwise_data[i+1][0]-1)
        results[i][1] = int(pairwise_data[i+1][1]-1)
        results[i][2] = int(round(prediction[i]))
        
    with open('%s.csv' % outFile, mode='w') as file:
        file_writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerows(results)


#### Reproduce paper results

To reproduce the results from the paper (up to small differences due to randomization of data splits and neural network
initialization), run

    python3 run.py --inFilePairStat <path_to_'pairwise-stat-data.csv'> --allPossibilities

The feature selection and design results from the paper can be reproduced by evaluating the feature selections discussed
in the paper using the corresponding feature selection files (see descriptions above), which are also created by the above run.


## Authors and acknowledgment
Imke Joormann, Andreas Tillmann, and Sabrina Ammann


## License
Copyright (c) <2023> Imke Joormann, Andreas Tillmann, and Sabrina Ammann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
