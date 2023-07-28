#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse

import data_functions


def neural_network(y_train, y_vali, x_train, x_vali, kfold, fileName, modelName):
    """
    trains a neural network and saves the scores and the trained model in a file. The training is
    performed on "normal" (not logarithmized) data that include the features defined below.

    INPUT:
    x_train, y_train: matrix and vector with (preprocessed) normal data and PAX for training
    x_vali, y_vali: matrix and vector with (preprocessed) normal data and (normal) PAX for
                    validation, if no validation should be done enter 0
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
    numFeatures = x_train.shape[1]
    # set batch size
    batch_size = 64

    # add a new dimension to get a 3-dimensional-Input for the neural network (required for training)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, numFeatures))
    x_vali = np.reshape(x_vali, (x_vali.shape[0], 1, numFeatures))

    y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
    y_vali = np.reshape(y_vali, (y_vali.shape[0], 1, 1))

    # create tensors
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valiSet = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    
    train_ds = dataset.repeat().shuffle(buffer_size=32).batch(batch_size=batch_size).prefetch(buffer_size=1)
    validation_ds = valiSet.repeat().shuffle(buffer_size=32).batch(batch_size=batch_size).prefetch(buffer_size=1)

    # build the model
    model = keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[1, numFeatures]))
    model.add(keras.layers.Dense(180, activation='tanh'))
    model.add(keras.layers.Dropout(0.05))
#    model.add(keras.layers.Dense(64, activation=keras.layers.LeakyReLU(), kernel_initializer='he_uniform'))
#    model.add(keras.layers.Dense(16, activation=keras.layers.LeakyReLU(), kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dropout(0.01))
#    model.add(keras.layers.Dense(1, activation=keras.layers.LeakyReLU(), kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(1, activation='relu', kernel_initializer='he_uniform'))
    
    # training stops if the loss hasn't improved in the last 300 epochs
    early_stop = keras.callbacks.EarlyStopping("val_loss", patience=300, restore_best_weights=True)
    best_model = tf.keras.callbacks.ModelCheckpoint(modelName, monitor='val_loss', save_best_only=True,
                                                    save_weights_only= False)


    # define the meanSquaredError as loss and Adam as the optimizer
    model.compile(loss=custom_loss_function, optimizer=keras.optimizers.Adam(learning_rate=0.002),
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsolutePercentageError()])
    
    # training; custom 0.001
    epochs = 10000
    if y_vali.shape[0] != 0:
        model.fit(train_ds, verbose=0, steps_per_epoch=x_train.shape[0]/batch_size, epochs=epochs,
                  validation_data=validation_ds, validation_steps=x_vali.shape[0]/batch_size,
                  callbacks=[early_stop, best_model])
        epochs = early_stop.stopped_epoch
    else:
        model.fit(train_ds, verbose=0, steps_per_epoch=x_train.shape[0]/batch_size, epochs=epochs)

    # evaluation and saving
    if kfold != 0.0:
        pred_vali = model.predict(x_vali)
    else:
        pred_vali = 0
    pred_train = model.predict(x_train)
    
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    data_functions.getScores(fileName, y_train, y_vali, pred_train, pred_vali, modelName, trainableParams, kfold, epochs = epochs)
    #model.save(modelName)


def custom_loss_function(y_true, y_pred):
    squared = keras.losses.MeanSquaredError()
    absPerc = keras.losses.MeanAbsolutePercentageError()
    return squared(y_true, y_pred)/300000000 + absPerc(y_true, y_pred)


def main(inFilePairStat, sos1, featureDecisionFile, fileName, modelName):
    decide = True if featureDecisionFile != '' else False
    x_data_grav_pre_bin, y_data_grav_pre_bin = \
        data_functions.getGravityPreprocessedDataScores(inFilePairStat, sos1, decide, featureDecisionFile, True)
    neural_network(y_data_grav_pre_bin, np.zeros([0]), x_data_grav_pre_bin, np.zeros([0]), 0, fileName, modelName)


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
