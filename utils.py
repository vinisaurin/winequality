import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.metrics import accuracy_score
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import keras.backend as K
import tensorflow as tf
import logging
import os


def balanced_split(df, with_dev=True):
    j=0
    X={}
    y={}
    for i in df.quality.unique():
        idx = pd.DataFrame(df.quality).query("quality == @i").index
        X_train, X_devtest, y_train, y_devtest = train_test_split(df.drop(['quality'], axis=1).loc[idx], df.quality.loc[idx], test_size=0.30, random_state=1)
        X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.50, random_state=1) 
        if j == 0:
            X['train'] = X_train
            X['dev'] = X_dev
            X['test'] = X_test
            y['train'] = y_train
            y['dev'] = y_dev
            y['test'] = y_test

        else:
            X['train'] = X['train'].append(X_train)
            X['dev'] = X['dev'].append(X_dev)
            X['test'] = X['test'].append(X_test)
            y['train'] = y['train'].append(y_train)
            y['dev'] = y['dev'].append(y_dev)
            y['test'] = y['test'].append(y_test)

        j += 1
    if not with_dev:
        return X['train'].append(X['dev']), X['test'], y['train'].append(y['dev']), y['test']
    return X['train'], X['dev'], X['test'], y['train'], y['dev'], y['test']


# evaluate a single mlp model
def evaluate_dnn(trainX, trainy, testX, testy, param):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
	# define model
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(trainX.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation='relu', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.1)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.1)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.1)),
    keras.layers.Dropout(rate=0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, kernel_initializer="GlorotNormal", activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adamax',
                  metrics=['accuracy'])
	# fit model
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.333, patience=3)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_tot_l2_1.h5", save_best_only=True)
    model.fit(trainX, trainy, epochs=80, validation_data=(testX, testy), batch_size=128, callbacks=[checkpoint_cb, lr_scheduler], verbose = 0)
	# evaluate the model
    _, test_acc = model.evaluate(testX, testy, batch_size = 128, verbose=0)
    pred = model.predict_proba(testX)
    return model, test_acc, pred

def evaluate_xgb(trainX, trainy, testX, testy, params):
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    model = xgb.XGBClassifier(**params)
    model.fit(trainX, trainy)
    test_acc = model.score(testX, testy)
    pred = model.predict_proba(testX)
    return model, test_acc, pred

def evaluate_rf(trainX, trainy, testX, testy, params):
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    model = RandomForestClassifier(**params)
    model.fit(trainX, trainy)
    test_acc = model.score(testX, testy)
    pred = model.predict_proba(testX)
    return model, test_acc, pred

def evaluate_cb(trainX, trainy, testX, testy, params):
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    model = CatBoostClassifier(**params)
    model.fit(trainX, trainy)
    test_acc = model.score(testX, testy)
    pred = model.predict_proba(testX)
    return model, test_acc, pred

def evaluate_lgbm(trainX, trainy, testX, testy, params):
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    model = LGBMClassifier(**params)
    model.fit(trainX, trainy)
    test_acc = model.score(testX, testy)
    pred = model.predict_proba(testX)
    return model, test_acc, pred

def evaluate_et(trainX, trainy, testX, testy, params):
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    model = ExtraTreesClassifier(**params)
    model.fit(trainX, trainy)
    test_acc = model.score(testX, testy)
    pred = model.predict_proba(testX)
    return model, test_acc, pred


def predicting_test(members, testX, n_class=10):
    yhats = list()
    for model in members:
        if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, CatBoostClassifier, LGBMClassifier, ExtraTreesClassifier)):
            predict = model.predict_proba(testX)
            predict = np.concatenate((np.zeros((testX.shape[0],n_class-predict.shape[1])), predict), axis=1)
            yhats.append(predict)
        else:
            yhats.append(model.predict(testX))
    #yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    return summed/10



def model_evaluation(model, testX, testy):
    if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, CatBoostClassifier, LGBMClassifier, ExtraTreesClassifier)):
        sc = StandardScaler()
        testX = sc.fit_transform(testX)
        return model.score(testX, testy)
    return model.evaluate(testX, testy, verbose=0)[1]


def kfold_validation(function, X_train, y_train, param, n_class=10, n_folds=10):
    # prepare the k-fold cross-validation configuration
    kfold = KFold(n_folds, True, 1)
    # cross validation estimation of performance
    scores, members = list(), list()
    prediction = np.zeros((X_train.shape[0], 10))
    for train_ix, test_ix in kfold.split(X_train):
        # select samples
        trainX, trainy = X_train.iloc[train_ix], y_train.iloc[train_ix]
        testX, testy = X_train.iloc[test_ix], y_train.iloc[test_ix]
        # evaluate model
        model, test_acc, pred = function(trainX, trainy, testX, testy, param)
        print('>%.3f' % test_acc)
        if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, CatBoostClassifier, LGBMClassifier, ExtraTreesClassifier)):
            pred = np.concatenate((np.zeros((testX.shape[0],n_class-pred.shape[1])), pred), axis=1)
        prediction[test_ix] = pred
        scores.append(test_acc)
        members.append(model)
    return scores, members, prediction



def adjusting_density(density):
    if density > 100:
        return density/100
    elif density > 10:
        return density/10
    return density

def filtering_alcohol(alcohol):
    if float(alcohol[0:6]) > 900:
        return float(alcohol[0:6])/100
    elif float(alcohol[0:6]) > 100:
        return float(alcohol[0:6])/10
    return float(alcohol[0:6])