# -*- coding: utf-8 -*-

import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from .features import PEFeatureExtractor
from sklearn.model_selection import GridSearchCV

numberoftrainsetfile = 1
numberoftrainsets = 1

def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                yield line


def vectorize(irow, raw_features_string, X_path, y_path, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    extractor = PEFeatureExtractor()
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    extractor = PEFeatureExtractor()
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir, rows):
    print("Vectorizing Dataset set")
    X_path = os.path.join(data_dir, "X.dat")
    y_path = os.path.join(data_dir, "y.dat")
    raw_feature_paths = [os.path.join(data_dir, "features.jsonl")]
    vectorize_subset(X_path, y_path, raw_feature_paths, rows)

def read_vectorized_features(data_dir, rows):
    ndim = PEFeatureExtractor.dim

    X_path = os.path.join(data_dir, "X.dat")
    y_path = os.path.join(data_dir, "y.dat")
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(rows, ndim))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=rows)

    return X, y

def read_metadata_record(raw_features_string):
    """
    Decode a raw features stringa and return the metadata fields
    """
    full_metadata = json.loads(raw_features_string)
    return {"sha256": full_metadata["sha256"], "appeared": full_metadata["appeared"], "label": full_metadata["label"]}


def create_metadata(data_dir):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()
    raw_feature_paths = [os.path.join(data_dir, "features.jsonl")]
    records = list(pool.imap(read_metadata_record, raw_feature_iterator(raw_feature_paths)))
    records = [dict(record, **{"subset": "train"}) for record in records]

    metadf = pd.DataFrame(records)[["sha256", "appeared", "subset", "label"]]
    metadf.to_csv(os.path.join(data_dir, "metadata.csv"))
    print("\n[Done] create_metadata\n")
    
    return metadf


def read_metadata(data_dir):
    """
    Read an already created metadata file and return its dataframe
    """
    return pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)


def train_model(data_dir, rows):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    X, y = read_vectorized_features(data_dir, rows)

    # Set params
    # Scores ~0.784 (without tuning and early stopping)
    params = {'boosting_type': 'gbdt',
            'max_depth' : -1,
            'objective': 'binary',
            'nthread': 3, # Updated from nthread
            'num_leaves': 64,
            'learning_rate': 0.05,
            'max_bin': 512,
            'subsample_for_bin': 200,
            'subsample': 1,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 5,
            'reg_lambda': 10,
            'min_split_gain': 0.5,
            'min_child_weight': 1,
            'min_child_samples': 5,
            'scale_pos_weight': 1,
            'num_class' : 1,
            'metric' : 'binary_error'}

    # Create parameters to search
    gridParams = {
        'learning_rate': [0.15, 0.2, 0.25, 0.3], #default = 0.1
        'n_estimators': [40],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt'],
        'objective' : ['binary'],
        'random_state' : [501], # Updated from 'seed'
        'colsample_bytree' : [0.65, 0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
    }

    # Create classifier to use. Note that parameters have to be input manually
    # not as a dict!
    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

    # Create the grid
    grid = GridSearchCV(mdl, gridParams,
                        verbose=0,
                        cv=4,
                        n_jobs=2)
    # train
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)


    # train
    lgbm_dataset = lgb.Dataset(X, y)
    lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

    return lgbm_model


def predict_sample(lgbm_model, file_data):
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return lgbm_model.predict([features])[0]

'''
    교차검증
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def cross_validation(data_dir, rows):
    X, y = read_vectorized_features(data_dir, rows)

    #Total 10th cross_validation
    for numberoftest in range(0, 5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        #train
        lgbm_dataset = lgb.Dataset(X_train, y_train)
        lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

        #predict
        predictions_lgbm_prob = lgbm_model.predict(X_test)
        predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.75, 1, 0)

        #print accuracy
        acc_lgbm = accuracy_score(y_test, predictions_lgbm_01)
        print("accuaracy : ", acc_lgbm)

        mt = confusion_matrix(y_test, predictions_lgbm_01)
        print(mt)

    return lgbm_model