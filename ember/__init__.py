# -*- coding: utf-8 -*-

import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from .features import PEFeatureExtractor

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

    #train
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