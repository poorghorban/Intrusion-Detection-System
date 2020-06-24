import numpy as np 
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.preprocessing import MinMaxScaler


"""Function to processing train and test dataset.
    (convert categorical variable to binary by OneHotEncoder method)

    Parameters
    ----------
    X_train : ndarray
    X_test : ndarray
    index columns categorical: list

    Returns
    -------
    X_train : ndarray 
    X_test : ndarray
"""
def one_hot_encoding(X_train , X_test , index):
    # create object OneHotEncoder
    ohc = OneHotEncoder() 
    # fit object 
    ohc.fit(X_train[:,index[0]:index[-1]+1])
    # transform train and test dataset
    binary_train = ohc.transform(X_train[:,index[0]:index[-1]+1]).toarray()
    binary_test = ohc.transform(X_test[:,index[0]:index[-1]+1]).toarray()
    # delete col by index 
    X_train = np.delete(X_train , index , axis=1)
    X_test = np.delete(X_test , index , axis=1)
    # concate by array 
    X_train = np.concatenate((X_train, binary_train), 1)
    X_test = np.concatenate((X_test, binary_test), 1)

    # return processed train and test 
    return X_train , X_test


"""Function to processing train and test labels.
    (convert categorical variable to binary by LabelEncoder method)

    Parameters
    ----------
    y_train : ndarray
    y_test : ndarray

    Returns
    -------
    y_train : ndarray 
    y_test: ndarray
"""
def label_encoder(y_train , y_test):
    # create object LabelEncoder
    le = LabelEncoder()
    # fit object 
    le.fit(y_train)
    # transform train and test labels 
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    # label names
    label_names = le.classes_
    # return processed labels
    return y_train , y_test , label_names



def min_max_scaler(X):
    # create object MinMaxScaler
    mms = MinMaxScaler((0,1))
    # fit object 
    X = mms.fit_transform(X)
    # return 
    return X

        