
import numpy as np 
from sklearn.utils import resample

"""Function to resample data .

    Parameters
    ----------
    X : ndarray
    y : ndarray

    Returns
    -------
    X : ndarray
    y: ndarray
"""
def resample_data(X , y):
    # concate X and y 
    data = np.concatenate((X, y.reshape(-1,1)), 1) 
    n_columns = data.shape[1]
    # unique y type and frequency 
    unique_y , counts_y = np.unique(y , return_counts=True)
    # max frequency and index y type 
    max_counts = np.max(counts_y)
    max_index = np.argmax(counts_y)


    # add sample to data 
    for i in range(len(unique_y)):
        if i == max_index:
            continue
        else:
            # get rows from data with label i 
            minority_data = data[:,:][data[:,n_columns-1] == unique_y[i]]
            n_samples = max_counts - counts_y[i]
            # get more sample 
            minority_upsampled = resample(minority_data ,n_samples=n_samples , random_state=144)
            # add sample to majority sampled 
            data = np.concatenate((data, minority_upsampled), 0) 

    n_columns = data.shape[1]
    # X and y resampled 
    y = data[:,n_columns-1]
    X = data[:,0:n_columns-1]

    # return new X and y 
    return X , y

