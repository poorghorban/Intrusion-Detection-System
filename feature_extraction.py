from sklearn.decomposition import PCA

"""Function to extract feature with pca method .

    Parameters
    ----------
    X_train : ndarray
    X_test : ndarray
    max_feature : int 
        number of features extract

    Returns
    -------
    X_train : ndarray
    X_test  : ndarray
"""
def principal_component_analysis(X_train , X_test , max_feature):
    # create object PCA 
    pca = PCA(n_components=max_feature)
    # fit object 
    pca.fit(X_train)
    # transform train and test 
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    # return new data 
    return X_train , X_test

