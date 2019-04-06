#!/usr/bin/python3.6
# -*-coding:utf-8 -

"""
File contatining functions for Functional Data Analysis.
"""
import numpy as np
import pandas as pd
import sklearn

def gaussian_kernel_smoothing(X, y, grid, cv=5):
    """
    Perform a gaussian kernel smoothing of data given as input.
    It fits a model y = f(X) + e where f is a unknown function.
        X: features - one dimensional array
        y: responses - one dimensional array
        grid: evaluation grid - one dimensonal array
        cv: cross validation splitting strategy - int
    NB: gamma = 1/h
    """
    
    # Perform a cross validation to find the best parameters
    cv = sklearn.model_selection.GridSearchCV(
            estimator=sklearn.kernel_ridge.KernelRidge(kernel='rbf'),  
            param_grid={"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                        "gamma": np.logspace(-2, 2, 5)},
            scoring='r2',
            cv=cv,
    )
    
    # Eventually convert X and y
    if X is not np.ndarray:
        X = np.array(X).reshape(-1, 1)
    
    if y is not np.ndarray:
        y = np.array(y)
        
    if grid is not np.ndarray:
        grid = np.array(grid).reshape(-1, 1)
    
    # Perform
    cv.fit(X, y)
    error = cv.score(X, y)
    y_hat = cv.predict(grid)
    
    return y_hat, error

def column_scaler(data, center=True, scale=True):
    """
    Standardize the data matrix with respoect to the rows.
        data: a matrix to scale
        center: should we center the matrix?
        scale: should we scale the matrix?
    """
    scaler = sklearn.preprocessing.StandardScaler(
        with_mean=center, 
        with_std=scale)

    data_scale = scaler.fit_transform(data)
    return pd.DataFrame(data_scale)

def row_scaler(data, center=True, scale=True):
    """
    Standardize the data matrix with respect to the columns.
        data: a matrix to scale
        center: should we center the data?
        scale: should we scale the data?
    """
    scaler = sklearn.preprocessing.StandardScaler(
        with_mean=center, 
        with_std=scale)

    data_scale = scaler.fit_transform(data.T)
    return pd.DataFrame(data_scale.T)

def mean(data):
    """
    Compute the mean curve from a set of curves. 
    """
    scaler = sklearn.preprocessing.StandardScaler(
        with_mean=True,
        with_std=True)

    return scaler.fit(data).mean_

def covariance(data):
    """
    Compute an estimation of the covariance of the data.
    """
    data_center = pd.DataFrame(column_scaler(data, center=True, scale=False))
    nb = len(data)
    return np.dot(data_center.T, data_center) / (nb - 1)

def univariate_fpca(data, percent_explained=1, whiten=False):
    """
    Perform a functional PCA on a matrix.
    Perform a functional PCA on a matrix. The matrix is build as follow: 
        - Each row represents a new observation of the stochastic process. 
        - Each column is a different time step at which the stochastic process is observed. Each time step is assumed to be in [0, 1].
    * percent_explained: Percentage of variance explained by the components.
    * whiten: Ensure uncorrelated components.  
    """

    # Perform the PCA
    pca = sklearn.decomposition.PCA(whiten=whiten)
    pca.fit(data)
    
    keep = pca.explained_variance_ratio_.cumsum() < percent_explained

    eigenfunctions = pd.DataFrame(pca.components_[keep]).transpose()
    projection = pd.DataFrame(pca.transform(data)[:, keep])

    return {'eigenfunction': eigenfunctions, 'coef': projection}

def compute_KL_expansion(mean, eigenfunctions, coefs):
    """
    Function that compute the Karhunen Loeve expansion given a set of coefficient and basis function. 
    Careful: use non standardized coefficients.
        mean: the mean function of the data.
        eigenfunctions: a dataframe of eigenfunctions
        coefs: a dataframe of coefficients
    """

    expansion = mean + coefs.dot(eigenfunctions.T)

    return expansion 

def mean_reconstruction_error(data, data_pred):
    """
    Compute the mean reconstruction error for a given curve.
        data: The true data.
        data_pred: The predicted data. 
    """
    error = {}
    for key in data.keys():
        error_rows = []
        for row in range(0, data[key].shape[0]):
            r2 = sklearn.metrics.r2_score(data[key].iloc[row], 
                                     data_pred[key].iloc[row])
            error_rows.append(r2)
        error[key] = np.mean(error_rows)
    return error