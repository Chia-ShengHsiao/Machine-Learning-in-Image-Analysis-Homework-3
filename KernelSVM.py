#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print('Creating easy synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print('Creating medium synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print('Creating hard easy synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print('Creating two moons dataset')
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print('Creating two circles dataset')
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print('Loading iris dataset')
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print('Loading digits dataset')
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print('Loading breast cancer dataset')
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print('Cannot find the requested data_name')
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

# draw results on 2D plan for binary classification
# this is a fake version (using a random linear classifier)
# modify it for your own usage (pass in parameter etc)
def draw_result_binary_fake(X_train, X_test, y_train, y_test):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]

    Z_class, Z_pred_val = get_prediction_fake(tmpX)

    Z_clapped = np.zeros(Z_pred_val.shape)
    Z_clapped[Z_pred_val>=0] = 1.5
    Z_clapped[Z_pred_val>=1.0] = 2.0
    Z_clapped[Z_pred_val<0] = -1.5
    Z_clapped[Z_pred_val<-1.0] = -2.0

    Z = Z_clapped.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    #    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth=3,
                label='Test Data')

    y_train_pred_class, y_train_pred_val = get_prediction_fake(X_train)
    sv_list_bool = np.logical_and(y_train_pred_val >= -1.0, y_train_pred_val <= 1.0)
    sv_list = np.where(sv_list_bool)[0]
    plt.scatter(X_train[sv_list, 0], X_train[sv_list, 1], s=100, facecolors='none', edgecolors='orange', linewidths = 3, label='Support Vectors')

    y_test_pred_class, y_test_pred_val = get_prediction_fake(X_test)
    score = myscore(y_test_pred_class, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    plt.legend()
    plt.show()

# predict labels using a random linear classifier
# returns a list of length N, each entry is either 0 or 1
def get_prediction_fake(X):
    np.random.seed(100)
    nfeatures = X.shape[1]
    # w = np.random.rand(nfeatures + 1) * 2.0
    w = [-1,0,0]

    assert len(w) == X.shape[1] + 1
    w_vec = np.reshape(w,(-1,1))
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    y_pred_val = np.ravel(np.dot(X_extended,w_vec))
    y_pred_class = np.maximum(np.zeros(y_pred_val.shape), y_pred_val)
    return y_pred_class, y_pred_val

####################################################
# binary label classification

# training kernel svm
# return sv_list: list of surport vector IDs
# alpha: alpha_i's
# b: the bias
def mytrain_binary(X_train, y_train, C, ker, kpar):
    #print('Start training ...')
    #start = time.time()
    if ker == 'linear':
        clf = svm.SVC(C=C, kernel='linear')
    elif ker == 'polynomial':
        clf = svm.SVC(C=C, kernel='poly', degree=kpar, gamma=1)
    elif ker == 'gaussian':
        clf = svm.SVC(C=C, kernel='rbf', gamma=0.5 / kpar)
    clf.fit(X_train, y_train)
    sv_list = clf.support_
    alpha = np.abs(clf.dual_coef_)
    b = clf.intercept_
    #print('Finished training.')
    #print('Train score: %s' % clf.score(X_train, y_train))
    #train_cost = time.time() - start
    #print('C[%s] kpar[%s] train_cost[%s]' % (C, kpar, train_cost))
    return sv_list, alpha, b

def compute_RBF(mat1, mat2, kpar):
    mat1 = np.mat(mat1)
    mat2 = np.mat(mat2)
    trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
    trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T
    k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T
    k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T
    k = k1 + k2 - 2 * np.mat(mat1 * mat2.T)
    rbf = np.exp(- 1. / (2 * kpar) * k)
    return np.asarray(rbf)

# predict given X_test data,
# need to use X_train, ker, kpar_opt to compute kernels
# need to use sv_list, y_train, alpha, b to make prediction
# return y_pred_class as classes (convert to 0/1 for evaluation)
# return y_pred_value as the prediction score
def mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar):
    y = [-1 if yy == 0 else yy for yy in y_train[sv_list]]
    x = X_train[sv_list]
    if ker == 'linear':
        zz = x.dot(X_test.T)
    elif ker == 'polynomial':
        zz = np.power(1 + x.dot(X_test.T), kpar)
    elif ker == 'gaussian':
        zz = compute_RBF(x, X_test, kpar)
    
    y_pred_value = np.dot(alpha * y, zz) + b
    y_pred_value = y_pred_value[0]
    y_pred_class = [0 if yy == -1 else yy for yy in np.sign(y_pred_value)]
    return y_pred_class, y_pred_value

# use cross validation to decide the optimal C and the kernel parameter kpar
# if linear, kpar = -1 (no meaning)
# if polynomial, kpar is the degree
# if gaussian, kpar is sigma-square
# k -- number of folds for cross-validation, default value = 5
def my_cross_validation(X_train, y_train, ker, k=5):
    assert ker == 'linear' or ker == 'polynomial' or ker == 'gaussian'
    Xs = np.array_split(X_train, k)
    ys = np.array_split(y_train, k)
    C_opts = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    if ker == 'linear':
        kpar_opts = [-1.0] # dummpy value
        C_best, kpar_best, score_best = grid_search(C_opts, kpar_opts, Xs, ys, k, ker)
    elif ker == 'polynomial':
        kpar_opts = [1, 2, 3]
        C_best, kpar_best, score_best = grid_search(C_opts, kpar_opts, Xs, ys, k, ker)
    elif ker == 'gaussian':
        kpar_opts = [0.001, 0.01, 0.1, 1.0, 10]
        C_best, kpar_best, score_best = grid_search(C_opts, kpar_opts, Xs, ys, k, ker)
    return C_best, kpar_best

def grid_search(C_opts, kpar_opts, Xs, ys, k, ker):
    print('Start grid search C and kpar')
    C_best = None
    kpar_best = None
    score_best = 0
    for kpar_opt in kpar_opts:
        for C_opt in C_opts:
            test_score = avg_score_by_cross_val(Xs, ys, k, ker, C_opt, kpar_opt)
            
            if test_score > score_best:
                C_best = C_opt
                kpar_best = kpar_opt
                score_best = test_score
    print('%s Best C[%s] kpar[%s] score[%.6f]' % (ker, C_best, kpar_best, score_best))
    return C_best, kpar_best, score_best

def avg_score_by_cross_val(Xs, ys, k, ker, C_opt, kpar_opt):
    test_score = 0
    train_cost = 0
    test_cost = 0
    for i in range(0, k):
        v_X_test = Xs[i]
        v_y_test = ys[i]
        if i == 0:
            v_X_train = np.concatenate(Xs[i+1:])
            v_y_train = np.concatenate(ys[i+1:])
        elif i == k - 1:
            v_X_train = np.concatenate(Xs[0:i])
            v_y_train = np.concatenate(ys[0:i])
        else:
            v_X_train = np.concatenate((np.concatenate(Xs[0:i]), np.concatenate(Xs[i+1:])))
            v_y_train = np.concatenate((np.concatenate(ys[0:i]), np.concatenate(ys[i+1:])))
        start = time.time()
        sv_list, alpha, b = mytrain_binary(v_X_train, v_y_train, C_opt, ker, kpar_opt)
        train_cost += time.time() - start
        start = time.time()
        y_test_pred_class, y_test_pred_val = mytest_binary(v_X_test, v_X_train, v_y_train, sv_list, alpha, b, ker, kpar_opt)
        test_cost += time.time() - start
        test_score += myscore(y_test_pred_class, v_y_test)
    test_score /= k
    train_cost /= k
    test_cost /= k
    print('%s C[%s] kpar[%s] score[%.6f] train_cost[%.6f] test_cost[%.6f]'
            % (ker, C_opt, kpar_opt, test_score, train_cost, test_cost))
    #print('%s,%s,%.6f,%.6f,%.6f' % (kpar_opt, C_opt, test_score, 1000 * train_cost, 1000 * test_cost))
    return test_score

################

def main():

    #######################
    # get data
    # only use binary labeled

    X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    #X_train, X_test, y_train, y_test = acquire_data('moons')
    #X_train, X_test, y_train, y_test = acquire_data('circles')
    #X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    #draw_data(X_train, X_test, y_train, y_test, nclasses)
    # a face function to draw svm results
    #draw_result_binary_fake(X_train, X_test, y_train, y_test)

    ker = 'linear'
    #ker = 'polynomial'
    #ker = 'gaussian'

    C_opt, kpar_opt = my_cross_validation(X_train, y_train, ker)
    sv_list, alpha, b = mytrain_binary(X_train, y_train, C_opt, ker, kpar_opt)

    y_test_pred_class, y_test_pred_val = mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar_opt)

    test_score = myscore(y_test_pred_class, y_test)

    print('Test Score:', test_score)

if __name__ == "__main__": main()