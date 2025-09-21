import numpy as np
import copy
import inspect
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation as MAD


class ConformalPvalues:
    """ Calculate conformal p-values """

    def __init__(self, method = 'MMDCP', inlier_kinds = 1):
        """ Initialize parameters: 
                method: the modals or methods to perform outlier detection and classification; 
                inlier_kinds: number of inlier classes
                cp: whether perform conformal prediction """
        self.m = method
        self.ik = inlier_kinds

    def compute_pvals(self, trainset, testset, tr_oracle_mean = None, tr_oracle_s = None):
        
        # p_kinds is used to store the results of all p-values
        p_kinds = []

        data_train = trainset[0]
        idx_train = trainset[1]
        data_calib = trainset[0]
        idx_calib = trainset[1]

        data_test = testset[0]
        idx_test = testset[1]

        if self.m == 'MMDCP':
            n_test = len(data_test)
            for i in range(1,self.ik+1):
                data_in = data_calib[idx_calib == i]
                data_tr = data_train[idx_train == i]
                scores = X_score(data_tr, data_test, data_in)
                scores_cal = scores[:-n_test]
                scores_test = scores[-n_test:]

                # conformal predictin codes referring to Bates(2023)
                scores_mat = np.tile(scores_cal, (len(scores_test),1))
                tmp = np.sum(scores_mat >= scores_test.reshape(len(scores_test),1), 1)
                pvals = (1.0+tmp)/(1.0+len(data_in))
                p_kinds.append(pvals)

        if self.m == 'MMDCP_oracle':
            n_test = len(data_test)
            for i in range(1,self.ik+1):
                data_in = data_calib[idx_calib == i]
                oracle_mean = tr_oracle_mean[i-1]
                oracle_s = [1]*len(oracle_mean)
                for p in range(len(oracle_mean)):
                    oracle_s[p] = tr_oracle_s[i-1][p][p]

                scores = X_oracle(data_in,data_test,oracle_mean,oracle_s)
                scores_cal = scores[:-n_test]
                scores_test = scores[-n_test:]

                # conformal predictin codes referring to Bates(2023)
                scores_mat = np.tile(scores_cal, (len(scores_test),1))
                tmp = np.sum(scores_mat >= scores_test.reshape(len(scores_test),1), 1)
                pvals = (1.0+tmp)/(1.0+len(data_in))
                p_kinds.append(pvals)

        return p_kinds

def X_score(x_train,x_test,x_cal):
    """ MMDCP """

    x = np.vstack((x_cal,x_test))
    std = np.std(x[:len(x_train)],axis=0)
    mad = MAD(x[:len(x_train)])
    m_std,m_mad = [],[]
    for i in range(len(std)):
        if std[i]!=0:
            m_std.append(std[i])  
        if mad[i]!=0:
            m_mad.append(mad[i]) 
    for i in range(len(std)):
        if std[i]==0:
            std[i]=np.mean(m_std)   
        if mad[i]==0:
            mad[i]=np.median(m_mad)
    x = (x-np.average(x[:len(x_train)],axis=0))/std

    X_scores = []
    for p in range(len(x)):
        X_scores.append(sum(x[p]*x[p]))

    return np.array(X_scores)

def X_oracle(x_cal,x_test,mu,sigma):
    """ MMDCP_Oracle """

    x = np.vstack((x_cal,x_test))
    x = (x-mu)/sigma

    X_scores = []
    for p in range(len(x)):
        X_scores.append(sum(x[p]*x[p]))

    return np.array(X_scores)
