import numpy as np
import pandas as pd


class data:
    """Mixture model used to generate simulated data """

    def __init__(self,mean,cov,mixture = True):
        """ Initialize parameters: 
                mean: mean vectors; 
                cov: covariance matrixs; 
                mixture: whether add uniform variables """
        
        self.mean = mean
        self.cov = cov
        self.mix = mixture
        # self.z is W contains uniform variables w in the paper
        self.z = np.random.uniform(low=-3, high=3, size=(len(self.mean[0]), len(self.mean[0])))
        
        # data_store is used to store all data generated from a instance of this class
        self.data_store = []
    
    def generate(self, n, purity, pow = 1):

        # number of inlier classes
        inliers = len(self.mean)-1

        # sample size of inliers and outliers
        n_inlier = int(n*purity/inliers)
        n_outlier = n-n_inlier*inliers

        # generate inliers
        data, idx = None, None
        for i in range(inliers):
            data_in = np.random.multivariate_normal(self.mean[i], self.cov[i], n_inlier, check_valid="ignore")
            # pow = 1 as fault, usually we don not consider the power of each variable
            if pow > 1:
                data_in = np.float_power(pow,data_in)
            idx_in = np.repeat(i+1, n*purity/inliers)
            if i == 0:
                data, idx = data_in, idx_in
            else:
                data = np.vstack((data,data_in))
                idx = np.concatenate((idx,idx_in))

        # generate outliers if purity < 1
        if(purity<1):
            data_out = np.random.multivariate_normal(self.mean[-1], self.cov[-1], n_outlier, check_valid="ignore")
            if pow > 1:
                data_out = np.float_power(pow,data_out)
            data = np.vstack((data,data_out))
            idx = np.concatenate((idx,np.repeat(0, n_outlier)))
        
        # add uniform variables if self.mix is TRUE
        if self.mix:
            cluster_idx = np.random.choice(self.z.shape[0], n, replace=True)
            data = data + self.z[cluster_idx,]
        
        # shuffle the order of data
        shuffle_ix = np.random.permutation(np.arange(len(idx)))
        data, idx = data[shuffle_ix], idx[shuffle_ix]
        self.data_store.append((data,idx,purity))

        return data, idx

    def stat(self):
        """ Perform statistical analysis on generated data """
        ### DO NOT perform this function after generate data more than once in one 'data' class

        class_num = len(self.mean)-1
        print("class num:", class_num, "| oneclass outlier detection" if class_num == 1 
              else "| multi-class prediction and outlier detection")
        print("feature num:", len(self.mean[0]))
        print("========================================")
        for i in range(len(self.data_store)):
            dataset = self.data_store[i]
            if(dataset[2] == 1):
                print("training set", i, ":")
                print("    data num:", len(dataset[0]))
                for k in range(class_num):
                    print("        class", k+1, "mean: %.3f" % np.mean(dataset[0][dataset[1]==k+1]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==k+1]))
                print("========================================")
            else:
                print("test set", i, ": ")
                print("    data num:", len(dataset[0]))
                print("    inliers per class:", len(dataset[0][dataset[1]==1]))
                for k in range(class_num):
                    print("        class", k+1, "mean: %.3f" % np.mean(dataset[0][dataset[1]==k+1]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==k+1]))
                print("    outliers:", len(dataset[0][dataset[1]==0]))
                print("        outlier", "mean: %.3f" % np.mean(dataset[0][dataset[1]==0]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==0]))
                print("========================================")
                