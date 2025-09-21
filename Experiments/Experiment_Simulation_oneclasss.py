import os
import sys
import random
import time
import numpy as np
import pandas as pd
cur = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur))

from generate import data
from methods import ConformalPvalues
from predict import evaluate

import warnings
warnings.filterwarnings("ignore")

# Generate data
## Features
p = 300
d = p                  # d=p: Non-sparse setting; d<p: Sparse setting 
mean1 = np.repeat(0,p)
mean0 = np.concatenate((np.repeat(0,d),np.repeat(0,p-d)))
mean = [mean1, mean0]

covv = 0
cov1 = np.diag(np.repeat(1,p))
cov0 = np.diag(np.concatenate((np.repeat(2.5,d),np.repeat(2.5, p-d))))
for c in [cov1,cov0]:
    for i in range(len(c)):
        for j in range(len(c)):
            if i!=j:
                c[i][j] = covv
cov = [cov1, cov0]

## Sample size
n_train = int((len(mean)-1)*max(p,200))    ## 2* When dataset split
n_test = int(5*n_train)

## Ratio of inliers in test set
purity_test = 0.7

## Mix or not
mixture = True

## Algorithm
method = 'MMDCP_oracle'

class_wise_fdr_loop,fdr_loop, power_loop = [[] for _ in range(len(mean)-1)],[],[]
scwise_fdr_loop, cov_loop, cxlen_loop, acc_loop = [],[],[],[]
flr_loop = []

loop_train = 10
loop_test = 10
alpha = 0.05

start = time.time()
for tr in range(loop_train):
    np.random.seed(2024*(tr))

    # initiate the mixture model
    X = data(mean, cov, mixture = mixture)
    # generate pure inliers
    X_train = X.generate(n_train, purity=1)

    for te in range(loop_test):
        np.random.seed(2024*(tr)+te)

        # generate test data
        X_test = X.generate(n_test, purity = purity_test)

        # calculate conformal p-values of test samples
        cp = ConformalPvalues(method = method, inlier_kinds = len(mean)-1)
        pvals_kinds = cp.compute_pvals(trainset = X_train, testset = X_test, tr_oracle_mean=mean, tr_oracle_s=cov)

        # calculate all metrics
        res = evaluate(pvals_kinds, X_test[1], alpha = alpha)

        for k in range(len(mean)-1):
            class_wise_fdr_loop[k].append(res['Class-wise FDR (c{})'.format(k+1)])
        scwise_fdr_loop.append(res['SCwise FDR'])
        fdr_loop.append(res['Global FDR'])
        power_loop.append(res['Power'])
        acc_loop.append(res['Accuracy'])
        cov_loop.append(res['Coverage'])
        cxlen_loop.append(res['Ambiguity'])
        flr_loop.append(res['FLR'])
end = time.time()
t = int(end - start)

# check how mamy prediction sets achieve the coverage rate and control FDR
cct = 0
for c in np.array(cov_loop):
    # print(c)
    if c[2] >= 1-alpha:
        cct += 1
print(cct)
fdrct = 0
for c in np.array(fdr_loop):
    if c[2] <= alpha:
        fdrct += 1
print(fdrct)

# Store the experimental results
for k in range(len(mean)-1):
    res['Class-wise FDR (c{})'.format(k+1)] = np.average(np.array(class_wise_fdr_loop[k]), axis = 0)
res['Global FDR'] = np.average(np.array(fdr_loop), axis = 0)
res['Power'] = np.average(np.array(power_loop), axis = 0)
res['Accuracy'] = np.average(np.array(acc_loop), axis = 0)
res['Coverage'] = np.average(np.array(cov_loop), axis = 0)
res['Ambiguity'] = np.average(np.array(cxlen_loop), axis = 0)
res['FLR'] = np.average(np.array(flr_loop), axis = 0)
res['Time'] = str(t//60)+"m"+str(t%60)+"s"

for k in range(len(mean)-1):
    res['Class-wise FDR (c{})_std'.format(k+1)] = np.std(np.array(class_wise_fdr_loop[k]), axis = 0)
res['Global FDR_std'] = np.std(np.array(fdr_loop), axis = 0)
res['Power_std'] = np.std(np.array(power_loop), axis = 0)
res['Accuracy_std'] = np.std(np.array(acc_loop), axis = 0)
res['Coverage_std'] = np.std(np.array(cov_loop), axis = 0)
res['Ambiguity_std'] = np.std(np.array(cxlen_loop), axis = 0)
res['FLR_std'] = np.std(np.array(flr_loop), axis = 0)


print("current method:", method)
print("oneclass outlier detection" if len(mean)<=2 else "%.f-class predition and outlier detection" % (len(mean)-1))
print("cov:", covv)
print("----------------------------------------------------------")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
print(res[1:].drop(columns=["Ineuqation"]))
