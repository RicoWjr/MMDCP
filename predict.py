import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def evaluate(p_kinds, idx_test, alpha = 0.05):
    """ Evaluate the performance
        p_kinds: number of inlier classes
        idx_test: index of test samples
        alpha: 1 - coverage rate """

    results = pd.DataFrame()
    n_test = len(idx_test)
    # When correction is True, use bh process; when ineq not in Marginal, use CCV in Bates(2023); 
    # When use_sbh the True, use storey correction. Actually, we just consider ineq == "Marginal" instead of CCV.
    for correction in [False,True]:
        for ineq in ['Marginal']:
            for use_sbh in [False,True]:
                reject = np.array([])
                for i in range(len(p_kinds)):
                    # p_kinds includes K lists, and each of them includes p-values of all test samples 
                    # to determine whether test samples belong to class i
                    cp = p_kinds[i]

                    # if use candes process, use (storey) BH to correct the p-values and get results
                    # else just use cp < alpha to get results
                    if correction:
                            reject_k = bh_correction(cp, alpha, ineq, use_sbh)
                    else:
                        use_sbh = False
                        reject_k = cp < alpha 
                    reject = np.concatenate((reject, reject_k))
    
                reject = reject.reshape(len(p_kinds), len(idx_test)).T
                
                # Calculate all metrics
                correct, cov1, pred0, pred00, pred01= 0,0,0,0,0
                pred1, pred11, cov0 = 0,0,0
                class_wise_pred0,class_wise_pred10 = [0]*len(p_kinds), [0]*len(p_kinds)
                pre_len = 0
                for j in range(n_test):
                    CX_i = np.where(reject[j]==False)[0]+1

                    for k in range(1,len(p_kinds)+1):
                        if k not in CX_i:
                            class_wise_pred0[k-1] += 1
                            if idx_test[j] == k:
                                class_wise_pred10[k-1] += 1

                    if len(CX_i) == 0:
                        pred0 += 1
                        if idx_test[j] == 0:
                            cov0 += 1
                            pred00 += 1
                            correct += 1 
                    if len(CX_i) >= 1:
                        pred1 += 1
                        if idx_test[j] in CX_i:
                            pre_len += len(CX_i)
                            cov1 += 1
                        if idx_test[j] == 0:
                            pred01 += 1
                    if len(CX_i) == 1 and CX_i[0] == idx_test[j]:
                        pred11 += 1
                        correct += 1

                
                if pre_len:
                    CX_len = pre_len/cov1
                else:
                    CX_len = 0

                class_wise_fdr = [0]*len(p_kinds)
                for k in range(len(p_kinds)):
                    if class_wise_pred0[k] != 0:
                        class_wise_fdr[k] = class_wise_pred10[k]/class_wise_pred0[k]
                        
                SCwise_fdr = np.sum(np.array(class_wise_pred10))/np.sum(np.array(class_wise_pred0))

                power = pred00/len(np.where(idx_test==0)[0])
                coverage = cov1/len(np.where(idx_test!=0)[0])
                # coverage = (cov1+cov0)/len(idx_test)
                acc = pred11/len(np.where(idx_test!=0)[0])
                # acc = correct/len(idx_test)
                if pred0 != 0:
                    fdr = 1-pred00/pred0
                else: 
                    fdr = 0

                if pred1 != 0:
                    flr = pred01/pred1
                else: 
                    flr = 0

                
                # store the results
                res_tmp = {'Candes':correction, 'Ineuqation':ineq, 'Storey':use_sbh, 'Class-wise FDR (c1)':class_wise_fdr[0]}
 
                if len(p_kinds)>1:
                    for k in range(1,len(p_kinds)):
                        res_tmp['Class-wise FDR (c{})'.format(k+1)] = class_wise_fdr[k]

                res_tmp['SCwise FDR'], res_tmp['Global FDR'], res_tmp['Power'] = SCwise_fdr, fdr, power
                res_tmp['Accuracy'], res_tmp['Coverage'], res_tmp['Ambiguity'] = acc, coverage, CX_len
                res_tmp['FLR'] = flr
                
                res_tmp = pd.DataFrame(res_tmp, index=[0])
                results = pd.concat([results, res_tmp])

    return results
        

def bh_correction(pvals, alpha = 0.05, inequation = 'Marginal', use_sbh = False):

    if use_sbh:
        lambda_par = 0.5
        pi = (1.0 + np.sum(pvals>lambda_par)) / (len(pvals)*(1.0 - lambda_par))
    else:
        pi = 1.0
    alpha_eff = alpha/pi
    reject_ccv, pvals_adj, _, _ = multipletests(pvals, alpha=alpha_eff, method='fdr_bh')

    return reject_ccv
