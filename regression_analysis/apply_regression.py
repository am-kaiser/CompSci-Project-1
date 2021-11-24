import numpy as np
import linear_regression
import franke
import seaborn as sns
def apply_regression(p, n, noise, r=np.zeros(1), reg_type="ols", ridge_lambda=np.ones(1), n_boots=np.ones(1, dtype=int), k_folds=np.ones(1, dtype=int)): 
    #applies regression for multiple parameter combos
    train_MSE_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    test_MSE_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    train_R2_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    test_R2_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    test_bias_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)]) #bias in test set
    test_var_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])  #variance in test set
    for j in range(len(n)):
        for k in range(len(noise)):
            x1 = np.linspace(0,1,n[j])
            x2 = np.linspace(0,1,n[j])
            xx1, xx2 = np.meshgrid(x1, x2)
            xx1 = xx1.reshape((n[j]*n[j]),1)
            xx2 = xx2.reshape((n[j]*n[j]),1)
            y = franke.Franke(xx1, xx2, var=noise[k])
            linear_reg = linear_regression.linear_regression2D(xx1, xx2, y) 
            
            for i in range(len(p)):
                for l in range(len(r)):
                    if(reg_type == "ols"):
                        linear_reg.apply_leastsquares(order=p[i], test_ratio=r[l])
                        train_MSE_arr[i,j,k,l,0,0,0] = linear_reg.trainMSE
                        test_MSE_arr[i,j,k,l,0,0,0] = linear_reg.testMSE
                        train_R2_arr[i,j,k,l,0,0,0] = linear_reg.trainR2
                        test_R2_arr[i,j,k,l,0,0,0] = linear_reg.testR2
                        test_bias_arr[i,j,k,l,0,0,0] = linear_reg.testbias
                        test_var_arr[i,j,k,l,0,0,0] = linear_reg.testvar

                        
                    elif(reg_type == "ols_bootstrap"):
                        for b in range(len(n_boots)):
                            linear_reg.apply_leastsquares_bootstrap(order=p[i], test_ratio=r[l], n_boots=n_boots[b])
                            train_MSE_arr[i,j,k,l,0,b,0] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,0,b,0] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,0,b,0] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,0,b,0] = linear_reg.testR2
                            test_bias_arr[i,j,k,l,0,b,0] = linear_reg.testbias
                            test_var_arr[i,j,k,l,0,b,0] = linear_reg.testvar
                            
                    elif(reg_type == "ols_crossvalidation"):
                        #note r is of length one for crossvalidation. we don't need test ratio
                        for c in range(len(k_folds)):
                            linear_reg.apply_leastsquares_crossvalidation(order=p[i], kfolds=k_folds[c])
                            train_MSE_arr[i,j,k,l,0,0,c] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,0,0,c] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,0,0,c] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,0,0,c] = linear_reg.testR2
                            test_bias_arr[i,j,k,l,0,0,c] = linear_reg.testbias
                            test_var_arr[i,j,k,l,0,0,c] = linear_reg.testvar     

                    elif(reg_type == "ridge"):
                        for rl in range(len(ridge_lambda)):
                            linear_reg.apply_leastsquares(order=p[i], test_ratio=r[l], ridge=True, lmbda=ridge_lambda[rl])
                            train_MSE_arr[i,j,k,l,rl,0,0] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,rl,0,0] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,rl,0,0] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,rl,0,0] = linear_reg.testR2
                            test_bias_arr[i,j,k,l,rl,0,0] = linear_reg.testbias
                            test_var_arr[i,j,k,l,rl,0,0] = linear_reg.testvar
                    elif(reg_type == "ridge_bootstrap"):
                        for rl in range(len(ridge_lambda)):
                            for b in range(len(n_boots)):       
                                linear_reg.apply_leastsquares_bootstrap(order=p[i], test_ratio=r[l], n_boots=n_boots[b], ridge=True, lmbda=ridge_lambda[rl])
                                train_MSE_arr[i,j,k,l,rl,b,0] = linear_reg.trainMSE
                                test_MSE_arr[i,j,k,l,rl,b,0] = linear_reg.testMSE
                                train_R2_arr[i,j,k,l,rl,b,0] = linear_reg.trainR2
                                test_R2_arr[i,j,k,l,rl,b,0] = linear_reg.testR2
                                test_bias_arr[i,j,k,l,rl,b,0] = linear_reg.testbias
                                test_var_arr[i,j,k,l,rl,b,0] = linear_reg.testvar
                    elif(reg_type == "ridge_crossvalidation"):
                        for rl in range(len(ridge_lambda)):
                            for c in range(len(k_folds)):
                                linear_reg.apply_leastsquares_crossvalidation(order=p[i], kfolds=k_folds[c], ridge=True, lmbda=ridge_lambda[rl])
                                train_MSE_arr[i,j,k,l,rl,0,c] = linear_reg.trainMSE
                                test_MSE_arr[i,j,k,l,rl,0,c] = linear_reg.testMSE
                                train_R2_arr[i,j,k,l,rl,0,c] = linear_reg.trainR2
                                test_R2_arr[i,j,k,l,rl,0,c] = linear_reg.testR2
                                test_bias_arr[i,j,k,l,rl,0,c] = linear_reg.testbias
                                test_var_arr[i,j,k,l,rl,0,c] = linear_reg.testvar
                    
    return train_MSE_arr, test_MSE_arr, train_R2_arr, test_R2_arr, test_bias_arr, test_var_arr

def apply_regression2(p, n, noise, r=np.zeros(1), reg_type="ols", ridge_lambda=np.ones(1), n_boots=np.ones(1, dtype=int), k_folds=np.ones(1, dtype=int)): 
    #applies regression for multiple parameter combos
    train_MSE_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    test_MSE_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    train_R2_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    test_R2_arr = np.zeros([len(p), len(n), len(noise), len(r), len(ridge_lambda), len(n_boots), len(k_folds)])
    for j in range(len(n)):
        for k in range(len(noise)):
            x1 = np.linspace(0,1,n[j])
            x2 = np.linspace(0,1,n[j])
            xx1, xx2 = np.meshgrid(x1, x2)
            xx1 = xx1.reshape((n[j]*n[j]),1)
            xx2 = xx2.reshape((n[j]*n[j]),1)
            y = franke.Franke(xx1, xx2, var=noise[k])
            linear_reg = linear_regression.linear_regression2D(xx1, xx2, y) 
            
            for i in range(len(p)):
                for l in range(len(r)):
                    if(reg_type == "ols"):
                        linear_reg.apply_leastsquares(order=p[i], test_ratio=r[l])
                        train_MSE_arr[i,j,k,l,0,0,0] = linear_reg.trainMSE
                        test_MSE_arr[i,j,k,l,0,0,0] = linear_reg.testMSE
                        train_R2_arr[i,j,k,l,0,0,0] = linear_reg.trainR2
                        test_R2_arr[i,j,k,l,0,0,0] = linear_reg.testR2
                        
                    elif(reg_type == "ols_bootstrap"):
                        for b in range(len(n_boots)):
                            linear_reg.apply_leastsquares_bootstrap(order=p[i], test_ratio=r[l], n_boots=n_boots[b])
                            train_MSE_arr[i,j,k,l,0,b,0] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,0,b,0] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,0,b,0] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,0,b,0] = linear_reg.testR2
                            
                    elif(reg_type == "ols_crossvalidation"):
                        #note r is of length one for crossvalidation. we don't need test ratio
                        for c in range(len(k_folds)):
                            linear_reg.apply_leastsquares_crossvalidation(order=p[i], kfolds=k_folds[c])
                            train_MSE_arr[i,j,k,l,0,0,c] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,0,0,c] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,0,0,c] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,0,0,c] = linear_reg.testR2
                    elif(reg_type == "ridge"):
                        for rl in range(len(ridge_lambda)):
                            linear_reg.apply_leastsquares(order=p[i], test_ratio=r[l], ridge=True, lmbda=ridge_lambda[rl])
                            train_MSE_arr[i,j,k,l,rl,0,0] = linear_reg.trainMSE
                            test_MSE_arr[i,j,k,l,rl,0,0] = linear_reg.testMSE
                            train_R2_arr[i,j,k,l,rl,0,0] = linear_reg.trainR2
                            test_R2_arr[i,j,k,l,rl,0,0] = linear_reg.testR2
                    elif(reg_type == "ridge_bootstrap"):
                        for rl in range(len(ridge_lambda)):
                            for b in range(len(n_boots)):       
                                linear_reg.apply_leastsquares_bootstrap(order=p[i], test_ratio=r[l], n_boots=n_boots[b], ridge=True, lmbda=ridge_lambda[rl])
                                train_MSE_arr[i,j,k,l,rl,b,0] = linear_reg.trainMSE
                                test_MSE_arr[i,j,k,l,rl,b,0] = linear_reg.testMSE
                                train_R2_arr[i,j,k,l,rl,b,0] = linear_reg.trainR2
                                test_R2_arr[i,j,k,l,rl,b,0] = linear_reg.testR2
                    elif(reg_type == "ridge_crossvalidation"):
                        for rl in range(len(ridge_lambda)):
                            for c in range(len(k_folds)):
                                linear_reg.apply_leastsquares_crossvalidation(order=p[i], kfolds=k_folds[c], ridge=True, lmbda=ridge_lambda[rl])
                                train_MSE_arr[i,j,k,l,rl,0,c] = linear_reg.trainMSE
                                test_MSE_arr[i,j,k,l,rl,0,c] = linear_reg.testMSE
                                train_R2_arr[i,j,k,l,rl,0,c] = linear_reg.trainR2
                                test_R2_arr[i,j,k,l,rl,0,c] = linear_reg.testR2
                    
    return train_MSE_arr, test_MSE_arr, train_R2_arr, test_R2_arr

def plot_stat(ratio=0.1, num=100, stat="testMSE", method="ols", n_boot=1000, k_fold=1000, ridge_lmb=122.0):
    p=np.load("data/p.npy")
    n=np.load("data/n.npy")
    noise=np.load("data/noise.npy")
    r=np.load("data/r.npy")
    ridge_lambda=np.load("data/ridge_lambda.npy")
    k_folds=np.load("data/k_folds.npy")
    n_boots=np.load("data/n_boots.npy")
    train_MSE=np.load("data/train_MSE"+method+".npy")
    test_MSE=np.load("data/test_MSE"+method+".npy")
    train_R2=np.load("data/train_R2"+method+".npy")
    test_R2=np.load("data/test_R2"+method+".npy")
    test_bias=np.load("data/test_bias"+method+".npy")
    test_var=np.load("data/test_var"+method+".npy")

    maxtrainMSE = np.amax(train_MSE)
    maxtestMSE = np.amax(test_MSE)

    n_ind = 0
    for i in range(len(n)):
        if num == n[i]:
            n_ind = i
    r_ind = 0
    for i in range(len(r)):
        if ratio == r[i]:
            r_ind = i
    lambda_ind = 0
    for i in range(len(ridge_lambda)):
        if ridge_lmb == ridge_lambda[i]:
            lambda_ind = i
    nb_ind = 0
    for i in range(len(n_boots)):
        if n_boot == n_boots[i]:
            nb_ind = i
    cv_ind = 0
    for i in range(len(k_folds)):
        if k_fold == k_folds[i]:
            cv_ind = i
    if(method=="ols_crossvalidation" or method=="ridge_crossvalidation"):
        r_ind = 0
    if(method=="ols" or method=="ols_crossvalidation" or method=="ridge" or method=="ridge_crossvalidation"):
        nb_ind=0
    if(method=="ols" or method=="ols_bootstrap" or method=="ridge" or method=="ridge_bootstrap"):
        cv_ind=0
    if(method=="ols" or method=="ols_bootstrap" or method=="ols_crossvalidation"):
        lambda_ind=0 
    
    if stat=="train MSE":
        trainMSE = train_MSE[:,n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
        sns.heatmap(trainMSE, annot=True, cmap="mako", vmax=np.amax(trainMSE), vmin = np.amin(trainMSE))
    elif stat=="test MSE":
        testMSE = test_MSE[:,n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
        sns.heatmap(testMSE, annot=True, cmap="mako", vmax=np.amax(testMSE), vmin = np.amin(testMSE))
    elif stat=="test R2":
        testR2 = test_R2[:,n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
        sns.heatmap(testR2, annot=True, cmap="mako", vmax=np.amax(testR2), vmin = np.amin(testR2))
    elif stat=="test bias":
        testbias = test_bias[:,n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
        sns.heatmap(testbias, annot=True, cmap="mako", vmax=np.amax(testbias), vmin = np.amin(testbias))
    elif stat=="test variance":
        testvar = test_var[:,n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
        sns.heatmap(testvar, annot=True, cmap="mako", vmax=np.amax(testvar), vmin = np.amin(testvar))
