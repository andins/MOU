#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:50:53 2017

@author: andrea
"""

import numpy as np
import scipy.linalg as spl
import scipy.stats as stt
from sklearn.base import BaseEstimator
import matplotlib.pyplot as pp
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA



class MOU(BaseEstimator):
    
    def __init__(self, n_nodes=10, C=None, Sigma=None, tau_x=1.0, mu=0.0, random_state=None):
        if random_state is not None:  # set seed for reproducibility
            np.random.seed(random_state)
        if C is None:
            self.C = np.zeros([n_nodes, n_nodes])
        else:
            self.C = C
        if Sigma is None:
            self.Sigma = np.eye(n_nodes) * 0.5 + np.eye(n_nodes) * 0.5 * np.random.rand(n_nodes)
        elif np.isscalar(Sigma):
            self.Sigma = np.eye(n_nodes) * np.sqrt(Sigma)
        else:
            raise("Only scalar values accepted corresponding to diagonal noise covariance matrix.")
        self.tau_x = tau_x
        self.n_nodes = n_nodes
        self.mu = mu

    def fit(self, X, y=None, SC_mask=None, method='lyapunov', norm_fc=None, 
            epsilon_EC=0.0005, epsilon_Sigma=0.05, min_val_EC=0.0, max_val_EC=0.4,
            n_opt=10000, regul_EC=0, regul_Sigma=0,
            true_model=None, verbose=0):
        """
        Estimation of MOU parameters (connectivity C, noise covariance Sigma,
        and time constant tau_x) with Lyapunov optimization as in: Gilson et al.
        Plos Computational Biology (2016).
        PARAMETERS:
            X: the timeseries data of the system to estimate, shape: T time points x P variables.
            y: needed to comply with BaseEstimator (not used here)
            SC_mask: mask of known non-zero values for connectivity matrix, for example 
            estimated by DTI
            method: 'lyapunov' or 'moments'
            norm_fc: normalization factor for FC. Normalization is needed to avoid high connectivity value that make
            the network activity explode. FC is normalized as FC *= 0.5/norm_fc. norm_fc can be specified to be for example
            the average over all entries of FC for all subjects or sessions in a given group. If not specified the normalization factor is
            the mean of 0-lag covariance matrix of the provided time series ts_emp. 
            epsilon_EC : learning rate for connectivity (this should be about n_nodes times smaller than epsilon_Sigma).
            epsilon_Sigma : learning rate for Sigma (this should be about n_nodes times larger than epsilon_EC).
            min_val_EC : minimum value to bound connectivity estimate. This should be zero or slightly negative (too negative limit can bring to an
            inhibition dominated system). If the empirical covariance has many negative entries then a slightly negative limit can improve the estimation
            accuracy.
            max_val_EC : maximum value to bound connectivity estimate. This is useful to avoid large weight that make the system unstable.
            If the estimated connectivity saturates toward this value (it usually doesn't happen) it can be increased.
            n_opt : number of maximum optimization steps. If final number of iterations reaches this maximum it means the algorithm has not converged.
            regul_EC : regularization parameter for connectivity (try a value of 0.5)
            regul_Sigma : regularization parameter for Sigma (try a value of 0.001)
            true_model: a tuple (true_C, true_S) of true connectivity matrix and noise covariance (for calculating the error of estimation iteratively)
            verbose: verbosity level; 0: no output; 1: prints regularization parameters,
            estimated \tau_x, used lag \tau for lagged-covariance and evolution of the iterative
            optimization with final values of data covariance (Functional Connectivity) fitting;
            2: create a diagnostic graphics with cost function over iteration 
            (distance between true and estimated connectivity is also shown 
            if true_C is not None) and fitting of data covariance matrix (fitting of connectivity
            and Sigma are also shown if true_C and true_S are not None).
        RETURN:
            C: estimated connectivity matrix, shape [P, P] with null-diagonal
            Sigma: estimated noise covariance, shape [P, P]
            tau_x: estimated time constant (scalar)
            d_fit: a dictionary with diagnostics of the fit; keys are: iterations, distance and correlation
        """
        # TODO: raise a warning if the algorithm does not converge
        # TODO: look into regularization
        # TODO: get rid of comparison with ground truth
        # TODO: move SC_make to __init__()
        # TODO: make better graphics (deal with axes separation, etc.)
        # FIXME: tau_x in Matt origina script is calculated as the mean tau_x over sessions for each subject: why? Is this import?
        # TODO: check consistent N between object init and time series X passed to fit()
    
        if not(true_model is None):  # if true model is known
            true_C = true_model[0]  # used later to calculate error iteratively
            true_S = true_model[1]

        n_T = X.shape[0]  # number of time samples
        N = X.shape[1] # number of ROIs
        d_fit = dict()  # a dictionary to store the diagnostics of fit
        
        # mask for existing connections for EC and Sigma
        mask_diag = np.eye(N,dtype=bool)
        if SC_mask is None:
            mask_EC = np.logical_not(mask_diag) # all possible connections except self
        else:
            mask_EC = SC_mask
        mask_Sigma = np.eye(N,dtype=bool) # independent noise
        #mask_Sigma = np.ones([N,N],dtype=bool) # coloured noise
            
        if method=='lyapunov':
            
            if verbose>0:
                print('regularization:', regul_EC, ';', regul_Sigma)
            
            n_tau = 3 # number of time shifts for FC_emp
            v_tau = np.arange(n_tau)
            i_tau_opt = 1 # time shift for optimization
            
            min_val_Sigma_diag = 0. # minimal value for Sigma
          
            # FC matrix 
            ts_emp = X - np.outer(np.ones(n_T), X.mean(0))
            FC_emp = np.zeros([n_tau,N,N])
            for i_tau in range(n_tau):
                FC_emp[i_tau,:,:] = np.tensordot(ts_emp[0:n_T-n_tau,:],ts_emp[i_tau:n_T-n_tau+i_tau,:],axes=(0,0)) / float(n_T-n_tau-1)
        
            # normalize covariances (to ensure the system does not explode)
            if norm_fc is None:
                norm_fc = FC_emp[0,:,:].mean()
            FC_emp *= 0.5/norm_fc
            if verbose>0:
                print('max FC value (most of the distribution should be between 0 and 1):', FC_emp.mean())
        
            # autocovariance time constant
            log_ac = np.log(np.maximum(FC_emp.diagonal(axis1=1,axis2=2),1e-10))
            lin_reg = np.polyfit(np.repeat(v_tau,N),log_ac.reshape(-1),1)
            tau_x = -1./lin_reg[0]
            if verbose>0:
                print('inverse of negative slope (time constant):', tau_x)
            
            # optimization
            if verbose>0:
                print('*opt*')
                print('i tau opt:', i_tau_opt)
            tau = v_tau[i_tau_opt]
            
            # objective FC matrices (empirical)
            FC0_obj = FC_emp[0,:,:]
            FCtau_obj = FC_emp[i_tau_opt,:,:]
            
            coef_0 = np.sqrt(np.sum(FCtau_obj**2)) / (np.sqrt(np.sum(FC0_obj**2))+np.sqrt(np.sum(FCtau_obj**2)))
            coef_tau = 1. - coef_0
            
            # initial network parameters
            EC = np.zeros([N,N])
            Sigma = np.eye(N)  # initial noise
            
            # best distance between model and empirical data
            best_dist = 1e10
            best_Pearson = 0.
            
            # record model parameters and outputs
            dist_FC_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
            Pearson_FC_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
            dist_EC_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
            Pearson_EC_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
            
            stop_opt = False
            i_opt = 0
            while not stop_opt:
                # calculate Jacobian of dynamical system
                J = -np.eye(N)/tau_x + EC
                		
                # calculate FC0 and FCtau for model
                FC0 = spl.solve_lyapunov(J,-Sigma)
                FCtau = np.dot(FC0,spl.expm(J.T*tau))
                
                # calculate error between model and empirical data for FC0 and FC_tau (matrix distance)
                err_FC0 = np.sqrt(np.sum((FC0-FC0_obj)**2))/np.sqrt(np.sum(FC0_obj**2))
                err_FCtau = np.sqrt(np.sum((FCtau-FCtau_obj)**2))/np.sqrt(np.sum(FCtau_obj**2))
                dist_FC_hist[i_opt] = 0.5*(err_FC0+err_FCtau)
                if not(true_model is None):
                    dist_EC_hist[i_opt] = np.sqrt(np.sum((EC-true_C)**2))/np.sqrt(np.sum(true_C**2))
                	
                # calculate Pearson corr between model and empirical data for FC0 and FC_tau
                Pearson_FC_hist[i_opt] = 0.5*(stt.pearsonr(FC0.reshape(-1),FC0_obj.reshape(-1))[0]+stt.pearsonr(FCtau.reshape(-1),FCtau_obj.reshape(-1))[0])
                if not(true_model is None):
                    Pearson_EC_hist[i_opt] = stt.pearsonr(EC.reshape(-1), true_C.reshape(-1))[0]
                
                # best fit given by best Pearson correlation coefficient for both FC0 and FCtau (better than matrix distance)
                if dist_FC_hist[i_opt]<best_dist:
                    	best_dist = dist_FC_hist[i_opt]
                    	best_Pearson = Pearson_FC_hist[i_opt]
                    	i_best = i_opt
                    	EC_best = np.array(EC)
                    	Sigma_best = np.array(Sigma)
                    	FC0_best = np.array(FC0)
                    	FCtau_best = np.array(FCtau)
                else:
                    stop_opt = i_opt>100
                
                # Jacobian update with weighted FC updates depending on respective error
                Delta_FC0 = (FC0_obj-FC0)*coef_0
                Delta_FCtau = (FCtau_obj-FCtau)*coef_tau
                Delta_J = np.dot(np.linalg.pinv(FC0),Delta_FC0+np.dot(Delta_FCtau,spl.expm(-J.T*tau))).T/tau
                # update conectivity and noise
                EC[mask_EC] += epsilon_EC * (Delta_J - regul_EC*EC)[mask_EC]
                EC[mask_EC] = np.clip(EC[mask_EC],min_val_EC,max_val_EC)
                
                Sigma[mask_Sigma] += epsilon_Sigma * (-np.dot(J,Delta_FC0)-np.dot(Delta_FC0,J.T) - regul_Sigma)[mask_Sigma]
                Sigma[mask_diag] = np.maximum(Sigma[mask_diag],min_val_Sigma_diag)
                
                # check if end optimization: if FC error becomes too large
                if stop_opt or i_opt==n_opt-1:
                    stop_opt = True
                    d_fit['iterations'] = i_opt
                    d_fit['distance'] = best_dist
                    d_fit['correlation'] = best_Pearson
                    if verbose>0:
                        print('stop at step', i_opt, 'with best dist', best_dist, ';best FC Pearson:', best_Pearson)
                else:
                    if (i_opt)%20==0 and verbose>0:
                        print('opt step:', i_opt)
                        print('current dist FC:', dist_FC_hist[i_opt], '; current Pearson FC:', Pearson_FC_hist[i_opt])
                    i_opt += 1
                
            if verbose>1:
                # plots
                
                mask_nodiag = np.logical_not(np.eye(N,dtype=bool))
                mask_nodiag_and_not_EC = np.logical_and(mask_nodiag,np.logical_not(mask_EC))
                mask_nodiag_and_EC = np.logical_and(mask_nodiag,mask_EC)
                
                pp.figure()
                gs = GridSpec(2, 3)
                    
                if not(true_model is None):
                    pp.subplot(gs[0,2])
                    pp.scatter(true_C, EC_best, marker='x')
                    pp.xlabel('original EC')
                    pp.ylabel('estimated EC')
                    pp.text(pp.xlim()[0]+.05, pp.ylim()[1]-.05,
                             r'$\rho$: ' + str(stt.pearsonr(true_C[mask_EC], EC_best[mask_EC])[0]))
                    pp.subplot(gs[1,2])
                    pp.scatter(true_S, Sigma_best,marker='x')
                    pp.xlabel('original Sigma')
                    pp.ylabel('estimated Sigma')
                    pp.text(pp.xlim()[0]+.05, pp.ylim()[1]-.05,
                             r'$\rho_{diag}$: ' + str(stt.pearsonr(true_S.diagonal(), Sigma_best.diagonal())[0])
                             + r'$\rho_{off-diag}$: ' + str(stt.pearsonr(true_S[mask_nodiag], Sigma_best[mask_nodiag])[0])
                             )
                
                    
                pp.subplot(gs[0,0:2])
                pp.plot(range(n_opt),dist_FC_hist, label='distance FC')
                pp.plot(range(n_opt),Pearson_FC_hist, label=r'$\rho$ FC')
                if not(true_model is None):
                    pp.plot(range(n_opt),dist_EC_hist, label='distance EC')
                    pp.plot(range(n_opt),Pearson_EC_hist, label=r'$\rho$ EC')
                pp.legend()
                pp.xlabel('optimization step')
                pp.ylabel('FC error')
                
                    
                pp.subplot(gs[1,0])
                pp.scatter(FC0_obj[mask_nodiag_and_not_EC], FC0_best[mask_nodiag_and_not_EC], marker='x', color='k', label='not(SC)')
                pp.scatter(FC0_obj[mask_nodiag_and_EC], FC0_best[mask_nodiag_and_EC], marker='.', color='b', label='SC')
                pp.scatter(FC0_obj.diagonal(), FC0_best.diagonal(), marker= '.', color='c', label='diagonal')
                pp.legend()
                pp.xlabel('FC0 emp')
                pp.ylabel('FC0 model')
                
                
                pp.subplot(gs[1,1])
                pp.scatter(FCtau_obj[mask_nodiag_and_not_EC], FCtau_best[mask_nodiag_and_not_EC], marker='x', color='k', label='not(SC)')
                pp.scatter(FCtau_obj[mask_nodiag_and_EC], FCtau_best[mask_nodiag_and_EC], marker='.', color='b', label='SC')
                pp.scatter(FCtau_obj.diagonal(), FCtau_best.diagonal(), marker= '.', color='c', label='diagonal')
                pp.xlabel('FCtau emp')
                pp.ylabel('FCtau model')

        elif method=='moments':
            n_tau = 2
            ts_emp = X - np.outer(np.ones(n_T), X.mean(0))  # subtract mean
            # empirical covariance (0 and 1 lagged)
            Q_emp = np.zeros([n_tau, self.n_nodes, self.n_nodes])
            for i_tau in range(n_tau):
                Q_emp[i_tau, :, :] = np.tensordot(ts_emp[0:n_T-n_tau,:],ts_emp[i_tau:n_T-n_tau+i_tau,:],axes=(0,0)) / float(n_T-n_tau-1)
            # Jacobian estimate
            J = spl.logm(np.dot(np.linalg.inv(Q_emp[0, :, :]), Q_emp[1, :, :])).T  # WARNING: tau is 1 here (if a different one is used C gets divided by tau)
            if np.any(np.iscomplex(J)):
                J = np.real(J)
                print("Warning: complex values in J; casting to real!")
            # Sigma estimate
            Sigma_best = -np.dot(J, Q_emp[0, :, :])-np.dot(Q_emp[0, :, :], J.T)
            # theoretical covariance
            Q0 = spl.solve_lyapunov(J, -Sigma_best)
            # theoretical 1-lagged covariance
            Qtau = np.dot(Q0, spl.expm(J.T))  # WARNING: tau is 1 here (if a different one is used J.T gets multiplied by tau)
            # average correlation between empirical and theoretical
            d_fit['correlation'] = 0.5 * (stt.pearsonr(Q0.flatten(), Q_emp[0, :, :].flatten())[0] +
                                         stt.pearsonr(Qtau.flatten(), Q_emp[1, :, :].flatten())[0])
            tau_x = -J.diagonal().copy()
            EC_best = np.zeros([N, N])
            EC_best[mask_EC] = J[mask_EC]
            
        elif method=='bayes':
            X = X.T  # here shape is [ROIs, timepoints] to be consistent with Singh paper
            N = X.shape[1]  # number of time steps
            X -= np.outer(X.mean(axis=1), np.ones(N))  # center the time series
            T1 = [np.dot(X[:, i+1:i+2], X[:, i+1:i+2].T) for i in range(N-1)]
            T1 = np.sum(T1, axis=0)
            T2 = [np.dot(X[:, i+1:i+2], X[:, i:i+1].T) for i in range(N-1)]
            T2 = np.sum(T2, axis=0)
            T3 = [np.dot(X[:, i:i+1], X[:, i:i+1].T) for i in range(N-1)]
            T3 = np.sum(T3, axis=0)
        #    T4 = np.dot(X[:, 0:1], X[:, 0:1].T)  # this is actually not used
            LAM_best = np.dot(T2, np.linalg.inv(T3))
            # Kappa_best can be useful for generating samples using (called Sigma in Singh paper)
            # x_(n+1) = dot(LAM, x_n) + dot(sqrt(Kappa_best), Xi_n)
            Kappa_best = (T1 - np.dot(np.dot(T2, np.linalg.inv(T3)), T2.T)) / N
            # J is -lambda in Singh paper
            # WARNING: tau is 1 here (if a different one is used J.T gets multiplied by tau)
            J = spl.logm(LAM_best)
            if not np.all(np.isclose(spl.expm(J), LAM_best, rtol=1e-01)):
                print("Warning: logarithm!")
            if np.any(np.iscomplex(J)):
                J = np.real(J)
                print("Warning: complex values in J; casting to real!")
            # TODO: implement bayes I for Q0 (called c in Singh paper)
            Q0 = T3 / N  # this here is bayes II solution (equivalent to sample covariance)
            Sigma_best = -np.dot(J, Q0)-np.dot(Q0, J.T)
            tau_x = J.diagonal()
            np.fill_diagonal(J, 0)
            EC_best = J
            
        else:
            raise('method should be either \'lyapunov\' or \'moments\'')

        self.C = EC_best
        self.Sigma = Sigma_best
        self.tau_x = tau_x
        self.d_fit = d_fit

        return self

    def score(self):
        try:
            return self.d_fit['correlation']
        except:
            print('the model has not been fit yet. Call the fit method first.')
            
    def model_covariance(self, tau=0):
        """
        Calculates theoretical (lagged) covariances of the model given the parameters (forward step).
        Notice that this is not the empirical covariance matrix as estimated from simulated time series.
        PARAMETERS:
            tau : the lag to calculate the covariance
        RETURNS:
            FC : the (lagged) covariance matrix.
        """
        J = -np.eye(self.n_nodes)/self.tau_x + self.C
        # calculate FC0 and FCtau for model
        FC0 = spl.solve_lyapunov(J, -self.Sigma)
        if tau==0:
            return FC0
        else:
            FCtau = np.dot(FC0, spl.expm(J.T * tau))
            return FCtau

    def simulate(self, T=9000, dt=0.05, verbose=0, random_state=None):
        """
        Simulate the model with simple Euler integration.
        -----------
        PARAMETERS:
        T : duration of simulation
        dt : integration time step
        -----------
        RETURNS:
        ts : time series of simulated network activity of shape [T, n_nodes]
        -----------
        NOTES:
        it is possible to include an acitvation function to
        give non linear effect of network input; here assumed to be identity

        """
        if random_state is not None:  # set seed for reproducibility
            np.random.seed(random_state)
        T0 = 100.  # initialization time for network dynamics
        n_sampl = int(1./dt)  # sampling to get 1 point every second
        n_T = int(np.ceil(T/dt))
        n_T0 = int(T0/dt)
        ts = np.zeros([n_T, self.n_nodes])  # to save results
        # initialization
        t_span = np.arange(n_T0 + n_T, dtype=int)
        x_tmp = np.random.rand(self.n_nodes)  # initial activity of nodes
        u_tmp = np.zeros([self.n_nodes])
        # sample all noise terms at ones
        noise = np.random.normal(size=[n_T0 + n_T, self.n_nodes], scale=(dt**0.5))
        # numerical simulations
        for t in t_span:
            u_tmp = np.dot(self.C, x_tmp) + self.mu
            x_tmp += dt * (-x_tmp / self.tau_x + u_tmp) + np.dot(self.Sigma, noise[t, :])
            if t > (n_T0-1):  # discard first n_T0 timepoints
                ts[t-n_T0, :] = x_tmp

        # subsample timeseries to approx match fMRI time resolution
        return ts[::n_sampl, :]


def make_rnd_connectivity(N, density=0.2, connectivity_strength=0.5):
    """
    Creates a random connnectivity matrix as the element-wise product $ C' = A \otimes W$,
    where A is a binary adjacency matrix samples from Bern(density) and W is sampled from log-normal
    $log(W) \sim \mathcal{N}(0,1)$.
    The matrix gets normalized in order to avoid explosion of activity when varying the number of nodes N.
    $ C = \frac{C' N}{\sum_{i,j} C'} $
    """
    C = np.exp(np.random.randn(N, N))  # log normal
    C[np.random.rand(N, N) > density] = 0
    C[np.eye(N, dtype=bool)] = 0
    C *= connectivity_strength * N / C.sum()
    return C


def classfy(X, y, zscore=False, pca=False):
    """
    Classify in X according to labels in y.
    PARAMETERS:
        X : data matrix shape [N, P] with samples on the rows
        y : labels shape [N]
        zscore : Bool turns z-score on or off
        PCA : Bool turns PCA on or off
    """
    # classifier instantiation
    clf = LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
    # corresponding pipeline: zscore and pca can be easily turned on or off
    if zscore:
        z = ('zscore', StandardScaler())
    else:
        z = ('identity_zscore', FunctionTransformer())
    if pca:
        p = ('PCA', PCA())
    else:
        p = ('identity_pca', FunctionTransformer())
    pipe = Pipeline([z, p, ('clf', clf)])
    repetitions = 100  # number of times the train/test split is repeated
    # shuffle splits for validation test accuracy
    shS = ShuffleSplit(n_splits=repetitions, test_size=None, train_size=.8, random_state=0)
    score = np.zeros([repetitions])
    i = 0  # counter for repetitions
    for train_idx, test_idx in shS.split(X):  # repetitions loop
        data_train = X[train_idx, :]
        y_train = y[train_idx]
        data_test = X[test_idx, :]
        y_test = y[test_idx]
        pipe.fit(data_train, y_train)
        score[i] = pipe.score(data_test, y_test)
        i += 1
    return score
