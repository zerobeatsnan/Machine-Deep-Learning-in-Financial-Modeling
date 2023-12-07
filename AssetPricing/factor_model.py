import numpy as np
import pandas as pd
from copy import copy
import statsmodels.api as sm
import gc
from tqdm import trange
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from tqdm import trange
import sklearn.metrics.pairwise as kernel
from copy import copy
import statsmodels.api as sm
import gc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import math
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore', message = 'Unused variable')

#=====================================================================================
# cross sectional ridge regression
#=====================================================================================

class ridge():
    def __init__(self, characteristics, ret):
        
        self.date_list = list(characteristics.keys())
                
        self.T = len(self.date_list)        
    
        self.stock_list = list(ret.columns)
        self.Nt = dict(zip(
            self.date_list,
            [characteristics[t].shape[0] for t in self.date_list]
        ))
        self.chanames = list(characteristics[self.date_list[0]].columns)
        self.l = len(self.chanames)
        
        self.xt = dict(zip(
            self.date_list,
            [
                characteristics[t].T.dot(ret.loc[t, characteristics[t].index]).values.reshape(self.l, 1)/self.Nt[t]
                for t in self.date_list
            ]
        ))

        self.zt_square=dict(zip(
            self.date_list,
            [characteristics[i].T.dot(characteristics[i]).values for i in self.date_list]
        ))
        
        gc.collect()

    def fit(self, lamb):
        self.lamb=lamb
        
        self.factor = pd.DataFrame(np.nan, columns = self.chanames, index = self.date_list)

        for t in trange(len(self.date_list)):
            key = self.date_list[t]
            try:
                self.factor.loc[key, :] = self.Nt[key]*np.linalg.inv(self.zt_square[key]+self.lamb*self.Nt[key]*np.eye(self.l)).dot(self.xt[key]).reshape(self.l,)
            except:
                pass

        def calc_NW_tvalue(factor_value):
            tsmod = sm.OLS(endog = factor_value, exog = np.ones(self.T), hasconst = False)\
                .fit(cov_type = 'HAC', cov_kwds = {'maxlags': 6})
            return tsmod.tvalues.values[0]
        
        self.mu_hat = pd.DataFrame(zip(\
                    [np.mean(self.factor[i]) for i in self.chanames],
                    [calc_NW_tvalue(self.factor[i]) for i in self.chanames]
                ), columns = ['risk premium', 't value'],\
               index = self.chanames).fillna(0)

    def predict(self, Z_t, mu_hat=None):
        if mu_hat is None:
            mu_hat = self.mu_hat.values[:, 0]
        if isinstance(Z_t, pd.DataFrame):
            Z_t = Z_t.values
        r_hat = Z_t.dot(mu_hat).reshape(Z_t.shape[0],)
        return r_hat

#=====================================================================================
# FM-LARS
#=====================================================================================

class FMLARS():
    def __init__(self,characteristics,ret):
        
        self.date_list = list(characteristics.keys())
                
        self.T = len(self.date_list)        
    
        self.stock_list = list(ret.columns)
        self.Nt = dict(zip(
            self.date_list,
            [characteristics[t].shape[0] for t in self.date_list]
        ))
        self.chanames = list(characteristics[self.date_list[0]].columns)
        self.l = len(self.chanames)
        
        self.xt = dict(zip(
            self.date_list,
            [
                characteristics[t].T.dot(ret.loc[t, characteristics[t].index]).values.reshape(self.l, 1)/self.Nt[t]
                for t in self.date_list
            ]
        ))

        self.zt_square=dict(zip(
                self.date_list,
                [characteristics[i].T.dot(characteristics[i]).values for i in self.date_list]
            ))
        
        gc.collect()

    def fit(self,phi2,lamb=0):
        
        def Icrp(Ak,t,lamb):
            icrp = np.zeros((self.l, self.l))
            k = len(Ak)
            
            try:
                icrp[np.ix_(Ak, Ak)] = np.linalg.inv(self.zt_square[t][np.ix_(Ak, Ak)]+lamb*self.Nt[t]*np.eye(k, k))
#                 icrp[np.ix_(Ak,Ak)] = np.linalg.inv(self.zt_square[t][np.ix_(Ak,Ak)])
            except:
                icrp[np.ix_(Ak, Ak)] = np.linalg.pinv(self.zt_square[t][np.ix_(Ak, Ak)]+lamb*self.Nt[t]*np.eye(k, k), 1e-100)
   
            return icrp

        def calc_gamma_t(Ak,proding_list,t,lamb):
            gamma_t = self.Nt[t]*Icrp(Ak, t, lamb).dot(proding_list[t]).dot(self.xt[t])
            return gamma_t

        def solve_alpha(coef2,coef1,cons):
            if coef2 == 0:
                return np.nan
            else:
                if 4*coef1**2-4*coef2*cons > 0:
                    x1=(-2*coef1 + np.sqrt(4*coef1**2 - 4*coef2*cons))/(2*coef2)
                    x2=(-2*coef1 - np.sqrt(4*coef1**2 - 4*coef2*cons))/(2*coef2)
                    if 0<x1<1:
                        return(float(x1))
                    else:
                        return(float(x2))
                else:
                    return np.nan

        def calc_NW_tvalue(factor_value):
            tsmod = sm.OLS(endog = factor_value, exog = np.ones(self.T), hasconst = False)\
                .fit(cov_type = 'HAC', cov_kwds = {'maxlags': 6})
            return tsmod.tvalues.values[0]

        self.lamb = lamb
        #k=0
        self.Ak_list = []
        self.alpha_list = []
        
        self.Ak_list.append(pd.DataFrame(
                    sum([(self.Nt[t]*self.xt[t])**2 for t in self.date_list])\
                    +phi2*sum([self.Nt[t]*self.xt[t] for t in self.date_list])**2
                ).idxmax().iloc[0])

        proding_list = dict(zip(
            self.date_list,
            [np.eye(self.l) for t in self.date_list]
        ))

        #k>0
        print('Estimating FM-LARS')
        for k in trange(self.l):
            try:
                Zr = []
                Zsq_gamma = []
                Ak = self.Ak_list[0:k+1]
                Ak_1 = self.Ak_list[0:k]
                for t in self.date_list:
                    if k>0:
                        proding_list[t] = proding_list[t]\
                        .dot(np.eye(self.l) - self.alpha_list[k-1]*self.zt_square[t].dot(Icrp(Ak_1, t, self.lamb)))
                gamma_t_list=dict(zip(
                    self.date_list,
                    [calc_gamma_t(Ak, proding_list, t, self.lamb) for t in self.date_list]
                ))

                for t in self.date_list:
                    Zr.append(self.Nt[t]*proding_list[t].dot(self.xt[t]))
                    Zsq_gamma.append(self.zt_square[t].dot(gamma_t_list[t]))

                coef2 = sum([Zg**2 for Zg in Zsq_gamma])\
                        +phi2*sum(Zsq_gamma)**2
                coef2 = coef2-coef2[self.Ak_list[0]]
                coef1 = -sum([zr * Zg for (zr, Zg) in zip(Zr,Zsq_gamma)])\
                        -phi2*sum(Zr)*sum(Zsq_gamma)                
                coef1 = coef1 - coef1[self.Ak_list[0]]
                cons = sum([zr**2 for zr in Zr]) + phi2*sum(Zr)**2                
                cons = cons - cons[self.Ak_list[0]]
                
                solving_alpha_list = np.array([solve_alpha(c2, c1, c) for (c2, c1, c) in zip(coef2, coef1, cons)])
                # solving_alpha_list[Ak] = np.nan
                solving_alpha_list[Ak] = 1

                if k < self.l-1:
                    j_ = pd.DataFrame(solving_alpha_list).idxmin().iloc[0]
                    alpha_k = solving_alpha_list[j_]
                else:
                    j_ = Ak[len(Ak)-1]
                    alpha_k = 1
                self.Ak_list.append(j_)
                self.alpha_list.append(alpha_k)

                factor_k = pd.DataFrame(np.nan, index = self.date_list, columns = self.chanames)
                for t in self.date_list:
                    factor_k.loc[t] = gamma_t_list[t].reshape(self.l,)
                if k == 0:
                    self.factor_list = [self.alpha_list[k]*factor_k]
                else:
                    self.factor_list.append(self.factor_list[k-1] + self.alpha_list[k]*factor_k)
            except:
                break
        
        self.maxk = len(self.factor_list)

        def get_mu_hat(k):
            mu_hat = pd.DataFrame(zip(\
                    [np.mean(self.factor_list[k][i]) for i in self.chanames],
                    [calc_NW_tvalue(self.factor_list[k][i]) for i in self.chanames]
                        ), columns = ['risk premium', 't value'],\
                       index = self.chanames).fillna(0)
            mu_hat = mu_hat[mu_hat.iloc[:, 1] != 0]
            return(mu_hat)
        
        self.mu_hat = [get_mu_hat(k) for k in range(self.maxk)]

    def predict(self, Z_t, maxk = None):
        if maxk == None:
            maxk = self.maxk
        prediction = []
        for k in range(maxk):
            r_hat = Z_t.loc[:, self.mu_hat[k].index].dot(self.mu_hat[k].iloc[:, 0].values)
            prediction.append(r_hat)
        return prediction

#=====================================================================================
# IPCA
#=====================================================================================

class IPCA():
    def __init__(self,characteristics,ret):
        
        self.date_list = list(characteristics.keys())
        self.Nt = dict(zip(
            self.date_list,
            [characteristics[i].shape[0] for i in self.date_list]
        ))
        self.chanames = list(characteristics[self.date_list[0]].columns)
        self.l = len(self.chanames)
        
        self.xt = pd.DataFrame(np.nan, index = self.date_list, columns = self.chanames)
        for t in self.date_list:
            self.xt.loc[t] = characteristics[t].T.dot(ret.loc[t, characteristics[t].index])

        self.zt_square = dict(zip(
            self.date_list,
            [characteristics[i].T.dot(characteristics[i]) for i in self.date_list]
        ))
        
        gc.collect()

    def fit(self, k, maxT=1000, tol=1e-3):
        self.k = k
        eigvalue, eigvector = np.linalg.eig(self.xt.T.dot(self.xt))
        self.gamma = eigvector[:, :self.k]
        
        def calc_ft(zt_square, xt, gamma):
            gamma_zts = gamma.T.dot(zt_square).dot(gamma)
            if np.linalg.det(gamma_zts) > 0:
                ft = np.linalg.inv(gamma_zts).dot(gamma.T).dot(xt)
                return ft
            else:
                pass

        def calc_denomt(ft, zt_square):
            return np.kron(ft.dot(ft.T), zt_square)

        def calc_numert(ft, xt):
            return np.kron(ft, xt)

        def calc_gamma(ft):
            denom = np.zeros((self.k*self.l, self.k*self.l))
            numer = np.zeros((self.k*self.l, 1))
            for t in self.date_list:
                ft_ = ft.loc[t].values.reshape(self.k, 1)
                if ft.loc[t].sum() != 0:
                    denom = denom+calc_denomt(ft_, self.zt_square[t])
                    numer = numer+calc_numert(ft_, self.xt.loc[t].values.reshape(self.l, 1))
            if np.linalg.det(denom) > 0:
                gamma = np.linalg.inv(denom).dot(numer)
            else:
                gamma = np.linalg.pinv(denom).dot(numer)
            gamma = gamma.reshape(self.k, self.l).T
            return gamma

        for i in range(maxT):
            self.ft = pd.DataFrame(np.nan, index = self.date_list, columns=['PC'+str(i) for i in range(1, self.k+1)])
            for t in self.date_list:
                self.ft.loc[t] = calc_ft(self.zt_square[t], self.xt.loc[t], self.gamma)

            gamma_ = copy(self.gamma)
            self.gamma = calc_gamma(self.ft)

            error = ((self.gamma-gamma_)**2).sum()
            if error <= tol:
                break
            else:
                print('round {}, error: {}'.format(i+1, error))

        Chol = np.linalg.cholesky(self.gamma.T.dot(self.gamma)).T
        fcov = self.ft.dropna().T.dot(self.ft.dropna())/self.ft.shape[0]
        eigvalue, Orth = np.linalg.eig(Chol.dot(fcov).dot(Chol.T))
        self.gamma = self.gamma.dot(np.linalg.inv(Chol)).dot(Orth)
        self.ft = self.ft.dot(Chol.T).dot(Orth)
        self.ft.columns = ['PC'+str(i) for i in range(1, self.k+1)]
        gc.collect()

    def predict(self,zt):
        mu = self.ft.mean(skipna=True).values.reshape(self.k, 1)
        return zt.dot(self.gamma).dot(mu)[0]

#==========================
# KERNEL IPCA
#==========================
class KERNEL_IPCA():
    def __init__(self,characteristics,ret, n_freedom = None, kernel_method = None, **kwargs):

        self.n_free = n_freedom
        self.kernel_met = kernel_method
        self.kernel_params = kwargs
        self.date_list=list(characteristics.keys())

        if self.n_free != None:

            data = pd.concat([characteristics[i] for i in self.date_list], axis = 0)

            char_array = data.values

            n_clusters = self.n_free

            kmeans = KMeans(n_clusters = n_clusters)

            kmeans.fit(char_array)

            print("clustering finish")

            self.centers = kmeans.cluster_centers_

            if self.kernel_met == "linear_kernel":

                kernel_use = kernel.linear_kernel
            
            elif self.kernel_met == "polynomial_kernel":

                kernel_use = kernel.polynomial_kernel

            elif self.kernel_met == "rbf_kernel":

                kernel_use = kernel.rbf_kernel

            elif self.kernel_met == "sigmoid_kernel":

                kernel_use = kernel.sigmoid_kernel


            _characteristics = copy(characteristics)

            for i in trange(len(self.date_list)):

                t = self.date_list[i]

                centers = self.centers
                index = _characteristics[t].index
                gram_train = kernel_use(_characteristics[t],centers, **self.kernel_params)
                gram_use = pd.DataFrame(gram_train,index = index, columns = [f"Feature{i}" for i in range(1,n_clusters + 1)])
                characteristics[t] = gram_use

        self.characteristics=copy(characteristics)
        self.ret=copy(ret)
        self.stock_list=list(self.ret.columns)
        self.Nt=dict(zip(
            self.date_list,
            [self.characteristics[i].shape[0] for i in self.date_list]
        ))
        self.chanames=list(self.characteristics[self.date_list[0]].columns)
        self.l=len(self.chanames)
        # self.kernel
        
        # for each t, it l*N_t * N_t * 1
        self.xt=pd.DataFrame(np.nan,index=self.date_list,columns=self.chanames)
        for t in self.date_list:
            self.xt.loc[t]=self.characteristics[t].T.dot(self.ret.loc[t,self.characteristics[t].index])
        # z_t square is l*N_t * N_t*l
        self.zt_square=dict(zip(
            self.date_list,
            [self.characteristics[i].T.dot(self.characteristics[i]) for i in self.date_list]
        ))
        self.error_ls = []
        
        gc.collect()
    
    def fit(self,k,maxT=1000,tol=1e-3):
        self.k=k
        eigvalue,eigvector=np.linalg.eig(self.xt.T.dot(self.xt))
        self.gamma=eigvector[:,:self.k]
        
        def calc_ft(zt_square,xt,gamma):
            gamma_zts=gamma.T.dot(zt_square).dot(gamma)
            if np.linalg.det(gamma_zts)>0:
                ft=np.linalg.inv(gamma_zts).dot(gamma.T).dot(xt)
                return ft
            else:
                pass


        def calc_error(characteristics, date_list, ret, ft, gamma, k):

            total_error = 0

            for t in date_list:

                x = characteristics[t].dot(gamma).dot(ft.loc["1963-01-31"].values.reshape(k,1)).values.squeeze()

                y = ret.loc[t, characteristics[t].index].values

                error = np.linalg.norm(x-y)**2

                total_error = total_error + error

            return total_error



        def calc_denomt(ft,zt_square):
            return np.kron(ft.dot(ft.T),zt_square)

        def calc_numert(ft,xt):
            return np.kron(ft,xt)

        def calc_gamma(ft):
            denom=np.zeros((self.k*self.l,self.k*self.l))
            numer=np.zeros((self.k*self.l,1))
            for t in self.date_list:
                ft_=ft.loc[t].values.reshape(self.k,1)
                if ft.loc[t].sum()!=0:
                    denom=denom+calc_denomt(ft_,self.zt_square[t])
                    numer=numer+calc_numert(ft_,self.xt.loc[t].values.reshape(self.l,1))
            if np.linalg.det(denom)>0:
                gamma=np.linalg.inv(denom).dot(numer)
            else:
                gamma=np.linalg.pinv(denom).dot(numer)
            gamma=gamma.reshape(self.k,self.l).T
            return gamma
        
        for i in trange(maxT):
            self.ft=pd.DataFrame(np.nan,index=self.date_list,columns=['PC'+str(i) for i in range(1,self.k+1)])
            for t in self.date_list:
                self.ft.loc[t]=calc_ft(self.zt_square[t],self.xt.loc[t],self.gamma)

            gamma_=copy(self.gamma)
            self.gamma=calc_gamma(self.ft)

            total_error = calc_error(self.characteristics, self.date_list, self.ret, self.ft, self.gamma, self.k)

            # print(f"The total error after round {i} is :", total_error)

            error=((self.gamma-gamma_)**2).sum()

            self.error_ls.append(total_error)
            if error<=tol:
                break
            # else:
            #     print('round {}, error: {}'.format(i+1,error))

        Chol=np.linalg.cholesky(self.gamma.T.dot(self.gamma)).T
        fcov=self.ft.dropna().T.dot(self.ft.dropna())/self.ft.shape[0]
        eigvalue,Orth=np.linalg.eig(Chol.dot(fcov).dot(Chol.T))
        self.gamma=self.gamma.dot(np.linalg.inv(Chol)).dot(Orth)
        self.ft=self.ft.dot(Chol.T).dot(Orth)
        self.ft.columns=['PC'+str(i) for i in range(1,self.k+1)]
        gc.collect()
        
    def predict(self,zt):
        mu=self.ft.mean(skipna=True).values.reshape(self.k,1)
        return zt.dot(self.gamma).dot(mu)[0]